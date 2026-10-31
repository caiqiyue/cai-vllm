from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs #最多2个样本
        self.max_num_batched_tokens = config.max_num_batched_tokens#一个batch 的 每个样本的token总和
        self.eos = config.eos#停止符号
        self.block_size = config.kvcache_block_size#每个样本 最长 256
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)#这个应该是管理 kv cache block 的 也就是那26个 36mb 的 kv cache 物理空间
        self.waiting: deque[Sequence] = deque()#管理请求的，也就是等待出来的 请求 query
        self.running: deque[Sequence] = deque()#正在处理的 请求 query

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:#等待队列中有请求（sequence对象），并且本batch的请求数少于 最大批次请求数
            seq = self.waiting[0]#每次都是从等待队列中取
            remaining = self.max_num_batched_tokens - num_batched_tokens# 512 代表一个 batch 所有样本token总和不能超过 512 
            if remaining == 0:
                break
            if not seq.block_table:#如果这个 sequence还没有被分配 block，就开始分配
                num_cached_blocks = self.block_manager.can_allocate(seq)# 返回 0 ，应该是判断是否可以给seq 分配kv cache block；返回 -1 应该表示 kv cache block 不够分了
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size # 12
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)#上面是判断可以不可以分配，现在应该是开始真正分配，分配的是 block 对象
            seq.num_scheduled_tokens = min(num_tokens, remaining)# 12 token-len
            num_batched_tokens += seq.num_scheduled_tokens #本批次的第一个样本有12个token
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING# 上面这个判断条件没有看懂，这里就是 修改样本的状态，从 waiting 变为 running
                self.waiting.popleft()#从等待队列中，弹出
                self.running.append(seq)#加入到 运行队列中
            scheduled_seqs.append(seq)#加入 管理队列中

        if scheduled_seqs:# 只在 prefill阶段 有返回
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()#从 运行队列中，弹出一个 sequence 对象  promptA+new-token1 + new-token2 + .....
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)#意思是在decode阶段，如果 生成token数超过了 block装载的最大数，就需要添加新的block
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):#2个sequence对象（被包住的样本） 和  新生成的 token 每个样本一个 token
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)#把新生成的 token 加入到 sequence对象中
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
