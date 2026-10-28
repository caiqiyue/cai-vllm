from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id # blockid=0,表示 代表着 第0块 物理上的 kv cache block
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []
        """
        block id
        Block 对象：记录这个物理 Block 的管理信息。
        kv_cache[..., block_id, ...]：存放这个 Block 对应的实际 K/V 数值

        ref_count
        两个请求具有相同的前缀，命中了 Prefix Cache，因此可以共享某个已有的 Block

        hash 和 token_ids
        Block 对应的 Token 内容是 [10, 20, 30, 40]，其内容哈希为 12345。
        这些信息可以用于判断前缀是否匹配，从而辅助复用已有的 KV Cache
        """

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1 #重置，只是删除了block的管理 元信息， 物理block 没有删除
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size#256
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]#用于管理 物理层面上的 kv cache block
        self.hash_to_block_id: dict[int, int] = dict()
        """
        表示哈希值为 12345 的缓存内容对应 Block 2，哈希值为 67890 的缓存内容对应 Block 5。
        它通常用于 Prefix Cache（前缀缓存）：如果新请求的某个 Token 前缀对应的缓存 Block 已经存在，就有机会直接复用，不必重新计算这部分 Token 的 K/V。
        """
        self.free_block_ids: deque[int] = deque(range(num_blocks))#哪些block还空着
        self.used_block_ids: set[int] = set()#哪些block 已经被用了

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()# 从空闲队列 弹出一个 block id
        block = self.blocks[block_id]# 根据 block id 选出真实 block，注意block不是真正的物理kv cache ，是管理物理kv cache 的对象
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()#重置 block，block 的 引用计数器，从0 变为 1 表示 那个sequence 引用了 这个 block
        self.used_block_ids.add(block_id)#在 已使用block 集合中 加入 这个 block
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks# 0 
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):#num_cached_blocks 0  seq.num_blocks 1
            seq.block_table.append(self._allocate_block())#self._allocate_block() 返回的是 block id
        seq.num_cached_tokens = num_cached_blocks * self.block_size#0

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        """
        Prefix Cache 缓存的不一定只有最初的 Prompt，也可以包含 Prompt + Decode 阶段已经生成的 Token
        所以只对 存满的block 计算 hash，用于 prompt cacheß
        """
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
