import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:
    
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config#配置
        hf_config = config.hf_config #模型结构
        self.block_size = config.kvcache_block_size #256 KV Cache 每个逻辑块能存储的 Token 数
        self.enforce_eager = config.enforce_eager#是否启用cuda计算图
        self.world_size = config.tensor_parallel_size #tensor并发的 进程数量
        self.rank = rank #当前进程在 进程组中的 编号
        self.event = event#用于同步的事件对象，具体用途取决于调用方， 现在进程数 是 1，events 是空列表

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        """
        NCCL：NVIDIA 的 GPU 通信库，常用于多 GPU 的数据传输和集合通信。

        init_process_group()：创建分布式通信进程组。

        torch.cuda.set_device(rank)：设置当前进程默认使用的 CUDA 设备
        """
        default_dtype = torch.get_default_dtype()#默认以 fp32
        torch.set_default_dtype(hf_config.dtype)#bf16
        torch.set_default_device("cuda")#默认使用cuda，gpu推理
        self.model = Qwen3ForCausalLM(hf_config)#这里怎么把模型结构给预定义好了？nano vllm 为了简单方便  
        load_model(self.model, config.model)#加载真实权重，前面属于 为每个gpu分配属于他们自己那一部分的权重空间，但不是真实权重值
        """
        hf_config：模型设计图纸。 记录隐藏层维度、Transformer 层数、Attention 头数等配置信息。

        Qwen3ForCausalLM(hf_config)：按照图纸搭建模型。 创建 Embedding、Attention、MLP、RMSNorm 等模块，并建立权重参数。

        load_model()：填入训练好的权重。 从模型文件读取权重，放到对应的模型参数中。
        """
        self.sampler = Sampler()#根据最后算出的 logits 中去挑，然后返回token，但具体是个什么策略，需要看一下（不是贪心）
        self.warmup_model()#意思是，虚构一个样本，跑一遍完整推理，激活cuda 的内核
        self.allocate_kv_cache()#预先分配kv cache空间
        if not self.enforce_eager:
            self.capture_cudagraph()
            """
            普通 Eager 执行时，CPU 需要不断发起 CUDA 操作。
                CUDA Graph 则可以提前记录一组 GPU 操作，后续对兼容的输入形状重复执行，减少 CPU 发起内核的开销。
                它主要优化执行调度开销，不是改变 Transformer 的数学计算公式
            """
        torch.set_default_device("cpu")#为啥又变回 cpu 了？
        torch.set_default_dtype(default_dtype)#fp32
        """
        因为模型和 KV Cache 等 GPU 对象已经创建好了，后面不一定需要让所有新建张量都默认放到 GPU。
            恢复默认设备可以减少后续代码意外创建 CUDA 张量的风险，也避免改变后续普通 PyTorch 代码的预期行为。
            注意：这不会把模型权重从 GPU 搬回 CPU
        """

        if self.world_size > 1:
            """
            - rank=0：创建约 1 MiB 的共享内存区域。
                - dist.barrier()：让所有参与的进程等待同步。
                - 其他 rank：打开同一块共享内存，进入 self.loop() 处理后续任务。
                这块共享内存主要用于进程间的数据交换，不是存放整个模型权重或 KV Cache
            """
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)#反射，获取model_runner 的 run函数
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()#empty_cache()：释放 PyTorch 缓存分配器中未被占用的显存，不会释放模型权重。
        torch.cuda.reset_peak_memory_stats()#reset_peak_memory_stats()：重置显存峰值统计，便于测量这次 Warmup 的显存使用峰值
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)#本次模拟样本的最大长度
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)#算出一个batch中最多几个样本
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)] #应该是为一个样本创建一个sequence对象
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len # 句子长度
        self.run(seqs, True)#true 代表 本次预热 是 用 prefill 阶段去预热
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()#free 28406133632  total 11344740352 单位是bytes free是gpu 还未用，total是gpu总共的显存
        used = total - free # 8504606720 已使用的量
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]#peak 8394689536。torch在 预热阶段记录的峰值
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]#current 8243397632 torch记录的当前使用量
        num_kv_heads = hf_config.num_key_value_heads // self.world_size# 8 
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)# 128
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize#用bf16存 ，37748736 一个逻辑 KV Cache Block，需要为 36 层的 K 和 V 总共分配 36 MiB 显存
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes   # 26 根据显存估算，去 算出 总共可以有 多少个 36mb 的 kv kvcache block
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)#[2, 36, 26, 256, 8, 128] 创建kv cache空间 2表示k和v两个，36表示36层，26 表示 可以有 26个 kv cache block，256代表每个block 能够存储的token数， 8代表 8个头，128 代表维度
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]# 26, 256, 8, 128 某一层的 k cache 空间 第一个维度 0 表示存k
                module.v_cache = self.kv_cache[1, layer_id]# 26, 256, 8, 128 某一层的 v cache 空间 第一个维度 1 表示存V
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):#warmup 阶段 seqs 是 【1个sqequence对象】
        input_ids = []#sequence 可以简单理解为 就是 在 样本input_id 外层有套了一层
        positions = []#和理解的位置 是一样的
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        """
        假设一个批次中有多个 样本，那么就需要标记能区分，不同样本的k 和 q 的边界
        告诉 Attention 计算内核，一个批次中每条请求的 Q 和 K 分别在哪里开始、在哪里结束。


        cu_seqlens_q = [0, 2, 5]
        cu_seqlens_k = [0, 5, 9]

        请求	Q 的范围	K 的范围
        A	    [0:2]	    [0:5]
        B	    [2:5]	    [5:9]
        """
        max_seqlen_q = 0# 记录本批次中 样本的q的最大长度
        max_seqlen_k = 0#记录本批次中，样本的k的最大长度
        slot_mapping = []
        """
        block_tables：查找映射

        告诉系统：某条 Sequence 的逻辑 Block 对应哪个物理 Block。

        例如：[5, 9] 表示逻辑 Block 0 对应物理 Block 5，逻辑 Block 1 对应物理 Block 9。


        slot_mapping：具体写入位置

            进一步记录本轮每个新 Token 对应的 K/V 缓存槽位。

            例如：[20, 21, 22] 表示本轮三个 Token 的 K/V 分别写入这些物理槽位。
        """
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens#0
            seqlen_q = seq.num_scheduled_tokens#512  第一个样本 12  第二样本 7
            end = start + seqlen_q#512
            seqlen_k = end# 512
            input_ids.extend(seq[start:end])#预热阶段 512   现在是真实跑 12个token 一开始 是 0 - 11 现在是 12 - 19
            positions.extend(range(start, end))#512 个位置信息。同理 从 0 - 11 再 0 - 6
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)#[0,512]。[0,12] 现在是 [0,12,19]
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)#[0,512]  [0,12]  [0,12,19]
            max_seqlen_q = max(seqlen_q, max_seqlen_q)#512
            max_seqlen_k = max(seqlen_k, max_seqlen_k)#512
            if not seq.block_table:    # warmup, 预热阶段不会做下面这段逻辑
                continue
            start_block = start // self.block_size #这2步 估计是在算如果一个 sequence token太长，要用几个block来装
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size# 0 这应该就是物理的 初始插入位置   第二个 样本的位置就是 从 256 开始
                if i == start_block:
                    slot_start += start % self.block_size #256
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size  # 初始 slot 是 256 ， prompt长度是7 结束位置是 263
                slot_mapping.extend(range(slot_start, slot_end))# 在 slot_mapping 中记录 [0-11,256-263]
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)#Python 列表转换成 PyTorch Tensor，再把 Tensor 从 CPU 传到 GPU，供后续模型推理使用
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)# input_ids 和 position 一样, 一个batch 2 个 样本，和在了一起
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        """
        slot_mapping 的意思是
        {0-11, 256-262} 意思就是 block0 的 0-11存promptA的kv；block1 的 0-6 存 promptB的kv，但是写256-262 记录的是全局 编号，因为全局编号，可以反推回自己存在哪个block中
        block0 ， block1，block2........
        0 - 255， 256-511， 512-783.......

        Block 是按整块分配、按 Token 槽位写入。一个请求可以使用多个 Block，但在你当前研究的 Nano-vLLM 中，不同请求通常不会共享一个 Block 的剩余空间
        一个batch 可以有多个 sequence
        一个 sequence 可以 分配多个 block
        一个 sequence 占一个 block ，即使这个block 没有满，也不会把它分配给其他block
        """
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)#组合成一个上下文
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]#每个样本的温度
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:#true  true false
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)#input_ids  512个0，positions 就是从0到511  在真实请求中，一个batch 2个请求，2个请求合并到了一起 所以现在 inputs_id 是19 position 是 19；在decode阶段 inputs_id 和 position_id 每个样本都上传一个数
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None#温度 目前为1，也就是什么都不变
        logits = self.run_model(input_ids, positions, is_prefill)#is_prefill 为 true，现在应该是 prefill 阶段. logits :1, 151936   run_model 用的是 torch 包里面的函数，有一个ctx工厂
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None #从 1，151936中选择一个logit，最后选择词表
        reset_context()#重制上下文？ 就是 新建一个 context 对象，但是 context是单例，所以新建一个对象，相当于 重置上下文
        return token_ids#返回本次选中的token  [362,21927] promptA 选择的next——token promptB 选择的第一个next——token

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
