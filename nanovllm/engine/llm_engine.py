import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)} # 可允许填入的所有配置字段
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields} #有值的 配置字段
        """
        config_kwargs:{'enforce_eager': True,  false 走的就是 cuda 计算图
        'tensor_parallel_size': 1,   张量并行使用的 GPU 数量。
        'max_model_len': 512,     允许输出的长度
        'max_num_batched_tokens': 512,   一批次中 所有 prompt 的token 总和
        'max_num_seqs': 2,     一个批次最多两个 请求  
        'gpu_memory_utilization': 0.85} ,  0.85 表示目标上限约为 GPU 总显存的 85%，留下约 15% 给：
                                                    - CUDA Context；
                                                    - 临时 Tensor；
                                                    - FlashAttention/Triton Workspace；
                                                    - NCCL；
                                                    - PyTorch 内存碎片；
                                                    - 其他进程。
        
        config_fields : {'tensor_parallel_size',
          'eos', 
          'kvcache_block_size', 
          'model', 
          'enforce_eager', 
          'max_num_seqs', 
          'hf_config', 
          'num_kvcache_blocks',
            'max_num_batched_tokens', 
            'gpu_memory_utilization', 
            'max_model_len'}
        """
        config = Config(model, **config_kwargs) #更新配置
        Sequence.block_size = config.kvcache_block_size #256  一个kv cache block 最多存储 256 个token
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")# mp 是 multiprocessing
        """
                        ```
                import multiprocessing as mpctx = mp.get_context("spawn")
                ```

                ### 1. 核心概念

                - `mp`：Python 的 `multiprocessing` 多进程模块，用于创建和管理子进程。
                - `ctx`：多进程上下文，决定使用什么方式创建子进程。
                - `spawn`：启动一个全新的 Python 解释器进程，执行指定任务。

                ### 2. 三种进程启动方式

                | 方式           | 原理              | 特点                  |
                | ------------ | --------------- | ------------------- |
                | `spawn`      | 启动新的 Python 解释器 | 相对安全，启动较慢           |
                | `fork`       | 复制父进程           | 启动快，但可能引发 CUDA、线程问题 |
                | `forkserver` | 由专门的服务进程创建子进程   | 减少多线程环境下直接 fork 的风险 |

                ### 3. 具体案例

                ```
                import multiprocessing as mpdef worker():    print("执行模型推理")if __name__ == "__main__":    ctx = mp.get_context("spawn")     # 选择 spawn 模式    p = ctx.Process(target=worker)    # 创建进程对象    p.start()                         # 启动子进程    p.join()                          # 等待子进程结束
                ```

                注意：`get_context()` 不会创建进程，`p.start()` 才会真正启动子进程。

                ### 4. 为什么 vLLM 等推理框架会使用 spawn？

                主要是为了避免 `fork` 复制已经初始化的 CUDA 运行时状态，从而导致 GPU 初始化异常、死锁等问题。

                一句话总结：`mp` 是多进程工具模块，`ctx` 是指定启动方式的上下文，`spawn` 是启动全新 Python 进程的一种方式。
        """
        for i in range(1, config.tensor_parallel_size): #允许并发数为1，说明就是没有并发，self.events就是一个空列表
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)#运行模型的
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)#词表，151643
        config.eos = self.tokenizer.eos_token_id    #151645 停止符号 的token id
        self.scheduler = Scheduler(config)#主要是对block进行 管理 ，block 并不是物理上的 kv cache block ，而是对 物理block 管理的 管理信息block类
        atexit.register(self.exit)#程序正常退出时，记得调用 self.exit()，做一些收尾工作。

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)#一个列表 12个 token
        seq = Sequence(prompt, sampling_params)# 创建一个sequence 对象包裹 tokenid，
        self.scheduler.add(seq)#加入到计划类的 等待 队列中，进行管理

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()#分配block ，请求都在 waiting 队列的时候，就prefill；如果都在running 队列的时候就 decode 阶段
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)#该批次的所有样本的token总和
        token_ids = self.model_runner.call("run", seqs, is_prefill)#【362，21297】 2个样本，每个样本选择的下一个token
        self.scheduler.postprocess(seqs, token_ids, is_prefill)#后处理，第一：如果block 存满了就依据token内容做hash，为了prefix cache；第二，sequence 中加入新token
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]#outputs不记录每次新生成的token，只在prompt最终生成完所有token之后才输出
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],#采样参数 温度0.6 最大token数16，要注意eos
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)#列表中，放多个 samplingparams 对象，方便zip使用
        for prompt, sp in zip(prompts, sampling_params):#'Explain what a KV cache is in one short sentence.', 'Say hello in one short sentence.'
            self.add_request(prompt, sp)#内部就是 变成sequence 对象，加入schedular 的 管理队列中
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()#3953217.747942597
            output, num_tokens = self.step()#output=[] num_tokens 19
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)#0.2830366316117231
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)#0.0  0.150258320766
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        """
        outputs
        1 =
        [362, 84648, 6500, 10533, 1376, 19083, 13530, 504, 3681, 3950, 19898, 311, 4628, 705, 17420, 3950]
        2 =
        [21927, 0, 26525, 232, 271, 4340, 646, 358, 7789, 498, 3351, 30, 26525, 232, 271, 3838]

        最终输出

        0 =
        {'text': ' A KV cache stores key-value pairs from previous sequences in a model to speed up', 'token_ids': [362, 84648, 6500, 10533, 1376, 19083, 13530, 504, 3681, 23700, 304, 264, 1614, 311, 4628, 705]}
        1 =
        {'text': ' Hello! 😊\n\nCan you help me with a 500-word', 'token_ids': [21927, 0, 26525, 232, 271, 6713, 498, 1492, 752, 448, 264, 220, 20, 15, 15, 37328]}
        """
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
