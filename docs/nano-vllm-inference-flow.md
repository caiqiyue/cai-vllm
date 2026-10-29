# nano-vLLM 推理全流程与关键组件机制

> **对应源码版本**：`cai-vllm` 仓库 `main` 分支，基准提交 `5bbbfa8`（`torch-fallback-smoke-scripts-debug`）。
> **覆盖范围**：`nanovllm/` 全部 20 个 Python 源文件所构成的完整推理链路。
> **性质**：学习笔记。文中所有实现性结论均给出源码位置；无法在本机复现的部分统一标注为「待验证」或「推测」。

---

## 0. 如何使用本文

本文回答一个问题：**一次 `LLM.generate()` 调用到底发生了什么，经过了哪些组件，每个组件的机制是什么。**

阅读顺序建议：

1. 先看 [第 1 章 全景](#1-全景一次-generate-调用链)，建立组件地图；
2. 再按 [第 3 章 阶段拆解](#3-阶段拆解一次推理的十个阶段) 建立时序感；
3. 最后按需钻 [第 4 章 核心机制详解](#4-核心机制详解)，每一节对应一个组件，可独立阅读。

文中出现的关键组件与源码的对应关系：

| 组件 | 源码位置 | 一句话职责 |
| --- | --- | --- |
| `LLM` | [`nanovllm/llm.py`](../nanovllm/llm.py) | 公开 API 外壳，继承 `LLMEngine`，自身无逻辑 |
| `LLMEngine` | [`nanovllm/engine/llm_engine.py`](../nanovllm/engine/llm_engine.py) | 引擎主循环：装配置、拉起模型、收请求、驱动 step |
| `Config` | [`nanovllm/config.py`](../nanovllm/config.py) | 运行期参数 + 读取 HF 模型结构配置 |
| `Scheduler` | [`nanovllm/engine/scheduler.py`](../nanovllm/engine/scheduler.py) | 决定「这一轮跑哪些序列、各跑多少 token」 |
| `Sequence` | [`nanovllm/engine/sequence.py`](../nanovllm/engine/sequence.py) | 单个请求的全生命周期状态 |
| `BlockManager` | [`nanovllm/engine/block_manager.py`](../nanovllm/engine/block_manager.py) | 物理块池管理 + 前缀缓存哈希表 |
| `ModelRunner` | [`nanovllm/engine/model_runner.py`](../nanovllm/engine/model_runner.py) | 把「这一轮」变成一个 batch 的张量并驱动前向 |
| `Context` | [`nanovllm/utils/context.py`](../nanovllm/utils/context.py) | 全局单例，承载单次前向的元数据 |
| `Attention` | [`nanovllm/layers/attention.py`](../nanovllm/layers/attention.py) | 注意力计算 + KV Cache 写入，屏蔽后端差异 |
| `Sampler` | [`nanovllm/layers/sampler.py`](../nanovllm/layers/sampler.py) | 从 logits 采样出 token |
| 并行层 | [`nanovllm/layers/linear.py`](../nanovllm/layers/linear.py)、[`embed_head.py`](../nanovllm/layers/embed_head.py) | 权重切分与跨卡通信 |
| 模型层 | [`nanovllm/models/qwen3.py`](../nanovllm/models/qwen3.py) | Qwen3 结构定义 |

---

## 1. 全景：一次 `generate()` 调用链

### 1.1 一次调用的完整路径

```
LLM(...).generate(prompts, sampling_params)
│
├─【引擎层】LLMEngine.__init__
│    ├─ Config(model, **kwargs)                  ← 读 HF config，定稿运行参数
│    ├─ Sequence.block_size = config.kvcache_block_size
│    ├─ ModelRunner(config, rank=0, events)       ← 见 §3.2
│    ├─ AutoTokenizer.from_pretrained(...)
│    ├─ config.eos = tokenizer.eos_token_id
│    ├─ Scheduler(config)                         ← 建 BlockManager
│    └─ atexit.register(self.exit)
│
├─【引擎层】add_request(prompt, sp)
│    ├─ tokenizer.encode(prompt)                  ← 注意：不套 chat template
│    ├─ Sequence(token_ids, sp)                   ← 状态 WAITING
│    └─ scheduler.add(seq)                        ← 进入 waiting 队列
│
└─【引擎层】主循环 while not scheduler.is_finished():
     │
     ├─【调度层】scheduler.schedule() → (seqs, is_prefill)
     │      waiting 非空 → 组 PREFILL 批
     │      否则         → 组 DECODE 批
     │
     ├─【执行层】model_runner.run(seqs, is_prefill)
     │    ├─ prepare_prefill / prepare_decode      ← 构造张量 + 元数据
     │    ├─ set_context(...)                      ← 元数据注入全局单例
     │    ├─ model(input_ids, positions)           ← 36 层前向，内部写 KV Cache
     │    ├─ compute_logits → logits              ← prefill 只算每序列最后一个 token
     │    ├─ Sampler(logits, temperatures)          ← 每序列 1 个新 token
     │    └─ reset_context()
     │
     ├─【调度层】scheduler.postprocess(seqs, token_ids, is_prefill)
     │    ├─ block_manager.hash_blocks(seq)        ← 满块登记进前缀哈希表
     │    ├─ num_cached_tokens += num_scheduled_tokens
     │    ├─ seq.append_token(token_id)
     │    └─ EOS / 达到 max_tokens → FINISHED + deallocate
     │
     └─ 收集 outputs（仅在序列 FINISHED 时一次性返回）
```

### 1.2 组件之间的依赖关系

```
                    ┌──────────────┐
                    │  Config      │  运行参数 + HF 模型结构
                    └──────┬───────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌───────▼────────┐ ┌───────▼────────┐ ┌───────▼────────┐
│  ModelRunner   │ │   Scheduler    │ │  BlockManager  │
│  执行层        │◄─┤   调度层       │─┤   内存层       │
│                │ │                │ │                │
│ 持有 model     │ │ 持有 waiting  │ │ 持有物理块池    │
│ 持有 kv_cache  │ │       running  │ │ 持有前缀哈希表  │
│ 持有 sampler   │ │                │ │                │
└───────┬────────┘ └───────┬────────┘ └───────┬────────┘
        │                  │                  │
        │        ┌─────────▼─────────┐        │
        │        │     Sequence      │◄───────┘
        │        │  请求生命周期状态  │
        │        └─────────┬─────────┘
        │                  │
        │          ┌───────▼────────┐
        │          │    Context     │  ← 全局单例，ModelRunner 写、Attention 读
        │          │ 单次前向元数据  │
        │          └───────┬────────┘
        │                  │
        │          ┌───────▼────────┐
        │          │   Attention    │  ← 后端抽象：flash / torch
        │          └────────────────┘
        ▼
   ┌─────────┐
   │  Qwen3   │  →  36 × DecoderLayer
   └─────────┘
```

**这张图上最重要的一条线是 `ModelRunner → Context → Attention`。**

调度层和内存层完全不知道算子层的存在；算子层也完全不知道 `Sequence`、`Block`、`Scheduler` 的存在。二者只通过 `Context` 这一个数据结构交换信息。

`Context` 的字段（[`nanovllm/utils/context.py`](../nanovllm/utils/context.py)）：

| 字段 | 类型 | 含义 | prefill | decode |
| --- | --- | --- | --- | --- |
| `is_prefill` | bool | 本轮是 prefill 还是 decode | `True` | `False` |
| `cu_seqlens_q` | int32 tensor | 每个序列在 q 缓冲区中的累计长度 | 用 | 不用 |
| `cu_seqlens_k` | int32 tensor | 每个序列在 k 缓冲区中的累计长度（含缓存前缀） | 用 | 不用 |
| `max_seqlen_q` | int | 批内最长 q 长度 | 用 | 不用 |
| `max_seqlen_k` | int | 批内最长 k 长度 | 用 | 不用 |
| `slot_mapping` | int32 tensor | 本轮每个 token 的 K/V 物理写入槽位 | 用 | 用 |
| `context_lens` | int32 tensor | 每个序列已有的历史长度 | 不用 | 用 |
| `block_tables` | int32 tensor | 逻辑块 → 物理块映射表 | 前缀命中时用 | 用 |

`Context` 是模块级全局单例，通过 `set_context()` 整体替换、`get_context()` 读取、`reset_context()` 复位。它是「执行层 → 算子层」唯一的通信通道。

---

## 2. 组件清单

| 组件 | 生命周期 | 输入 | 输出 / 状态 |
| --- | --- | --- | --- |
| `Config` | 启动时一次 | 模型路径 + kwargs | 运行参数、`hf_config`、`eos` |
| `ModelRunner` | 启动时创建，退出时销毁 | `Config`、rank | `model`、`kv_cache`、`sampler`、`scheduler` 依赖的元数据构造能力 |
| `Scheduler` | 启动时创建 | `Config` | `waiting`/`running` 两个队列、`block_manager` |
| `BlockManager` | 启动时创建 | 块总数、块大小 | 物理块池 `free_block_ids`、`used_block_ids`、前缀哈希表 `hash_to_block_id` |
| `Sequence` | 每请求一个 | token_ids、`SamplingParams` | 状态、token 列表、块表、计数 |
| `Context` | 每次前向刷新 | 由 `ModelRunner` 写入 | 单次前向的元数据 |
| `Attention` | 每层一个 | q/k/v | 注意力输出；同时负责写 KV Cache |

---

## 3. 阶段拆解：一次推理的十个阶段

### 阶段 0：引擎启动前的参数收敛

`LLMEngine.__init__` 只接受 `Config` 声明过的字段：

```python
config_fields = {field.name for field in fields(Config)}
config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
config = Config(model, **config_kwargs)
```

未知 kwargs 会被静默丢弃。`Config.__post_init__` 做三件事：

1. 断言模型目录存在；
2. 断言 `kvcache_block_size % 256 == 0`（块大小必须是 256 的倍数）；
3. 断言 `1 <= tensor_parallel_size <= 8`；
4. 用 `AutoConfig.from_pretrained(model)` 读出 HF 模型结构，写入 `self.hf_config`；
5. `self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)`——**用户请求的上下文长度会被模型自身支持的上限截断**。

对于 Qwen3-4B，`hf_config` 给出：36 层、32 个 Q 头、8 个 KV 头、`head_dim=128`、`hidden_size=2560`、`intermediate_size=9728`、`vocab_size=151936`、`max_position_embeddings=262144`、`rms_norm_eps=1e-6`、`tie_word_embeddings=True`。

### 阶段 1：模型装入与显存规划（`ModelRunner.__init__`）

顺序是固定的，**每一步都为下一步提供前提**：

```
1. dist.init_process_group("nccl", "tcp://localhost:2333", world_size, rank)
2. torch.cuda.set_device(rank)
3. torch.set_default_dtype(hf_config.dtype)      # bf16
4. torch.set_default_device("cuda")
5. self.model = Qwen3ForCausalLM(hf_config)       # 建结构，权重为空
6. load_model(self.model, config.model)           # 填真实权重
7. self.sampler = Sampler()
8. self.warmup_model()                            # 跑一次假 prefill
9. self.allocate_kv_cache()                       # 按实测峰值规划 KV Cache
10. if not enforce_eager: self.capture_cudagraph()
11. torch.set_default_device("cpu") + 恢复默认 dtype
```

第 11 步的用意：`model` 和 `kv_cache` 已经在 GPU 上，之后新建的普通张量不应该默认落到 GPU，避免误用显存。

**为什么 warmup 必须在 `allocate_kv_cache` 之前**：warmup 会真实触发一次完整前向，把中间激活值等**瞬时开销**都跑到，于是 `torch.cuda.memory_stats()` 里同时有了 `peak`（瞬时峰值）和 `current`（稳态存活量）。`allocate_kv_cache` 依赖这两个数字做显存预算（见 §4.7）。

**warmup 造了什么**：

```python
seq_len = min(max_num_batched_tokens, max_model_len)          # 512
num_seqs = min(max_num_batched_tokens // seq_len, max_num_seqs) # 1
seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
for seq in seqs: seq.num_scheduled_tokens = seq_len
self.run(seqs, True)
```

即：**1 个长度 512 的假序列，走一遍 prefill 路径**。因为 `max_num_batched_tokens == max_model_len == 512`，这个 warmup 的形状正好是真实 prefill 的最大形状，起到「预热内核 + 固定显存峰值」的双重作用。

> **注意副作用**：`Sequence.counter` 是全局递增计数器，warmup 创建的假序列占用了 `seq_id = 0`，因此**用户请求的 `seq_id` 从 1 开始**。

### 阶段 2：请求接纳

```python
prompt = self.tokenizer.encode(prompt)   # 无 chat template
seq = Sequence(prompt, sampling_params)
self.scheduler.add(seq)                  # → waiting 队列
```

**这里有一个容易被忽略的事实**：`add_request` 直接对原始文本做 `tokenizer.encode`，**没有套用 Qwen3-Instruct 的 chat template**。也就是说，模型虽然是指令微调版本，但请求走的是「裸文本补全」模式。

### 阶段 3：调度（`Scheduler.schedule`）

调度层做四个决策：**谁参与、每人几个 token、能不能复用前缀、显存够不够**。详见 [§4.2](#42-scheduler调度层)。

返回 `(scheduled_seqs, is_prefill)`：

- `is_prefill=True`：`waiting` 队列非空，本次是 prefill 批；
- `is_prefill=False`：`waiting` 已空，本次是 decode 批。

### 阶段 4：批次构建（`ModelRunner.prepare_*`）

把 `Sequence` 列表翻译成 GPU 张量。prefill 与 decode 用两套完全不同的元数据：

| | prefill | decode |
| --- | --- | --- |
| `input_ids` | 本轮要算的**全部** token（prompt 剩余部分） | 每序列**1 个** token（`last_token`） |
| `positions` | 每 token 的位置下标 | 每序列 1 个（`len(seq) - 1`） |
| `cu_seqlens_q/k` | 构造 varlen 边界 | 不构造 |
| `slot_mapping` | 本轮所有 token 的写入槽位 | 每序列 1 个槽位 |
| `context_lens` | 不构造 | 每序列已有的历史长度 |
| `block_tables` | 仅前缀命中时构造 | 总是构造 |

详见 [§4.5](#45-modelrunner执行层)。

### 阶段 5：模型前向

`Qwen3ForCausalLM.forward` → `Qwen3Model.forward` → 36 × `Qwen3DecoderLayer.forward`。

每层内部顺序：

```
hidden_states, residual = input_layernorm(hidden_states, residual)
qkv = qkv_proj(hidden_states)          # 融合矩阵，一次 GEMM 出 q|k|v
q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
q = q_norm(q); k = k_norm(k)           # Qwen3 特有 QK-Norm
q, k = rotary_emb(positions, q, k)     # RoPE
o = attn(q, k, v)                      # ← KV Cache 在这里被写入
output = o_proj(o.flatten(1, -1))
hidden_states, residual = post_attention_layernorm(hidden_states, residual)
hidden_states = mlp(hidden_states)     # gate_up → SiLU(gate)*up → down
```

`residual` 作为一个累加器在层间流动：每层把「本层的输出」和「累计残差」相加后再归一化，最后一层之后再做一次 `norm(hidden, residual)`。

### 阶段 6：KV Cache 写入

写入发生在 `Attention.forward` 内部，**先写后算**：

```python
if k_cache.numel() and v_cache.numel():
    store_kvcache_torch(k, v, k_cache, v_cache, context.slot_mapping)   # torch 后端
```

物理写入位置完全由 `slot_mapping` 决定，后端只负责「按扁平下标搬运」。详见 [§4.4](#44-paged-kv-cache物理布局与双层索引) 与 [§4.6](#46-attention后端抽象)。

### 阶段 7：Logits 与采样

`compute_logits` → `ParallelLMHead.forward`：

- **prefill 时只取每序列最后一个 token** 的隐状态去算 logits（用 `cu_seqlens_q[1:] - 1` 索引）。512 个 token 只用最后 1 个算分类头，把 `512 × 151936` 的矩阵乘降到 `1 × 151936`；
- decode 时全部 token（本就只有 1 个/序列）都要算。

采样用 Gumbel-max trick，详见 [§4.10](#410-采样与结束判定)。

### 阶段 8：后处理（`Scheduler.postprocess`）

```python
for seq, token_id in zip(seqs, token_ids):
    self.block_manager.hash_blocks(seq)                    # 满块登记前缀缓存
    seq.num_cached_tokens += seq.num_scheduled_tokens
    seq.num_scheduled_tokens = 0
    if is_prefill and seq.num_cached_tokens < seq.num_tokens:
        continue                                          # chunked prefill 尚未完成，不产出 token
    seq.append_token(token_id)
    if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
        seq.status = SequenceStatus.FINISHED
        self.block_manager.deallocate(seq)
        self.running.remove(seq)
```

**prefill 阶段本身也会产出 1 个 token**——因为 prefill 的最后一步就是「算出第 1 个生成 token」，`append_token` 把它追加进 `token_ids`，`completion_token_ids` 随之增长。

### 阶段 9：回收与终止

`deallocate` 把序列占用的物理块归还池中，但**保留前缀哈希条目**（见 [§4.3](#43-blockmanager内存层)），使已完成请求的前缀仍可被后续请求复用。

### 阶段 10：收尾

`atexit.register(self.exit)` → `LLMEngine.exit` → `model_runner.call("exit")` → 各进程 `dist.destroy_process_group()`，TP worker 进程 join。

---

## 4. 核心机制详解

### 4.1 Sequence：请求的生命周期状态机

`Sequence` 是「一个请求」的完整聚合体，不是「一个 token」。

**状态机**：

```
   add_request
       │
       ▼
   ┌─────────┐   prefill 完成（全部分片算完）   ┌─────────┐
   │ WAITING │ ────────────────────────────► │ RUNNING │
   └─────────┘                                 └─────────┘
       ▲   ▲                                        │
       │   └────────────────────────────────────────┤
       │        抢占（显存不足，释放全部块）          │
       │                                        ▼
       │                                  ┌──────────┐
       └───────────────────────────────── │ FINISHED │
              （不会回到 WAITING）           └──────────┘
```

**三种状态的区别**：

- `WAITING`：还没完成 prefill。**可能已经在部分计算中**（chunked prefill 场景下，它会被多次调度，每次只算一段）。
- `RUNNING`：prefill 已完成，正在逐 token decode。
- `FINISHED`：收到 EOS 或达到 `max_tokens`，块已归还。

**关键字段**：

| 字段 | 含义 | 用途 |
| --- | --- | --- |
| `seq_id` | 全局递增编号 | 输出排序、跨进程标识 |
| `token_ids` | prompt + 已生成 token 的完整列表 | `completion_token_ids` 的来源 |
| `num_prompt_tokens` | prompt 长度 | 切分 prompt / completion 的分界 |
| `num_tokens` | 总长度 | 当前已有多少 token 的 KV |
| `num_cached_tokens` | 已经算过的 token 数 | prefill 起点；decode 时是历史长度 |
| `num_scheduled_tokens` | 本轮计划计算的 token 数 | 调度器写入，前向读取 |
| `block_table` | 逻辑块 → 物理块 | 定位 KV Cache |
| `is_prefill` | 当前处于哪个阶段 | 序列化压缩的依据 |
| `temperature` / `max_tokens` / `ignore_eos` | 从 `SamplingParams` 拷来 | 采样与结束判定 |

**两个派生属性**：

```python
num_blocks = (num_tokens + block_size - 1) // block_size      # 向上取整的块数
last_block_num_tokens = num_tokens - (num_blocks - 1) * block_size   # 末块内的 token 数（1-based）
```

**序列化压缩（为 TP 服务）**：

```python
def __getstate__(self):
    last_state = self.last_token if not self.is_prefill else self.token_ids
    return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
            self.num_scheduled_tokens, self.block_table, last_state)
```

TP>1 时 `Sequence` 要跨进程 pickle 传输。**decode 阶段 worker 只需要最后一个 token**（历史全在 KV Cache 里），所以只发 `last_token`；**prefill 阶段 worker 需要完整 token_ids**（要算 KV），所以发全量。这把 decode 阶段的 IPC 体积从 O(seq_len) 降到 O(1)。

### 4.2 Scheduler：调度层

#### 4.2.1 两个队列

```python
self.waiting: deque[Sequence]   # 还没完成 prefill
self.running: deque[Sequence]   # 已完成 prefill，正在 decode
```

`schedule()` 的第一判断是 `waiting` 是否非空：

- 非空 → 组 **prefill 批**；
- 为空 → 组 **decode 批**。

> **这是一个简化**：decode 请求永远优先，新请求必须在 `waiting` 排空后才有机会被调度。在负载高的场景下会造成 waiting 队列饥饿。成熟实现会把 decode 与 prefill 混在同一批次里。

#### 4.2.2 Prefill 分支

```python
while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.waiting[0]
    remaining = self.max_num_batched_tokens - num_batched_tokens
    if remaining == 0: break
    if not seq.block_table:
        num_cached_blocks = self.block_manager.can_allocate(seq)   # -1 = 装不下
        if num_cached_blocks == -1: break
        num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
    else:
        num_tokens = seq.num_tokens - seq.num_cached_tokens
    if remaining < num_tokens and scheduled_seqs: break           # chunked prefill 保护
    if not seq.block_table:
        self.block_manager.allocate(seq, num_cached_blocks)
    seq.num_scheduled_tokens = min(num_tokens, remaining)
    num_batched_tokens += seq.num_scheduled_tokens
    if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
        seq.status = SequenceStatus.RUNNING
        self.waiting.popleft()
        self.running.append(seq)
    scheduled_seqs.append(seq)
```

逐句机制：

- `can_allocate` 返回值有三重含义：**`-1` 装不下**、**`0` 无前缀可复用**、**`n>0` 命中 n 个块**。
- 命中 n 个块时，本次要算的 token 数是 `num_tokens - n × 256`——**前缀部分直接跳过**。
- `if remaining < num_tokens and scheduled_seqs: break` 是 **chunked prefill 保护**：只有本批的**第一个**序列允许被切分；后续序列装不下就整体推迟到下一批。
- `min(num_tokens, remaining)` 实现切分：prompt 比 batch 预算长时，只算一段。
- 状态迁移条件 `num_cached_tokens + num_scheduled_tokens == num_tokens` 表示「这一轮算完，整个 prompt 就齐了」→ 转 RUNNING。chunked prefill 中途不满足，序列留在 waiting。

#### 4.2.3 Decode 分支

```python
while self.running and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.running.popleft()
    while not self.block_manager.can_append(seq):
        if self.running: self.preempt(self.running.pop())
        else:           self.preempt(seq); break
    else:
        seq.num_scheduled_tokens = 1
        seq.is_prefill = False
        self.block_manager.may_append(seq)
        scheduled_seqs.append(seq)
assert scheduled_seqs
self.running.extendleft(reversed(scheduled_seqs))
```

- 每个序列本轮只算 **1 个 token**（`num_scheduled_tokens = 1`）。
- `can_append` 判断「还需不需要新块」，不够就**抢占**（`preempt`）：把某个 running 序列踢回 waiting 队首，释放它的全部块。
- `while...else` 结构：`else` 块只在 `while` 条件为假且未被 `break` 打断时执行——即「块够用」的正常路径。
- 末尾 `extendleft(reversed(...))` 把本轮参与解码的序列**按原顺序放回 running 队首**，保证公平（不会饿死靠后的请求）。

**抢占的代价**：`preempt` 把序列退回 waiting，`deallocate` 清空 `block_table` 且 `num_cached_tokens` 归零。被抢占的序列下一轮要从头 prefill（虽然前缀哈希还在，`can_allocate` 可能又命中）。

#### 4.2.4 调度层的输出语义

`LLMEngine.step` 用 `is_prefill` 区分吞吐统计：

```python
num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
```

prefill 用**正数**（本轮处理的 token 总数），decode 用**负数**（本轮生成的 token 数），靠符号区分两种吞吐。

### 4.3 BlockManager：内存层

#### 4.3.1 三张表

```python
self.blocks: list[Block]              # 物理块对象数组，索引即 block_id
self.free_block_ids: deque[int]       # 空闲块队列
self.used_block_ids: set[int]         # 已占用块集合
self.hash_to_block_id: dict[int, int] # 前缀哈希 → block_id
```

`Block` 对象只存管理信息（`ref_count`、`hash`、`token_ids`），**不存任何 KV 数据**。真正的 KV 数据在 `ModelRunner.kv_cache` 这个大张量里。

#### 4.3.2 分配与回收

```python
def _allocate_block(self) -> int:
    block_id = self.free_block_ids.popleft()
    block = self.blocks[block_id]
    assert block.ref_count == 0
    if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
        del self.hash_to_block_id[block.hash]     # 这个块要被新内容覆盖了
    block.reset()                                  # 清 hash / token_ids，ref_count = 1
    self.used_block_ids.add(block_id)
    return block_id

def _deallocate_block(self, block_id: int):
    assert self.blocks[block_id].ref_count == 0
    self.used_block_ids.remove(block_id)
    self.free_block_ids.append(block_id)          # 放回池，但【不清 hash / token_ids】
```

**这里有一个决定前缀缓存生命周期的关键不对称**：

- `_deallocate_block`（请求结束）**不删** `hash_to_block_id` 条目，也**不清** `block.token_ids`；
- `_allocate_block`（块被新内容占用）才删除哈希条目并 `reset()`。

所以**已完成的请求，其前缀依然留在哈希表里，可以被新请求复用**。

`reset()` 在 `_allocate_block` 里被调用：

```python
def reset(self):
    self.ref_count = 1
    self.hash = -1
    self.token_ids = []
```

顺序很重要：必须先删哈希再 reset，否则会出现「哈希指向一个内容已被覆盖的块」。

#### 4.3.3 引用计数

`ref_count` 表达「**有几个序列正在使用这个物理块**」。前缀命中时，两个序列共享同一个块，`ref_count = 2`；其中一个结束，`ref_count` 减到 1，块**不归还**；都结束了才归零归还。

`allocate` 的实现体现了这一点：

```python
for i in range(num_cached_blocks):
    block_id = self.hash_to_block_id[h]
    block = self.blocks[block_id]
    if block_id in self.used_block_ids:
        block.ref_count += 1          # 已在被别人用 → 共享，引用计数 +1
    else:
        block.ref_count = 1
        self.free_block_ids.remove(block_id)   # 从空闲池捞回
        self.used_block_ids.add(block_id)
    seq.block_table.append(block_id)
```

#### 4.3.4 前缀缓存的链式哈希

```python
@classmethod
def compute_hash(cls, token_ids: list[int], prefix: int = -1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()
```

**关键点：`prefix` 是上一块的哈希**，所以第 i 块的哈希实际覆盖了 `token[0 : (i+1)×256]` 整条前缀，而不是只有块 i 的内容。

这带来两个性质：

1. **同样的 token 出现在不同位置，哈希不同**——因为前缀链不同。这正是前缀缓存需要的语义；
2. **前缀相同，后续块的哈希必然相同**——链式传递保证了这一点。

如果改用纯内容哈希，会出现「块 2 内容匹配但块 1 不匹配」的错误复用。链式哈希从根上杜绝了这种误命中。

**双重校验**：哈希表命中后还要比对内容：

```python
block_id = self.hash_to_block_id.get(h, -1)
if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
    break
```

`token_ids` 逐元素比对是哈希碰撞时的兜底。xxhash64 碰撞概率极低，但这个校验是**必须**的。

#### 4.3.5 只缓存「满块」

```python
def hash_blocks(self, seq: Sequence):
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
```

只有**写满 256 个 token 的块**才进入哈希表。原因是块是分配单位：一个块被分配就必须整块使用，不能让两个序列共享同一块的不同部分（否则 `slot_mapping` 会冲突）。因此只有满块才满足「块内容 == 完整前缀片段」这个不变式。

> **直接推论**：前缀缓存有 256 token 的粒度成本。300 token 的公共前缀只有第 0 块（满块）可复用，第 1 块（44 token）不可复用，因此 `reusable_prefix_blocks = 1`。

#### 4.3.6 `can_allocate` 的三重返回值

```python
def can_allocate(self, seq: Sequence) -> int:
    h = -1
    num_cached_blocks = 0
    num_new_blocks = seq.num_blocks
    for i in range(seq.num_blocks - 1):          # 注意是 -1：最后一个块永远不复用
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)
        block_id = self.hash_to_block_id.get(h, -1)
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            break
        num_cached_blocks += 1
        if block_id in self.used_block_ids:
            num_new_blocks -= 1        # 这个块会被共享，不需要新分配
    if len(self.free_block_ids) < num_new_blocks:
        return -1
    return num_cached_blocks
```

- 循环范围是 `num_blocks - 1`：**最后一个块永远不复用**。因为最后一个块可能没写满（内容会变），复用它会导致后续写入覆盖别人的数据。
- `num_new_blocks` 的递减只在该块**已被别人占用**时发生（共享 vs 新占）。
- 返回 `-1` 表示物理块不足，调度器据此放弃本轮调度该序列。

### 4.4 Paged KV Cache：物理布局与双层索引

#### 4.4.1 物理布局

```python
self.kv_cache = torch.empty(2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)
#                      ↑K/V ↑层    ↑块       ↑块内token ↑KV头   ↑维度
```

每层的 `Attention` 拿到自己那一层的**视图**：

```python
layer_id = 0
for module in self.model.modules():
    if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
        module.k_cache = self.kv_cache[0, layer_id]
        module.v_cache = self.kv_cache[1, layer_id]
        layer_id += 1
```

由于 `kv_cache[0, layer_id]` 的第 0 维被固定为 0，取出的切片在内存中是连续的，因此后续 `reshape` 不会触发拷贝。

**隐含契约**：`self.model.modules()` 的遍历顺序必须与 `forward` 的层执行顺序一致，否则第 0 层会拿到第 5 层的 cache。`nn.ModuleList` 保证顺序遍历，因此 `layers.0 … layers.35` 与 `Attention` 实例的出现顺序一致。

#### 4.4.2 双层索引

这是 Paged KV Cache 最核心的概念区分。

**逻辑层（页表）**：`seq.block_table = [5, 9, 2]`

```
逻辑块0 → 物理块5
逻辑块1 → 物理块9
逻辑块2 → 物理块2
```

**物理层（扁平槽位）**：`slot_mapping = [1280, 1281, …, 1285, 2304, 2305, …]`

```
物理块5 的第 0..255 槽位 → 扁平地址 5×256+0 … 5×256+255 = 1280 … 1535
物理块9 的第 0..255 槽位 → 扁平地址 9×256+0 … 9×256+255 = 2304 … 2559
```

- `block_table` 回答「**逻辑块 → 物理块**」
- `slot_mapping` 回答「**本轮要写的物理扁平地址**」

两者不能互相替代：`block_table` 是整个序列的持久映射，`slot_mapping` 只是本轮这一小撮 token 的写入位置。

#### 4.4.3 `slot_mapping` 的构造（prefill）

```python
start_block = start // self.block_size
end_block = (end + self.block_size - 1) // self.block_size
for i in range(start_block, end_block):
    slot_start = seq.block_table[i] * self.block_size
    if i == start_block:
        slot_start += start % self.block_size      # 跳过已缓存的部分
    if i != end_block - 1:
        slot_end = seq.block_table[i] * self.block_size + self.block_size   # 整块
    else:
        slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size  # 截断
    slot_mapping.extend(range(slot_start, slot_end))
```

三种情况：

| 情况 | 处理 |
| --- | --- |
| 首块且部分命中 | `slot_start += start % block_size`，跳过已缓存 token |
| 中间块 | 整块 256 个槽位 |
| 末块 | 截断到 `end - i × block_size`，不写满 |

#### 4.4.4 `slot_mapping` 的构造（decode）

```python
slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
```

`last_block_num_tokens` 是最后一个 token 在其块内的 1-based 位置，减 1 转成 0-based 扁平偏移。

**跨块时序示例**（block_size = 256，prompt 12 token）：

| `num_tokens` | `num_blocks` | `last_block_num_tokens` | slot | 含义 |
| --- | --- | --- | --- | --- |
| 12 | 1 | 12 | `0×256+11` | 写 token #11 |
| 13 | 1 | 13 | `0×256+12` | 写 token #12 |
| 255 | 1 | 255 | `0×256+254` | 写 token #254 |
| 256 | 1 | 256 | `0×256+255` | 写 token #255（块 0 写满） |
| **257** | **2** | 1 | `1×256+0` | **写 token #256 → 需要块 1** |

「什么时候需要新块」由这条规则判定：

```python
def can_append(self, seq): return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
def may_append(self, seq):
    if len(seq) % self.block_size == 1:
        seq.block_table.append(self._allocate_block())
```

`len(seq) % 256 == 1` 就是「刚跨过块边界」的信号。注意 `may_append` 在**追加 token 之前**调用，此时 `num_tokens` 是上一轮结束后的长度。

#### 4.4.5 读取（`_gather_cache`）

```python
def _gather_cache(self, cache, block_table, seqlen):
    block_size = cache.size(1)                                   # 256
    num_blocks = (seqlen + block_size - 1) // block_size
    block_ids = block_table[:num_blocks].to(torch.long)
    return cache.index_select(0, block_ids).reshape(-1, cache.size(2), cache.size(3))[:seqlen]
```

`cache` 形状 `[num_blocks, 256, 8, 128]` → `index_select` 取出用到的块 → `reshape(-1, 8, 128)` 扁平成「按 token 连续」 → 截断到 `seqlen`。

**代价**：这是**物化**操作，会把整个前缀复制出来。`torch` 后端下这是 O(prefix_len) 的额外开销。

### 4.5 ModelRunner：执行层

#### 4.5.1 `prepare_prefill`

产出四个张量 + 一份元数据：

| 产出 | 内容 |
| --- | --- |
| `input_ids` | 本轮所有要算的 token 拼接（各序列按 `num_cached_tokens` 切片） |
| `positions` | 每个 token 的位置下标 |
| `cu_seqlens_q` / `cu_seqlens_k` | varlen 边界 |
| `slot_mapping` | 每个 token 的物理写入槽位 |
| `block_tables` | **仅当存在前缀命中时**才构造 |

```python
if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
    block_tables = self.prepare_block_tables(seqs)
```

**为什么用这个条件判断是否需要 `block_tables`**：`cu_seqlens_k` 的最后一项是「所有序列的 k 长度总和」，`cu_seqlens_q` 的是「q 长度总和」。两者相等 ⟺ 没有任何序列带缓存前缀。

`cu_seqlens_k` 与 `cu_seqlens_q` 的差异来自：

```python
seqlen_q = seq.num_scheduled_tokens
end = start + seqlen_q
seqlen_k = end                     # = start + seqlen_q，比 seqlen_q 多一个 start
cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
```

`start = seq.num_cached_tokens`，即已缓存的 token 数。所以 **k 缓冲区的语义长度包含了缓存前缀**，这正是 §4.6 里 torch 后端推导前缀长度的依据。

#### 4.5.2 `prepare_decode`

```python
for seq in seqs:
    input_ids.append(seq.last_token)
    positions.append(len(seq) - 1)
    context_lens.append(len(seq))
    slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
```

四个张量各司其职：`input_ids` 是新 token，`positions` 是它的位置，`context_lens` 是历史长度（决定读多少 KV），`slot_mapping` 是它的写入位置。

`block_tables` 在 decode 中**总是**构造（读 cache 必需）。

#### 4.5.3 `run` 的骨架

```python
def run(self, seqs, is_prefill):
    input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
    temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
    logits = self.run_model(input_ids, positions, is_prefill)
    token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
    reset_context()
    return token_ids
```

- **只有 rank 0 做采样**：其他 rank 的 logits 是分片，需要 gather 到 rank 0（见 §4.9）。
- `reset_context()` 在每次前向后把全局单例复位，防止上一轮的元数据泄漏到下一轮（比如 `block_tables` 残留会让下一次 prefill 误以为有前缀命中）。

### 4.6 Attention：后端抽象

#### 4.6.1 后端选择

```python
def select_attention_backend() -> str:
    backend = os.getenv("CAI_VLLM_ATTENTION_BACKEND", "auto").lower()
    if backend not in {"auto", "flash", "torch"}:
        raise ValueError(...)
    if backend == "flash":
        if flash_attn_varlen_func is None or flash_attn_with_kvcache is None:
            raise RuntimeError("FlashAttention is requested but flash-attn is not installed")
        return "flash"
    if backend == "torch": return "torch"
    return "flash" if 已安装 else "torch"
```

环境变量 `CAI_VLLM_ATTENTION_BACKEND` 取值 `auto` / `flash` / `torch`：

- `auto`：装了 `flash_attn` 就用，否则退回 torch；
- `flash`：强制使用，未安装直接报错（**不静默降级**）；
- `torch`：强制使用纯 PyTorch 实现。

#### 4.6.2 三条路径的分工

| 场景 | flash 路径 | torch 路径 |
| --- | --- | --- |
| 写 KV Cache | Triton kernel `store_kvcache_kernel` | `flat_k_cache[slots[valid]] = key[valid]` |
| prefill 计算 | `flash_attn_varlen_func(q, k_cache, v_cache, cu_seqlens_q/k, block_table=...)` | 手工 gather 前缀 + 造因果 mask + `F.scaled_dot_product_attention` |
| decode 计算 | `flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens, block_table=...)` | 手工 gather 全量 KV + `F.scaled_dot_product_attention` |

**写 cache 的两种实现**：

```python
# torch
slots = slot_mapping.to(torch.long)
valid = slots >= 0
flat_k_cache = k_cache.reshape(-1, key.size(1), key.size(2))
flat_k_cache[slots[valid]] = key[valid]
```

```python
# triton
idx = tl.program_id(0)
slot = tl.load(slot_mapping_ptr + idx)
if slot == -1: return                    # ← 跳过 padding 槽位
tl.store(k_cache_ptr + slot * D + tl.arange(0, D), key)
```

`slot == -1` 的分支是 CUDA Graph 的前提（见 §4.8）。

#### 4.6.3 torch prefill 路径的语义对齐

```python
for i in range(context.cu_seqlens_q.numel() - 1):
    q_start, q_end = context.cu_seqlens_q[i:i + 2].tolist()
    q_i = q[q_start:q_end]
    q_len = q_i.size(0)
    k_i, v_i = k[q_start:q_end], v[q_start:q_end]
    k_len = context.cu_seqlens_k[i + 1].item() - context.cu_seqlens_k[i].item()
    prefix_len = k_len - q_len
    if prefix_len:
        cached_k = self._gather_cache(self.k_cache, context.block_tables[i], prefix_len)
        cached_v = self._gather_cache(self.v_cache, context.block_tables[i], prefix_len)
        k_i = torch.cat((cached_k, k_i), dim=0)
        v_i = torch.cat((cached_v, v_i), dim=0)
    q_i = q_i.transpose(0, 1).unsqueeze(0)
    k_i = self._repeat_kv(k_i, self.num_heads).transpose(0, 1).unsqueeze(0)
    v_i = self._repeat_kv(v_i, self.num_heads).transpose(0, 1).unsqueeze(0)
    query_positions = torch.arange(q_len, device=q.device) + prefix_len
    key_positions = torch.arange(k_len, device=q.device)
    causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    output = F.scaled_dot_product_attention(q_i, k_i, v_i, attn_mask=causal_mask, dropout_p=0.0, scale=self.scale)
```

对应关系：

| 代码 | 语义 |
| --- | --- |
| `cu_seqlens_q` 分段 | varlen 布局：每个序列的 q 在缓冲区中的区间 |
| `k_len - q_len = prefix_len` | 精确等于 `seq.num_cached_tokens`，即需要从 cache 补回来的前缀长度 |
| `cat((cached_k, k_i))` | 把缓存前缀接在本轮新算的 K/V 前面，恢复完整历史 |
| `query_positions = arange(q_len) + prefix_len` | 新 token 的绝对位置（不是从 0 开始） |
| `causal_mask = key_pos <= query_pos` | 因果掩码，布尔语义（`True` = 可见） |
| `_repeat_kv` | GQA 展开：8 个 KV 头 → 32 个 Q 头 |

**torch 后端的固有代价**（重要，影响选型）：

| 开销 | flash 路径 | torch 路径 |
| --- | --- | --- |
| 前缀读取 | 内核内按页表访问，不物化 | Python 循环 + `index_select`，**物化整个前缀** |
| GQA | 内核原生支持 | `repeat_interleave`，**实打实复制 4 倍 KV** |
| 因果掩码 | 内核融合计算 | 物化 `[q_len, k_len]` 布尔矩阵 |
| 写 cache | 单个 Triton kernel | torch 高级索引（生成临时索引张量） |

torch 路径的开销随前缀长度线性增长。

### 4.7 显存预算（`allocate_kv_cache`）

```python
free, total = torch.cuda.mem_get_info()
used = total - free
peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
num_kv_heads = hf_config.num_key_value_heads // self.world_size
head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
```

**块大小公式**逐项拆解（以 Qwen3-4B 为例）：

```
2                          ← K 和 V 各一份
× 36 (num_hidden_layers)   ← 每层都要独立缓存
× 256 (block_size)         ← 每块存 256 个 token 的 KV
× 8  (num_kv_heads)        ← GQA：8 个 KV 头，不是 32
× 128 (head_dim)
× 2  (bf16)
= 37,748,736 B ≈ 36 MiB
```

**显存预算公式**的含义：

```
可用预算 = total × util        ← 允许进程占用的显存上限
       − used                 ← 减去当前已占用（CUDA context、torch 已缓存块、模型权重）
       − (peak − current)     ← 减去 warmup 实测到的瞬时峰值余量
```

`peak − current` 是「稳态存活量之上还需要预留的瞬时开销」。因为 warmup 已经真实跑过一次前向，这个差值是被**实测**出来的，而不是猜的。效果是：**保证在最坏的瞬时时刻，总占用也不超过 `util × total`**。

**一次 `torch.empty` 分配全部**：

```python
self.kv_cache = torch.empty(2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)
```

不用动态增长的 allocator，而是启动时一次性大块分配：

- 物理地址连续 → `index_select` / `reshape` 效率高；
- 显存占用可预测，不会碎片化；
- 避免推理中途的 `cudaMalloc`（同步点 + 延迟尖刺）。

**代价**：KV Cache 容量在启动时就**静态锁死**。`max_model_len` 与 `gpu_memory_utilization` 决定了能同时服务多少序列：

```
可并发序列数 ≈ num_kvcache_blocks × block_size / max_model_len
```

### 4.8 CUDA Graph

#### 4.8.1 问题

decode 阶段每步只算 1 个 token，却要跑 36 层 × 数十个 kernel。每个 kernel 的 CPU 端 launch 开销约 5–10 μs，1000+ 次 launch 就是**数毫秒纯 CPU 开销**，而 GPU 上的实际计算只有几十微秒 → **CPU 成为瓶颈，GPU 空转**。

#### 4.8.2 做法

```python
self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
for bs in reversed(self.graph_bs):
    graph = torch.cuda.CUDAGraph()
    set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
    outputs[:bs] = self.model(input_ids[:bs], positions[:bs])              # warmup
    with torch.cuda.graph(graph, self.graph_pool):
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])          # capture
    if self.graph_pool is None: self.graph_pool = graph.pool()
    self.graphs[bs] = graph
```

- **按 batch size 分档**，每档一个 graph。replay 时取「最小的够用的档」：`next(x for x in self.graph_bs if x >= bs)`。
- **共享 memory pool**：所有 graph 用同一个 pool，避免 N 倍显存。
- **capture 之前必须 warmup**：触发内核编译与显存池稳定，否则 capture 会失败。
- **逆序捕获**（`reversed`）：先捕获大 batch 再捕获小 batch，让显存池逐步建立。

#### 4.8.3 Replay 的输入搬运

```python
graph_vars["input_ids"][:bs] = input_ids
graph_vars["positions"][:bs] = positions
graph_vars["slot_mapping"].fill_(-1)
graph_vars["slot_mapping"][:bs] = context.slot_mapping
graph_vars["context_lens"].zero_()
graph_vars["context_lens"][:bs] = context.context_lens
graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
graph.replay()
```

**`slot_mapping.fill_(-1)` 是这里的关键**：graph 是按最大 batch size 录制的，padding 部分的槽位设为 `-1`，Triton kernel 里的 `if slot == -1: return` 会直接跳过 → **padding 不会污染 KV Cache**。

同理 `context_lens.zero_()` 让 padding 序列的历史长度为 0。

#### 4.8.4 何时启用

```python
if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
    return self.model.compute_logits(self.model(input_ids, positions))
```

- prefill 不用 graph（形状变化大，收益低）；
- `enforce_eager=True` 时全都不用；
- batch 超过 512 也不走 graph（超出录制范围）。

**为什么 nano-vLLM 默认 `enforce_eager=True`**：CUDA Graph 要求形状固定，而 continuous batching 导致每轮 batch size 都不同。分档能缓解，但 `bs × context_len × block 数` 的组合会让 graph 数量与显存迅速膨胀。

### 4.9 Tensor Parallel 与进程模型

#### 4.9.1 权重切分的两种基本形式

```
ColumnParallelLinear  (tp_dim = 0，切【输出】维)
    W: [out, in] → [out/tp, in]
    X @ Wᵀ → 每个 GPU 算出【不同的输出特征】
    后续若接逐元素算子 → 无需通信
    后续若需 concat 语义 → 最后 gather

RowParallelLinear  (tp_dim = 1，切【输入】维)
    W: [out, in] → [out, in/tp]
    X 沿输入维切开，每 GPU 算出输出的【部分和】
    → 必须 dist.all_reduce 才能得到完整结果
```

**判据**：切完之后，下一个算子是「逐元素」还是「求和」。

#### 4.9.2 Qwen3 中的具体应用

| 模块 | 类型 | 切分 | 通信 |
| --- | --- | --- | --- |
| `qkv_proj` | QKVParallelLinear（Column） | 输出维，q/k/v 段长不同 | 无（后接 reshape） |
| `o_proj` | RowParallelLinear | 输入维（32×128 = 4096） | `all_reduce` |
| `gate_up_proj` | MergedColumnParallelLinear（Column） | 输出维，gate/up 合并 | 无（后接逐元素 SiLU×up） |
| `down_proj` | RowParallelLinear | 输入维（9728） | `all_reduce` |
| `embed_tokens` | VocabParallelEmbedding | 词表行切 | `all_reduce`（带 mask） |
| `lm_head` | ParallelLMHead | 词表行切 | `gather` 到 rank 0 |

#### 4.9.3 QKVParallelLinear 的不等长分片

Qwen3 的 Q/K/V 大小不同（32 头 / 8 头 / 8 头），因此加载时必须知道当前这块属于哪一段：

```python
if loaded_shard_id == "q":   shard_size = num_heads * head_size;    shard_offset = 0
elif loaded_shard_id == "k": shard_size = num_kv_heads * head_size; shard_offset = num_heads * head_size
else:                        shard_size = num_kv_heads * head_size; shard_offset = (num_heads + num_kv_heads) * head_size
param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
```

这要求加载器知道「源权重名 → 目标参数名 + 分片 id」的映射，由 `packed_modules_mapping` 提供：

```python
packed_modules_mapping = {
    "q_proj": ("qkv_proj", "q"), "k_proj": ("qkv_proj", "k"), "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0), "up_proj": ("gate_up_proj", 1),
}
```

#### 4.9.4 两阶段权重加载

```
阶段一（建"图纸"）：Qwen3ForCausalLM(hf_config)
    → 按 hf_config 造出所有 nn.Parameter 的形状，torch.empty，不填值
    → 每个 param 挂一个 weight_loader 属性（自带 TP 切分逻辑）

阶段二（填"真值"）：load_model(model, path)
    → 遍历 safetensors → 名字重映射 → 调用该 param 自己的 weight_loader
    → weight_loader 内部完成 narrow + copy_
```

**参数自带加载器**是这里的核心设计：加载逻辑与切分逻辑写在同一个类里，新增一种并行方式不需要改加载器。

#### 4.9.5 进程模型：共享内存 RPC

```
主进程 (rank 0)                       worker 进程 (rank 1..N-1)
────────────────                      ──────────────────────────
ctx.Process(target=ModelRunner) ────►  同样的 ModelRunner.__init__
  rank=1, event                            world_size > 1 → 不是 rank 0
  dist.init_process_group(nccl)            ├─ dist.barrier()
  dist.barrier()                           ├─ shm = SharedMemory("nanovllm")
  shm = SharedMemory(create)               └─ loop(): 永远等消息
        │                                           ▲
        │  call("run", seqs, is_prefill)             │
        ▼                                           │
  write_shm("run", seqs, is_prefill)                 │
    pickle → shm.buf[4 : n+4]                        │
    event.set() ───────────────────────────────────► │
                                                读 shm → unpickle → self.call(...)
```

```python
def write_shm(self, method_name, *args):
    data = pickle.dumps([method_name, *args])
    n = len(data)
    self.shm.buf[0:4] = n.to_bytes(4, "little")
    self.shm.buf[4:n+4] = data
    for event in self.event: event.set()
```

**协议极其简单**：4 字节长度 + pickle payload + `Event` 同步。

**为什么用 `spawn` 而不是 `fork`**：

```python
ctx = mp.get_context("spawn")
```

`fork` 会复制父进程内存，包括已初始化的 CUDA context，导致子进程 CUDA 状态错乱（典型症状是 `Cannot re-initialize CUDA in forked subprocess`）。`spawn` 启动全新解释器，慢但安全。

**共享内存只有 1 MiB**：`SharedMemory(name="nanovllm", create=True, size=2**20)`。这限制了单次 RPC 的 payload 上限，也正是 `Sequence.__getstate__` 要做压缩的原因。

#### 4.9.6 TP 下的数据流

- **rank 0** 是「主进程」，`ModelRunner` 直接跑在主进程里；
- **rank > 0** 是 worker，只通过共享内存接收命令；
- 每一步：rank 0 `write_shm` → 其他 rank 收到 → 各自执行 `run` → 各 rank 内部用 NCCL 做 `all_reduce` 同步；
- **采样只在 rank 0 做**：其他 rank 的 logits 是词表分片，需要 `dist.gather` 到 rank 0 拼回完整词表。

### 4.10 采样与结束判定

#### 4.10.1 Gumbel-max 采样

```python
@torch.compile
def forward(self, logits, temperatures):
    logits = logits.float().div_(temperatures.unsqueeze(dim=1))
    probs = torch.softmax(logits, dim=-1)
    sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
```

- **温度是「除」不是「乘」**：`temperature = 0.6` → logits 放大 1.67 倍 → 分布更尖锐（更确定）。
- `probs / Exponential(1)` 取 argmax，等价于从 softmax 分布中**精确采样**（Gumbel-max trick），比 multinomial 少一次随机数与归一化。
- `clamp_min_(1e-10)` 防止 `Exponential(1)` 采到 0 导致除零。

#### 4.10.2 结束判定

```python
if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
    seq.status = SequenceStatus.FINISHED
    self.block_manager.deallocate(seq)
    self.running.remove(seq)
```

两个终止条件：**EOS** 与 **达到 max_tokens**。`ignore_eos=True` 时忽略 EOS（强行跑满 max_tokens）。

#### 4.10.3 `SamplingParams` 禁止贪心

```python
assert self.temperature > 1e-10, "greedy sampling is not permitted"
```

采样策略被强制为随机采样，不允许 `temperature → 0` 的贪心模式。

#### 4.10.4 输出的粒度

```python
outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
...
outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
```

**只有序列完全结束才返回**，一次性给出全部 completion token（不是流式增量）。输出按 `seq_id` 排序，因此顺序等于请求提交顺序。

---

## 5. 张量与形状契约表

以 Qwen3-4B（36 层 / 32 Q 头 / 8 KV 头 / head_dim 128 / hidden 2560 / vocab 151936）+ `block_size=256` 为例。

| 张量 | 生产者 | 形状 | dtype | 含义 |
| --- | --- | --- | --- | --- |
| `input_ids` (prefill) | `prepare_prefill` | `[sum(seqlen_q)]` | int64 | 本轮所有要算的 token |
| `input_ids` (decode) | `prepare_decode` | `[batch]` | int64 | 每序列 1 个新 token |
| `positions` | `prepare_*` | 同 `input_ids` | int64 | 位置下标 |
| `cu_seqlens_q` | `prepare_prefill` | `[batch+1]` | int32 | q 的 varlen 边界 |
| `cu_seqlens_k` | `prepare_prefill` | `[batch+1]` | int32 | k 的 varlen 边界（含前缀） |
| `slot_mapping` | `prepare_*` | `[sum(seqlen_q)]` / `[batch]` | int32 | 物理写入槽位 |
| `context_lens` | `prepare_decode` | `[batch]` | int32 | 每序列历史长度 |
| `block_tables` | `prepare_block_tables` | `[batch, max_blocks]` | int32 | 逻辑块 → 物理块 |
| `temperatures` | `prepare_sample` | `[batch]` | float32 | 采样温度 |
| `kv_cache` | `allocate_kv_cache` | `[2, 36, num_blocks, 256, 8, 128]` | bf16 | 全局 KV 池 |
| 隐状态 | 模型 | `[tokens, 2560]` | bf16 | — |
| `logits` (prefill) | `lm_head` | `[batch, 151936]` | bf16 | 只算每序列最后 token |
| `logits` (decode) | `lm_head` | `[batch, 151936]` | bf16 | — |

---

## 6. 关键常量与来源

| 常量 | 值 | 来源 | 作用 |
| --- | --- | --- | --- |
| `kvcache_block_size` | 256 | `Config` 默认（必须是 256 的倍数） | KV 块内 token 数，也是前缀缓存粒度 |
| `max_num_batched_tokens` | 16384 默认 | `Config` | 单批 token 预算，chunked prefill 的切片大小 |
| `max_num_seqs` | 512 默认 | `Config` | 单批最大序列数 |
| `max_model_len` | 4096 默认 | `Config`，并被 `max_position_embeddings` 截断 | 单序列最大长度 |
| `gpu_memory_utilization` | 0.9 默认 | `Config` | 显存预算比例 |
| `enforce_eager` | False 默认 | `Config` | 是否跳过 CUDA Graph |
| `tensor_parallel_size` | 1 默认 | `Config`（1–8） | 并行卡数 |
| `eos` | tokenizer 填充 | `LLMEngine.__init__` | 结束符 token id |
| `num_kvcache_blocks` | 运行时推算 | `allocate_kv_cache` | 可用物理块数 |
| `CUDA_DEVICE_ORDER` | `PCI_BUS_ID` | 脚本设置 | 固定物理枚举顺序，保证 rank 映射稳定 |
| `CUDA_VISIBLE_DEVICES` | `0` | 脚本设置 | 只暴露一张卡 |
| `CAI_VLLM_ATTENTION_BACKEND` | `torch` | 脚本设置 | 选择注意力后端 |

---

## 7. 一次完整推理的逐帧走查

以下以 `scripts/torch_fallback_smoke.py` 的参数为例：

```python
LLM(model, enforce_eager=True, tensor_parallel_size=1,
    max_model_len=512, max_num_batched_tokens=512, max_num_seqs=2,
    gpu_memory_utilization=0.85)
prompts = ["Explain what a KV cache is in one short sentence.",
           "Say hello in one short sentence."]
SamplingParams(temperature=0.6, max_tokens=16)
```

> prompt 长度按脚本调试时记录的数值：A = 12 token，B = 7 token。**本机无 CUDA，未复现验证**。

### 帧 0：启动

| 步骤 | 结果 |
| --- | --- |
| `Config.__post_init__` | 读到 Qwen3 结构；`max_model_len = min(512, 262144) = 512` |
| `ModelRunner.__init__` | `dist.init_process_group` (world_size=1)、dtype=bf16、device=cuda |
| `Qwen3ForCausalLM(hf_config)` | 建结构，参数为空 |
| `load_model` | 遍历 safetensors，按 `packed_modules_mapping` 重映射并填值 |
| `warmup_model` | `seq_len = min(512, 512) = 512`，`num_seqs = min(512//512, 2) = 1` → 1 条 512 token 的假序列走一遍 prefill。**占用 seq_id = 0** |
| `allocate_kv_cache` | 按显存预算算出 `num_kvcache_blocks`（记录值 26），分配 `kv_cache[2, 36, 26, 256, 8, 128]` |
| `enforce_eager=True` | 跳过 `capture_cudagraph` |
| `Scheduler` | `BlockManager(26, 256)`，26 个块全部空闲 |

### 帧 1：prefill 批（A + B 同时）

**调度**：

| 序列 | `num_blocks` | `can_allocate` | 分配 | `num_scheduled_tokens` | 状态 |
| --- | --- | --- | --- | --- | --- |
| A | 1 | 0（无前缀） | block_table = [0] | 12 | WAITING → RUNNING |
| B | 1 | 0（无前缀） | block_table = [1] | 7 | WAITING → RUNNING |

`num_batched_tokens = 19 ≤ 512` ✓

**批次构建**：

```
input_ids     = [A 的 12 个] + [B 的 7 个]           → 长度 19
positions     = [0..11] + [0..6]
cu_seqlens_q  = [0, 12, 19]
cu_seqlens_k  = [0, 12, 19]        （无前缀 → 与 q 相同）
max_seqlen_q  = 12,  max_seqlen_k = 12
slot_mapping  = [0..11] + [256..262]
block_tables  = None                （cu_seqlens_k[-1] == cu_seqlens_q[-1]）
```

`slot_mapping` 的来历：A 占物理块 0 → 扁平 0×256+0 … 0×256+11；B 占物理块 1 → 扁平 1×256+0 … 1×256+6。

**前向**：

```
input_ids [19] → embed → [19, 2560]
36 层 DecoderLayer
  每层：qkv_proj → [19, 6144] → split → q[19,32,128], k[19,8,128], v[19,8,128]
        q_norm / k_norm (RMSNorm over head_dim)
        rotary_emb (positions, q, k)
        attn:
          写 cache: flat_k_cache[[0..11, 256..262]] = k
          prefill 计算: 2 段循环，prefix_len = 0，因果 mask
        o_proj → [19, 2560]
        mlp: gate_up [19, 19456] → SiLU(x)*y → [19, 9728] → down_proj → [19, 2560]
final norm → [19, 2560]
```

**logits**：`cu_seqlens_q[1:] - 1 = [11, 18]` → 取隐状态第 11、18 行 → `lm_head` → `logits[2, 151936]`

**采样**：temperatures `[0.6, 0.6]` → Gumbel-max → 2 个 token

**后处理**：

| 序列 | `hash_blocks` | `num_cached_tokens` | 追加 token | `num_completion_tokens` | 结果 |
| --- | --- | --- | --- | --- | --- |
| A | start=0, end=12//256=0 → **不登记** | 0→12 | ✅ | 1 | 继续 RUNNING |
| B | start=0, end=7//256=0 → **不登记** | 0→7 | ✅ | 1 | 继续 RUNNING |

> 两条序列的 prompt 都不足 256 token，因此**没有任何块被登记进前缀哈希表**。

### 帧 2 ~ 16：decode 批

每帧调度：A、B 各 1 token。

**批次构建**（以某帧为例，A 已有 12+N token，B 已有 7+N token）：

```
input_ids    = [A.last_token, B.last_token]
positions    = [len(A)-1, len(B)-1]
context_lens = [len(A), len(B)]
slot_mapping = [A.block_table[-1]*256 + last_block_num_tokens - 1, ...]
block_tables = [[0], [1]]
```

**前向**：decode 分支，`input_ids` 只有 2 个 token；每层：

```
q[2,32,128], k[2,8,128], v[2,8,128]
attn:
  写 cache: 2 个 token 分别写入 slot
  decode 计算: 逐序列 gather context_lens[i] 个历史 KV → SDPA（无因果 mask，全可见）
logits[2, 151936] → 2 个新 token
```

**后处理**：`num_cached_tokens` 继续累加；当 `num_completion_tokens == 16` 时 → FINISHED → `deallocate` → 块归还（**哈希条目保留**，但本例从未登记过）。

### 帧 17：终止

`waiting` 与 `running` 都空 → `is_finished()` 为真 → 收集 outputs → 按 `seq_id` 排序 → decode 文本。

---

## 8. 典型路径与边界行为

### 路径 A：单请求短 prompt

最简路径。prefill 1 帧 + decode N 帧（N = 生成 token 数）。占用 1 个物理块（若 prompt + 生成 ≤ 256 token）。

### 路径 B：多请求同批 prefill

`max_num_seqs=2` 且两条请求同时在 waiting 队列 → 同一次 prefill 一起算，`slot_mapping` 交错排列（A 的连续一段 + B 的连续一段），`cu_seqlens_q` 给出分界。这是最能体现「批处理」价值的路径。

### 路径 C：前缀缓存命中

1. 请求 1 的 prompt 长度 > 256 → 第 0 块写满 → `hash_blocks` 把它登记进哈希表；
2. 请求 1 结束 → `deallocate` → 块归还空闲池，**哈希条目与 `token_ids` 保留**；
3. 请求 2 的 prompt 前缀与请求 1 相同 → `can_allocate` 命中 → `num_cached_blocks = 1` → 本次只需算 `num_tokens - 256` 个 token；
4. `prepare_prefill` 中 `cu_seqlens_k[-1] > cu_seqlens_q[-1]` 成立 → 构造 `block_tables`；
5. torch 后端由 `prefix_len = k_len - q_len` 反推出前缀长度，从 cache 中 gather 前缀 K/V 拼到本轮 K/V 前面。

**边界**：只有**满块**可复用。前缀 300 token → 复用 1 块；前缀 200 token → 复用 0 块。

### 路径 D：chunked prefill

prompt 长度 > `max_num_batched_tokens`：

- 第 1 轮：`num_scheduled_tokens = max_num_batched_tokens`，`num_cached_tokens` 从 0 增到 512，**不满足完成条件**，序列留在 waiting；
- `postprocess` 中 `is_prefill and num_cached_tokens < num_tokens` 成立 → **不追加 token**（这一轮不产出新 token）；
- 第 2 轮：`seq.block_table` 非空 → `num_tokens = seq.num_tokens - seq.num_cached_tokens` → 继续算下一段。

**保护**：`if remaining < num_tokens and scheduled_seqs: break` 保证只有本批第一个序列能被切分，后续序列整体推迟。

### 路径 E：抢占

decode 阶段 `can_append` 返回 False（空闲块不足）：

```
while not self.block_manager.can_append(seq):
    if self.running: self.preempt(self.running.pop())   # 踢掉后面的序列
    else:           self.preempt(seq); break             # 只能踢自己
```

`preempt` 的效果：状态回 WAITING，`block_table` 清空，`num_cached_tokens` 归零，序列被推到 waiting 队首。被抢占的序列需要重新 prefill。

> **注意**：抢占是「完全抢占」（全部块归还，从头重算），没有保留已算部分的机制。

### 路径 F：TP > 1

见 §4.9.5。额外开销：每步 decode 都要跨进程 pickle 一遍 `Sequence`（已通过 `__getstate__` 压缩到 O(1)）。

---

## 9. 失败模式与诊断线索

| 现象 | 可能原因 | 定位位置 |
| --- | --- | --- |
| 启动即 `OSError: ... torch_python.dll` | 本机 PyTorch 运行时损坏 | 环境问题，与本仓库代码无关 |
| `FlashAttention is requested but flash-attn is not installed` | `CAI_VLLM_ATTENTION_BACKEND=flash` 但未安装 | `attention.py` `select_attention_backend` |
| `assert config.num_kvcache_blocks > 0` | `gpu_memory_utilization` 太小，或模型权重已占满显存 | `allocate_kv_cache` |
| `AssertionError` in `can_allocate` 路径 | 块池耗尽且无序列可抢占 | `BlockManager` |
| `RuntimeError: No space left on device` | KV Cache 之外的开销超预算（激活值、CUDA context） | `allocate_kv_cache` 的预算公式 |
| 输出重复 / 乱码 | `slot_mapping` 或 `block_tables` 不一致 | `prepare_prefill` / `prepare_decode` |
| 前缀复用结果与首次生成不一致 | `block.token_ids` 被提前 `reset()`，哈希指向脏块 | `_allocate_block` / `reset` 的顺序 |
| CUDA Graph 报错 | 未 warmup、shape 超出录制档位、`slot_mapping` 含脏数据 | `capture_cudagraph` |

---

## 10. 推测与待验证清单

以下内容在本机（无 CUDA、PyTorch 运行时不可用）**未能验证**，标注为待验证：

| # | 结论 | 来源 |
| --- | --- | --- |
| 1 | `allocate_kv_cache` 在目标机器上算出 26 个 KV 块 | 脚本调试记录，本机未复现 |
| 2 | prompt A = 12 token、B = 7 token | 脚本调试记录，本机未复现 |
| 3 | `torch` 后端与 `flash` 后端在数值上完全等价 | 由代码语义推导；缺 flash 侧对照实验 |
| 4 | `cu_seqlens_k` 的绝对偏移语义在 flash paged 路径下与 torch 路径一致 | 代码推导；flash 路径未运行验证 |
| 5 | 抢占后重新 prefill 时，`can_allocate` 可能再次命中原前缀 | 代码推导；未构造该场景实验 |
| 6 | `num_seqs = min(max_num_batched_tokens // seq_len, max_num_seqs)` 在其他参数下的取值 | 纯算术，可自行验算 |
| 7 | TP>1 下 `SharedMemory` 1 MiB 是否足以承载大 `batch` 的 `Sequence` 列表 | 未在 TP 环境实测 |

**另需注意的实现细节（已确认）**：

- `LLMEngine.add_request` **不套 chat template**，请求走裸文本补全模式；
- `Sequence.seq_id` 从 **1** 开始（0 被 warmup 的假序列占用）；
- `Scheduler.schedule` 中 decode 优先于 prefill，`waiting` 队列存在饥饿风险；
- 前缀缓存表**永不淘汰**，块被覆盖时才移除对应条目。

---

## 附：相关文件索引

| 文件 | 职责 |
| --- | --- |
| [`nanovllm/llm.py`](../nanovllm/llm.py) | 公开 API 外壳 |
| [`nanovllm/config.py`](../nanovllm/config.py) | 运行参数与 HF 配置读取 |
| [`nanovllm/sampling_params.py`](../nanovllm/sampling_params.py) | 采样参数 |
| [`nanovllm/engine/llm_engine.py`](../nanovllm/engine/llm_engine.py) | 引擎主循环 |
| [`nanovllm/engine/scheduler.py`](../nanovllm/engine/scheduler.py) | 调度 |
| [`nanovllm/engine/sequence.py`](../nanovllm/engine/sequence.py) | 请求状态 |
| [`nanovllm/engine/block_manager.py`](../nanovllm/engine/block_manager.py) | 块池与前缀缓存 |
| [`nanovllm/engine/model_runner.py`](../nanovllm/engine/model_runner.py) | 批次构建与前向驱动 |
| [`nanovllm/utils/context.py`](../nanovllm/utils/context.py) | 前向元数据单例 |
| [`nanovllm/utils/loader.py`](../nanovllm/utils/loader.py) | safetensors 权重加载 |
| [`nanovllm/models/qwen3.py`](../nanovllm/models/qwen3.py) | Qwen3 结构 |
| [`nanovllm/layers/attention.py`](../nanovllm/layers/attention.py) | 注意力与 KV 写入 |
| [`nanovllm/layers/linear.py`](../nanovllm/layers/linear.py) | TP 线性层 |
| [`nanovllm/layers/embed_head.py`](../nanovllm/layers/embed_head.py) | TP 词表并行 |
| [`nanovllm/layers/sampler.py`](../nanovllm/layers/sampler.py) | 采样 |
| [`nanovllm/layers/rotary_embedding.py`](../nanovllm/layers/rotary_embedding.py) | RoPE |
| [`nanovllm/layers/layernorm.py`](../nanovllm/layers/layernorm.py) | RMSNorm |
| [`nanovllm/layers/activation.py`](../nanovllm/layers/activation.py) | SwiGLU |
| [`scripts/torch_fallback_smoke.py`](../scripts/torch_fallback_smoke.py) | 端到端验证脚本 |
| [`scripts/prefix_cache_smoke.py`](../scripts/prefix_cache_smoke.py) | 前缀缓存验证脚本 |
