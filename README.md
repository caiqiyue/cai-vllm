# CAI-vLLM

CAI-vLLM 是一个以源码实践为主、学习笔记为辅的 AI Infra 推理学习仓库。项目通过一套紧凑的 Python 实现，展示现代大模型推理引擎从请求进入、调度、KV Cache 管理、模型执行到 Token 采样的核心链路。

这个仓库适合希望：

- 从代码层面理解 vLLM 类推理引擎工作原理；
- 学习 Continuous Batching、Paged KV Cache、Prefix Caching 和 Tensor Parallelism；
- 阅读 Qwen3 的 Attention、RoPE、RMSNorm、MLP 与采样实现；
- 在小型代码库中定位推理性能优化点。

> 仓库中的 `nanovllm` 是 Python 导入名；`cai-vllm` 是发行包和仓库名称。

## 核心实现

| 路径 | 作用 | 建议关注点 |
| --- | --- | --- |
| [`nanovllm/llm.py`](nanovllm/llm.py) | 用户侧 API 入口 | `LLM` 与推理引擎的接口边界 |
| [`nanovllm/engine/llm_engine.py`](nanovllm/engine/llm_engine.py) | 请求管理与生成循环 | `add_request`、`step`、`generate` |
| [`nanovllm/engine/scheduler.py`](nanovllm/engine/scheduler.py) | Prefill/Decode 调度 | 批处理、抢占与序列状态迁移 |
| [`nanovllm/engine/block_manager.py`](nanovllm/engine/block_manager.py) | KV Cache 块管理 | 块分配、释放与前缀缓存 |
| [`nanovllm/engine/model_runner.py`](nanovllm/engine/model_runner.py) | 模型执行与 CUDA Graph | 输入准备、KV Cache 分配、并行执行 |
| [`nanovllm/models/qwen3.py`](nanovllm/models/qwen3.py) | Qwen3 模型结构 | GQA、QK Norm、RoPE、SwiGLU |
| [`nanovllm/layers/`](nanovllm/layers/) | 推理基础算子 | Attention、线性并行层、归一化与采样 |
| [`nanovllm/utils/`](nanovllm/utils/) | 执行上下文与权重加载 | 推理上下文传递、Safetensors 加载 |

## 项目架构

CAI-vLLM 是一个离线批量生成型推理引擎，调用链从用户 API 开始，经过请求调度、KV Cache 管理、GPU 模型执行和采样，最后返回生成文本：

```text
LLM.generate(prompts, sampling_params)
        │
        ▼
LLMEngine：请求队列与生成循环
        │
        ├── Scheduler：Prefill/Decode 调度、批处理、抢占
        │       │
        │       └── BlockManager：Paged KV Cache、Prefix Cache、块生命周期
        │
        ▼
ModelRunner：输入准备、模型执行、采样、CUDA Graph、Tensor Parallel
        │
        ├── Qwen3ForCausalLM：Qwen3 Transformer
        │       ├── Embedding / LM Head
        │       ├── RMSNorm
        │       ├── Attention / RoPE / GQA
        │       └── SwiGLU MLP
        │
        └── Sampler：根据 logits 和温度选择下一个 Token
```

### 1. 用户接口层

- `nanovllm/llm.py`：暴露 `LLM` 类，保持 `from nanovllm import LLM` 的使用方式。
- `nanovllm/sampling_params.py`：定义温度、最大生成长度和 EOS 行为等采样参数。

### 2. 推理引擎层

- `nanovllm/engine/llm_engine.py`：创建 Tokenizer、Scheduler 和 ModelRunner，接收请求并循环执行 `step()`，直到所有序列结束。
- `nanovllm/engine/sequence.py`：保存一条序列的 Prompt、已生成 Token、状态、缓存 Token 数和 Block Table。
- `nanovllm/engine/scheduler.py`：维护 waiting/running 队列，在 Prefill 和 Decode 之间切换，支持批处理和显存不足时的抢占。
- `nanovllm/engine/block_manager.py`：把 KV Cache 划分成固定大小的块，负责分配、释放、引用计数、Prefix Cache 哈希和序列到物理块的映射。

### 3. 模型执行层

- `nanovllm/engine/model_runner.py`：把序列状态转换成 GPU 输入，分别准备 Prefill/Decode 张量，分配 KV Cache，执行模型、采样和多进程 Tensor Parallel 调用。
- `nanovllm/utils/context.py`：在 Attention 算子和 ModelRunner 之间传递当前批次的 cu-seqlens、slot mapping、context lengths 和 block tables。
- `nanovllm/utils/loader.py`：从模型目录加载 Safetensors 权重并映射到实现中的参数。

### 4. Qwen3 模型层

- `nanovllm/models/qwen3.py`：实现 Qwen3 的 Decoder Layer、Attention、MLP、Embedding 和 LM Head。
- Attention 使用 GQA、RoPE 和 Q/K 归一化；MLP 使用 SwiGLU；线性层通过 Tensor Parallel 层支持多 GPU 切分。

### 5. 算子与后端层

- `nanovllm/layers/attention.py`：实现 KV Cache 写入、Paged Attention，以及 FlashAttention 和 PyTorch SDPA 两种后端。
- `nanovllm/layers/linear.py`：实现列并行、行并行和 QKV 并行线性层。
- `nanovllm/layers/rotary_embedding.py`：实现 RoPE 位置编码。
- `nanovllm/layers/layernorm.py`：实现 RMSNorm。
- `nanovllm/layers/activation.py`：实现 SwiGLU 所需的 SiLU 与乘法融合。
- `nanovllm/layers/embed_head.py`：实现词表并行 Embedding 和并行 LM Head。
- `nanovllm/layers/sampler.py`：把 logits 转换为下一个 Token。

Attention 后端可通过环境变量选择：

```bash
export CAI_VLLM_ATTENTION_BACKEND=torch  # 学习和调试用的 PyTorch SDPA 路径
export CAI_VLLM_ATTENTION_BACKEND=flash  # 已安装 FlashAttention 时使用快速路径
export CAI_VLLM_ATTENTION_BACKEND=auto   # 默认：优先 FlashAttention，否则回退到 PyTorch
```

### 6. 一次生成的执行顺序

1. `LLM.generate` 把 Prompt 编码成 Token，并加入 Scheduler 的 waiting 队列。
2. Scheduler 在 Prefill 阶段为序列分配 Block，并计算需要处理的 Prompt Token。
3. ModelRunner 执行 Qwen3，Attention 将 K/V 写入 Paged KV Cache。
4. Scheduler 进入 Decode 阶段，每轮为运行中的序列准备一个 Token。
5. Attention 从 KV Cache 读取历史上下文，Sampler 选择新的 Token。
6. 生成到 EOS 或达到 `max_tokens` 后释放 Block，返回文本和 Token ID。

当前实现覆盖：

- 离线批量推理；
- Prefill 与 Decode 两阶段执行；
- Paged KV Cache 与 Prefix Caching；
- Tensor Parallelism；
- FlashAttention、Triton、Torch 编译与 CUDA Graph；
- Qwen3 模型加载和温度采样。

Attention 后端默认使用 `auto`：安装 FlashAttention 时走高性能 CUDA 路径，否则自动回退到 PyTorch SDPA。也可以显式选择普通 PyTorch 路径：

```bash
export CAI_VLLM_ATTENTION_BACKEND=torch
```

`torch` 后端保留 Paged KV Cache 和 GQA，但速度通常低于 FlashAttention；`flash` 后端在依赖缺失时会给出明确错误。

## 源码阅读路线

建议先跑通一次生成，再沿调用链向下阅读：

1. 从 [`example.py`](example.py) 了解模型、采样参数与 Prompt 的使用方式。
2. 阅读 [`nanovllm/llm.py`](nanovllm/llm.py) 和 `LLM.generate` 的实现入口。
3. 进入 [`llm_engine.py`](nanovllm/engine/llm_engine.py)，理解请求如何被循环调度和逐 Token 推进。
4. 对照 [`scheduler.py`](nanovllm/engine/scheduler.py) 与 [`block_manager.py`](nanovllm/engine/block_manager.py)，理解序列调度和 KV Cache 生命周期。
5. 阅读 [`model_runner.py`](nanovllm/engine/model_runner.py)，区分 Prefill、Decode、采样和 CUDA Graph 路径。
6. 进入 [`qwen3.py`](nanovllm/models/qwen3.py)，追踪 Transformer 层和 Attention 数据流。
7. 最后阅读 [`nanovllm/layers/`](nanovllm/layers/)，理解张量并行线性层、RoPE、RMSNorm、Attention 与 Sampler。

配套概念和阅读记录统一收录在 [AI Infra 推理学习笔记](docs/README.md)。

## 环境要求

- Python 3.10-3.12
- NVIDIA GPU 与匹配的 CUDA/PyTorch 环境
- 推荐 Linux；FlashAttention 和 Triton 在原生 Windows 环境通常不可用

## 安装

直接安装：

```bash
pip install git+https://github.com/caiqiyue/cai-vllm.git
```

源码学习建议使用开发模式：

```bash
git clone https://github.com/caiqiyue/cai-vllm.git
cd cai-vllm
pip install -e .
```

## 下载模型

```bash
huggingface-cli download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/
```

## 快速开始

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(["Hello, CAI-vLLM."], sampling_params)
print(outputs[0]["text"])
```

完整生成示例见 [`example.py`](example.py)，性能实验入口见 [`bench.py`](bench.py)。运行两者均需要兼容的 NVIDIA GPU、CUDA 运行时和模型权重。

## 学习笔记

[`docs/`](docs/README.md) 专门用于保存面向学习者的 AI Infra 推理笔记，包括推理基础、模型执行、显存与 KV Cache、调度与批处理、算子优化以及服务化工程等主题。

学习笔记服务于源码理解，但不替代源码。涉及实现行为时，以当前代码和可复现验证结果为准。

## Agent Harness

仓库内置 WIP=1、状态持久化和验证证据机制。Agent 开始工作前运行：

```bash
cat AGENTS.md
cat feature_list.json
bash ./init.sh
```

PowerShell 环境可运行：

```powershell
./init.ps1
```

轻量检查覆盖 Python 版本、Harness 状态、源码语法、包元数据和仓库卫生。缺少 PyTorch/CUDA 时会明确显示 `SKIP`；涉及 GPU 行为的验证必须在支持的环境中运行：

```bash
CAI_VLLM_REQUIRE_GPU=1 bash ./init.sh
```

## 项目来源

本仓库基于 MIT 许可的 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 进行学习和维护，不代表上游官方项目。仓库不沿用上游 Git 历史，新增学习笔记、Harness 和后续实验均在 CAI-vLLM 中独立维护。

## License

本项目遵循 [MIT License](LICENSE)。
