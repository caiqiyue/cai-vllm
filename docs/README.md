# AI Infra 推理学习笔记

本目录用于保存面向学习者的 AI Infra 与大模型推理笔记。仓库以 [`nanovllm/`](../nanovllm/) 源码实践为主，笔记负责补充背景知识、关键概念、源码阅读记录和实验结论。

开发方案、Agent 执行计划、临时分析和交接材料不属于本目录内容。

## 使用方式

建议将笔记与源码交叉阅读：

1. 从根目录 [`README.md`](../README.md) 获取源码地图和阅读路线。
2. 运行 [`example.py`](../example.py)，观察一次完整生成过程。
3. 选择下方主题补齐概念，再回到对应源码定位实现。
4. 对性能结论进行本机复测，记录硬件、依赖版本、参数和可复现命令。

## 笔记路线

当前目录刚建立，以下主题均为规划内容；在正文完成并经过源码或实验核对之前，一律标记为“待补充”。

| 方向 | 计划内容 | 对应源码 | 状态 |
| --- | --- | --- | --- |
| 推理基础 | Autoregressive Decoding、Prefill/Decode、吞吐与时延 | [`engine/llm_engine.py`](../nanovllm/engine/llm_engine.py) | 待补充 |
| 模型执行 | Qwen3 结构、GQA、QK Norm、RoPE、SwiGLU | [`models/qwen3.py`](../nanovllm/models/qwen3.py) | 待补充 |
| 显存与 KV Cache | Paged KV Cache、Block Table、Prefix Caching | [`engine/block_manager.py`](../nanovllm/engine/block_manager.py) | 待补充 |
| 调度与批处理 | Continuous Batching、抢占、序列状态 | [`engine/scheduler.py`](../nanovllm/engine/scheduler.py) | 待补充 |
| 算子与性能 | FlashAttention、Triton、CUDA Graph、Torch 编译 | [`layers/`](../nanovllm/layers/) | 待补充 |
| 并行推理 | Tensor Parallelism、权重切分与进程协作 | [`layers/linear.py`](../nanovllm/layers/linear.py) | 待补充 |
| 服务化工程 | 动态批处理、流式输出、指标与稳定性 | 当前源码尚未覆盖 | 待补充 |

## 笔记约定

- 每篇笔记必须说明对应源码版本或提交。
- 实现相关结论必须链接到具体源码位置。
- 性能数据必须记录硬件、软件版本、参数和执行命令。
- 未实际验证的内容标记为“推测”或“待验证”。
- 笔记只保存稳定的学习成果，不提交开发过程中的临时文档。
