# feat-004 nano-vLLM 推理全流程学习笔记 — 2026-10-29

## Goal

为 `nanovllm/` 推理链路产出一篇可独立阅读的机制说明笔记，覆盖组件全景、阶段拆解与关键组件内部机制，使读者不逐行读源码也能建立整条链路的正确心智模型。

## Implementation

- 新增 `docs/nano-vllm-inference-flow.md`。
- 文档结构：
  - 第 1 章 全景：一次 `generate()` 的调用链与组件依赖图；
  - 第 3 章 阶段拆解：参数收敛 → 模型装入 → 显存规划 → 请求接纳 → 调度 → 批次构建 → 前向 → KV 写入 → logits → 后处理 → 回收 → 收尾，共 10 个阶段；
  - 第 4 章 核心机制详解：`Sequence` 状态机、`Scheduler`、`BlockManager`、Paged KV Cache 双层索引、`ModelRunner` 与 `Context` 契约、Attention 后端抽象、显存预算、CUDA Graph、Tensor Parallel 与进程模型、采样与结束判定，共 10 节；
  - 第 5 章 张量与形状契约表；第 6 章 关键常量与来源；
  - 第 7 章 逐帧走查（以 `scripts/torch_fallback_smoke.py` 参数为例，帧 0 → 帧 1 prefill → 帧 2–16 decode → 帧 17 终止）；
  - 第 8 章 六条典型路径（单请求 / 多请求同批 / 前缀命中 / chunked prefill / 抢占 / TP>1）；
  - 第 9 章 失败模式与诊断线索；第 10 章 推测与待验证清单。
- 更新 `docs/README.md`：把新笔记登记为笔记路线的公共基础入口，各专题状态标注为「机制部分见推理全流程，专题待补充」。
- 更新根 `README.md`：在「学习笔记」与「源码阅读路线」两处加入新笔记入口。
- 更新 `harness.md`：把学习笔记入口登记进本仓库实际文件表。
- 新增 `feat-004` 到 `feature_list.json` 与 `.ai/state/feature-list.json`，两份保持逐字节一致。

## Scope

- 只新增/修改文档与 harness 状态文件。
- 未改动 `nanovllm/` 任何源码、测试或运行时脚本。
- 未改变 `feat-002` 的状态、验证条件或证据链。

## Verification

- 本地 Markdown 链接检查（覆盖 `docs/**/*.md` 与根 `README.md`）：**0 broken**。

  ```bash
  python - <<'PY'
  import re, pathlib
  root = pathlib.Path('.')
  bad = []
  for md in sorted(root.glob('docs/**/*.md')) + [root/'README.md']:
      text = md.read_text(encoding='utf-8')
      for m in re.finditer(r'\[([^\]]*)\]\(([^)]+)\)', text):
          target = m.group(2)
          if target.startswith(('http://','https://','#','mailto:')):
              continue
          p = (md.parent / target.split('#')[0]).resolve()
          if not p.exists():
              bad.append(f"{md}: {target}")
  print("broken links:", len(bad))
  for b in bad: print("  ", b)
  PY
  ```

  输出：`broken links: 0`

- `bash ./init.sh`：

  ```
  [L0] Environment                                    [PASS] Python 3.11
  [L1] Harness state                                  [PASS] one active feature: feat-002
  [L2] Python syntax and package metadata            [PASS]
  [L3] Repository hygiene                            [PASS] no whitespace errors
  [L4] Runtime dependency and import smoke            [SKIP] GPU/runtime smoke unavailable
  === Verification complete ===
  ```

  L4 为显式 `SKIP`，原因是本机 PyTorch 无法加载 `torch_python.dll`（`WinError 127`），与本次改动无关，也未被转写为静默通过。

- `git status` 仅包含预期的文档与 harness 状态文件改动。

### Note-to-source spot check

仓库笔记规则要求「笔记改动的 feature 必须抽样核对 ≥3 处与源码一致」。本次抽样 10 处，全部一致：

| # | 笔记中的结论 | 源码位置 | 实际代码 |
| --- | --- | --- | --- |
| 1 | `num_blocks` 向上取整 | `nanovllm/engine/sequence.py:57` | `(self.num_tokens + self.block_size - 1) // self.block_size` |
| 2 | `last_block_num_tokens` 公式 | `nanovllm/engine/sequence.py:61` | `self.num_tokens - (self.num_blocks - 1) * self.block_size` |
| 3 | `hash_blocks` 只处理满块 | `nanovllm/engine/block_manager.py:131-133` | `start = num_cached_tokens // block_size`，`end = (num_cached_tokens + num_scheduled_tokens) // block_size`，`if start == end: return` |
| 4 | `_deallocate_block` 不删哈希条目 | `nanovllm/engine/block_manager.py:69-72` | 仅 `used_block_ids.remove` + `free_block_ids.append`，无 `hash_to_block_id` 操作 |
| 5 | 跨块判定 `len(seq) % block_size == 1` | `nanovllm/engine/block_manager.py:119-124` | `can_append` / `may_append` 一致 |
| 6 | decode 的 `slot_mapping` 公式 | `nanovllm/engine/model_runner.py:248` | `seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1` |
| 7 | 前缀命中才构造 `block_tables` | `nanovllm/engine/model_runner.py:218` | `if cu_seqlens_k[-1] > cu_seqlens_q[-1]:` |
| 8 | 显存预算公式 | `nanovllm/engine/model_runner.py:143` | `int(total * gpu_memory_utilization - used - peak + current) // block_bytes` |
| 9 | `add_request` 不套 chat template | `nanovllm/engine/llm_engine.py:104` | `prompt = self.tokenizer.encode(prompt)` |
| 10 | 采样禁止贪心 | `nanovllm/sampling_params.py:11` | `assert self.temperature > 1e-10, "greedy sampling is not permitted"` |

附带确认：warmup 在 `nanovllm/engine/model_runner.py:127` 创建一个 `Sequence`，而 `seq_id` 来自 `nanovllm/engine/sequence.py:19` 的全局计数器，因此用户请求的 `seq_id` 从 1 开始；`nanovllm/engine/model_runner.py:57` 在模型初始化末尾把默认设备复位为 CPU。

## Unverified / Not covered

- 笔记中「26 个 KV 块」「prompt A=12 token / B=7 token」等数值来自服务器端脚本调试记录，本机无 CUDA，未复现。
- torch 与 flash 两个 Attention 后端的数值等价性、flash paged 路径下 `cu_seqlens_k` 的绝对偏移语义，均为代码推导结论，标记为待验证。
- 文档不包含与生产级推理框架的功能对照，也不包含本仓库的逐行代码走读，两者按需求明确排除。

## Related

- `feat-002` 仍是唯一 `in_progress` feature，本笔记不改变其阻塞原因（本地 PyTorch 运行时不可用，GPU 验证需在受支持环境执行）。
