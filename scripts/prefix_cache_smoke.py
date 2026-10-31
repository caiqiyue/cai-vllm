"""Prefix Cache smoke test.

The earlier version of this script only pre-checked the BlockManager hash index
with a hand-built Sequence, so it could never prove that the second request
actually reused the cached KV block. This version observes the real scheduler
steps and verifies the engine-level prefix cache hit:

  cold request : prefill = full prompt tokens, cached tokens = 0
  warm request : prefill = prompt tokens - block_size, cached tokens >= block_size

Run on a supported CUDA host:
    python scripts/prefix_cache_smoke.py [model_path]
"""
import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "/mnt/public/gw/LLM_model/Qwen3-4B-Instruct-2507"
FILLER = "cache "  # 填充词：重复足够多次，保证 prompt 能填满至少一个完整 block


def configure_environment():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CAI_VLLM_ATTENTION_BACKEND"] = "torch"
    os.environ.setdefault("PYTHONPATH", str(PROJECT_ROOT))
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def build_prompt_ids(tokenizer, block_size: int, suffix: str) -> list[int]:
    """构造 prompt 的 token id 列表。

    前 block_size 个 token 完全一致（同一个完整 block），后缀不同。
    用 token id 而不是字符串拼接，保证两个 prompt 的首个 block 一定逐 token 相同。
    """
    shared = tokenizer.encode(FILLER * (block_size + 64))[:block_size]
    if len(shared) < block_size:
        raise RuntimeError(f"shared prefix too short: {len(shared)} < {block_size}")
    return shared + tokenizer.encode(suffix)


def observe_scheduler(llm) -> list[dict]:
    """给 scheduler.schedule 打一个只读观测钩子，记录每一步真实的 prefill 情况。

    不修改 nanovllm 源码，只在本脚本内包装一层。
    """
    records: list[dict] = []
    original_schedule = llm.scheduler.schedule

    def schedule():
        seqs, is_prefill = original_schedule()
        for seq in seqs:
            records.append({
                "seq_id": seq.seq_id,
                "is_prefill": bool(seq.is_prefill),
                "cached_tokens": seq.num_cached_tokens,
                "scheduled_tokens": seq.num_scheduled_tokens,
            })
        return seqs, is_prefill

    llm.scheduler.schedule = schedule
    return records


def summarize_prefix_cache(prefills: list[dict], block_size: int, prompt_token_counts: list[int]) -> dict:
    """纯函数：校验引擎级的 prefix cache 命中，不满足就抛出可读的失败原因。"""
    if len(prefills) < 2:
        raise RuntimeError(f"expected two prefill steps, observed {len(prefills)}")

    cold, warm = prefills[0], prefills[1]
    expected_cold, expected_warm = prompt_token_counts[0], prompt_token_counts[1]
    problems = []

    if cold["cached_tokens"] != 0:
        problems.append(f"cold request unexpectedly started from {cold['cached_tokens']} cached tokens")
    if cold["scheduled_tokens"] != expected_cold:
        problems.append(f"cold request prefilled {cold['scheduled_tokens']} tokens, expected {expected_cold}")
    if warm["cached_tokens"] < block_size:
        problems.append(f"warm request reused only {warm['cached_tokens']} cached tokens, expected >= {block_size}")
    if warm["scheduled_tokens"] != expected_warm - block_size:
        problems.append(
            f"warm request prefilled {warm['scheduled_tokens']} tokens, expected {expected_warm - block_size}"
        )
    if warm["scheduled_tokens"] >= cold["scheduled_tokens"]:
        problems.append("warm request did not prefill fewer tokens than the cold request")

    if problems:
        raise RuntimeError("prefix cache verification failed: " + "; ".join(problems))

    return {
        "block_size": block_size,
        "cold_prompt_tokens": expected_cold,
        "cold_prefill_tokens": cold["scheduled_tokens"],
        "warm_prompt_tokens": expected_warm,
        "warm_prefill_tokens": warm["scheduled_tokens"],
        "warm_cached_tokens": warm["cached_tokens"],
        "saved_prefill_tokens": cold["scheduled_tokens"] - warm["scheduled_tokens"],
    }


def main(model=DEFAULT_MODEL):
    configure_environment()
    from nanovllm import LLM, SamplingParams

    llm = LLM(
        model,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=512,
        max_num_batched_tokens=512,
        max_num_seqs=1,
        gpu_memory_utilization=0.85,
    )
    block_size = llm.scheduler.block_manager.block_size
    sampling_params = SamplingParams(temperature=0.6, max_tokens=2)

    first_ids = build_prompt_ids(llm.tokenizer, block_size, "Explain the first question.")
    second_ids = build_prompt_ids(llm.tokenizer, block_size, "Explain the second question.")
    if first_ids[:block_size] != second_ids[:block_size]:
        raise RuntimeError("the two prompts do not share an identical leading block")

    records = observe_scheduler(llm)

    llm.generate([first_ids], sampling_params, use_tqdm=False)
    llm.generate([second_ids], sampling_params, use_tqdm=False)

    prefills = [r for r in records if r["is_prefill"]]
    report = summarize_prefix_cache(prefills, block_size, [len(first_ids), len(second_ids)])

    # 附带的块管理级别证据：哈希索引里确实存在该前缀块
    cached_hash_blocks = len(llm.scheduler.block_manager.hash_to_block_id)

    print(f"block_size={report['block_size']}")
    print(
        f"cold: prompt_tokens={report['cold_prompt_tokens']} "
        f"prefill_tokens={report['cold_prefill_tokens']} cached_tokens=0"
    )
    print(
        f"warm: prompt_tokens={report['warm_prompt_tokens']} "
        f"prefill_tokens={report['warm_prefill_tokens']} "
        f"cached_tokens={report['warm_cached_tokens']}"
    )
    print(f"saved_prefill_tokens={report['saved_prefill_tokens']}")
    print(f"hash_index_blocks={cached_hash_blocks}")
    print("prefix_cache_hit=YES")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    args = parser.parse_args()
    main(args.model)
