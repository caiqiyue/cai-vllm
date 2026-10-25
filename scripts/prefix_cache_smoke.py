from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "/mnt/public/gw/LLM_model/Qwen3-4B-Instruct-2507"


def configure_environment():
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CAI_VLLM_ATTENTION_BACKEND"] = "torch"
    os.environ.setdefault("PYTHONPATH", str(PROJECT_ROOT))
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def main(model=DEFAULT_MODEL):
    configure_environment()
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.sequence import Sequence

    llm = LLM(
        model,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=512,
        max_num_batched_tokens=512,
        max_num_seqs=1,
        gpu_memory_utilization=0.85,
    )
    sampling_params = SamplingParams(temperature=0.6, max_tokens=2)
    common_prefix = "cache " * 300
    first_prompt = common_prefix + "Explain the first question."
    second_prompt = common_prefix + "Explain the second question."

    first_output = llm.generate([first_prompt], sampling_params, use_tqdm=False)[0]
    second_token_ids = llm.tokenizer.encode(second_prompt)
    second_sequence = Sequence(second_token_ids, sampling_params)
    reusable_blocks = llm.scheduler.block_manager.can_allocate(second_sequence)
    second_output = llm.generate([second_prompt], sampling_params, use_tqdm=False)[0]

    print(f"reusable_prefix_blocks={reusable_blocks}")
    print(f"first={first_output['text']!r}")
    print(f"second={second_output['text']!r}")
    if reusable_blocks <= 0:
        raise RuntimeError("expected the second prompt to reuse at least one prefix block")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    args = parser.parse_args()
    main(args.model)
