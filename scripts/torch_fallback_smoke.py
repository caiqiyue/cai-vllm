import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "/mnt/public/gw/LLM_model/Qwen3-4B-Instruct-2507"


def configure_environment():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CAI_VLLM_ATTENTION_BACKEND"] = "torch"
    os.environ.setdefault("PYTHONPATH", str(PROJECT_ROOT))
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def main():
    configure_environment()
    from nanovllm import LLM, SamplingParams

    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=16)
    args = parser.parse_args()
    llm = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=512,
        max_num_batched_tokens=512,
        max_num_seqs=2,
        gpu_memory_utilization=0.85,
    )
    outputs = llm.generate(
        [
            "Explain what a KV cache is in one short sentence.",
            "Say hello in one short sentence.",
        ],
        SamplingParams(temperature=0.6, max_tokens=args.max_tokens),
        use_tqdm=False,
    )
    for index, output in enumerate(outputs, start=1):
        print(f"[{index}] {output['text']}")


if __name__ == "__main__":
    main()
