import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1] #'/mnt/public/caiqiyue_file/cai-vllm'
DEFAULT_MODEL = "/mnt/public/gw/LLM_model/Qwen3-4B-Instruct-2507"


def configure_environment():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CAI_VLLM_ATTENTION_BACKEND"] = "torch"
    os.environ.setdefault("PYTHONPATH", str(PROJECT_ROOT))
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def main():
    configure_environment() #配置 环境变量
    from nanovllm import LLM, SamplingParams

    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default=DEFAULT_MODEL) # /mnt/public/gw/LLM_model/Qwen3-4B-Instruct-2507
    parser.add_argument("--max-tokens", type=int, default=16) #最大输出token数 16
    args = parser.parse_args()
    llm = LLM( # llm 对象，
        args.model,  # 底层的模型
        enforce_eager=True, # false 的话就是 使用cuda 计算图；true ，就是 pytorch 一个算子一个算子给cuda去算
        tensor_parallel_size=1, #
        max_model_len=512, #
        max_num_batched_tokens=512, #
        max_num_seqs=2, #
        gpu_memory_utilization=0.85, #
    )
    outputs = llm.generate(
        [
            "Explain what a KV cache is in one short sentence.",
            "Say hello in one short sentence.",
        ],
        SamplingParams(temperature=0.6, max_tokens=args.max_tokens),
        use_tqdm=False,
    )
    """
    
        0 = {'text': ' A KV cache stores the key-value pairs from previous tokens in a sequence, allowing', 'token_ids': [362, 84648, 6500, 10533, 279, 1376, 19083, 13530, 504, 3681, 11211, 304, 264, 8500, 11, 10693]}
        special variables
        function variables
        'text' =
        ' A KV cache stores the key-value pairs from previous tokens in a sequence, allowing'
        'token_ids' =
        [362, 84648, 6500, 10533, 279, 1376, 19083, 13530, 504, 3681, 11211, 304, 264, 8500, 11, 10693]
        len() = 2
    """
    for index, output in enumerate(outputs, start=1):
        print(f"[{index}] {output['text']}")


if __name__ == "__main__":
    main()
