import argparse
import os

from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


DEFAULT_MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B/")


def main():
    # 模型路径可从命令行传入，服务器上使用 /mnt/public/gw/LLM_model/Qwen3-4B-Instruct-2507
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    path = args.model
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(
        path,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    sampling_params = SamplingParams(temperature=0.6, max_tokens=args.max_tokens)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
