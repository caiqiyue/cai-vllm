import argparse

from nanovllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()
    llm = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=512,
        max_num_batched_tokens=512,
        max_num_seqs=1,
        gpu_memory_utilization=0.85,
    )
    outputs = llm.generate(
        ["Say hello in one short sentence."],
        SamplingParams(temperature=0.0, max_tokens=8),
        use_tqdm=False,
    )
    print(outputs)


if __name__ == "__main__":
    main()
