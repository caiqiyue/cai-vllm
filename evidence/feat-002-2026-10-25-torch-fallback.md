# feat-002 torch Attention fallback verification — 2026-10-25

## Implementation

- Added `CAI_VLLM_ATTENTION_BACKEND=auto|flash|torch`.
- `auto` uses FlashAttention when installed and otherwise selects PyTorch SDPA.
- `torch` preserves GQA, Paged KV Cache gathering, and KV Cache writes using regular PyTorch indexing.
- The original FlashAttention and Triton fast paths remain available.

## Verification

- Server commit: `4939b17`
- GPU restriction: `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=0`
- Server GPU: NVIDIA GeForce RTX 2080 Ti
- Environment: `/mnt/public/caiqiyue_file/conda-envs/cai-vllm`
- `CAI_VLLM_REQUIRE_GPU=1 bash ./init.sh`: PASS
- `python -m unittest tests.test_attention_backend`: PASS, 2 tests
- Real Qwen3 generation using `/mnt/public/gw/LLM_model/Qwen3-4B-Instruct-2507`, `tensor_parallel_size=1`, `enforce_eager=True`, and `CAI_VLLM_ATTENTION_BACKEND=torch`: PASS
- Generated output: `Hello! 😊` (short response)

## Remaining limitation

- FlashAttention installation/build remains unverified on this server. The normal PyTorch path is now usable without the `flash_attn` package; FlashAttention remains an optional acceleration path.
