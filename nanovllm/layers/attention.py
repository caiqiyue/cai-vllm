import os

import torch
from torch import nn
import torch.nn.functional as F
import triton
import triton.language as tl

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except ImportError:
    flash_attn_varlen_func = flash_attn_with_kvcache = None
from nanovllm.utils.context import get_context


def select_attention_backend() -> str:
    backend = os.getenv("CAI_VLLM_ATTENTION_BACKEND", "auto").lower()
    if backend not in {"auto", "flash", "torch"}:
        raise ValueError("CAI_VLLM_ATTENTION_BACKEND must be auto, flash, or torch")
    if backend == "flash":
        if flash_attn_varlen_func is None or flash_attn_with_kvcache is None:
            raise RuntimeError("FlashAttention is requested but flash-attn is not installed")
        return "flash"
    if backend == "torch":
        return "torch"
    return "flash" if flash_attn_varlen_func is not None and flash_attn_with_kvcache is not None else "torch"


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


def store_kvcache_torch(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    slots = slot_mapping.to(torch.long)
    valid = slots >= 0
    flat_k_cache = k_cache.reshape(-1, key.size(1), key.size(2))
    flat_v_cache = v_cache.reshape(-1, value.size(1), value.size(2))
    flat_k_cache[slots[valid]] = key[valid]
    flat_v_cache[slots[valid]] = value[valid]


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    @staticmethod
    def _repeat_kv(x: torch.Tensor, num_heads: int) -> torch.Tensor:
        if x.size(1) == num_heads:
            return x
        assert num_heads % x.size(1) == 0
        return x.repeat_interleave(num_heads // x.size(1), dim=1)

    def _gather_cache(self, cache: torch.Tensor, block_table: torch.Tensor, seqlen: int):
        block_size = cache.size(1)
        num_blocks = (seqlen + block_size - 1) // block_size
        block_ids = block_table[:num_blocks].to(torch.long)
        return cache.index_select(0, block_ids).reshape(-1, cache.size(2), cache.size(3))[:seqlen]

    def _forward_torch(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            outputs = []
            for i in range(context.cu_seqlens_q.numel() - 1):
                q_start, q_end = context.cu_seqlens_q[i:i + 2].tolist()
                q_i = q[q_start:q_end]
                q_len = q_i.size(0)
                k_i, v_i = k[q_start:q_end], v[q_start:q_end]
                k_len = context.cu_seqlens_k[i + 1].item() - context.cu_seqlens_k[i].item()
                prefix_len = k_len - q_len
                if prefix_len:
                    cached_k = self._gather_cache(self.k_cache, context.block_tables[i], prefix_len)
                    cached_v = self._gather_cache(self.v_cache, context.block_tables[i], prefix_len)
                    k_i = torch.cat((cached_k, k_i), dim=0)
                    v_i = torch.cat((cached_v, v_i), dim=0)
                q_i = q_i.transpose(0, 1).unsqueeze(0)
                k_i = self._repeat_kv(k_i, self.num_heads).transpose(0, 1).unsqueeze(0)
                v_i = self._repeat_kv(v_i, self.num_heads).transpose(0, 1).unsqueeze(0)
                query_positions = torch.arange(q_len, device=q.device) + prefix_len
                key_positions = torch.arange(k_len, device=q.device)
                causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
                output = F.scaled_dot_product_attention(
                    q_i,
                    k_i,
                    v_i,
                    attn_mask=causal_mask,
                    dropout_p=0.0,
                    scale=self.scale,
                )
                outputs.append(output.squeeze(0).transpose(0, 1))
            return torch.cat(outputs, dim=0)

        outputs = []
        for i, seqlen in enumerate(context.context_lens.tolist()):
            k_i = self._gather_cache(self.k_cache, context.block_tables[i], seqlen)
            v_i = self._gather_cache(self.v_cache, context.block_tables[i], seqlen)
            q_i = q[i].unsqueeze(0).unsqueeze(2)
            k_i = self._repeat_kv(k_i, self.num_heads).transpose(0, 1).unsqueeze(0)
            v_i = self._repeat_kv(v_i, self.num_heads).transpose(0, 1).unsqueeze(0)
            output = F.scaled_dot_product_attention(q_i, k_i, v_i, dropout_p=0.0, scale=self.scale)
            outputs.append(output.squeeze(0).squeeze(1))
        return torch.stack(outputs, dim=0)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        backend = select_attention_backend()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            if backend == "torch":
                store_kvcache_torch(k, v, k_cache, v_cache, context.slot_mapping)
            else:
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if backend == "torch":
            return self._forward_torch(q, k, v)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True)
        return o
