import os
import unittest

import torch
import torch.nn.functional as F

from nanovllm.layers.attention import Attention
from nanovllm.utils.context import reset_context, set_context


class TorchAttentionBackendTest(unittest.TestCase):
    def setUp(self):
        os.environ["CAI_VLLM_ATTENTION_BACKEND"] = "torch"

    def tearDown(self):
        os.environ.pop("CAI_VLLM_ATTENTION_BACKEND", None)
        reset_context()

    def test_torch_prefill_matches_causal_sdpa(self):
        torch.manual_seed(0)
        tokens, heads, head_dim = 3, 2, 4
        q = torch.randn(tokens, heads, head_dim)
        k = torch.randn(tokens, heads, head_dim)
        v = torch.randn(tokens, heads, head_dim)
        set_context(
            True,
            cu_seqlens_q=torch.tensor([0, tokens], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, tokens], dtype=torch.int32),
            max_seqlen_q=tokens,
            max_seqlen_k=tokens,
        )

        actual = Attention(heads, head_dim, head_dim ** -0.5, heads)(q, k, v)
        causal_mask = torch.ones(tokens, tokens, dtype=torch.bool).tril()
        expected = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            attn_mask=causal_mask,
            dropout_p=0.0,
        ).squeeze(0).transpose(0, 1)
        torch.testing.assert_close(actual, expected)

    def test_torch_decode_writes_and_reads_paged_cache(self):
        torch.manual_seed(1)
        heads, kv_heads, head_dim, block_size = 2, 1, 2, 4
        q = torch.randn(1, heads, head_dim)
        k = torch.randn(1, kv_heads, head_dim)
        v = torch.randn(1, kv_heads, head_dim)
        cached_k = torch.randn(3, kv_heads, head_dim)
        cached_v = torch.randn(3, kv_heads, head_dim)
        set_context(
            False,
            slot_mapping=torch.tensor([3], dtype=torch.int32),
            context_lens=torch.tensor([4], dtype=torch.int32),
            block_tables=torch.tensor([[0]], dtype=torch.int32),
        )
        attention = Attention(heads, head_dim, head_dim ** -0.5, kv_heads)
        attention.k_cache = torch.zeros(1, block_size, kv_heads, head_dim)
        attention.v_cache = torch.zeros_like(attention.k_cache)
        attention.k_cache[0, :3] = cached_k
        attention.v_cache[0, :3] = cached_v

        actual = attention(q, k, v)
        full_k = torch.cat((cached_k, k), dim=0).repeat_interleave(heads, dim=1)
        full_v = torch.cat((cached_v, v), dim=0).repeat_interleave(heads, dim=1)
        expected = F.scaled_dot_product_attention(
            q.unsqueeze(2),
            full_k.transpose(0, 1).unsqueeze(0),
            full_v.transpose(0, 1).unsqueeze(0),
            dropout_p=0.0,
            scale=head_dim ** -0.5,
        ).squeeze(0).squeeze(1)
        expected = expected.unsqueeze(0)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
