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


if __name__ == "__main__":
    unittest.main()
