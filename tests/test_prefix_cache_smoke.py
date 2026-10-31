import unittest

from scripts.prefix_cache_smoke import build_prompt_ids, summarize_prefix_cache


class FakeTokenizer:
    """Whitespace tokenizer: every distinct word gets one stable token id."""

    def __init__(self):
        self._ids: dict[str, int] = {}

    def encode(self, text: str) -> list[int]:
        token_ids = []
        for word in text.split():
            if word not in self._ids:
                self._ids[word] = len(self._ids)
            token_ids.append(self._ids[word])
        return token_ids


class PrefixCacheScriptTest(unittest.TestCase):

    def test_build_prompt_ids_shares_exactly_one_leading_block(self):
        tokenizer = FakeTokenizer()
        block_size = 16

        first = build_prompt_ids(tokenizer, block_size, "Explain the first question.")
        second = build_prompt_ids(tokenizer, block_size, "Explain the second question.")

        self.assertEqual(first[:block_size], second[:block_size])
        self.assertNotEqual(first[block_size:], second[block_size:])
        self.assertEqual(len(first), len(second))

    def test_summarize_prefix_cache_accepts_a_real_hit(self):
        prefills = [
            {"seq_id": 0, "is_prefill": True, "cached_tokens": 0, "scheduled_tokens": 261},
            {"seq_id": 1, "is_prefill": True, "cached_tokens": 256, "scheduled_tokens": 5},
        ]

        report = summarize_prefix_cache(prefills, block_size=256, prompt_token_counts=[261, 261])

        self.assertEqual(report["warm_cached_tokens"], 256)
        self.assertEqual(report["warm_prefill_tokens"], 5)
        self.assertEqual(report["saved_prefill_tokens"], 256)

    def test_summarize_prefix_cache_rejects_a_missed_hit(self):
        prefills = [
            {"seq_id": 0, "is_prefill": True, "cached_tokens": 0, "scheduled_tokens": 261},
            {"seq_id": 1, "is_prefill": True, "cached_tokens": 0, "scheduled_tokens": 261},
        ]

        with self.assertRaises(RuntimeError) as ctx:
            summarize_prefix_cache(prefills, block_size=256, prompt_token_counts=[261, 261])

        self.assertIn("reused only 0 cached tokens", str(ctx.exception))

    def test_summarize_prefix_cache_rejects_a_partial_reuse(self):
        prefills = [
            {"seq_id": 0, "is_prefill": True, "cached_tokens": 0, "scheduled_tokens": 261},
            {"seq_id": 1, "is_prefill": True, "cached_tokens": 256, "scheduled_tokens": 261},
        ]

        with self.assertRaises(RuntimeError):
            summarize_prefix_cache(prefills, block_size=256, prompt_token_counts=[261, 261])


if __name__ == "__main__":
    unittest.main()
