import os
import unittest

from scripts.torch_fallback_smoke import configure_environment


class TorchFallbackSmokeConfigTest(unittest.TestCase):
    def test_configure_environment_selects_single_gpu_and_torch_attention(self):
        for name in (
            "CUDA_DEVICE_ORDER",
            "CUDA_VISIBLE_DEVICES",
            "CAI_VLLM_ATTENTION_BACKEND",
        ):
            os.environ.pop(name, None)

        configure_environment()

        self.assertEqual(os.environ["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(os.environ["CAI_VLLM_ATTENTION_BACKEND"], "torch")


if __name__ == "__main__":
    unittest.main()
