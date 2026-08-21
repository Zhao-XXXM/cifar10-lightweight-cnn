import sys
import unittest
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "notecode"))

from d20_light_cnn import count_trainable_params  # noqa: E402
from d27_width_multiplier import WidthLightVGGSlimGAP  # noqa: E402


class ModelSmokeTest(unittest.TestCase):
    def test_width_models_return_ten_logits(self):
        torch.manual_seed(0)
        inputs = torch.randn(2, 3, 32, 32)
        expected_params = {
            0.5: 10875,
            1.0: 37579,
            1.5: 80155,
        }

        for width, params in expected_params.items():
            with self.subTest(width=width):
                model = WidthLightVGGSlimGAP(width_mult=width, num_classes=10)
                logits = model(inputs)
                self.assertEqual(tuple(logits.shape), (2, 10))
                self.assertEqual(count_trainable_params(model), params)
                self.assertTrue(torch.isfinite(logits).all().item())


if __name__ == "__main__":
    unittest.main()
