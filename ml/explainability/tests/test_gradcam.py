"""Unit test for Hook-based Grad-CAM."""

import os
import sys
from pathlib import Path
import unittest

# Ensure repo root is on sys.path for direct test execution
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

class TestGradCAMHook(unittest.TestCase):
    def test_hook_mechanism(self):
        try:
            import numpy as np
            import torch
            import torch.nn as nn
            from ml.explainability.gradcam import HookBasedGradCAM
        except ImportError as e:
            self.skipTest(f"Skipping PyTorch/numpy test: {e}")

        class SimpleCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
                self.fc = nn.Linear(32 * 16 * 16, 6)

            def forward(self, x):
                x = nn.functional.relu(self.conv1(x))
                x = nn.functional.max_pool2d(x, 2)
                x = nn.functional.relu(self.conv2(x))
                x = nn.functional.max_pool2d(x, 2)
                x = x.view(x.size(0), -1)
                return self.fc(x)

        model = SimpleCNN()
        gradcam = HookBasedGradCAM(model, target_layer=model.conv2)

        fake_input = torch.randn(1, 3, 64, 64, requires_grad=True)
        heatmap, pred_cls = gradcam.generate_heatmap(fake_input, target_class=0)

        self.assertEqual(heatmap.shape, (32, 32))
        self.assertTrue(0.0 <= np.max(heatmap) <= 1.0)
        gradcam.remove_hooks()
        print("✅ Grad-CAM hook test passed successfully!")

if __name__ == "__main__":
    unittest.main()

