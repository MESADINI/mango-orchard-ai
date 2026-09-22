"""
AI Mango Orchard Health & Pest Intelligence — Hook-based Grad-CAM
Rule: Must use hook-based implementation (not guided backprop) so it
works on any PyTorch model without architectural modification.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, Tuple

class HookBasedGradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        Registers forward and backward hooks on the target convolutional layer.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handles = []

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self.handles.extend([h1, h2])

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Generates Grad-CAM heatmap for the given input tensor and target class.
        Returns:
            heatmap: 2D numpy array [0, 1] normalized.
            predicted_class: index of target or top class.
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)
        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        # Backward pass for the target class score
        score = logits[:, target_class]
        score.backward(retain_graph=True)

        # Global average pooling of gradients
        # Gradients: [B, C, H, W] -> pooled_grads: [C]
        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])

        # Weight the channels by corresponding gradients
        activations = self.activations[0]  # [C, H, W]
        for i in range(activations.size(0)):
            activations[i, :, :] *= pooled_grads[i]

        # Heatmap is the channel-wise sum followed by ReLU
        heatmap = torch.sum(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)

        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap /= max_val

        return heatmap, target_class

    def overlay_on_image(
        self,
        original_image_rgb: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Resizes heatmap to original image dimensions and applies color overlay.
        """
        h, w, _ = original_image_rgb.shape
        resized_heatmap = cv2.resize(heatmap, (w, h))
        colored_heatmap = cv2.applyColorMap(np.uint8(255 * resized_heatmap), colormap)
        colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

        overlay = np.uint8((1.0 - alpha) * original_image_rgb + alpha * colored_heatmap)
        return overlay

    def remove_hooks(self):
        """Cleanly deregisters all PyTorch hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __del__(self):
        self.remove_hooks()
