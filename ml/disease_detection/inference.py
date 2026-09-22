"""
AI Mango Orchard Health & Pest Intelligence — Disease Inference Pipeline
Architecture: EfficientNet-B3
"""

import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import timm
from typing import Dict, Any, List

DISEASE_CLASSES = [
    "Anthracnose",
    "Bacterial Canker",
    "Powdery Mildew",
    "Sooty Mold",
    "Leaf Spot",
    "Healthy"
]

class DiseaseDetector:
    def __init__(self, weights_path: str = "weights/efficientnet_b3_best.pth", device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=len(DISEASE_CLASSES))
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            print(f"Loaded weights from {weights_path}")
        else:
            print(f"Warning: Weights file {weights_path} not found. Running in untrained/dummy mode.")
            
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Runs forward pass and returns top prediction, confidence, and class probabilities."""
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        top_idx = int(probs.argmax())
        disease_name = DISEASE_CLASSES[top_idx]
        confidence = float(probs[top_idx])

        return {
            "disease_name": disease_name,
            "confidence": confidence,
            "is_healthy": disease_name == "Healthy",
            "all_probabilities": {cls_name: float(probs[i]) for i, cls_name in enumerate(DISEASE_CLASSES)}
        }
