"""
AI Mango Orchard Health & Pest Intelligence — Disease Inference Pipeline
Architecture: EfficientNet-B3

If no trained checkpoint is available the detector reports `is_trained = False`
and `predict()` returns an explicit "unavailable" result instead of the output
of a randomly initialised network.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

# Canonical class order. Model output index i always means DISEASE_CLASSES[i].
DISEASE_CLASSES = [
    "Anthracnose",
    "Bacterial Canker",
    "Powdery Mildew",
    "Sooty Mold",
    "Leaf Spot",
    "Healthy"
]

UNAVAILABLE_LABEL = "Unavailable"
IMG_SIZE = 300


def select_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_eval_transform(img_size: int = IMG_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_checkpoint(weights_path: str, map_location) -> Dict[str, Any]:
    """
    Loads a checkpoint and returns {"state_dict": ..., "classes": [...]}.

    Supports two formats:
    - New format written by train.py: {"state_dict": ..., "classes": [...]}
    - Legacy raw state_dict written by the original train.py, which used
      torchvision ImageFolder's alphabetical folder order as label indices.
    """
    ckpt = torch.load(weights_path, map_location=map_location)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        classes = list(ckpt.get("classes") or DISEASE_CLASSES)
        return {"state_dict": ckpt["state_dict"], "classes": classes}

    legacy_classes = sorted(DISEASE_CLASSES)
    logger.warning(
        "Checkpoint %s is a legacy raw state_dict; assuming alphabetical class order %s. "
        "Retrain with ml/disease_detection/train.py to embed the class list.",
        weights_path, legacy_classes
    )
    return {"state_dict": ckpt, "classes": legacy_classes}


class DiseaseDetector:
    def __init__(self, weights_path: Optional[str] = None, device: Optional[str] = None):
        self.device = select_device(device)
        self.weights_path = weights_path
        self.classes: List[str] = list(DISEASE_CLASSES)
        self.model: Optional[torch.nn.Module] = None
        self.transform = build_eval_transform()

        if weights_path and os.path.isfile(weights_path):
            ckpt = load_checkpoint(weights_path, map_location=self.device)
            self.classes = ckpt["classes"]
            model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=len(self.classes))
            model.load_state_dict(ckpt["state_dict"])
            model.to(self.device)
            model.eval()
            self.model = model
            logger.info("Loaded disease model weights from %s", weights_path)
        else:
            logger.warning(
                "Disease model weights not found (%s). Disease classification is disabled "
                "until a model is trained with ml/disease_detection/train.py.", weights_path
            )

    @property
    def is_trained(self) -> bool:
        return self.model is not None

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Runs a forward pass and returns top prediction, confidence and class probabilities."""
        if self.model is None:
            return {
                "disease_name": UNAVAILABLE_LABEL,
                "confidence": 0.0,
                "is_healthy": False,
                "model_available": False,
                "all_probabilities": {}
            }

        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        top_idx = int(probs.argmax())
        disease_name = self.classes[top_idx]

        return {
            "disease_name": disease_name,
            "confidence": float(probs[top_idx]),
            "is_healthy": disease_name == "Healthy",
            "model_available": True,
            "all_probabilities": {cls_name: float(probs[i]) for i, cls_name in enumerate(self.classes)}
        }
