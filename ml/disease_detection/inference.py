"""
AI Mango Orchard Health & Pest Intelligence — Disease Inference Pipeline
Architecture: any timm backbone; the checkpoint records arch, input size and normalisation (legacy
checkpoints without these fields are EfficientNet-B3 at 300 px with ImageNet normalisation).

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
# Leaf Spot was dropped: MangoLeafBD has no Leaf Spot images and no openly licensed, verifiable
# supplementary source was found (see docs/datasets/mangoleafbd.md).
DISEASE_CLASSES = [
    "Anthracnose",
    "Bacterial Canker",
    "Powdery Mildew",
    "Sooty Mold",
    "Healthy"
]

UNAVAILABLE_LABEL = "Unavailable"
IMG_SIZE = 300


def select_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    # is_available() can be True while no device is visible (e.g. CUDA_VISIBLE_DEVICES=""); loading a checkpoint
    # onto "cuda" then fails, so a visible device is required as well
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_eval_transform(img_size: int = IMG_SIZE, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(mean), std=list(std))
    ])


def load_checkpoint(weights_path: str, map_location) -> Dict[str, Any]:
    """
    Loads a checkpoint and returns {"state_dict", "classes", "arch", "img_size", "mean", "std", "temperature"}.

    Supports two formats:
    - New format written by train.py: {"state_dict": ..., "classes": [...]}
    - Legacy raw state_dict written by the original train.py, which used
      torchvision ImageFolder's alphabetical folder order as label indices.
    """
    ckpt = torch.load(weights_path, map_location=map_location)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return {
            "state_dict": ckpt["state_dict"],
            "classes": list(ckpt.get("classes") or DISEASE_CLASSES),
            "arch": ckpt.get("timm_id") or ckpt.get("arch") or "efficientnet_b3",
            "img_size": int(ckpt.get("img_size") or IMG_SIZE),
            "mean": tuple(ckpt.get("mean") or IMAGENET_MEAN),
            "std": tuple(ckpt.get("std") or IMAGENET_STD),
            "temperature": float(ckpt.get("temperature") or 1.0),
            "ood_threshold": ckpt.get("ood_threshold"),
        }

    legacy_classes = sorted(DISEASE_CLASSES)
    logger.warning(
        "Checkpoint %s is a legacy raw state_dict; assuming alphabetical class order %s. "
        "Retrain with ml/disease_detection/train.py to embed the class list.",
        weights_path, legacy_classes
    )
    return {"state_dict": ckpt, "classes": legacy_classes, "arch": "efficientnet_b3", "img_size": IMG_SIZE,
            "mean": IMAGENET_MEAN, "std": IMAGENET_STD, "temperature": 1.0, "ood_threshold": None}


class DiseaseDetector:
    def __init__(self, weights_path: Optional[str] = None, device: Optional[str] = None):
        self.device = select_device(device)
        self.weights_path = weights_path
        self.classes: List[str] = list(DISEASE_CLASSES)
        self.model: Optional[torch.nn.Module] = None
        self.transform = build_eval_transform()
        self.arch = None
        self.temperature = 1.0
        self.ood_threshold: Optional[float] = None

        if weights_path and os.path.isfile(weights_path):
            ckpt = load_checkpoint(weights_path, map_location=self.device)
            self.classes = ckpt["classes"]
            self.arch = ckpt["arch"]
            self.temperature = ckpt["temperature"]
            self.ood_threshold = ckpt["ood_threshold"]
            self.transform = build_eval_transform(ckpt["img_size"], ckpt["mean"], ckpt["std"])
            model = timm.create_model(self.arch, pretrained=False, num_classes=len(self.classes))
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
                "low_confidence": False,
                "all_probabilities": {}
            }

        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits.float() / self.temperature, dim=1).squeeze(0).cpu().numpy()

        top_idx = int(probs.argmax())
        disease_name = self.classes[top_idx]

        return {
            "disease_name": disease_name,
            "confidence": float(probs[top_idx]),
            "is_healthy": disease_name == "Healthy",
            "model_available": True,
            # below the threshold that keeps 95% of validation images: likely not a clear leaf photo of a known
            # class (unknown disease, pest damage, other plant, bad photo) -> the app should ask for a retake
            "low_confidence": self.ood_threshold is not None and float(probs[top_idx]) < float(self.ood_threshold),
            "all_probabilities": {cls_name: float(probs[i]) for i, cls_name in enumerate(self.classes)}
        }
