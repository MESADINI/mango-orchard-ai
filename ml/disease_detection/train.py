"""
AI Mango Orchard Health & Pest Intelligence — Disease Detection Training
Architecture: EfficientNet-B3 (PyTorch / timm)
Target Classes: Anthracnose, Bacterial Canker, Powdery Mildew, Sooty Mold, Healthy

Expected dataset layout (produced by ml/datasets/build_disease_dataset.py):
    <data_dir>/{train,val}/<Class Name>/*.jpg
"""

import argparse
import os
import sys
from pathlib import Path

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.disease_detection.inference import DISEASE_CLASSES, build_eval_transform, select_device

DEFAULT_DATA_DIR = REPO_ROOT / "ml" / "datasets" / "processed" / "disease"
DEFAULT_SAVE_DIR = REPO_ROOT / "ml" / "disease_detection" / "weights"
CHECKPOINT_NAME = "efficientnet_b3_best.pth"


def build_transforms(img_size: int = 300):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, build_eval_transform(img_size)


def load_split(split_dir: str, transform) -> datasets.ImageFolder:
    """
    Loads an ImageFolder split whose labels are remapped to the canonical DISEASE_CLASSES order.

    ImageFolder assigns indices alphabetically by folder name, which does not match
    DISEASE_CLASSES. Without remapping, a trained model's outputs would be mislabelled
    at inference time.
    """
    dataset = datasets.ImageFolder(split_dir, transform=transform, allow_empty=True)
    unknown = [c for c in dataset.classes if c not in DISEASE_CLASSES]
    if unknown:
        raise ValueError(f"Unknown class folders in {split_dir}: {unknown}. Expected only {DISEASE_CLASSES}.")

    remap = {folder_idx: DISEASE_CLASSES.index(name) for name, folder_idx in dataset.class_to_idx.items()}
    dataset.samples = [(path, remap[idx]) for path, idx in dataset.samples]
    dataset.targets = [idx for _, idx in dataset.samples]
    dataset.classes = list(DISEASE_CLASSES)
    dataset.class_to_idx = {name: i for i, name in enumerate(DISEASE_CLASSES)}
    return dataset


def save_checkpoint(model: nn.Module, path: Path, val_acc: float, epoch: int):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": list(DISEASE_CLASSES),
            "arch": "efficientnet_b3",
            "val_acc": val_acc,
            "epoch": epoch,
        },
        path,
    )


def train_model(
    data_dir: str = str(DEFAULT_DATA_DIR),
    epochs: int = 25,
    batch_size: int = 16,
    lr: float = 1e-4,
    save_dir: str = str(DEFAULT_SAVE_DIR),
    pretrained: bool = True,
    num_workers: int = None,
):
    device = select_device()
    print(f"Training EfficientNet-B3 on device: {device}")

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        print(f"Dataset directories {train_dir} / {val_dir} do not exist. Run ml/datasets/build_disease_dataset.py first.")
        return None

    train_tf, val_tf = build_transforms(img_size=300)
    train_dataset = load_split(train_dir, train_tf)
    val_dataset = load_split(val_dir, val_tf)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Training or validation split is empty; nothing to train on.")
        return None

    counts = {c: 0 for c in DISEASE_CLASSES}
    for t in train_dataset.targets:
        counts[DISEASE_CLASSES[t]] += 1
    print(f"Training images per class: {counts}")
    missing = [c for c, n in counts.items() if n == 0]
    if missing:
        print(f"WARNING: no training images for {missing}; the model cannot learn these classes.")

    if num_workers is None:
        num_workers = 0 if sys.platform in ("darwin", "win32") else 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = None
    if pretrained:
        try:
            model = timm.create_model("efficientnet_b3", pretrained=True, num_classes=len(DISEASE_CLASSES))
        except Exception as e:
            print(f"Notice: ImageNet-pretrained weights unavailable ({e}); training from scratch.")
    if model is None:
        model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=len(DISEASE_CLASSES))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_path / CHECKPOINT_NAME
    best_acc = -1.0

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = correct / total if total else 0.0

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                val_correct += (model(images).argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total if val_total else 0.0
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {running_loss / max(total, 1):.4f} "
              f"- Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, checkpoint_path, val_acc, epoch + 1)
            print(f"  Saved new best checkpoint to {checkpoint_path} (Val Acc: {val_acc:.4f})")

    return {"best_val_acc": best_acc, "checkpoint": str(checkpoint_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNet-B3 Disease Detection Model")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--save_dir", type=str, default=str(DEFAULT_SAVE_DIR))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                        help="Do not download ImageNet-pretrained weights")
    args = parser.parse_args()
    train_model(args.data_dir, args.epochs, args.batch_size, args.lr, args.save_dir, args.pretrained)
