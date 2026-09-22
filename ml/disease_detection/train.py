"""
AI Mango Orchard Health & Pest Intelligence — Disease Detection Training
Architecture: EfficientNet-B3 (PyTorch / timm)
Target Classes: Anthracnose, Bacterial Canker, Powdery Mildew, Sooty Mold, Leaf Spot, Healthy
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

DISEASE_CLASSES = [
    "Anthracnose",
    "Bacterial Canker",
    "Powdery Mildew",
    "Sooty Mold",
    "Leaf Spot",
    "Healthy"
]

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
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

def train_model(data_dir: str, epochs: int = 25, batch_size: int = 16, lr: float = 1e-4, save_dir: str = "weights"):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training EfficientNet-B3 on device: {device}")

    train_tf, val_tf = build_transforms(img_size=300)
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if not os.path.exists(train_dir):
        print(f"Dataset directory {train_dir} does not exist yet. Run ml/datasets/phase1_disease.py first.")
        return

    train_dataset = datasets.ImageFolder(train_dir, transform=train_tf)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_tf)

    # Use num_workers=0 on macOS to avoid spawn multiprocessing issues in DataLoader
    workers = 0 if sys.platform == "darwin" else 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    # Required Architecture: EfficientNet-B3
    try:
        model = timm.create_model("efficientnet_b3", pretrained=True, num_classes=len(DISEASE_CLASSES))
    except Exception as e:
        print(f"Notice: Pretrained download skipped ({e}). Initializing architecture.")
        model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=len(DISEASE_CLASSES))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    os.makedirs(save_dir, exist_ok=True)
    best_acc = 0.0

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
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = correct / total if total > 0 else 0

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/total:.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "efficientnet_b3_best.pth"))
            print(f"  ⭐ Saved new best checkpoint (Val Acc: {val_acc:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNet-B3 Disease Detection Model")
    parser.add_argument("--data_dir", type=str, default="../datasets/processed/disease")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    train_model(args.data_dir, args.epochs, args.batch_size, args.lr)
