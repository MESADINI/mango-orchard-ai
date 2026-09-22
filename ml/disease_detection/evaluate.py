"""
AI Mango Orchard Health & Pest Intelligence — Model Evaluation Pipeline
Computes precision, recall, F1-score, and confusion matrix on test split.
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from sklearn.metrics import classification_report, confusion_matrix

DISEASE_CLASSES = [
    "Anthracnose",
    "Bacterial Canker",
    "Powdery Mildew",
    "Sooty Mold",
    "Leaf Spot",
    "Healthy"
]

def evaluate_model(data_dir: str, weights_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    val_tf = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dir = os.path.join(data_dir, "test")
    if not os.path.exists(test_dir):
        print(f"Test directory {test_dir} not found.")
        return

    test_dataset = datasets.ImageFolder(test_dir, transform=val_tf)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=len(DISEASE_CLASSES))
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("\n" + "=" * 60)
    print("📋 EfficientNet-B3 Disease Detection Classification Report")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=test_dataset.classes))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../datasets/processed/disease")
    parser.add_argument("--weights", type=str, default="weights/efficientnet_b3_best.pth")
    args = parser.parse_args()
    evaluate_model(args.data_dir, args.weights)
