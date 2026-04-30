import os
import shutil
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np

# =========================
# AUTO DATASET SPLIT (FIXED)
# =========================
def split_dataset(source_dir, train_dir, val_dir, split_ratio=0.8):
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        print("Dataset already split. Skipping...")
        return

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for class_name in os.listdir(source_dir):

        # 🚨 Ignore already created folders
        if class_name in ["train", "val"]:
            continue

        class_path = os.path.join(source_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        # Only take valid images
        images = [
            img for img in os.listdir(class_path)
            if img.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        random.shuffle(images)

        split_index = int(len(images) * split_ratio)

        train_images = images[:split_index]
        val_images = images[split_index:]

        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

        for img in train_images:
            shutil.copy(
                os.path.join(class_path, img),
                os.path.join(train_dir, class_name, img)
            )

        for img in val_images:
            shutil.copy(
                os.path.join(class_path, img),
                os.path.join(val_dir, class_name, img)
            )

    print("✅ Dataset split completed!")

# =========================
# CONFIG
# =========================
DATASET_PATH = "dataset"
TRAIN_PATH = "dataset/train"
VAL_PATH = "dataset/val"

BATCH_SIZE = 32
EPOCHS = 15
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# RUN SPLIT
# =========================
split_dataset(DATASET_PATH, TRAIN_PATH, VAL_PATH)

# =========================
# TRANSFORMS
# =========================
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =========================
# DATA
# =========================
train_data = datasets.ImageFolder(TRAIN_PATH, transform=train_transforms)
val_data = datasets.ImageFolder(VAL_PATH, transform=val_transforms)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

# =========================
# CLASS WEIGHTS
# =========================
class_counts = np.bincount(train_data.targets)
class_weights = 1. / class_counts
class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

# =========================
# MODEL
# =========================
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

num_classes = len(train_data.classes)
model.classifier[1] = nn.Linear(1280, num_classes)

model = model.to(DEVICE)

# =========================
# LOSS + OPTIMIZER
# =========================
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# =========================
# TRAINING LOOP
# =========================
best_val_acc = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total

    # =========================
    # VALIDATION
    # =========================
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total

    scheduler.step()

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Validation Accuracy: {val_acc:.2f}%")

    # =========================
    # SAVE BEST MODEL
    # =========================
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print("✅ Best model saved!")

print(f"\n🔥 Training Complete! Best Val Accuracy: {best_val_acc:.2f}%")