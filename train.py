import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from PIL import ImageFile
from tqdm import tqdm

# ----------------------------------------------
# CONFIG
# ----------------------------------------------
DATA_DIR = r"tomato_diseases"
BATCH = 16
EPOCHS = 10
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ImageFile.LOAD_TRUNCATED_IMAGES = True  # prevent PIL freeze

# ----------------------------------------------
# CUSTOM DATASET THAT SKIPS CORRUPTED IMAGES
# ----------------------------------------------
class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except Exception:
            print(f"[WARN] Skipping corrupted image at index {index}")
            return self.__getitem__((index + 1) % len(self.samples))

# ----------------------------------------------
# TRANSFORMS
# ----------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ----------------------------------------------
# LOAD DATASET
# ----------------------------------------------
dataset = SafeImageFolder(DATA_DIR, transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

num_classes = len(dataset.classes)
print("Classes:", dataset.classes)
print("Total Images:", len(dataset))

# ----------------------------------------------
# MODEL
# ----------------------------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

best_val_acc = 0.0

# ----------------------------------------------
# TRAINING LOOP
# ----------------------------------------------
for epoch in range(EPOCHS):
    print(f"\n🔵 Epoch {epoch + 1}/{EPOCHS}")
    model.train()
    train_correct = 0

    # Progress bar for training
    for imgs, labels in tqdm(train_loader, desc="Training", ncols=80):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_correct += (outputs.argmax(1) == labels).sum().item()

    train_acc = train_correct / len(train_ds)

    # ---------------- Validation ----------------
    model.eval()
    val_correct = 0

    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Validating", ncols=80):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            val_correct += (outputs.argmax(1) == labels).sum().item()

    val_acc = val_correct / len(val_ds)
    scheduler.step()

    print(f"Epoch {epoch+1}: Train Acc = {train_acc:.4f} | Val Acc = {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_tomato_model.pth")
        print("✔ Saved new BEST model")

# ----------------------------------------------
# SAVE FINAL MODEL
# ----------------------------------------------
torch.save(model.state_dict(), "tomato_model.pth")
print("\n🎉 Training completed! Model saved as tomato_model.pth")
print(f"🏆 Best Validation Accuracy: {best_val_acc:.4f}")