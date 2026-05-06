import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("results/figures", exist_ok=True)

class DNADataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DNAClassifierCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DNAClassifierCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=4, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=4, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 64, kernel_size=4, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        conv_out_size = self._get_conv_output(input_size)
        self.classifier = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def _get_conv_output(self, input_size):
        dummy = torch.zeros(1, 1, input_size)
        out = self.conv_block(dummy)
        return int(out.view(1, -1).shape[1])

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

df = pd.read_csv("data/features.csv")
X = df.drop("label", axis=1).to_numpy(dtype=np.float32)
y_raw = df["label"].to_numpy()

scaler = StandardScaler()
X = scaler.fit_transform(X).astype(np.float32)

le = LabelEncoder()
y = le.fit_transform(y_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

train_loader = DataLoader(DNADataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader  = DataLoader(DNADataset(X_test, y_test), batch_size=32)

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = DNAClassifierCNN(X.shape[1], len(le.classes_)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

EPOCHS = 80
train_losses, train_accs = [], []

print(f"Training on {device} for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (out.argmax(1) == yb).sum().item()
        total      += len(yb)
    scheduler.step()
    acc = correct / total
    train_losses.append(total_loss / len(train_loader))
    train_accs.append(acc)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} — loss: {total_loss/len(train_loader):.4f}  acc: {acc:.3f}")

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        preds = model(xb).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(yb.numpy())

all_preds  = le.inverse_transform(all_preds)
all_labels = le.inverse_transform(all_labels)

print("\n=== CNN Test Set Results ===")
print(classification_report(all_labels, all_preds))

cm = confusion_matrix(all_labels, all_preds, labels=le.classes_)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_,
            yticklabels=le.classes_, cmap="Purples")
plt.title("CNN - Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("results/figures/CNN_confusion_matrix.png")
plt.close()

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.subplot(1, 2, 2)
plt.plot(train_accs)
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.tight_layout()
plt.savefig("results/figures/CNN_training_curve.png")
plt.close()
print("Plots saved to results/figures/")