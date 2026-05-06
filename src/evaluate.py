import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import os

os.makedirs("results/figures", exist_ok=True)

# Load and prepare data
df = pd.read_csv("data/features.csv")
X = df.drop("label", axis=1).to_numpy()
y = df["label"].to_numpy()

classes = sorted(list(set(y)))
n_classes = len(classes)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

y_test_bin = label_binarize(y_test, classes=classes)

models = {
    "SVM": SVC(kernel="rbf", C=10, gamma="scale", probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42)
}

# Plot ROC curves
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, (name, model) in zip(axes, models.items()):
    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)

    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_title(f"{name} — ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=7)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig("results/figures/ROC_curves.png", dpi=150)
plt.close()
print("ROC curves saved to results/figures/ROC_curves.png")

# Print AUC summary table
print("\n=== AUC Summary ===")
print(f"{'Model':<22} {'Species':<20} {'AUC':>6}")
print("-" * 50)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        print(f"{name:<22} {cls:<20} {roc_auc:.4f}")
    print()