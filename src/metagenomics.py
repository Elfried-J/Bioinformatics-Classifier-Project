import pandas as pd
import numpy as np
import random
from itertools import product
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from Bio import SeqIO
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("results/figures", exist_ok=True)

def get_kmers(sequence, k=4):
    all_kmers = [''.join(p) for p in product('ACGT', repeat=k)]
    counts = {km: 0 for km in all_kmers}
    sequence = str(sequence).upper()
    for i in range(len(sequence) - k + 1):
        sub = sequence[i:i+k]
        if sub in counts:
            counts[sub] += 1
    total = max(len(sequence) - k + 1, 1)
    return [counts[km] / total for km in all_kmers]

def gc_content(sequence):
    sequence = str(sequence).upper()
    return (sequence.count('G') + sequence.count('C')) / max(len(sequence), 1)

def extract_features(sequence):
    kmers = get_kmers(sequence)
    gc = gc_content(sequence)
    return kmers + [gc]

def introduce_errors(sequence, error_rate=0.01):
    bases = list(sequence)
    for i in range(len(bases)):
        if random.random() < error_rate:
            bases[i] = random.choice([b for b in "ACGT" if b != bases[i]])
    return "".join(bases)

def simulate_reads(sequence, read_length=150, num_reads=10, error_rate=0.01):
    reads = []
    seq = str(sequence).upper()
    if len(seq) < read_length:
        return reads
    for _ in range(num_reads):
        start = random.randint(0, len(seq) - read_length)
        read = seq[start:start + read_length]
        read = introduce_errors(read, error_rate)
        reads.append(read)
    return reads

fasta_files = {
    "E.coli":         "data/Escherichia_coli.fasta",
    "S.cerevisiae":   "data/Saccharomyces_cerevisiae.fasta",
    "D.melanogaster": "data/Drosophila_melanogaster.fasta",
    "H.sapiens":      "data/Homo_sapiens.fasta",
    "M.tuberculosis": "data/Mycobacterium_tuberculosis.fasta",
    "A.thaliana":     "data/Arabidopsis_thaliana.fasta",
}

# Train on short reads
print("Building training set from short reads...")
X_train_list, y_train_list = [], []
for label, fasta_path in fasta_files.items():
    sequences = list(SeqIO.parse(fasta_path, "fasta-pearson"))[:150]
    for record in sequences:
        reads = simulate_reads(record.seq, read_length=150, num_reads=5, error_rate=0.01)
        for read in reads:
            X_train_list.append(extract_features(read))
            y_train_list.append(label)

X_train_arr = np.array(X_train_list)
y_train_arr = np.array(y_train_list)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train_arr)

model = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
model.fit(X_scaled, y_train_arr)
print(f"Model trained on {len(y_train_arr)} short reads.")

# Test at different error rates
error_rates = [0.0, 0.01, 0.02, 0.05]
results = {er: {"true": [], "pred": []} for er in error_rates}

print("Simulating reads at different error rates...")
for label, fasta_path in fasta_files.items():
    sequences = list(SeqIO.parse(fasta_path, "fasta-pearson"))[:20]
    for record in sequences:
        for error_rate in error_rates:
            reads = simulate_reads(record.seq, read_length=150, num_reads=5, error_rate=error_rate)
            for read in reads:
                features = extract_features(read)
                features_scaled = scaler.transform([features])
                pred = model.predict(features_scaled)[0]
                results[error_rate]["true"].append(label)
                results[error_rate]["pred"].append(pred)

print("\n=== Metagenomic Read Classification Results ===")
print(f"{'Error Rate':<15} {'Accuracy':>10} {'Sequences':>12}")
print("-" * 40)

accuracies = []
for er in error_rates:
    true = results[er]["true"]
    pred = results[er]["pred"]
    acc = sum(t == p for t, p in zip(true, pred)) / len(true)
    accuracies.append(acc)
    print(f"{er:.1%}{'':8} {acc:.3f}{'':8} {len(true)}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot([f"{e:.0%}" for e in error_rates], accuracies, marker="o", color="#5DCAA5", linewidth=2, markersize=8)
axes[0].set_title("Classification accuracy vs sequencing error rate")
axes[0].set_xlabel("Error rate")
axes[0].set_ylabel("Accuracy")
axes[0].set_ylim(0, 1)
axes[0].axhline(y=0.85, color="#E85D24", linestyle="--", label="85% threshold")
axes[0].legend()

cm = confusion_matrix(results[0.01]["true"], results[0.01]["pred"], labels=list(fasta_files.keys()))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=list(fasta_files.keys()),
            yticklabels=list(fasta_files.keys()),
            cmap="Blues", ax=axes[1])
axes[1].set_title("Confusion matrix at 1% error rate")
axes[1].set_ylabel("True Label")
axes[1].set_xlabel("Predicted Label")

plt.tight_layout()
plt.savefig("results/figures/metagenomic_results.png", dpi=150)
plt.close()
print("Plots saved to results/figures/metagenomic_results.png")

print("\n=== Detailed report at 1% error rate ===")
print(classification_report(results[0.01]["true"], results[0.01]["pred"]))