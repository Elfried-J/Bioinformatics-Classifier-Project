from itertools import product
from Bio import SeqIO
import numpy as np
import pandas as pd

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
    gc = sequence.count('G') + sequence.count('C')
    return gc / max(len(sequence), 1)

def extract_features(fasta_file, label, k=4):
    records = []
    for record in SeqIO.parse(fasta_file, "fasta-pearson"):
        kmers = get_kmers(record.seq, k)
        gc = gc_content(record.seq)
        features = kmers + [gc, len(record.seq)]
        records.append(features + [label])
    return records

if __name__ == "__main__":
    all_records = []
    files = [
        ("data/Escherichia_coli.fasta", "E.coli"),
        ("data/Saccharomyces_cerevisiae.fasta", "S.cerevisiae"),
        ("data/Drosophila_melanogaster.fasta", "D.melanogaster"),
        ("data/Homo_sapiens.fasta", "H.sapiens"),
        ("data/Mycobacterium_tuberculosis.fasta", "M.tuberculosis"),
        ("data/Arabidopsis_thaliana.fasta", "A.thaliana")
    ]

    k = 4
    kmer_cols = [''.join(p) for p in product('ACGT', repeat=k)]
    columns = kmer_cols + ["gc_content", "seq_length", "label"]

    for fasta_file, label in files:
        print(f"Extracting features from {label}...")
        records = extract_features(fasta_file, label, k=k)
        all_records.extend(records)

    df = pd.DataFrame(all_records, columns=columns)
    df.to_csv("data/features.csv", index=False)
    print(f"Done. Dataset shape: {df.shape}")
    print(df["label"].value_counts())