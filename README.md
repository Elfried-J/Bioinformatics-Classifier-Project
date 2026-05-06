# 🧬 DNA Species Classifier

A machine learning pipeline that classifies DNA sequences across 6 species using k-mer frequency analysis. Built with Python, scikit-learn, PyTorch, and Streamlit.

**Live demo:** [huggingface.co/spaces/elf222/dna-species-classifier](https://huggingface.co/spaces/elf222/dna-species-classifier)

---

## What it does

Given a raw DNA sequence (a string of A, T, C, G letters), the classifier predicts which organism it came from. It works by extracting k-mer frequency profiles — counting how often every possible 4-letter substring appears — and using those as features for machine learning models.

Different organisms have distinct k-mer frequency fingerprints due to differences in GC content, codon usage, and regulatory motifs. The classifier learns to recognize these patterns without any prior biological knowledge.

---

## Species

| Species | Kingdom | GC Content |
|---------|---------|------------|
| *Escherichia coli* | Bacteria | ~50% |
| *Mycobacterium tuberculosis* | Bacteria | ~65% |
| *Saccharomyces cerevisiae* | Fungi | ~38% |
| *Arabidopsis thaliana* | Plant | ~36% |
| *Drosophila melanogaster* | Animal | ~43% |
| *Homo sapiens* | Animal | ~41% |

---

## Models & Results

| Model | Accuracy | AUC (avg) |
|-------|----------|-----------|
| SVM (RBF kernel) | 97% | 0.997 |
| Logistic Regression | 95% | 0.993 |
| 1D CNN (PyTorch) | 96% | — |
| Random Forest | 92% | 0.994 |

All models trained on 256 4-mer features + GC content extracted from 1,192 sequences downloaded from NCBI GenBank.

---

## Metagenomic simulation

Tested classifier robustness on short 150bp reads with simulated sequencing errors — matching real Illumina sequencing conditions.

| Error Rate | Accuracy |
|------------|----------|
| 0% | 93.8% |
| 1% | 93.7% |
| 2% | 91.7% |
| 5% | 89.2% |

The classifier maintains 89%+ accuracy even at 5% sequencing error rate — comparable to published metagenomic classification tools.

---

## App features

- **Classify Sequence** — paste any DNA sequence and get a species prediction with confidence scores and k-mer importance visualization
- **Compare Sequences** — compare two sequences side by side, showing which k-mers differ most between them
- **Upload FASTA** — upload a `.fasta` file to classify all sequences at once and download results as CSV
- **Metagenomic Simulation** — fragment a sequence into short noisy reads and see how the classifier performs under realistic sequencing conditions

---

## Project structure
