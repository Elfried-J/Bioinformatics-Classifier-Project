from Bio import Entrez, SeqIO
import os

Entrez.email = "alexturner0000007@gmail.com"

def download_sequences(organism, num_sequences=200, output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Searching for {organism} sequences...")
    handle = Entrez.esearch(db="nucleotide", term=f"{organism}[Organism] AND 500:2000[SLEN]", retmax=num_sequences)
    record = Entrez.read(handle)
    handle.close()

    ids = record["IdList"]
    print(f"Found {len(ids)} sequences. Downloading...")

    handle = Entrez.efetch(db="nucleotide", id=ids, rettype="fasta", retmode="text")
    filename = os.path.join(output_dir, f"{organism.replace(' ', '_')}.fasta")
    with open(filename, "w") as f:
        f.write(handle.read())
    handle.close()

    print(f"Saved to {filename}")
    return filename

if __name__ == "__main__":
    organisms = [
        "Escherichia coli",
        "Saccharomyces cerevisiae",
        "Drosophila melanogaster",
        "Homo sapiens",
        "Mycobacterium tuberculosis",
        "Arabidopsis thaliana"
    ]
    for org in organisms:
        download_sequences(org, num_sequences=200)