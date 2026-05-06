import streamlit as st
import pandas as pd
import numpy as np
import random
from itertools import product
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from Bio import SeqIO
import io

st.set_page_config(page_title="DNA Species Classifier", page_icon="🧬", layout="wide")
st.title("🧬 DNA Species Classifier")
st.markdown("Classify DNA sequences across 6 species using k-mer frequency analysis.")

# Feature extraction
def get_kmers(sequence, k=4):
    all_kmers = [''.join(p) for p in product('ACGT', repeat=k)]
    counts = {km: 0 for km in all_kmers}
    sequence = str(sequence).upper().replace(" ", "").replace("\n", "")
    for i in range(len(sequence) - k + 1):
        sub = sequence[i:i+k]
        if sub in counts:
            counts[sub] += 1
    total = max(len(sequence) - k + 1, 1)
    return {km: counts[km] / total for km in all_kmers}

def gc_content(sequence):
    sequence = str(sequence).upper()
    gc = sequence.count('G') + sequence.count('C')
    return gc / max(len(sequence), 1)

def extract_features(sequence):
    kmer_dict = get_kmers(sequence)
    kmers = list(kmer_dict.values())
    gc = gc_content(sequence)
    return np.array(kmers + [gc, len(sequence)]).reshape(1, -1), kmer_dict

# Load models
@st.cache_resource
def load_models():
    df = pd.read_csv("data/features.csv")
    X = df.drop("label", axis=1).to_numpy()
    y = df["label"].to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
    svm.fit(X_scaled, y)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)
    kmer_cols = [''.join(p) for p in product('ACGT', repeat=4)]
    return svm, rf, scaler, y, kmer_cols

svm_model, rf_model, scaler, y_all, kmer_cols = load_models()

species_info = {
    "E.coli":         "🦠 Escherichia coli — common gut bacteria",
    "S.cerevisiae":   "🍞 Saccharomyces cerevisiae — baker's yeast",
    "D.melanogaster": "🪰 Drosophila melanogaster — fruit fly",
    "H.sapiens":      "🧑 Homo sapiens — human",
    "M.tuberculosis": "🫁 Mycobacterium tuberculosis — TB bacteria",
    "A.thaliana":     "🌱 Arabidopsis thaliana — thale cress (plant)",
}

species_profiles = {
    "E.coli":         {"GC": 0.50, "bias": {"GCGC": 2.0, "CGCG": 2.0}},
    "S.cerevisiae":   {"GC": 0.38, "bias": {"ATAT": 2.0, "TATA": 2.0}},
    "D.melanogaster": {"GC": 0.43, "bias": {"AATC": 1.5, "GATT": 1.5}},
    "H.sapiens":      {"GC": 0.41, "bias": {"AAAA": 1.8, "TTTT": 1.8}},
    "M.tuberculosis": {"GC": 0.65, "bias": {"GCGG": 2.5, "CGGC": 2.5}},
    "A.thaliana":     {"GC": 0.36, "bias": {"AAAT": 2.0, "TTTA": 2.0}},
}

def generate_random_sequence(species, length=500):
    profile = species_profiles[species]
    gc = profile["GC"]
    at = 1 - gc
    weights = [at/2, gc/2, gc/2, at/2]
    bases = random.choices("ACGT", weights=weights, k=length)
    return "".join(bases)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔬 Classify Sequence", "🔄 Compare Sequences", "📁 Upload FASTA", "🧫 Metagenomic Simulation"])

# Tab 1 - single sequence
with tab1:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### Enter or generate a sequence")
        gen_col1, gen_col2 = st.columns([2, 1])
        with gen_col1:
            selected_species = st.selectbox("Generate random sequence from:", list(species_profiles.keys()))
        with gen_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🎲 Generate"):
                st.session_state["generated_seq"] = generate_random_sequence(selected_species)

        default_seq = st.session_state.get("generated_seq",
            "ATGCGATCGATCGATCGAATTCGATCGATCGATCGAATTCGATCGATCGATCGAATTCGATCGATCG")
        sequence = st.text_area("DNA Sequence (A, T, C, G only)", value=default_seq, height=150)
        model_choice = st.radio("Model", ["SVM", "Random Forest"], horizontal=True)
        classify_btn = st.button("Classify Sequence", type="primary")

    with col_right:
        if classify_btn:
            clean = sequence.upper().replace(" ", "").replace("\n", "")
            invalid = set(clean) - set("ACGT")
            if len(clean) < 50:
                st.error("Sequence too short — please enter at least 50 bases.")
            elif invalid:
                st.error(f"Invalid characters: {invalid}")
            else:
                features, kmer_dict = extract_features(clean)
                features_scaled = scaler.transform(features)
                model = svm_model if model_choice == "SVM" else rf_model
                prediction = model.predict(features_scaled)[0]
                probabilities = model.predict_proba(features_scaled)[0]
                classes = model.classes_
                top_conf = max(probabilities)

                if top_conf < 0.60:
                    st.warning(f"⚠️ Low confidence ({top_conf:.1%}) — sequence may be ambiguous or from an organism not in the training set.")
                else:
                    st.success(f"**Predicted: {prediction}**")
                    st.info(species_info.get(prediction, prediction))

                st.markdown("#### Confidence scores")
                prob_df = pd.DataFrame({"Species": classes, "Confidence": probabilities}).sort_values("Confidence", ascending=False)
                fig, ax = plt.subplots(figsize=(6, 3))
                colors = ["#5DCAA5" if c == prediction else "#AFA9EC" for c in prob_df["Species"]]
                ax.barh(prob_df["Species"], prob_df["Confidence"], color=colors)
                ax.set_xlabel("Confidence")
                ax.set_xlim(0, 1)
                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                st.markdown("#### Top 20 most important k-mers")
                kmer_importance = rf_model.feature_importances_[:256]
                top_idx = np.argsort(kmer_importance)[-20:][::-1]
                top_kmers = [kmer_cols[i] for i in top_idx]
                top_vals = [kmer_importance[i] for i in top_idx]
                top_freq = [kmer_dict.get(km, 0) for km in top_kmers]

                fig2, ax2 = plt.subplots(figsize=(6, 4))
                colors2 = ["#E85D24" if f > np.mean(top_freq) else "#AFA9EC" for f in top_freq]
                ax2.barh(top_kmers[::-1], top_vals[::-1], color=colors2[::-1])
                ax2.set_xlabel("Feature importance (Random Forest)")
                ax2.set_title("Top 20 discriminative k-mers")
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()
                st.caption("🔴 Red = above-average frequency in your sequence.")

                st.markdown("#### Sequence stats")
                c1, c2, c3 = st.columns(3)
                c1.metric("Length", f"{len(clean)} bp")
                c2.metric("GC Content", f"{gc_content(clean):.1%}")
                c3.metric("Valid bases", f"{len(clean)}")

# Tab 2 - sequence comparison
with tab2:
    st.markdown("### Compare two DNA sequences")
    col_a, col_b = st.columns(2)
    with col_a:
        seq_a = st.text_area("Sequence A", height=120,
            value="ATGCGATCGATCGATCGAATTCGATCGATCGATCGAATTCGATCGATCG")
    with col_b:
        seq_b = st.text_area("Sequence B", height=120,
            value="GCGCGCGGCGCGGCGCGCGGCGCGGCGCGCGGCGCGGCGCGCGGCGCGG")

    if st.button("Compare sequences", type="primary"):
        clean_a = seq_a.upper().replace(" ", "").replace("\n", "")
        clean_b = seq_b.upper().replace(" ", "").replace("\n", "")

        if len(clean_a) < 50 or len(clean_b) < 50:
            st.error("Both sequences must be at least 50 bases.")
        else:
            fa, kd_a = extract_features(clean_a)
            fb, kd_b = extract_features(clean_b)
            pred_a = svm_model.predict(scaler.transform(fa))[0]
            pred_b = svm_model.predict(scaler.transform(fb))[0]

            col1, col2 = st.columns(2)
            col1.success(f"Sequence A → **{pred_a}**")
            col2.success(f"Sequence B → **{pred_b}**")

            all_kmers = list(kd_a.keys())
            diff = {km: abs(kd_a[km] - kd_b[km]) for km in all_kmers}
            top_diff = sorted(diff.items(), key=lambda x: x[1], reverse=True)[:20]
            kmers_d, vals_d = zip(*top_diff)

            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            axes[0].bar(range(len(kmers_d)), [kd_a[k] for k in kmers_d], color="#5DCAA5")
            axes[0].set_title(f"Seq A ({pred_a})")
            axes[0].set_xticks(range(len(kmers_d)))
            axes[0].set_xticklabels(kmers_d, rotation=90, fontsize=8)
            axes[0].set_ylabel("Frequency")

            axes[1].bar(range(len(kmers_d)), [kd_b[k] for k in kmers_d], color="#7F77DD")
            axes[1].set_title(f"Seq B ({pred_b})")
            axes[1].set_xticks(range(len(kmers_d)))
            axes[1].set_xticklabels(kmers_d, rotation=90, fontsize=8)

            axes[2].bar(range(len(kmers_d)), vals_d, color="#E85D24")
            axes[2].set_title("Absolute difference")
            axes[2].set_xticks(range(len(kmers_d)))
            axes[2].set_xticklabels(kmers_d, rotation=90, fontsize=8)

            plt.suptitle("Top 20 most different k-mers between sequences", fontsize=13)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            stats_df = pd.DataFrame({
                "Metric": ["Length (bp)", "GC Content", "Predicted species"],
                "Sequence A": [len(clean_a), f"{gc_content(clean_a):.1%}", pred_a],
                "Sequence B": [len(clean_b), f"{gc_content(clean_b):.1%}", pred_b],
            })
            st.table(stats_df)

# Tab 3 - FASTA file upload
with tab3:
    st.markdown("### Upload a FASTA file")
    st.caption("Upload a .fasta file to classify all sequences at once.")
    uploaded = st.file_uploader("Choose a FASTA file", type=["fasta", "fa", "txt"])

    if uploaded:
        content = uploaded.read().decode("utf-8")
        handle = io.StringIO(content)
        records = list(SeqIO.parse(handle, "fasta-pearson"))

        if len(records) == 0:
            st.error("No sequences found in the file.")
        else:
            st.info(f"Found {len(records)} sequences. Classifying...")
            results = []
            for rec in records:
                clean = str(rec.seq).upper()
                if len(clean) < 50:
                    continue
                features, _ = extract_features(clean)
                features_scaled = scaler.transform(features)
                pred = svm_model.predict(features_scaled)[0]
                prob = max(svm_model.predict_proba(features_scaled)[0])
                results.append({
                    "ID": rec.id,
                    "Length (bp)": len(clean),
                    "GC Content": f"{gc_content(clean):.1%}",
                    "Predicted Species": pred,
                    "Confidence": f"{prob:.1%}",
                    "Flag": "⚠️ Low confidence" if prob < 0.60 else "✅"
                })

            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)

            fig, ax = plt.subplots(figsize=(7, 4))
            results_df["Predicted Species"].value_counts().plot(kind="bar", ax=ax, color="#5DCAA5", edgecolor="none")
            ax.set_title("Species distribution in uploaded file")
            ax.set_xlabel("Species")
            ax.set_ylabel("Count")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            csv = results_df.to_csv(index=False)
            st.download_button("Download results as CSV", csv, "classification_results.csv", "text/csv")

# Tab 4 - metagenomic simulation
with tab4:
    st.markdown("### Metagenomic read simulation")
    st.caption("Test how well the classifier handles short noisy reads — simulating real Illumina sequencing conditions.")

    col1, col2 = st.columns(2)
    with col1:
        read_length = st.slider("Read length (bp)", min_value=50, max_value=500, value=150, step=50)
        error_rate = st.slider("Sequencing error rate", min_value=0.0, max_value=0.10, value=0.01, step=0.01, format="%.2f")
        num_reads = st.slider("Reads per sequence", min_value=1, max_value=20, value=5)

    with col2:
        st.markdown("#### What this simulates")
        st.markdown(f"""
        - **Read length:** {read_length}bp fragments (Illumina short reads are typically 150bp)
        - **Error rate:** {error_rate:.1%} random base substitutions per position
        - **Real world:** Clinical metagenomics tools like Kraken2 face exactly this challenge
        """)

    sequence_input = st.text_area("Paste a DNA sequence to fragment and classify:",
        height=100,
        value="ATGCGATCGATCGATCGAATTCGATCGATCGATCGAATTCGATCGATCGATCGAATTCGATCGATCGATCGAATTCGATCGATCGATCGAATTCGATCGATCGATCGAATTCGATCGATCGATCGAATTCGATCGATCGATCGAATTCGATCGATCGATCG")

    if st.button("Run simulation", type="primary"):
        clean = sequence_input.upper().replace(" ", "").replace("\n", "")
        if len(clean) < read_length:
            st.error(f"Sequence must be at least {read_length}bp for this read length.")
        else:
            reads = []
            for _ in range(num_reads):
                start = random.randint(0, len(clean) - read_length)
                read = clean[start:start + read_length]
                noisy = []
                for base in read:
                    if random.random() < error_rate:
                        noisy.append(random.choice([b for b in "ACGT" if b != base]))
                    else:
                        noisy.append(base)
                reads.append("".join(noisy))

            predictions = []
            confidences = []
            for read in reads:
                features, _ = extract_features(read)
                features_scaled = scaler.transform(features)
                pred = svm_model.predict(features_scaled)[0]
                conf = max(svm_model.predict_proba(features_scaled)[0])
                predictions.append(pred)
                confidences.append(conf)

            results_df = pd.DataFrame({
                "Read #": range(1, len(reads) + 1),
                "Length (bp)": [len(r) for r in reads],
                "Predicted Species": predictions,
                "Confidence": [f"{c:.1%}" for c in confidences],
                "Flag": ["⚠️" if c < 0.60 else "✅" for c in confidences]
            })

            st.dataframe(results_df, use_container_width=True)

            vote_counts = pd.Series(predictions).value_counts()
            winner = vote_counts.index[0]
            win_pct = vote_counts.iloc[0] / len(predictions)

            st.success(f"**Consensus prediction: {winner}** ({win_pct:.0%} of reads agree)")
            st.caption(f"Average confidence: {np.mean(confidences):.1%} — Error rate applied: {error_rate:.1%}")

            fig, ax = plt.subplots(figsize=(7, 3))
            vote_counts.plot(kind="bar", ax=ax, color="#5DCAA5", edgecolor="none")
            ax.set_title("Vote distribution across reads")
            ax.set_xlabel("Predicted species")
            ax.set_ylabel("Number of reads")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()