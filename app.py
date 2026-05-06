import streamlit as st
import pandas as pd
import numpy as np
import pickle
from itertools import product
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(
    page_title="DNA Species Classifier",
    page_icon="🧬",
    layout="centered"
)

st.title("🧬 DNA Species Classifier")
st.markdown("Paste a DNA sequence and the model will predict which organism it came from.")

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
    return [counts[km] / total for km in all_kmers]

def gc_content(sequence):
    sequence = str(sequence).upper()
    gc = sequence.count('G') + sequence.count('C')
    return gc / max(len(sequence), 1)

def extract_features(sequence):
    kmers = get_kmers(sequence)
    gc = gc_content(sequence)
    return np.array(kmers + [gc, len(sequence)]).reshape(1, -1)

# Train model on load
@st.cache_resource
def load_model():
    df = pd.read_csv("data/features.csv")
    X = df.drop("label", axis=1).to_numpy()
    y = df["label"].to_numpy()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    model = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
    model.fit(X, y)
    return model, scaler

model, scaler = load_model()

# UI
st.markdown("### Enter a DNA sequence")
example = "ATGCGATCGATCGATCGAATTCGATCGATCGATCGAATTCGATCGATCGATCGAATTCGATCGATCG"
sequence = st.text_area("DNA Sequence (A, T, C, G only)", value=example, height=150)

species_info = {
    "E.coli":         "🦠 Escherichia coli — common gut bacteria",
    "S.cerevisiae":   "🍞 Saccharomyces cerevisiae — baker's yeast",
    "D.melanogaster": "🪰 Drosophila melanogaster — fruit fly",
    "H.sapiens":      "🧑 Homo sapiens — human",
    "M.tuberculosis": "🫁 Mycobacterium tuberculosis — TB bacteria",
    "A.thaliana":     "🌱 Arabidopsis thaliana — thale cress (plant)",
}

if st.button("Classify Sequence", type="primary"):
    clean = sequence.upper().replace(" ", "").replace("\n", "")
    invalid = set(clean) - set("ACGT")

    if len(clean) < 50:
        st.error("Sequence too short — please enter at least 50 bases.")
    elif invalid:
        st.error(f"Invalid characters found: {invalid}. Only A, T, C, G allowed.")
    else:
        features = extract_features(clean)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        classes = model.classes_

        st.success(f"**Predicted species: {prediction}**")
        st.info(species_info.get(prediction, prediction))

        st.markdown("### Confidence scores")
        prob_df = pd.DataFrame({
            "Species": classes,
            "Confidence": probabilities
        }).sort_values("Confidence", ascending=False)

        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#5DCAA5" if c == prediction else "#AFA9EC" for c in prob_df["Species"]]
        ax.barh(prob_df["Species"], prob_df["Confidence"], color=colors)
        ax.set_xlabel("Confidence")
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("### Sequence stats")
        col1, col2, col3 = st.columns(3)
        col1.metric("Length", f"{len(clean)} bp")
        col2.metric("GC Content", f"{gc_content(clean):.1%}")
        col3.metric("Valid bases", f"{len(clean)}")