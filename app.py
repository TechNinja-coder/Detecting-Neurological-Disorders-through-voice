import streamlit as st
import torch
import joblib
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
from scipy.stats import variation

# -----------------------------
# Neural Network
# -----------------------------
class ParkinsonNet(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Load scaler + features
# -----------------------------
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("features.pkl")

model = ParkinsonNet(input_dim=len(feature_columns))
model.load_state_dict(torch.load("parkinsons_model.pth", map_location="cpu"))
model.eval()


# -----------------------------
# TXT feature loader (fixed size)
# -----------------------------
def load_txt_features(txt_file, feature_columns):
    text = txt_file.read().decode("utf-8")
    values = [float(x) for x in text.replace(",", " ").split()]

    expected = len(feature_columns)

    if len(values) > expected:
        values = values[:expected]
    elif len(values) < expected:
        values += [0.0] * (expected - len(values))

    return np.array(values, dtype=float).reshape(1, -1)


# -----------------------------
# Audio feature extraction
# -----------------------------
def audio_to_features(uploaded_audio, feature_columns):
    # Save uploaded audio temporarily for librosa
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_audio.read())
        tmp_path = tmp.name

    try:
        y, sr = librosa.load(tmp_path, sr=22050, mono=True)

        if len(y) < sr * 0.3:
            raise ValueError("Audio too short.")

        features = {}

        # Pitch
        f0, _, _ = librosa.pyin(y, fmin=75, fmax=500, sr=sr)
        f0_voiced = f0[~np.isnan(f0)]

        if len(f0_voiced) == 0:
            raise ValueError("No voiced segments detected.")

        features["MDVP:Fo(Hz)"] = float(np.mean(f0_voiced))
        features["MDVP:Fhi(Hz)"] = float(np.max(f0_voiced))
        features["MDVP:Flo(Hz)"] = float(np.min(f0_voiced))
        features["MDVP:Jitter(%)"] = float(variation(f0_voiced) * 100)
        features["Jitter:DDP"] = float(
            np.mean(np.abs(np.diff(f0_voiced, n=2))) if len(f0_voiced) > 2 else 0.0
        )

        # Shimmer
        rms = librosa.feature.rms(y=y)[0]
        mean_rms = float(np.mean(rms))
        shimmer = float(np.mean(np.abs(np.diff(rms))) / mean_rms) if mean_rms > 1e-10 else 0.0
        features["MDVP:Shimmer"] = shimmer

        # HNR / NHR
        y_harm, y_noise = librosa.effects.hpss(y)
        harm = float(np.sum(y_harm ** 2))
        noise = float(np.sum(y_noise ** 2))
        ratio = harm / (noise + 1e-10)
        features["HNR"] = float(10 * np.log10(max(ratio, 1e-10)))
        if "NHR" in feature_columns:
            features["NHR"] = float(noise / (harm + 1e-10))

        # Align with training features
        all_features = {col: features.get(col, 0.0) for col in feature_columns}
        values = np.array(list(all_features.values()), dtype=float)
        return pd.DataFrame([values], columns=feature_columns)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Parkinson's Detection")
st.title("🧠 Parkinson’s Disease Detection")

st.write("Upload BOTH:")
st.write("- Voice recording (.wav)")
st.write("- TXT feature file")

audio_file = st.file_uploader("Upload WAV audio", type=["wav"])
txt_file = st.file_uploader("Upload TXT features", type=["txt"])

if audio_file and txt_file:
    try:
        audio_features = audio_to_features(audio_file, feature_columns)
        txt_features = load_txt_features(txt_file, feature_columns)

        # Mean fusion
        X = np.vstack([audio_features.values, txt_features])
        X = np.mean(X, axis=0).reshape(1, -1)

        X = scaler.transform(X)

        with torch.no_grad():
            logits = model(torch.tensor(X, dtype=torch.float32))
            prob = torch.sigmoid(logits).item()

        st.success(f"Parkinson’s Probability: **{prob:.2%}**")

        if prob > 0.5:
            st.warning("Model suggests Parkinson’s-like speech patterns.")
        else:
            st.info("Model suggests healthy speech patterns.")

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Upload both files to run prediction.")
