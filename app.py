import torch
import streamlit as st
import joblib
def load_txt_features(txt_file):
    text = txt_file.read().decode("utf-8")
    values = [float(x) for x in text.replace(",", " ").split()]
    return np.array(values, dtype=float).reshape(1, -1)





# Neural network (moved here from cell 3f959ffd for app.py)
class ParkinsonNet(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
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
# Load model, scaler, features
# -----------------------------
# X_train is not available in app.py context, so load scaler directly if it exists, otherwise it will fail here.
# The scaler should be saved in `scaler.pkl` from the notebook's execution.
# If not found, `FileNotFoundError` will be caught and the app will indicate a warning.

# The notebook state shows that X_train, model, scaler, and feature_columns were already defined and saved correctly.
# We need to make sure `scaler` is loaded from the saved file, not re-initialized.

def load_model(model_path: str):
    model = ParkinsonNet()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

try:
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    st.error("scaler.pkl not found. Please ensure the training and saving steps are executed correctly.")
    st.stop() # Stop the app if essential components are missing

try:
    feature_columns = joblib.load("features.pkl")
except FileNotFoundError:
    st.error("features.pkl not found. Please ensure the training and saving steps are executed correctly.")
    st.stop()

model = ParkinsonNet(input_dim=len(feature_columns))
model.load_state_dict(torch.load("parkinsons_model.pth", map_location="cpu"))
model.eval()

# Define the predict_from_audio function directly in app.py for Streamlit
import librosa
import numpy as np
import pandas as pd
from scipy.stats import variation

def audio_to_features(audio_path, feature_columns, sr=22050):
    # Load audio (mono)
    y, sr = librosa.load(audio_path, sr=sr, mono=True)

    if len(y) < sr * 0.3:
        raise ValueError("Audio too short for reliable feature extraction.")

    audio_features_np = audio_features.to_numpy()
    features = {}

    # Pitch extraction (modern replacement for Praat pitch)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=75,
        fmax=500,
        sr=sr
    )

    f0_voiced = f0[~np.isnan(f0)]

    if len(f0_voiced) == 0:
        raise ValueError("No voiced segments detected.")

    features["MDVP:Fo(Hz)"] = float(np.mean(f0_voiced))
    features["MDVP:Fhi(Hz)"] = float(np.max(f0_voiced))
    features["MDVP:Flo(Hz)"] = float(np.min(f0_voiced))

    # Jitter approximations (from F0 instability)
    features["MDVP:Jitter(%)"] = float(variation(f0_voiced) * 100)

    if len(f0_voiced) > 2:
        ddp = np.mean(np.abs(np.diff(f0_voiced, n=2)))
    else:
        ddp = 0.0
    features["Jitter:DDP"] = float(ddp)

    # Shimmer approximations (from amplitude envelope)
    rms = librosa.feature.rms(y=y)[0]

    if len(rms) > 1:
        shimmer_local = np.mean(np.abs(np.diff(rms))) / np.mean(rms)
    else:
        shimmer_local = 0.0

    features["MDVP:Shimmer"] = float(shimmer_local)
    features["Shimmer:APQ3"] = float(np.mean(np.abs(np.diff(rms, n=1))) if len(rms) > 3 else 0.0)
    features["Shimmer:APQ5"] = float(np.mean(np.abs(np.diff(rms, n=2))) if len(rms) > 5 else 0.0)
    features["Shimmer:DDA"] = float(3 * features["Shimmer:APQ3"])

    # Harmonics-to-Noise Ratio (HNR approximation)
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    harm_energy = np.sum(y_harmonic ** 2)
    noise_energy = np.sum(y_percussive ** 2) + 1e-10

    features["HNR"] = float(10 * np.log10(harm_energy / noise_energy))

    # Noise-to-Harmonics Ratio (NHR)
    if "NHR" in feature_columns:
        if features["HNR"] > 0:
            features["NHR"] = float(1.0 / features["HNR"])
        else:
            features["NHR"] = 0.0

    # Final safety checks
    all_features = {col: features.get(col, 0.0) for col in feature_columns}
    values = np.array(list(all_features.values()), dtype=float)

    if np.isnan(values).any() or np.all(values == 0):
        raise ValueError("Invalid recording: sustained vowel required.")

    return pd.DataFrame([values], columns=feature_columns)

def predict_from_audio(audio_path, model, scaler, feature_columns):
    features_df = audio_to_features(audio_path, feature_columns)
    features_scaled = scaler.transform(features_df)

    with torch.no_grad():
        tensor = torch.tensor(features_scaled, dtype=torch.float32)
        prob = torch.sigmoid(model(tensor)).item() # Apply sigmoid to get probability

    return prob, features_df


# -----------------------------
# Streamlit App
# -----------------------------

import streamlit as st
import torch
import joblib
import numpy as np


st.set_page_config(page_title="Parkinson's Detection", layout="centered")

st.title("🧠 Parkinson’s Disease Detection")
st.write("Upload **both** an audio recording and a TXT feature file.")

# Uploads
audio_file = st.file_uploader("Upload voice recording (.wav)", type=["wav"])
txt_file = st.file_uploader("Upload feature TXT file", type=["txt"])

if audio_file and txt_file:
    try:
        audio_features = audio_to_features(audio_file, feature_columns)
        txt_features = load_txt_features(txt_file)

        audio_np = audio_features.to_numpy()
        txt_np = txt_features

        X = np.vstack([audio_np, txt_np])
        X = np.mean(X, axis=0).reshape(1, -1)

        # Combine (mean fusion)
        X = np.vstack([audio_features_np, txt_features])
        X = np.mean(X, axis=0).reshape(1, -1)

        scaler = joblib.load("scaler.pkl")
        X = scaler.transform(X)

        model = load_model("parkinson_model.pt")

        with torch.no_grad():
            logits = model(torch.tensor(X, dtype=torch.float32))
            prob = torch.sigmoid(logits).item()

        st.success(f"🧪 Parkinson’s Probability: **{prob:.2%}**")

        if prob > 0.5:
            st.warning("⚠️ Model indicates Parkinson’s characteristics.")
        else:
            st.info("✅ Model indicates healthy speech patterns.")

    except Exception as e:
        st.error(str(e))
else:
    st.info("Please upload **both** files to continue.")
