from pathlib import Path

import joblib
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"

RANDOM_STATE = 42
TEST_SIZE = 0.2
POS_WEIGHT = 11.47
EPOCHS = 300
PATIENCE = 15


def load_txt_features(txt_path):
    df = pd.read_csv(txt_path)
    y = df["Class"]
    x_raw = df.drop(columns=["Subject_ID", "UPDRS", "Class"], errors="ignore")

    col_map = {
        "Jitter_local": "Jitter(%)",
        "Jitter_local_absolute": "Jitter(Abs)",
        "Jitter_rap": "Jitter:RAP",
        "Jitter_ppq5": "Jitter:PPQ5",
        "Jitter_ddp": "Jitter:DDP",
        "Shimmer_local": "Shimmer",
        "Shimmer_local_dB": "Shimmer(dB)",
        "Shimmer_apq3": "Shimmer:APQ3",
        "Shimmer_apq5": "Shimmer:APQ5",
        "Shimmer_dda": "Shimmer:DDA",
        "NTH": "NHR",
        "HTN": "HNR",
    }

    existing = [column for column in col_map if column in x_raw.columns]
    x = x_raw[existing].rename(columns=col_map)
    return x, y


def fetch_uci_dataset(dataset_id, drop_cols=None):
    dataset = fetch_ucirepo(id=dataset_id)
    x = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    if drop_cols:
        x = x.drop(columns=drop_cols, errors="ignore")

    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    return x, y, dataset.metadata


def align_feature_frames(*frames):
    cleaned = [frame.loc[:, ~frame.columns.duplicated()] for frame in frames]
    all_columns = sorted(set().union(*(frame.columns for frame in cleaned)))
    return [frame.reindex(columns=all_columns) for frame in cleaned]


def load_training_data(data_dir=DATA_DIR):
    x1, y1, _ = fetch_uci_dataset(174, drop_cols=["name"])
    x2, y2_raw, _ = fetch_uci_dataset(189, drop_cols=["subject#"])
    y2 = (y2_raw > 0).astype(int)
    x3, y3 = load_txt_features(data_dir / "train_data_with_headers.txt")

    x1, x2, x3 = align_feature_frames(x1, x2, x3)

    x = pd.concat([x1, x2, x3], ignore_index=True)
    y = pd.concat([y1, y2, y3], ignore_index=True)
    x = x.fillna(x.mean(numeric_only=True))
    return x, y


def split_scale_balance(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    smote = SMOTE(random_state=random_state, k_neighbors=5)
    x_train_smote, y_train_smote = smote.fit_resample(x_train_scaled, y_train)

    return {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "x_train_scaled": x_train_scaled,
        "x_test_scaled": x_test_scaled,
        "x_train_smote": x_train_smote,
        "y_train_smote": y_train_smote,
        "scaler": scaler,
    }


class ParkinsonNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, inputs):
        return self.net(inputs)


def train_model(
    x_train_smote,
    y_train_smote,
    x_test_scaled,
    y_test,
    pos_weight=POS_WEIGHT,
    epochs=EPOCHS,
    patience=PATIENCE,
):
    x_train_t = torch.tensor(x_train_smote, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_smote, dtype=torch.float32)
    x_test_t = torch.tensor(x_test_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32)

    model = ParkinsonNet(input_dim=x_train_smote.shape[1])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        train_logits = model(x_train_t).squeeze()
        train_loss = criterion(train_logits, y_train_t.float())
        train_loss.backward()
        optimizer.step()
        train_losses.append(train_loss.item())

        model.eval()
        with torch.no_grad():
            val_logits = model(x_test_t).squeeze()
            val_loss = criterion(val_logits, y_test_t.float())
        val_losses.append(val_loss.item())

        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {train_loss.item():.4f} | "
                f"Val Loss: {val_loss.item():.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "x_train_t": x_train_t,
        "y_train_t": y_train_t,
        "x_test_t": x_test_t,
        "y_test_t": y_test_t,
    }


def evaluate_model(model, x_train_t, y_train_smote, x_test_t, y_test):
    model.eval()
    with torch.no_grad():
        train_logits = model(x_train_t).squeeze()
        train_probs = torch.sigmoid(train_logits).cpu().numpy()
        test_logits = model(x_test_t).squeeze()
        test_probs = torch.sigmoid(test_logits).cpu().numpy()

    train_preds = (train_probs > 0.5).astype(int)
    test_preds = (test_probs > 0.5).astype(int)

    return {
        "train_accuracy": accuracy_score(y_train_smote, train_preds),
        "train_auc": roc_auc_score(y_train_smote, train_probs),
        "train_report": classification_report(
            y_train_smote, train_preds, target_names=["Healthy", "Parkinson's"]
        ),
        "test_accuracy": accuracy_score(y_test, test_preds),
        "test_auc": roc_auc_score(y_test, test_probs),
        "test_report": classification_report(
            y_test, test_preds, target_names=["Healthy", "Parkinson's"]
        ),
        "test_confusion_matrix": confusion_matrix(y_test, test_preds),
    }


def save_artifacts(model, scaler, feature_columns, models_dir=MODELS_DIR):
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), models_dir / "parkinsons_model.pth")
    joblib.dump(scaler, models_dir / "scaler.pkl")
    joblib.dump(list(feature_columns), models_dir / "features.pkl")


def load_artifacts(models_dir=MODELS_DIR):
    feature_columns = joblib.load(models_dir / "features.pkl")
    scaler = joblib.load(models_dir / "scaler.pkl")
    model = ParkinsonNet(input_dim=len(feature_columns))
    model.load_state_dict(torch.load(models_dir / "parkinsons_model.pth", map_location="cpu"))
    model.eval()
    return model, scaler, feature_columns
