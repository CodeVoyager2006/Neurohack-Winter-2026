"""
EEG Transformer — Stroke Rehab BCI Model Training
===================================================
Dataset  : resources/final_bci_master.csv
Channels : TP9 (left temporal) · AF7 (left frontal) · AF8 (right frontal) · TP10 (right temporal)
Features : 13 per channel -> 52 total  (bandpower theta/alpha/beta/gamma/emg + 8 statistical features)
Classes  : 0=rest · 1=left_blink · 2=right_blink
Validation: Session-based leave-one-out cross-validation

Run:
    python train_eeg_transformer.py
    python train_eeg_transformer.py --final   # train on ALL sessions for deployment
"""

import os
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -- MNE for EEG channel bookkeeping and epoch sanity checks --------------------
import mne
mne.set_log_level("WARNING")

# -- PyTorch ---------------------------------------------------------------------
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

# -- Scikit-learn ----------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
)

# -------------------------------------------------------------------------------
# 1.  CONFIGURATION
# -------------------------------------------------------------------------------

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "resources", "final_bci_master.csv")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# Muse headset channel names (MNE-compatible labels)
EEG_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]

# ---------- Features extracted per channel ----------
# Bandpower (5 bands — delta excluded: always 0 in 250 ms windows)
BP_BANDS = ["theta", "alpha", "beta", "gamma", "emg"]
# Statistical descriptors
STAT_FEATURES = ["mean", "std", "rms", "mav", "peak2peak", "kurtosis", "skewness", "zcr"]
# TP9_theta, TP9_alpha, ..., TP9_skewness, TP9_zcr  (13 features × 4 channels = 52)
PER_CHANNEL_FEATS = [f"bp_{b}" for b in BP_BANDS] + STAT_FEATURES  # 5 + 8 = 13

CLASS_NAMES  = ["rest", "left_blink", "right_blink"]
SESSIONS     = ["LeftEye", "RightEye", "MixedBlink"]

# Training hyper-parameters
D_MODEL      = 64
N_HEADS      = 8
N_LAYERS     = 4
DROPOUT      = 0.30
LR           = 3e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 256
MAX_EPOCHS   = 100
PATIENCE     = 12          # early-stopping patience on val macro-F1
SEED         = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# -------------------------------------------------------------------------------
# 2.  DATA LOADING & FEATURE MATRIX CONSTRUCTION
# -------------------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame):
    """
    Returns
    -------
    X : ndarray [N, n_channels=4, n_feat_per_ch=13]
    y : ndarray [N]  int labels
    """
    channel_arrays = []
    for ch in EEG_CHANNELS:
        cols = [f"{ch}_{f}" for f in PER_CHANNEL_FEATS]
        channel_arrays.append(df[cols].values)          # [N, 13]
    X = np.stack(channel_arrays, axis=1).astype(np.float32)  # [N, 4, 13]
    y = df["label_int"].values.astype(np.int64)
    return X, y


def load_dataset():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna()
    print(f"Loaded {len(df):,} windows  |  Classes: {df['label'].value_counts().to_dict()}")
    return df


# -------------------------------------------------------------------------------
# 3.  MNE CHANNEL INFO — for neuroscientific validity & epoch inspection
# -------------------------------------------------------------------------------

def make_mne_info(sfreq: float = 256.0) -> mne.Info:
    """Create MNE Info object for the 4 Muse channels."""
    ch_names = EEG_CHANNELS
    ch_types = ["eeg"] * 4
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    # Muse standard 10-20 positions
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, on_missing="ignore")
    return info


def mne_sanity_check(df: pd.DataFrame):
    """
    Build a small MNE EpochsArray from the raw EEG snapshot columns to verify
    channel amplitude ranges are physiologically plausible (<100 uV).
    Saves a topographic plot to model/topo_check.png.
    """
    raw_cols = {
        "TP9":  "EEG_Behind_Left_Ear",
        "AF7":  "EEG_Frontal_1",
        "AF8":  "EEG_Frontal_2",
        "TP10": "EEG_Behind_Right_Ear",
    }
    # Create pseudo-epochs: each window is 1 sample -> shape [N, 4, 1]
    data = np.stack([df[raw_cols[ch]].values for ch in EEG_CHANNELS], axis=0)  # [4, N]
    data_uv = data * 1e-6   # convert uV -> V for MNE

    info = make_mne_info()
    raw = mne.io.RawArray(data_uv, info)

    print("\n[MNE] Channel amplitudes (uV):")
    for ch, col in raw_cols.items():
        vals = df[col].values
        print(f"  {ch:5s}: mean={vals.mean():+7.2f}  std={vals.std():6.2f}  "
              f"min={vals.min():+8.2f}  max={vals.max():+8.2f}")

    # Band-power summary using MNE PSD
    try:
        spectrum = raw.compute_psd(method="welch", fmin=1, fmax=100, n_fft=256)
        fig = spectrum.plot(show=False)
        fig.savefig(os.path.join(MODEL_DIR, "psd_check.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"[MNE] PSD plot saved -> model/psd_check.png")
    except Exception as e:
        print(f"[MNE] PSD plot skipped ({e})")


# -------------------------------------------------------------------------------
# 4.  EEG TRANSFORMER MODEL
# -------------------------------------------------------------------------------

class ChannelTokenEmbedding(nn.Module):
    """Projects each channel's feature vector to d_model."""
    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)

    def forward(self, x):
        # x : [B, n_channels, n_features]
        return self.proj(x)   # [B, n_channels, d_model]


class EEGTransformer(nn.Module):
    """
    Channel-wise Transformer for EEG classification.

    Architecture
    -------------
    Linear projection  ->  learnable channel pos-embedding
    -> Transformer Encoder (n_layers × n_heads)
    -> Global average pool (over channels)
    -> Classification head
    """
    def __init__(
        self,
        n_channels: int,
        n_features: int,
        n_classes: int,
        d_model: int   = D_MODEL,
        n_heads: int   = N_HEADS,
        n_layers: int  = N_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.n_channels = n_channels

        self.embedding   = ChannelTokenEmbedding(n_features, d_model)
        self.pos_embed   = nn.Parameter(torch.randn(1, n_channels, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,          # Pre-LayerNorm (more stable training)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.norm        = nn.LayerNorm(d_model)
        self.dropout     = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, x):
        # x : [B, n_channels, n_features]
        tokens = self.embedding(x) + self.pos_embed   # [B, C, d_model]
        tokens = self.transformer(tokens)              # [B, C, d_model]
        pooled = tokens.mean(dim=1)                    # [B, d_model]  global avg pool
        pooled = self.dropout(self.norm(pooled))
        return self.classifier(pooled)                 # [B, n_classes]


# -------------------------------------------------------------------------------
# 5.  TRAINING & EVALUATION UTILITIES
# -------------------------------------------------------------------------------

def compute_class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights to counteract severe class imbalance."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    weights = counts.sum() / (n_classes * counts)
    weights = weights / weights.min()    # normalise so lightest class = 1.0
    print(f"Class weights: { {CLASS_NAMES[i]: f'{w:.2f}' for i, w in enumerate(weights)} }")
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


def make_loaders(X_tr, y_tr, X_val, y_val):
    def to_tensor(X, y):
        return TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.int64),
        )
    train_ds = to_tensor(X_tr, y_tr)
    val_ds   = to_tensor(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss    += loss.item() * len(y_batch)
        total_correct += (logits.argmax(1) == y_batch).sum().item()
        total         += len(y_batch)
    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, return_preds=False):
    model.eval()
    total_loss, total = 0.0, 0
    all_preds, all_targets = [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        total_loss    += loss.item() * len(y_batch)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_targets.extend(y_batch.cpu().numpy())
        total += len(y_batch)
    all_preds   = np.array(all_preds)
    all_targets = np.array(all_targets)
    macro_f1    = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    acc         = (all_preds == all_targets).mean()
    if return_preds:
        return total_loss / total, acc, macro_f1, all_preds, all_targets
    return total_loss / total, acc, macro_f1


def run_fold(fold_name, df_train, df_val, df_test, scaler=None):
    """Train + evaluate one CV fold. Returns (scaler, best_macro_f1, test_preds, test_targets)."""
    print(f"\n{'-'*60}")
    print(f"  FOLD: held-out session = {fold_name}")
    print(f"  Train: {len(df_train):,}  |  Val: {len(df_val):,}  |  Test: {len(df_test):,}")

    X_tr, y_tr   = build_feature_matrix(df_train)
    X_val, y_val = build_feature_matrix(df_val)
    X_te, y_te   = build_feature_matrix(df_test)

    # Flatten for scaler, then reshape
    N_tr,  C, F = X_tr.shape
    N_val       = X_val.shape[0]
    N_te        = X_te.shape[0]

    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(X_tr.reshape(N_tr, -1))

    X_tr  = scaler.transform(X_tr.reshape(N_tr,   -1)).reshape(N_tr,  C, F)
    X_val = scaler.transform(X_val.reshape(N_val,  -1)).reshape(N_val, C, F)
    X_te  = scaler.transform(X_te.reshape(N_te,    -1)).reshape(N_te,  C, F)

    class_weights = compute_class_weights(y_tr, n_classes=3)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    model = EEGTransformer(
        n_channels=C, n_features=F, n_classes=3
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)

    train_loader, val_loader = make_loaders(X_tr, y_tr, X_val, y_val)

    best_f1, best_epoch, best_state = 0.0, 0, None
    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc, va_f1 = evaluate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_f1"].append(va_f1)

        if va_f1 > best_f1:
            best_f1    = va_f1
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or patience_counter == 0:
            print(f"  Epoch {epoch:3d}  "
                  f"tr_loss={tr_loss:.4f}  va_loss={va_loss:.4f}  "
                  f"va_f1={va_f1:.4f}  {'*best*' if patience_counter == 0 else ''}")

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} (best epoch={best_epoch})")
            break

    # Restore best weights and evaluate on test set
    model.load_state_dict(best_state)
    te_loss, te_acc, te_f1, preds, targets = evaluate(
        model, DataLoader(
            TensorDataset(torch.tensor(X_te, dtype=torch.float32),
                          torch.tensor(y_te, dtype=torch.int64)),
            batch_size=BATCH_SIZE),
        criterion, return_preds=True,
    )

    print(f"\n  [Test] loss={te_loss:.4f}  acc={te_acc:.4f}  macro-F1={te_f1:.4f}")
    print(classification_report(targets, preds, target_names=CLASS_NAMES, zero_division=0))

    return scaler, model, best_f1, te_f1, preds, targets, history


# -------------------------------------------------------------------------------
# 6.  CROSS-VALIDATION LOOP
# -------------------------------------------------------------------------------

def stratified_kfold_cv(df, n_splits=5):
    """
    5-fold stratified cross-validation across all sessions combined.

    Why this replaces session-LOO:
    - Session-LOO trained on only 1 session (~12k windows, 1 dominant blink type)
    - Each session has biased label distribution (LeftEye=left_blinks, RightEye=right_blinks)
    - This caused near-zero blink recall because the model never saw balanced examples
    - Stratified k-fold ensures every fold has the same class ratio as the full dataset
    - Train size per fold: ~80% of 38k = 30k windows with all blink types represented
    """
    X, y = build_feature_matrix(df)
    cv_results = []
    all_preds, all_targets = [], []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(X, y)):
        # Further split train_val into 80% train / 20% val (stratified)
        X_tv, y_tv = X[train_val_idx], y[train_val_idx]
        inner_skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        tr_idx, va_idx = next(inner_skf.split(X_tv, y_tv))

        X_tr,  y_tr  = X_tv[tr_idx],  y_tv[tr_idx]
        X_val, y_val = X_tv[va_idx],  y_tv[va_idx]
        X_te,  y_te  = X[test_idx],   y[test_idx]

        print(f"\n{'-'*60}")
        print(f"  FOLD {fold_idx+1}/{n_splits}")
        print(f"  Train: {len(y_tr):,}  |  Val: {len(y_val):,}  |  Test: {len(y_te):,}")
        print(f"  Train class dist: { {CLASS_NAMES[i]: int((y_tr==i).sum()) for i in range(3)} }")

        # Normalise
        N_tr,  C, F = X_tr.shape
        scaler = StandardScaler()
        scaler.fit(X_tr.reshape(N_tr, -1))
        X_tr  = scaler.transform(X_tr.reshape(N_tr,  -1)).reshape(N_tr,  C, F)
        X_val = scaler.transform(X_val.reshape(len(y_val), -1)).reshape(len(y_val), C, F)
        X_te  = scaler.transform(X_te.reshape(len(y_te),  -1)).reshape(len(y_te),  C, F)

        class_weights = compute_class_weights(y_tr, n_classes=3)
        criterion     = nn.CrossEntropyLoss(weight=class_weights)

        model = EEGTransformer(n_channels=C, n_features=F, n_classes=3).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)

        train_loader, val_loader = make_loaders(X_tr, y_tr, X_val, y_val)

        best_f1, best_epoch, best_state = 0.0, 0, None
        history = {"train_loss": [], "val_loss": [], "val_f1": []}
        patience_counter = 0

        for epoch in range(1, MAX_EPOCHS + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            va_loss, va_acc, va_f1 = evaluate(model, val_loader, criterion)
            scheduler.step()

            history["train_loss"].append(tr_loss)
            history["val_loss"].append(va_loss)
            history["val_f1"].append(va_f1)

            if va_f1 > best_f1:
                best_f1    = va_f1
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 10 == 0 or patience_counter == 0:
                print(f"  Epoch {epoch:3d}  tr_loss={tr_loss:.4f}  va_loss={va_loss:.4f}"
                      f"  va_f1={va_f1:.4f}  {'*best*' if patience_counter == 0 else ''}")

            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} (best epoch={best_epoch})")
                break

        model.load_state_dict(best_state)
        te_loss, te_acc, te_f1, preds, targets = evaluate(
            model,
            DataLoader(TensorDataset(torch.tensor(X_te, dtype=torch.float32),
                                     torch.tensor(y_te, dtype=torch.int64)),
                       batch_size=BATCH_SIZE),
            criterion, return_preds=True,
        )

        print(f"\n  [Test] loss={te_loss:.4f}  acc={te_acc:.4f}  macro-F1={te_f1:.4f}")
        print(classification_report(targets, preds, target_names=CLASS_NAMES, zero_division=0))

        all_preds.extend(preds)
        all_targets.extend(targets)
        cv_results.append({"fold": fold_idx+1, "val_f1": best_f1, "test_f1": te_f1, "history": history})

    print("\n" + "="*60)
    print("  CROSS-VALIDATION SUMMARY  (5-fold stratified)")
    print("="*60)
    for r in cv_results:
        print(f"  Fold {r['fold']}  val_f1={r['val_f1']:.4f}  test_f1={r['test_f1']:.4f}")
    mean_f1 = np.mean([r["test_f1"] for r in cv_results])
    print(f"  MEAN                      test_f1={mean_f1:.4f}")

    cm = confusion_matrix(all_targets, all_preds)
    print(f"\n  Overall Classification Report:")
    print(classification_report(all_targets, all_preds, target_names=CLASS_NAMES, zero_division=0))

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, colorbar=False)
    ax.set_title(f"CV Confusion Matrix ({n_splits}-fold stratified)")
    fig.tight_layout()
    fig.savefig(os.path.join(MODEL_DIR, "cv_confusion_matrix.png"), dpi=120)
    plt.close(fig)
    print(f"  Confusion matrix saved -> model/cv_confusion_matrix.png")

    return cv_results


# -------------------------------------------------------------------------------
# 7.  FINAL MODEL TRAINING (all sessions)
# -------------------------------------------------------------------------------

def train_final_model(df):
    """Train on all data with a small held-out validation split for early stopping."""
    print("\n" + "="*60)
    print("  TRAINING FINAL DEPLOYMENT MODEL (all sessions)")
    print("="*60)

    # 90/10 stratified-by-session split
    val_idx   = df.groupby("session").apply(
        lambda g: g.sample(frac=0.10, random_state=SEED)
    ).index.get_level_values(1)
    df_val   = df.loc[val_idx]
    df_train = df.drop(index=val_idx)

    scaler, model, best_val_f1, _, preds, targets, history = run_fold(
        "ALL_SESSIONS", df_train, df_val, df_val   # test = val (no held-out for final)
    )

    # Save artefacts
    model_path  = os.path.join(MODEL_DIR, "eeg_transformer.pt")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    config_path = os.path.join(MODEL_DIR, "model_config.pkl")

    torch.save(model.state_dict(), model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(config_path, "wb") as f:
        pickle.dump({
            "n_channels":  len(EEG_CHANNELS),
            "n_features":  len(PER_CHANNEL_FEATS),
            "n_classes":   3,
            "d_model":     D_MODEL,
            "n_heads":     N_HEADS,
            "n_layers":    N_LAYERS,
            "dropout":     DROPOUT,
            "channels":    EEG_CHANNELS,
            "features":    PER_CHANNEL_FEATS,
            "class_names": CLASS_NAMES,
        }, f)

    print(f"\n  Saved model   -> {model_path}")
    print(f"  Saved scaler  -> {scaler_path}")
    print(f"  Saved config  -> {config_path}")

    # Training curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="train")
    ax1.plot(history["val_loss"],   label="val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.set_title("Loss")
    ax2.plot(history["val_f1"], color="green")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Macro-F1"); ax2.set_title("Val Macro-F1")
    fig.tight_layout()
    fig.savefig(os.path.join(MODEL_DIR, "training_curve.png"), dpi=120)
    plt.close(fig)
    print(f"  Training curve -> model/training_curve.png")

    return model, scaler


# -------------------------------------------------------------------------------
# 8.  REAL-TIME INFERENCE HELPER  (imported by openbci_stream.py / gui.py)
# -------------------------------------------------------------------------------

class EEGIntentDecoder:
    """
    Lightweight wrapper for real-time deployment.

    Usage
    -----
    decoder = EEGIntentDecoder.load()
    # features_dict: { 'TP9': [...13 values...], 'AF7': [...], 'AF8': [...], 'TP10': [...] }
    label, confidence = decoder.predict(features_dict)
    """

    def __init__(self, model: EEGTransformer, scaler: StandardScaler, config: dict):
        self.model  = model.eval().to("cpu")
        self.scaler = scaler
        self.config = config

    @classmethod
    def load(cls, model_dir=MODEL_DIR):
        config_path = os.path.join(model_dir, "model_config.pkl")
        model_path  = os.path.join(model_dir, "eeg_transformer.pt")
        scaler_path = os.path.join(model_dir, "scaler.pkl")

        with open(config_path, "rb") as f:
            config = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        model = EEGTransformer(
            n_channels=config["n_channels"],
            n_features=config["n_features"],
            n_classes=config["n_classes"],
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_layers=config["n_layers"],
            dropout=0.0,    # no dropout at inference
        )
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        return cls(model, scaler, config)

    def predict(self, features_dict: dict):
        """
        Parameters
        ----------
        features_dict : dict
            Keys = channel names (TP9/AF7/AF8/TP10),
            Values = list/array of 13 feature values in PER_CHANNEL_FEATS order.

        Returns
        -------
        label      : str  ("rest" / "left_blink" / "right_blink")
        confidence : float (0–1)
        probs      : ndarray [3]
        """
        channels = self.config["channels"]
        X = np.array([features_dict[ch] for ch in channels], dtype=np.float32)  # [4, 13]
        X_flat  = self.scaler.transform(X.reshape(1, -1))                        # [1, 52]
        X_input = torch.tensor(X_flat.reshape(1, len(channels), -1))             # [1, 4, 13]

        with torch.no_grad():
            logits = self.model(X_input)
            probs  = torch.softmax(logits, dim=-1).squeeze().numpy()

        class_idx  = int(probs.argmax())
        label      = self.config["class_names"][class_idx]
        confidence = float(probs[class_idx])
        return label, confidence, probs


# -------------------------------------------------------------------------------
# 9.  MAIN
# -------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--final", action="store_true",
        help="Also train final deployment model on all sessions after CV"
    )
    parser.add_argument(
        "--skip-cv", action="store_true",
        help="Skip cross-validation, go straight to final model training"
    )
    parser.add_argument(
        "--skip-mne", action="store_true",
        help="Skip MNE sanity check (useful if mne not installed)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  EEG Transformer — BCI Stroke Rehab")
    print("=" * 60)
    print(f"  Channels  : {EEG_CHANNELS}")
    print(f"  Features  : {len(PER_CHANNEL_FEATS)} per channel  "
          f"({[f for f in PER_CHANNEL_FEATS]})")
    print(f"  Classes   : {CLASS_NAMES}")
    print(f"  Data      : {DATA_PATH}")
    print()

    df = load_dataset()

    if not args.skip_mne:
        try:
            mne_sanity_check(df)
        except ImportError:
            print("[MNE] Not installed — skipping sanity check. pip install mne")

    if not args.skip_cv:
        cv_results = stratified_kfold_cv(df)

    if args.final or args.skip_cv:
        model, scaler = train_final_model(df)
        print("\n  Final model ready for real-time deployment.")
        print("  Load with:  decoder = EEGIntentDecoder.load()")


if __name__ == "__main__":
    main()
