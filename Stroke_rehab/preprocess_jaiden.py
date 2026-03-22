"""
preprocess_jaiden.py
====================
Converts accurate_jaiden.csv (raw capture) into windowed EEG features,
relabels the existing final_bci_master.csv to match, and combines both
into resources/combined_bci_master.csv.

Label scheme (unified):
    NEUTRAL      -> neutral      (0)
    LOOKING_UP   -> looking_up   (1)
    LOOKING_DOWN -> looking_down (2)

Channel mapping:
    EEG_Behind_Left_Ear  -> TP9
    EEG_Frontal_1        -> AF7
    EEG_Frontal_2        -> AF8
    EEG_Behind_Right_Ear -> TP10

Run:
    python Stroke_rehab/preprocess_jaiden.py
"""

import os
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import kurtosis as sp_kurtosis, skew as sp_skew
from collections import Counter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV    = os.path.join(ROOT, "resources", "accurate_jaiden.csv")
MASTER_CSV   = os.path.join(ROOT, "resources", "final_bci_master.csv")
JAIDEN_OUT   = os.path.join(ROOT, "resources", "jaiden_master.csv")
COMBINED_OUT = os.path.join(ROOT, "resources", "combined_bci_master.csv")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHANNEL_MAP = {
    "EEG_Behind_Left_Ear":  "TP9",
    "EEG_Frontal_1":        "AF7",
    "EEG_Frontal_2":        "AF8",
    "EEG_Behind_Right_Ear": "TP10",
}

LABEL_MAP = {
    "NEUTRAL":     ("neutral",      0),
    "LOOKING_UP":  ("looking_up",   1),
    "LOOKING_DOWN": ("looking_down", 2),
}

# Relabel existing master: old -> new
RELABEL_MAP = {
    "rest":        ("neutral",      0),
    "left_blink":  ("looking_up",   1),
    "right_blink": ("looking_down", 2),
}

WINDOW_SEC = 0.250   # 250 ms window (matches existing master)
STEP_SEC   = 0.044   # ~44 ms step   (matches existing master overlap)

BANDS = {
    "delta": (1,   4),
    "theta": (4,   8),
    "alpha": (8,  13),
    "beta":  (13, 30),
    "gamma": (30, 80),
    "emg":   (80, 200),
}

# Features the training model uses (order matters for PER_CHANNEL_FEATS)
STAT_FEATURES = ["mean", "std", "rms", "mav", "peak2peak",
                 "var", "kurtosis", "skewness", "zcr", "iemg"]
BP_FEATURES   = [f"bp_{b}" for b in BANDS]   # delta included for CSV completeness


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _bandpower(data: np.ndarray, sfreq: float, fmin: float, fmax: float) -> float:
    nperseg = min(len(data), max(8, int(sfreq * 0.25)))
    freqs, psd = signal.welch(data, fs=sfreq, nperseg=nperseg)
    idx = (freqs >= fmin) & (freqs <= fmax)
    return float(np.trapezoid(psd[idx], freqs[idx])) if idx.any() else 0.0


def extract_channel_features(window: np.ndarray, sfreq: float) -> dict | None:
    """Return dict of all features for one channel window. None if too short."""
    if len(window) < 4:
        return None
    w = window.astype(float)
    feats = {}
    # Band powers
    for band, (fmin, fmax) in BANDS.items():
        feats[f"bp_{band}"] = _bandpower(w, sfreq, fmin, fmax)
    # Statistics
    feats["mean"]      = float(np.mean(w))
    feats["std"]       = float(np.std(w))
    feats["rms"]       = float(np.sqrt(np.mean(w ** 2)))
    feats["mav"]       = float(np.mean(np.abs(w)))
    feats["peak2peak"] = float(np.ptp(w))
    feats["var"]       = float(np.var(w))
    feats["kurtosis"]  = float(sp_kurtosis(w))
    feats["skewness"]  = float(sp_skew(w))
    feats["zcr"]       = float(np.sum(np.diff(np.sign(w)) != 0) / len(w))
    feats["iemg"]      = float(np.sum(np.abs(w)))
    return feats


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_jaiden(input_csv: str = INPUT_CSV) -> pd.DataFrame:
    print(f"\n[1/3] Reading {input_csv} ...")
    df = pd.read_csv(input_csv).dropna()
    df = df.sort_values("timestamp").reset_index(drop=True)

    t = df["timestamp"].values
    dt = np.diff(t)
    sfreq = 1.0 / np.median(dt[dt > 0])
    t_rel = t - t[0]
    total_dur = t_rel[-1]

    print(f"      Sampling rate : {sfreq:.1f} Hz")
    print(f"      Samples       : {len(df)}")
    print(f"      Duration      : {total_dur:.1f} s")
    print(f"      Label dist    : {dict(df['action_left'].value_counts())}")

    windows = []
    t0 = 0.0
    while t0 + WINDOW_SEC <= total_dur:
        t1 = t0 + WINDOW_SEC
        mask = (t_rel >= t0) & (t_rel < t1)
        chunk = df[mask]

        if len(chunk) < 4:
            t0 += STEP_SEC
            continue

        # Majority-vote label from action_left
        label_raw = Counter(chunk["action_left"].values).most_common(1)[0][0]
        label_str, label_int = LABEL_MAP.get(label_raw, ("neutral", 0))

        row = {
            "session":            "Jaiden",
            "window_start_s":     round(t0, 6),
            "window_end_s":       round(t1, 6),
            "label":              label_str,
            "label_int":          label_int,
            "eye_distance_left":  float(chunk["ear_left"].mean()),
            "eye_distance_right": float(chunk["ear_right"].mean()),
            "action_left":        label_raw,
            "action_right":       Counter(chunk["action_right"].values).most_common(1)[0][0],
        }

        # Per-channel features
        ok = True
        for raw_col, ch in CHANNEL_MAP.items():
            sig = chunk[raw_col].values.astype(float)
            feats = extract_channel_features(sig, sfreq)
            if feats is None:
                ok = False
                break
            for feat_name, val in feats.items():
                row[f"{ch}_{feat_name}"] = val

        if ok:
            windows.append(row)

        t0 += STEP_SEC

    out = pd.DataFrame(windows)
    print(f"\n      Windows generated : {len(out)}")
    print(f"      Label dist        : {dict(out['label'].value_counts())}")
    return out


def relabel_master(master_csv: str = MASTER_CSV) -> pd.DataFrame:
    print(f"\n[2/3] Relabelling {master_csv} ...")
    df = pd.read_csv(master_csv).dropna()
    print(f"      Original labels: {dict(df['label'].value_counts())}")

    df["label"] = df["label"].map(
        lambda x: RELABEL_MAP.get(x, (x, None))[0]
    )
    df["label_int"] = df["label"].map(
        {"neutral": 0, "looking_up": 1, "looking_down": 2}
    ).fillna(df["label_int"]).astype(int)

    print(f"      Relabelled     : {dict(df['label'].value_counts())}")
    return df


def combine_and_save(jaiden_df: pd.DataFrame, master_df: pd.DataFrame):
    print(f"\n[3/3] Combining datasets ...")

    # Align columns — master may have extra or fewer columns
    common_cols = [c for c in master_df.columns if c in jaiden_df.columns]
    extra_cols  = [c for c in jaiden_df.columns if c not in master_df.columns]

    combined = pd.concat(
        [master_df[common_cols],
         jaiden_df[common_cols + extra_cols]],
        ignore_index=True,
        sort=False,
    )

    combined.to_csv(COMBINED_OUT, index=False)
    jaiden_df.to_csv(JAIDEN_OUT, index=False)

    print(f"      Total windows  : {len(combined):,}")
    print(f"      Label dist     : {dict(combined['label'].value_counts())}")
    print(f"\n  Saved jaiden features -> {JAIDEN_OUT}")
    print(f"  Saved combined master -> {COMBINED_OUT}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    jaiden_df = preprocess_jaiden()
    master_df = relabel_master()
    combine_and_save(jaiden_df, master_df)
    print("\nDone. Next step:")
    print("  python Stroke_rehab/train_eeg_transformer.py --final --skip-cv")
