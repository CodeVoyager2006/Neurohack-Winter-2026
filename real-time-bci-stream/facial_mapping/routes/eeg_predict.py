"""
routes/eeg_predict.py  –  EEG Model Prediction Endpoint
========================================================
Simulates real-time EEG input by sampling a random row from the
training CSV, running it through the saved EEG Transformer model,
and returning the predicted expression together with enough debug
info to verify functionality without a live headset.

GET  /api/eeg-predict
---------------------
Response JSON (success)
    {
        "success":        true,
        "expression":     "raise",          // mapped expression name
        "prediction":     "left_blink",     // raw model class name
        "confidence":     0.78,             // softmax probability 0-1
        "true_label":     "left_blink",     // ground-truth label from CSV
        "csv_index":      20726,            // integer row index in CSV
        "session":        "RightEye",       // recording session name
        "window_start_s": 372.41,           // time window start (seconds)
        "window_end_s":   372.67,           // time window end (seconds)
        "probs": {                          // full softmax distribution
            "rest":        0.10,
            "left_blink":  0.78,
            "right_blink": 0.12
        }
    }

Response JSON (failure)
    { "success": false, "error": "<reason>" }

Expression mapping
------------------
    rest        →  neutral
    left_blink  →  raise
    right_blink →  knit
"""

import os
import sys
import random

import numpy as np
import pandas as pd
from flask import Blueprint, jsonify

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
_ROUTES_DIR = os.path.dirname(os.path.abspath(__file__))          # routes/
_FACIAL_DIR = os.path.dirname(_ROUTES_DIR)                         # facial_mapping/
_STREAM_DIR = os.path.dirname(_FACIAL_DIR)                         # real-time-bci-stream/
_ROOT       = os.path.dirname(_STREAM_DIR)                         # Neurohack-Winter-2026/
_STROKE_DIR = os.path.join(_ROOT, "Stroke_rehab")
_MODEL_DIR  = os.path.join(_STROKE_DIR, "model")
_DATA_PATH  = os.path.join(_ROOT, "resources", "final_bci_master.csv")

# Add Stroke_rehab to path so we can import the decoder
if _STROKE_DIR not in sys.path:
    sys.path.insert(0, _STROKE_DIR)

# ---------------------------------------------------------------------------
# Load model + dataset (once at import time — shared across all requests)
# ---------------------------------------------------------------------------
_decoder     = None
_df          = None
_df_by_class = {}   # { label_str: sub-DataFrame } for stratified sampling
_load_error  = None

try:
    from train_eeg_transformer import EEGIntentDecoder, EEG_CHANNELS, PER_CHANNEL_FEATS, CLASS_NAMES

    _decoder = EEGIntentDecoder.load(_MODEL_DIR)
    _df      = pd.read_csv(_DATA_PATH).dropna().reset_index(drop=True)
    # Pre-split by class for O(1) stratified sampling at request time
    for label in CLASS_NAMES:
        _df_by_class[label] = _df[_df["label"] == label]
    print(f"[EEG] Model loaded from {_MODEL_DIR}")
    print(f"[EEG] Dataset loaded — {len(_df):,} rows from {_DATA_PATH}")
    print(f"[EEG] Class counts: { {k: len(v) for k, v in _df_by_class.items()} }")
except Exception as exc:
    _load_error = str(exc)
    print(f"[EEG] WARNING: Could not load model/dataset: {exc}")
    # Define fallbacks so the route can still report the error cleanly
    CLASS_NAMES     = ["rest", "left_blink", "right_blink"]
    EEG_CHANNELS    = ["TP9", "AF7", "AF8", "TP10"]
    PER_CHANNEL_FEATS = []

# ---------------------------------------------------------------------------
# Expression mapping
# ---------------------------------------------------------------------------
_EXPRESSION_MAP = {
    "rest":        "neutral",
    "left_blink":  "raise",
    "right_blink": "knit",
}

# ---------------------------------------------------------------------------
# Blueprint
# ---------------------------------------------------------------------------
eeg_predict_bp = Blueprint("eeg_predict", __name__)


def _features_from_row(row: pd.Series) -> dict:
    """Extract the 13-feature dict from a single CSV row (matches training format)."""
    return {
        ch: [float(row[f"{ch}_{f}"]) for f in PER_CHANNEL_FEATS]
        for ch in EEG_CHANNELS
    }


@eeg_predict_bp.route("/api/eeg-predict", methods=["GET"])
def eeg_predict():
    """
    Sample a random row from the training CSV, run the EEG Transformer,
    and return the predicted expression with full debug metadata.

    Uses stratified sampling — picks a class uniformly at random first,
    then picks a random row from that class. This ensures all 3 expressions
    (neutral / raise / knit) are exercised during testing even though the
    dataset is 93% rest rows.
    """
    if _load_error:
        return jsonify({"success": False, "error": f"Model not loaded: {_load_error}"}), 500

    # Stratified: equal chance of each class regardless of dataset imbalance
    chosen_class = random.choice(CLASS_NAMES)
    subset       = _df_by_class[chosen_class]
    row          = subset.iloc[random.randint(0, len(subset) - 1)]

    # Extract features and predict
    try:
        features  = _features_from_row(row)
        label, confidence, probs_arr = _decoder.predict(features)
    except Exception as exc:
        return jsonify({"success": False, "error": f"Prediction failed: {exc}"}), 500

    expression = _EXPRESSION_MAP.get(label, "neutral")
    true_label = str(row["label"])

    return jsonify({
        "success":        True,
        "expression":     expression,
        "prediction":     label,
        "confidence":     round(float(confidence), 4),
        "true_label":     true_label,
        "csv_index":      int(row.name),
        "session":        str(row["session"]),
        "window_start_s": round(float(row["window_start_s"]), 3),
        "window_end_s":   round(float(row["window_end_s"]), 3),
        "probs": {
            CLASS_NAMES[i]: round(float(probs_arr[i]), 4)
            for i in range(len(CLASS_NAMES))
        },
    })
