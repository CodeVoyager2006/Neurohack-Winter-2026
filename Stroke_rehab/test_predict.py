"""
test_predict.py
===============
Interactive inference tester for the EEG Transformer BCI model.

Two ways to use it:

  1. Demo mode (default) — loads one real sample per class from the CSV
     and runs predictions, showing the EMG cross-check each time:

         python test_predict.py

  2. Custom input — pass your own EEG feature values as a JSON string:

         python test_predict.py --features '{
             "TP9":  [bp_theta, bp_alpha, bp_beta, bp_gamma, bp_emg, mean, std, rms, mav, peak2peak, kurtosis, skewness, zcr],
             "AF7":  [...13 values...],
             "AF8":  [...13 values...],
             "TP10": [...13 values...]
         }'

Feature order per channel (13 values):
    [bp_theta, bp_alpha, bp_beta, bp_gamma, bp_emg,
     mean, std, rms, mav, peak2peak, kurtosis, skewness, zcr]

EMG cross-check:
    Uses bp_emg (100-200 Hz band power) from TP9 and TP10.
    These temporal channels sit behind the ears and pick up facial
    muscle activation as high-frequency artifact — a blink should
    show elevated bp_emg; rest should be low.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths — resolve relative to this file so the script works from any cwd
# ---------------------------------------------------------------------------
_HERE      = os.path.dirname(os.path.abspath(__file__))
_ROOT      = os.path.dirname(_HERE)
_MODEL_DIR = os.path.join(_HERE, "model")
_DATA_PATH = os.path.join(_ROOT, "resources", "final_bci_master.csv")

# Add Stroke_rehab to sys.path so we can import the training module
sys.path.insert(0, _HERE)

# EEGIntentDecoder and config constants live in train_eeg_transformer
from train_eeg_transformer import (
    EEGIntentDecoder,
    EEG_CHANNELS,
    PER_CHANNEL_FEATS,
    CLASS_NAMES,
)

# ---------------------------------------------------------------------------
# EMG cross-verification
# ---------------------------------------------------------------------------

#: Index of bp_emg within PER_CHANNEL_FEATS  (position 4 → 0-indexed)
_BP_EMG_IDX: int = PER_CHANNEL_FEATS.index("bp_emg")

#: Channels most sensitive to facial EMG (behind ears, close to jaw/brow muscles)
_EMG_CHANNELS: list[str] = ["TP9", "TP10"]

#: bp_emg threshold (µV²) above which we consider muscular activity elevated.
#: Derived from dataset median of the blink-class bp_emg distribution.
#: Adjust this if your hardware has different noise floor.
EMG_ACTIVITY_THRESHOLD: float = 500.0


def verify_emg(features_dict: dict) -> dict:
    """Cross-check the EEG prediction using the high-frequency (EMG) band power.

    Parameters
    ----------
    features_dict:
        Same dict passed to predict() — keys = channel names,
        values = list of 13 feature values in PER_CHANNEL_FEATS order.

    Returns
    -------
    dict with keys:
        emg_power_tp9   : float  — bp_emg from TP9 (µV²)
        emg_power_tp10  : float  — bp_emg from TP10 (µV²)
        emg_mean_power  : float  — average of the two
        muscle_active   : bool   — True if mean > EMG_ACTIVITY_THRESHOLD
        verdict         : str    — human-readable agreement note
    """
    tp9_emg  = float(features_dict["TP9"][_BP_EMG_IDX])
    tp10_emg = float(features_dict["TP10"][_BP_EMG_IDX])
    mean_emg = (tp9_emg + tp10_emg) / 2.0
    active   = mean_emg > EMG_ACTIVITY_THRESHOLD

    return {
        "emg_power_tp9":  tp9_emg,
        "emg_power_tp10": tp10_emg,
        "emg_mean_power": mean_emg,
        "muscle_active":  active,
        "threshold_used": EMG_ACTIVITY_THRESHOLD,
    }


def _emg_verdict(prediction: str, muscle_active: bool) -> str:
    """Return a human-readable agreement string."""
    is_blink = prediction in ("left_blink", "right_blink")
    if is_blink and muscle_active:
        return "CONFIRMED  — EEG predicts blink and facial EMG is elevated"
    if is_blink and not muscle_active:
        return "UNCERTAIN  — EEG predicts blink but facial EMG is low (possible false positive)"
    if not is_blink and not muscle_active:
        return "CONFIRMED  — EEG predicts rest and facial EMG is quiet"
    # rest predicted but EMG elevated
    return "UNCERTAIN  — EEG predicts rest but facial EMG is elevated (artifact or missed blink)"


# ---------------------------------------------------------------------------
# Main prediction entry point
# ---------------------------------------------------------------------------

def predict_intent(
    features_dict: dict,
    decoder: EEGIntentDecoder | None = None,
) -> dict:
    """Run the full prediction + EMG cross-check pipeline.

    Parameters
    ----------
    features_dict:
        Dict mapping each EEG channel name to a list/array of 13 feature
        values in this exact order:
            [bp_theta, bp_alpha, bp_beta, bp_gamma, bp_emg,
             mean, std, rms, mav, peak2peak, kurtosis, skewness, zcr]

    decoder:
        Optional pre-loaded EEGIntentDecoder.  If None, the model is loaded
        from ``Stroke_rehab/model/`` on every call (slow — pass a decoder
        object when calling in a loop).

    Returns
    -------
    dict with keys:
        prediction  : str    — "rest" / "left_blink" / "right_blink"
        confidence  : float  — softmax probability of winning class (0-1)
        probs       : dict   — {class_name: probability} for all 3 classes
        emg         : dict   — output of verify_emg()
        verdict     : str    — EMG agreement string
    """
    if decoder is None:
        decoder = EEGIntentDecoder.load(_MODEL_DIR)

    label, confidence, probs_arr = decoder.predict(features_dict)

    emg_info = verify_emg(features_dict)
    verdict  = _emg_verdict(label, emg_info["muscle_active"])

    return {
        "prediction": label,
        "confidence": confidence,
        "probs":      {CLASS_NAMES[i]: float(probs_arr[i]) for i in range(len(CLASS_NAMES))},
        "emg":        emg_info,
        "verdict":    verdict,
    }


# ---------------------------------------------------------------------------
# Helper: build features_dict from a CSV row
# ---------------------------------------------------------------------------

def features_from_csv_row(row: pd.Series) -> dict:
    """Extract the 13-feature dict from a single CSV row."""
    return {
        ch: [float(row[f"{ch}_{f}"]) for f in PER_CHANNEL_FEATS]
        for ch in EEG_CHANNELS
    }


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def print_result(result: dict, true_label: str | None = None) -> None:
    """Print a prediction result in a readable format."""
    pred  = result["prediction"]
    conf  = result["confidence"]
    probs = result["probs"]
    emg   = result["emg"]

    label_str = f"  True label  : {true_label}" if true_label else ""

    print("=" * 58)
    if label_str:
        print(label_str)
    print(f"  Prediction  : {pred.upper()}")
    print(f"  Confidence  : {conf:.1%}")
    print(f"  Class probs : rest={probs['rest']:.3f}  "
          f"left={probs['left_blink']:.3f}  "
          f"right={probs['right_blink']:.3f}")
    print(f"  EMG TP9     : {emg['emg_power_tp9']:.1f} µV²")
    print(f"  EMG TP10    : {emg['emg_power_tp10']:.1f} µV²")
    threshold = emg.get("threshold_used", EMG_ACTIVITY_THRESHOLD)
    print(f"  EMG active  : {'YES' if emg['muscle_active'] else 'NO'} "
          f"(threshold={threshold:.0f} µV²)")
    print(f"  Verdict     : {result['verdict']}")
    print("=" * 58)


# ---------------------------------------------------------------------------
# Demo: sample one row per class from the CSV
# ---------------------------------------------------------------------------

def run_demo(decoder: EEGIntentDecoder) -> None:
    print("\n[DEMO] Sampling one window per class from the training CSV...\n")

    df = pd.read_csv(_DATA_PATH).dropna()

    for class_label in CLASS_NAMES:
        subset = df[df["label"] == class_label]
        if subset.empty:
            print(f"  No rows found for class '{class_label}' — skipping.")
            continue

        row = subset.sample(1, random_state=42).iloc[0]
        features = features_from_csv_row(row)
        result   = predict_intent(features, decoder=decoder)

        print(f"\n--- Sample class: {class_label} ---")
        print_result(result, true_label=class_label)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="EEG BCI prediction tester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--features", "-f", type=str, default=None,
        help=(
            "JSON string with custom EEG features. "
            "Keys: TP9, AF7, AF8, TP10. "
            "Each value is a list of 13 floats: "
            "[bp_theta, bp_alpha, bp_beta, bp_gamma, bp_emg, "
            "mean, std, rms, mav, peak2peak, kurtosis, skewness, zcr]."
        ),
    )
    parser.add_argument(
        "--threshold", type=float, default=EMG_ACTIVITY_THRESHOLD,
        help=f"EMG activity threshold in µV² (default: {EMG_ACTIVITY_THRESHOLD})",
    )
    args = parser.parse_args()

    import test_predict as _self
    _self.EMG_ACTIVITY_THRESHOLD = args.threshold

    print(f"Loading model from {_MODEL_DIR} ...")
    decoder = EEGIntentDecoder.load(_MODEL_DIR)
    print("Model loaded.\n")

    if args.features:
        # Custom input mode
        try:
            features_dict = json.loads(args.features)
        except json.JSONDecodeError as e:
            print(f"ERROR: Could not parse --features JSON: {e}")
            sys.exit(1)

        missing = [ch for ch in EEG_CHANNELS if ch not in features_dict]
        if missing:
            print(f"ERROR: Missing channels in --features: {missing}")
            print(f"Required channels: {EEG_CHANNELS}")
            sys.exit(1)

        for ch in EEG_CHANNELS:
            if len(features_dict[ch]) != len(PER_CHANNEL_FEATS):
                print(
                    f"ERROR: {ch} has {len(features_dict[ch])} values, "
                    f"expected {len(PER_CHANNEL_FEATS)}."
                )
                print(f"Feature order: {PER_CHANNEL_FEATS}")
                sys.exit(1)

        result = predict_intent(features_dict, decoder=decoder)
        print_result(result)
    else:
        # Demo mode — sample from CSV
        if not os.path.exists(_DATA_PATH):
            print(f"ERROR: Dataset not found at {_DATA_PATH}")
            print("Run with --features to use custom input instead.")
            sys.exit(1)
        run_demo(decoder)


if __name__ == "__main__":
    main()
