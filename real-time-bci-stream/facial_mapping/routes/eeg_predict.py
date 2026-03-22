"""
routes/eeg_predict.py  –  Live EEG Inference via OpenBCI Cyton
==============================================================
Streams live EEG from a Cyton board, extracts features on every request,
and runs the trained EEG Transformer model for real-time intent prediction.

Channel mapping (matches Stroke_rehab/backend/openbci_stream.py):
    Cyton ch5 (EXG[4]) → EEG_Frontal_1        → AF7
    Cyton ch6 (EXG[5]) → EEG_Frontal_2        → AF8
    Cyton ch7 (EXG[6]) → EEG_Behind_Left_Ear  → TP9
    Cyton ch8 (EXG[7]) → EEG_Behind_Right_Ear → TP10

GET  /api/eeg-predict   →  run inference on latest 256 ms window
GET  /api/eeg-status    →  board connection status
POST /api/eeg-connect   →  attempt (re)connection to Cyton board
"""

import os
import sys
import threading
from collections import deque, Counter

import numpy as np
from scipy import signal
from scipy.stats import kurtosis as sp_kurtosis, skew as sp_skew
from flask import Blueprint, jsonify

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
_ROUTES_DIR = os.path.dirname(os.path.abspath(__file__))
_FACIAL_DIR = os.path.dirname(_ROUTES_DIR)
_STREAM_DIR = os.path.dirname(_FACIAL_DIR)
_ROOT       = os.path.dirname(_STREAM_DIR)
_STROKE_DIR = os.path.join(_ROOT, "Stroke_rehab")
_MODEL_DIR  = os.path.join(_STROKE_DIR, "model")

if _STROKE_DIR not in sys.path:
    sys.path.insert(0, _STROKE_DIR)

# ---------------------------------------------------------------------------
# Load trained model (once at import time)
# ---------------------------------------------------------------------------
_decoder    = None
_load_error = None

# Defaults — overwritten on successful import from training module
CLASS_NAMES       = ["neutral", "looking_up", "looking_down"]
EEG_CHANNELS      = ["TP9", "AF7", "AF8", "TP10"]
PER_CHANNEL_FEATS = [
    "bp_theta", "bp_alpha", "bp_beta", "bp_gamma", "bp_emg",
    "mean", "std", "rms", "mav", "peak2peak", "kurtosis", "skewness", "zcr",
]

try:
    from train_eeg_transformer import (
        EEGIntentDecoder, EEG_CHANNELS, PER_CHANNEL_FEATS, CLASS_NAMES,
    )
    _decoder = EEGIntentDecoder.load(_MODEL_DIR)
    print(f"[EEG] Model loaded  |  classes: {CLASS_NAMES}")
except Exception as exc:
    _load_error = str(exc)
    print(f"[EEG] WARNING: Model not loaded — {exc}")

# ---------------------------------------------------------------------------
# Board / inference constants
# ---------------------------------------------------------------------------
_CYTON_SAMPLE_RATE = 250      # Hz
_WINDOW_SAMPLES    = 64       # ~256 ms of EEG per inference window
_CONFIDENCE_FLOOR  = 0.40     # below this → fall back to neutral

# EXG channel offsets within the 8-channel Cyton layout:
#   Indices 0-3 → EMG_1..4   (channels 1-4 on the board)
#   Indices 4-7 → EEG        (channels 5-8 on the board)
_EXG_OFFSETS = {
    "TP9":  6,   # ch7 → EEG_Behind_Left_Ear
    "AF7":  4,   # ch5 → EEG_Frontal_1
    "AF8":  5,   # ch6 → EEG_Frontal_2
    "TP10": 7,   # ch8 → EEG_Behind_Right_Ear
}

# Frequency bands for Welch bandpower (must match preprocessing)
_BANDS = {
    "bp_theta": (4,   8),
    "bp_alpha": (8,  13),
    "bp_beta":  (13, 30),
    "bp_gamma": (30, 80),
    "bp_emg":   (80, 200),
}

# ---------------------------------------------------------------------------
# Expression map — model class → frontend expression key
# ---------------------------------------------------------------------------
_EXPRESSION_MAP = {
    "neutral":      "neutral",
    "looking_up":   "lookup",
    "looking_down": "lookdown",
}

# ---------------------------------------------------------------------------
# Prediction smoother — majority vote over the last 3 windows
# ---------------------------------------------------------------------------
_pred_history: deque = deque(maxlen=3)


def _smooth(label: str) -> str:
    _pred_history.append(label)
    return Counter(_pred_history).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _bandpower(data: np.ndarray, fmin: float, fmax: float) -> float:
    """Welch-based band power estimate for one EEG channel window."""
    nperseg = min(len(data), max(8, int(_CYTON_SAMPLE_RATE * 0.25)))
    freqs, psd = signal.welch(data, fs=_CYTON_SAMPLE_RATE, nperseg=nperseg)
    idx = (freqs >= fmin) & (freqs <= fmax)
    if not idx.any():
        return 0.0
    # Support both NumPy ≥2.0 (trapezoid) and older (trapz)
    try:
        return float(np.trapezoid(psd[idx], freqs[idx]))
    except AttributeError:
        return float(np.trapz(psd[idx], freqs[idx]))


def _extract_features(window: dict) -> dict | None:
    """
    Extract 13 features per channel from a raw EEG window.

    Parameters
    ----------
    window : dict
        { channel_name: np.ndarray of shape (n_samples,) }

    Returns
    -------
    dict | None
        { channel_name: [13 floats] } in PER_CHANNEL_FEATS order,
        or None if the window is too short.
    """
    features = {}
    for ch in EEG_CHANNELS:
        w = window[ch].astype(float)
        if len(w) < 8:
            return None

        ch_feats: list[float] = []

        # 5 band powers (in PER_CHANNEL_FEATS order)
        for band_key in ["bp_theta", "bp_alpha", "bp_beta", "bp_gamma", "bp_emg"]:
            fmin, fmax = _BANDS[band_key]
            ch_feats.append(_bandpower(w, fmin, fmax))

        # 8 statistical descriptors
        ch_feats += [
            float(np.mean(w)),
            float(np.std(w)),
            float(np.sqrt(np.mean(w ** 2))),          # RMS
            float(np.mean(np.abs(w))),                 # MAV
            float(np.ptp(w)),                          # peak-to-peak
            float(sp_kurtosis(w)),
            float(sp_skew(w)),
            float(np.sum(np.diff(np.sign(w)) != 0) / len(w)),  # ZCR
        ]
        features[ch] = ch_feats

    return features


# ---------------------------------------------------------------------------
# Cyton board manager (thread-safe singleton)
# ---------------------------------------------------------------------------

class _CytonManager:
    """
    Wraps BrainFlow BoardShim for the OpenBCI Cyton board.

    connect()      — (re)connect, auto-detecting the serial port if needed
    get_window()   — peek at the latest _WINDOW_SAMPLES without clearing buffer
    disconnect()   — stop stream and release session
    """

    def __init__(self):
        self._lock   = threading.Lock()
        self._board  = None
        self._exg_ch = None   # row indices in the BrainFlow data array
        self.status  = "disconnected"
        self.error   = ""
        self.port    = ""

    # ------------------------------------------------------------------
    def connect(self, port: str | None = None) -> bool:
        """
        Connect to the Cyton.  If port is None, probe all serial ports.
        Returns True on success.
        """
        try:
            from brainflow.board_shim import BoardShim, BoardIds, BrainFlowInputParams
            from brainflow.exit_codes import BrainFlowError
            import serial.tools.list_ports
        except ImportError as exc:
            self.error  = f"BrainFlow not installed: {exc}"
            self.status = "error"
            return False

        BoardShim.disable_board_logger()

        with self._lock:
            self._release_locked()   # clean up any previous session

            exg_ch = BoardShim.get_exg_channels(BoardIds.CYTON_BOARD.value)
            ports_to_try = [port] if port else [
                p.device for p in serial.tools.list_ports.comports()
            ]

            for p in ports_to_try:
                params = BrainFlowInputParams()
                params.serial_port = p
                board = BoardShim(BoardIds.CYTON_BOARD.value, params)
                try:
                    board.prepare_session()
                    board.start_stream(45000)
                    self._board  = board
                    self._exg_ch = exg_ch
                    self.status  = "connected"
                    self.error   = ""
                    self.port    = p
                    print(f"[EEG] Cyton connected on {p}  |  EXG rows: {exg_ch}")
                    return True
                except Exception:
                    try:
                        board.release_session()
                    except Exception:
                        pass

            self.status = "disconnected"
            self.error  = "No Cyton board found — check USB dongle and drivers"
            print(f"[EEG] {self.error}")
            return False

    # ------------------------------------------------------------------
    def get_window(self) -> dict | None:
        """
        Non-destructively peek at the last _WINDOW_SAMPLES of EEG data.
        Returns { ch_name: np.ndarray } or None if not ready.
        """
        with self._lock:
            if self._board is None or self.status != "connected":
                return None
            try:
                data = self._board.get_current_board_data(_WINDOW_SAMPLES)
            except Exception:
                self.status = "disconnected"
                self._board = None
                return None

            if data.shape[1] < _WINDOW_SAMPLES // 2:
                return None   # buffer hasn't filled yet after connect

            result = {}
            for ch_name, offset in _EXG_OFFSETS.items():
                row = self._exg_ch[offset]
                result[ch_name] = data[row, :]
            return result

    # ------------------------------------------------------------------
    def disconnect(self):
        with self._lock:
            self._release_locked()

    def _release_locked(self):
        if self._board is not None:
            try:
                self._board.stop_stream()
            except Exception:
                pass
            try:
                self._board.release_session()
            except Exception:
                pass
            self._board  = None
            self._exg_ch = None
            self.status  = "disconnected"


# Singleton — connects at Flask startup
_board = _CytonManager()
_board.connect()

# ---------------------------------------------------------------------------
# Blueprint
# ---------------------------------------------------------------------------
eeg_predict_bp = Blueprint("eeg_predict", __name__)


@eeg_predict_bp.route("/api/eeg-predict", methods=["GET"])
def eeg_predict():
    """
    Acquire the latest 256 ms EEG window, extract features, run the
    EEG Transformer, and return the smoothed intent prediction.
    """
    # ── 0. Model check ─────────────────────────────────────────────────
    if _load_error:
        return jsonify({
            "success": False,
            "error": f"Model not loaded: {_load_error}",
            "board_status": _board.status,
        }), 500

    # ── 1. Board check ──────────────────────────────────────────────────
    if _board.status != "connected":
        return jsonify({
            "success":      False,
            "error":        _board.error or "Cyton board not connected",
            "board_status": _board.status,
        }), 503

    # ── 2. Acquire EEG window ───────────────────────────────────────────
    window = _board.get_window()
    if window is None:
        return jsonify({
            "success":      False,
            "error":        "EEG buffer not ready — wait ~0.5 s after connecting",
            "board_status": _board.status,
        }), 503

    # ── 3. Feature extraction ───────────────────────────────────────────
    try:
        features = _extract_features(window)
    except Exception as exc:
        return jsonify({
            "success": False,
            "error": f"Feature extraction failed: {exc}",
            "board_status": _board.status,
        }), 500

    if features is None:
        return jsonify({
            "success":      False,
            "error":        "Insufficient samples in window",
            "board_status": _board.status,
        }), 503

    # ── 4. Model inference ──────────────────────────────────────────────
    try:
        label, confidence, probs_arr = _decoder.predict(features)
    except Exception as exc:
        return jsonify({
            "success": False,
            "error": f"Prediction failed: {exc}",
            "board_status": _board.status,
        }), 500

    # ── 5. Confidence floor + majority-vote smoothing ───────────────────
    raw_label = label
    if confidence < _CONFIDENCE_FLOOR:
        label = "neutral"

    smoothed   = _smooth(label)
    expression = _EXPRESSION_MAP.get(smoothed, "neutral")

    return jsonify({
        "success":      True,
        "expression":   expression,          # → frontend setExpression()
        "prediction":   smoothed,            # smoothed class (3-window majority)
        "raw":          raw_label,           # single-window raw class
        "confidence":   round(float(confidence), 4),
        "probs": {
            CLASS_NAMES[i]: round(float(probs_arr[i]), 4)
            for i in range(len(CLASS_NAMES))
        },
        "samples":      int(list(window.values())[0].shape[0]),
        "board_status": _board.status,
        "port":         _board.port,
    })


@eeg_predict_bp.route("/api/eeg-status", methods=["GET"])
def eeg_status():
    """Return current Cyton board connection status."""
    return jsonify({
        "board_status": _board.status,
        "port":         _board.port,
        "error":        _board.error,
        "model_loaded": _decoder is not None,
    })


@eeg_predict_bp.route("/api/eeg-connect", methods=["POST"])
def eeg_connect():
    """Attempt to (re)connect to the Cyton board."""
    ok = _board.connect()
    return jsonify({
        "success":      ok,
        "board_status": _board.status,
        "port":         _board.port,
        "error":        _board.error,
    })
