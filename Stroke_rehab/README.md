# Stroke Rehab BCI — Neural Intent Decoder

> **Brain-Computer Interface for stroke recovery patients.**
> Uses a 4-channel EEG headset and facial EMG to detect voluntary motor intent in real time,
> driving a visual feedback loop that reinforces neuropathway training when physical muscle
> activation is impaired or absent.

---

## Table of Contents

- [How It Works](#how-it-works)
- [System Architecture](#system-architecture)
- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
  - [1. Install Dependencies](#1-install-dependencies)
  - [2. Record Training Data](#2-record-training-data)
  - [3. Train the Model](#3-train-the-model)
  - [4. Run Real-Time Inference](#4-run-real-time-inference)
- [EEG Channel Guide](#eeg-channel-guide)
- [Model Deep Dive](#model-deep-dive)
  - [Feature Engineering](#feature-engineering)
  - [Architecture](#architecture)
  - [Training Strategy](#training-strategy)
- [Dataset Reference](#dataset-reference)
- [Training Script Reference](#training-script-reference)
- [Understanding the Outputs](#understanding-the-outputs)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)

---

## How It Works

The system operates in three stages that form a closed loop:

```
  PATIENT INTENT                    BRAIN SIGNAL                   VISUAL FEEDBACK
  ─────────────                     ────────────                   ───────────────

  "I want to move              EEG motor-planning signal      Animation shows the limb
   my left hand"        ──►   detected in frontal/temporal    moving as if the patient
                              channels (AF7, TP9)        ──►  had full motor control
                                        │
                              EMG channel checks if any
                              actual muscle signal fired
                                        │
                              Comparison drives rehab
                              feedback intensity
```

**Why this matters for stroke patients:**
- After a stroke, the neural pathway between intent and muscle may be severed or degraded.
- The brain still *attempts* to fire the signal — EEG captures that attempt.
- Pairing the *attempted* signal with immediate visual feedback re-trains the neuropathway
  through neuroplasticity (the Hebbian "fire together, wire together" principle).
- When EMG also detects partial muscle activation, the system confirms the loop is closing.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA CAPTURE PIPELINE                        │
│                                                                     │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │  Muse / Cyton│    │  Webcam (30 fps) │    │  OpenBCI Cyton   │  │
│  │  EEG Headset │    │  MediaPipe Face  │    │  EMG Electrodes  │  │
│  │  256 Hz      │    │  Mesh Tracking   │    │  (facial / limb) │  │
│  └──────┬───────┘    └────────┬─────────┘    └────────┬─────────┘  │
│         │                     │                        │            │
│         └──────────────────┬──┘                        │            │
│                            │    Zero-order hold sync   │            │
│                     ┌──────▼──────────────────────────▼──────┐     │
│                     │         DataRecorder (250 Hz rows)      │     │
│                     └──────────────────┬────────────────────-─┘     │
│                                        │                            │
│                                        ▼                            │
│                              final_bci_master.csv                   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         MODEL TRAINING                              │
│                                                                     │
│   final_bci_master.csv                                              │
│          │                                                          │
│          ▼                                                          │
│   Feature Extraction  ──►  [N, 4 channels, 13 features]            │
│   (per 250ms window)        bandpower × 5 + statistics × 8         │
│          │                                                          │
│          ▼                                                          │
│   StandardScaler  ──►  Session-based Cross Validation              │
│          │                                                          │
│          ▼                                                          │
│   EEGTransformer                                                    │
│   4 channel tokens  ──►  Multi-Head Attention  ──►  Softmax(3)     │
│          │                                                          │
│          ▼                                                          │
│   model/eeg_transformer.pt  +  model/scaler.pkl                    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      REAL-TIME INFERENCE                            │
│                                                                     │
│  EEG Stream  ──►  250ms buffer  ──►  extract_features()            │
│                                           │                         │
│                                    EEGIntentDecoder                 │
│                                           │                         │
│                          ┌────────────────┼───────────────┐        │
│                          ▼                ▼               ▼        │
│                        rest          left_blink      right_blink   │
│                          │                │               │        │
│                          └────────────────┼───────────────┘        │
│                                           ▼                         │
│                                  Visual Feedback UI                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Hardware Requirements

| Component | Specification | Notes |
|---|---|---|
| **EEG Headset** | Muse (4-ch: TP9, AF7, AF8, TP10) | 256 Hz, Bluetooth |
| **or EEG Board** | OpenBCI Cyton (8-ch) + USB dongle | 250 Hz, channels 5–8 used for EEG |
| **EMG Electrodes** | OpenBCI Cyton channels 1–4 | Facial or limb placement |
| **Webcam** | Any USB/built-in (720p+) | Used for eye-tracking calibration |
| **CPU** | Any modern x64 processor | GPU optional — model runs in <5 ms on CPU |
| **RAM** | 4 GB minimum, 8 GB recommended | Training uses ~1.5 GB peak |
| **OS** | Windows 10/11 | COM port required for OpenBCI |

### Electrode Placement

```
  OpenBCI Cyton — 8-channel assignment:

  Channels 1–4  →  EMG (facial / target limb)
  Channel  5    →  EEG Frontal 1    (Fp1 or F7 position)
  Channel  6    →  EEG Frontal 2    (Fp2 or F8 position)
  Channel  7    →  EEG Behind Left Ear   (TP9 / P9 position)
  Channel  8    →  EEG Behind Right Ear  (TP10 / P10 position)

  Muse Headset — fixed positions:
  AF7  ·  AF8  ·  TP9  ·  TP10  (+ reference at Fpz)
```

---

## Software Requirements

### Python Version
```
Python 3.10 or newer  (3.13 confirmed working)
```

### Core Dependencies

| Package | Version | Purpose |
|---|---|---|
| `torch` | ≥ 2.0 | EEG Transformer model |
| `mne` | ≥ 1.0 | EEG channel validation, PSD analysis |
| `scikit-learn` | ≥ 1.3 | StandardScaler, metrics |
| `pandas` | ≥ 2.0 | Dataset loading |
| `numpy` | ≥ 1.24 | Numerical operations |
| `matplotlib` | ≥ 3.7 | Training curves, confusion matrix |
| `brainflow` | ≥ 5.10 | OpenBCI Cyton streaming |
| `mediapipe` | ≥ 0.10 | Face mesh / eye tracking |
| `opencv-python` | ≥ 4.8 | Webcam capture |
| `Pillow` | ≥ 10.0 | GUI image rendering |
| `pyserial` | ≥ 3.5 | COM port communication |

### Install all at once

```bash
# Data capture app + model training (everything)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install mne scikit-learn pandas numpy matplotlib
pip install brainflow mediapipe opencv-python Pillow pyserial

# Or use the provided requirements file (capture app only):
pip install -r requirements.txt
```

> **GPU acceleration (optional):** Replace the torch install line with the appropriate CUDA version
> from https://pytorch.org/get-started/locally/ — training will be 5–10x faster.

---

## Project Structure

```
Stroke_rehab/
│
├── backend/                        # Data capture application
│   ├── main.py                     # Entry point — wires all modules together
│   ├── camera_tracker.py           # MediaPipe face-mesh eye/eyebrow tracking
│   ├── openbci_stream.py           # BrainFlow Cyton EEG/EMG acquisition
│   ├── data_recorder.py            # Thread-safe multi-stream CSV recorder
│   └── gui.py                      # Tkinter control panel
│
├── frontend/                       # Visual feedback web interface
│   ├── index.html
│   ├── css/style.css
│   └── js/app.js
│
├── train_eeg_transformer.py        # Model training script  ← START HERE
├── BCI_MODEL_PLAN.md               # Full neuroscience rationale & architecture plan
├── requirements.txt                # Capture app Python dependencies
│
├── model/                          # Generated after training (git-ignored)
│   ├── eeg_transformer.pt          # Saved model weights
│   ├── scaler.pkl                  # Fitted StandardScaler
│   ├── model_config.pkl            # Architecture config for inference
│   ├── training_curve.png          # Loss + F1 over epochs
│   ├── cv_confusion_matrix.png     # Cross-validation confusion matrix
│   └── psd_check.png               # MNE channel PSD sanity check
│
└── README.md                       # This file

../resources/
└── final_bci_master.csv            # Training dataset (38,343 windows)
```

---

## Quick Start

### 1. Install Dependencies

```bash
cd Stroke_rehab

# Install PyTorch (CPU build — works on any machine)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining packages
pip install mne scikit-learn pandas numpy matplotlib
pip install -r requirements.txt
```

---

### 2. Record Training Data

> Skip this step if you are using the provided `final_bci_master.csv` dataset.

```bash
cd Stroke_rehab/backend
python main.py
```

The GUI will open. Follow these steps:

```
  ┌─────────────────────────────────────────────┐
  │  STEP 1  Select COM port → click Connect     │
  │  STEP 2  Choose eye tracking: Left/Right/Both│
  │  STEP 3  Look at camera → click Calibrate    │
  │          (hold still for 3 seconds)          │
  │  STEP 4  Click  [Start Recording]            │
  │          Perform the target action           │
  │  STEP 5  Click  [Stop & Export CSV]          │
  │          Choose a save location              │
  └─────────────────────────────────────────────┘
```

**Recording tips for quality training data:**
- Record at least 5 minutes of `rest` baseline
- Record 50+ clear, deliberate left blinks for `left_blink`
- Record 50+ clear, deliberate right blinks for `right_blink`
- Keep the headset firmly in place — movement artifacts are the #1 source of noise
- Sit 40–60 cm from the camera in consistent lighting

---

### 3. Train the Model

The training script is self-contained and handles everything automatically.

**Option A — Cross-validation only** (evaluate model quality, no deployment artifact):
```bash
cd Stroke_rehab
python train_eeg_transformer.py
```

**Option B — Cross-validation + save deployment model** (recommended):
```bash
python train_eeg_transformer.py --final
```

**Option C — Skip CV, train deployment model only** (fastest, ~5 min on CPU):
```bash
python train_eeg_transformer.py --skip-cv
```

**Option D — Skip MNE channel check** (if MNE is not installed):
```bash
python train_eeg_transformer.py --final --skip-mne
```

#### What to expect during training

```
============================================================
  EEG Transformer -- BCI Stroke Rehab
============================================================
  Channels  : ['TP9', 'AF7', 'AF8', 'TP10']
  Features  : 13 per channel
  Classes   : ['rest', 'left_blink', 'right_blink']

Loaded 38,343 windows  |  Classes: rest: 35703, right_blink: 1330, ...

[MNE] Channel amplitudes (uV):
  TP9  : mean= -22.90  std= 14.69  min= -122.95  max=  74.31
  AF7  : mean=-177.15  std=247.31  ...
  ...

------------------------------------------------------------
  FOLD: held-out session = LeftEye
  Train: 25,564  |  Val: 12,782  |  Test: 12,782
  Class weights: {'rest': '1.00', 'left_blink': '28.11', 'right_blink': '25.35'}
  Epoch   1  tr_loss=1.1103  va_loss=1.0903  va_f1=0.2341  *best*
  Epoch  10  tr_loss=0.8451  va_loss=0.8112  va_f1=0.6823  *best*
  ...
  Early stopping at epoch 47

  [Test] macro-F1=0.72
              precision    recall  f1-score
        rest       0.97      0.98      0.98
  left_blink       0.71      0.68      0.69
 right_blink       0.73      0.71      0.72
```

Training time estimates (CPU, 38k windows):
| Mode | Time |
|---|---|
| Full CV (3 folds × 100 epochs max) | 15–30 min |
| Final model only (`--skip-cv`) | 5–10 min |

---

### 4. Run Real-Time Inference

After training, load the decoder in your streaming script:

```python
from train_eeg_transformer import EEGIntentDecoder

# Load the trained model (runs on CPU, < 5ms per prediction)
decoder = EEGIntentDecoder.load()

# Inside your EEG streaming loop:
while streaming:
    # features_dict keys must match the 4 channel names
    # each value = list of 13 features in this order:
    #   bp_theta, bp_alpha, bp_beta, bp_gamma, bp_emg,
    #   mean, std, rms, mav, peak2peak, kurtosis, skewness, zcr
    features_dict = {
        "TP9":  extract_features(window_ch0),
        "AF7":  extract_features(window_ch1),
        "AF8":  extract_features(window_ch2),
        "TP10": extract_features(window_ch3),
    }

    label, confidence, probs = decoder.predict(features_dict)
    # label      → "rest" | "left_blink" | "right_blink"
    # confidence → float 0–1 (probability of predicted class)
    # probs      → [p_rest, p_left_blink, p_right_blink]

    if label != "rest" and confidence > 0.70:
        trigger_visual_feedback(label)
```

---

## EEG Channel Guide

### Muse Headset — 10-20 Positions

```
                    FRONT
                      │
              AF7 ────┼──── AF8
            (left)    │    (right)
         Left         │         Right
         Frontal    [Fpz]      Frontal

              TP9 ────┼──── TP10
            (left)    │    (right)
         Left         │         Right
         Temporal   [Oz]      Temporal

                    BACK
```

### What Each Channel Captures

| Channel | Brain Region | What It Detects in This Task |
|---|---|---|
| **AF7** | Left prefrontal cortex | Left hemisphere motor planning; Bereitschaftspotential |
| **AF8** | Right prefrontal cortex | Right hemisphere motor planning |
| **TP9** | Left temporal / mastoid | Left eye EOG artifact; left visual cortex signal |
| **TP10** | Right temporal / mastoid | Right eye EOG artifact; right visual cortex signal |

### For Future Motor Imagery Upgrade (OpenBCI)

| Channel to Add | Location | Why |
|---|---|---|
| **C3** | Left motor cortex | Alpha/beta ERD during right-limb motor imagery |
| **C4** | Right motor cortex | Alpha/beta ERD during left-limb motor imagery |
| **CP5, CP6** | Sensorimotor | Somatosensory feedback loop |

---

## Model Deep Dive

### Feature Engineering

Each 250 ms window is converted into a **[4 channels × 13 features]** matrix:

```
  Per-channel features (13 total):

  ┌─ Bandpower (5) ──────────────────────────────┐
  │  theta   4–8 Hz    motor preparation         │
  │  alpha   8–13 Hz   ERD during movement       │  ← most important
  │  beta   13–30 Hz   ERD pre / ERS post move   │  ← most important
  │  gamma  30–100 Hz  fine motor / binding      │
  │  emg   100–200 Hz  actual muscle firing      │
  └──────────────────────────────────────────────┘
  ┌─ Statistics (8) ─────────────────────────────┐
  │  mean        DC offset / drift               │
  │  std         signal variability              │
  │  rms         signal power                   │
  │  mav         mean absolute value            │
  │  peak2peak   amplitude range                │
  │  kurtosis    spike impulsiveness            │
  │  skewness    waveform asymmetry             │
  │  zcr         zero-crossing rate             │
  └──────────────────────────────────────────────┘

  NOTE: Delta band (0.5–4 Hz) is excluded.
  A 250 ms window at 256 Hz = 64 samples → frequency resolution = 4 Hz.
  Delta requires at least 500 ms for reliable estimation.
```

### Architecture

```
  Input  [Batch, 4 channels, 13 features]
         │
         ▼
  ┌─────────────────────────────────────────┐
  │  Linear Projection  (13 → d_model=64)  │   one projection per channel
  └─────────────────────────────────────────┘
         │
         + Learnable Channel Positional Embedding  [1, 4, 64]
         │
         ▼
  ┌─────────────────────────────────────────┐
  │  Transformer Encoder                   │
  │  4 layers · 8 attention heads          │
  │  Pre-LayerNorm · dropout=0.30          │
  │                                        │
  │  Self-attention across channels:       │
  │  AF7 ←→ AF8  (hemispheric coherence)  │
  │  TP9 ←→ TP10 (bilateral temporal)     │
  │  AF7 ←→ TP9  (ipsilateral)            │
  └─────────────────────────────────────────┘
         │
         ▼
  Global Average Pool (over channel dim)  →  [Batch, 64]
         │
         LayerNorm → Dropout(0.30)
         │
         ▼
  FC(64 → 32) → GELU → Dropout
         │
         ▼
  FC(32 → 3) → Softmax
         │
         ▼
  [ p_rest · p_left_blink · p_right_blink ]
```

**Why a Transformer over CNN/LSTM?**
- Attention over 4 channel tokens directly captures *inter-channel relationships*
  (e.g., when AF7 activates, does TP9 co-activate?)
- No hardcoded spatial filter — the model learns which channel pairs matter per class
- Inference latency: **< 5 ms on CPU** — suitable for real-time use at 256 Hz
- Total parameters: ~85,000 — lightweight, no overfitting on 38k samples

### Training Strategy

| Aspect | Approach | Reason |
|---|---|---|
| **Class imbalance** | Inverse-frequency weights (~27× for blink classes) | 97% rest / 3% blink |
| **Evaluation metric** | Macro-F1 (not accuracy) | Accuracy is misleading on imbalanced data |
| **Validation** | Session-based leave-one-out CV | Sliding windows overlap → naive split leaks data |
| **Optimizer** | AdamW (lr=3e-4, wd=1e-4) | Standard for transformers |
| **Scheduler** | CosineAnnealingLR | Smooth decay, avoids sharp val-loss spikes |
| **Early stopping** | Patience=12 on val macro-F1 | Stops at best generalisation |
| **Gradient clipping** | max_norm=1.0 | Stabilises transformer training |

---

## Dataset Reference

The included dataset `resources/final_bci_master.csv` was recorded using the Muse headset
during 3 sessions of deliberate eye blink tasks.

| Property | Value |
|---|---|
| Total windows | 38,343 |
| Window length | ~250 ms (sliding, 44 ms step) |
| Sampling rate | 256 Hz (Muse) |
| Sessions | LeftEye · RightEye · MixedBlink |

**Class distribution:**

```
  rest          ████████████████████████████████████████  35,703  (93.1%)
  left_blink    █                                          1,310   (3.4%)
  right_blink   █                                          1,330   (3.5%)
```

**Column reference:**

| Column | Description |
|---|---|
| `session` | Recording session name |
| `window_start_s`, `window_end_s` | Window time boundaries (seconds) |
| `label` | Class name: `rest` / `left_blink` / `right_blink` |
| `label_int` | Integer label: 0 / 1 / 2 |
| `eye_distance_left/right` | Eyebrow-eyelid pixel distance (MediaPipe) |
| `action_left/right` | `OPEN` or `BLINK` |
| `EEG_Frontal_1/2` | Raw EEG sample at window midpoint (uV) — Cyton ch 5/6 |
| `EEG_Behind_Left/Right_Ear` | Raw EEG sample (uV) — Cyton ch 7/8 |
| `{CH}_mean` … `{CH}_zcr` | Statistical features per Muse channel |
| `{CH}_bp_delta` … `{CH}_bp_emg` | Bandpower per Muse channel (uV²/Hz) |

---

## Training Script Reference

```
train_eeg_transformer.py

Arguments:
  --final      Also train and save a final deployment model after CV
  --skip-cv    Skip cross-validation, go straight to final model training
  --skip-mne   Skip MNE channel sanity check

Output files (all in Stroke_rehab/model/):
  eeg_transformer.pt      PyTorch model state dict
  scaler.pkl              Fitted StandardScaler (must use same scaler at inference)
  model_config.pkl        Channel names, feature names, class names, hyperparams
  training_curve.png      Loss and val macro-F1 over epochs
  cv_confusion_matrix.png Confusion matrix across all 3 CV folds
  psd_check.png           MNE channel power spectral density plot

Key constants (edit at top of script to tune):
  D_MODEL      = 64      # transformer hidden dimension
  N_HEADS      = 8       # attention heads
  N_LAYERS     = 4       # transformer encoder layers
  DROPOUT      = 0.30    # regularisation
  LR           = 3e-4    # learning rate
  BATCH_SIZE   = 256
  MAX_EPOCHS   = 100
  PATIENCE     = 12      # early stopping patience
```

---

## Understanding the Outputs

### Training Curve (`model/training_curve.png`)
- **Loss** should decrease smoothly on both train and val
- A large train/val gap = overfitting → increase `DROPOUT` or reduce `N_LAYERS`
- Val loss rising while train loss falls = overfitting → reduce `MAX_EPOCHS` or increase `DROPOUT`

### Confusion Matrix (`model/cv_confusion_matrix.png`)
- `rest` class will have the most samples and highest accuracy by default
- Focus on **recall for blink classes** — missing a blink is worse than a false positive
- If `left_blink` recall is low, check that TP9/AF7 features are not saturated in your data

### PSD Check (`model/psd_check.png`)
Generated by MNE. Useful checks:
- **TP9/TP10** should show a clear **alpha peak** (8–13 Hz) during rest
- **AF7/AF8** may show more EMG contamination (flat spectrum at >30 Hz)
- A completely flat or noisy spectrum suggests electrode contact issues

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: torch` | PyTorch not installed | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| `ModuleNotFoundError: mne` | MNE not installed | `pip install mne` or run with `--skip-mne` |
| `FileNotFoundError: final_bci_master.csv` | Wrong working directory | Run from inside `Stroke_rehab/` folder |
| Val macro-F1 stuck at ~0.33 | Class weights not working | Check `y_tr` is integer not float; verify `label_int` column exists |
| Training very slow | CPU only, large batch | Reduce `BATCH_SIZE` to 64, or install CUDA PyTorch |
| Blink class recall = 0.00 | Model predicts all rest | Normal in first 5 epochs — wait for class weights to take effect |
| `UnicodeEncodeError` in terminal | Windows CP1252 terminal | Set `PYTHONUTF8=1` env var or use Windows Terminal |
| Camera not found | Wrong index | Edit `CameraTracker(camera_index=N)` in `backend/main.py` |
| COM port not listed | Dongle not connected / driver missing | Install FTDI or CP210x driver; check Device Manager |
| BrainFlow connection error | Wrong COM port or board off | Verify in Device Manager; power-cycle the Cyton board |
| CSV has all-NaN camera columns | Camera failed during recording | Check webcam, relaunch application |
| MNE PSD plot skipped | MNE compute_psd API version | Update MNE: `pip install --upgrade mne` |

---

## Roadmap

```
  Phase 1  (current)  ─────────────────────────────────────────────
  Eye blink intent detection — left / right voluntary blink
  Hardware: Muse headset (TP9, AF7, AF8, TP10)
  Status: COMPLETE — model trains and saves deployment artifacts

  Phase 2  ─────────────────────────────────────────────────────────
  Motor imagery classification — hand open/close, arm raise/lower
  Hardware: OpenBCI Cyton with C3, C4, CP5, CP6 electrodes
  Key signal: Alpha/beta ERD at motor cortex channels
  What to add: New label set + retrain transformer on motor imagery data

  Phase 3  ─────────────────────────────────────────────────────────
  EEG + EMG fusion — compare intended vs. actual muscle activation
  Hardware: Phase 2 + surface EMG on target limb
  What to add: Dual-stream model (EEG branch + EMG branch + fusion head)
  Clinical value: Quantify neuropathway recovery over sessions

  Phase 4  ─────────────────────────────────────────────────────────
  Personalised per-patient fine-tuning
  Method: Transfer learning from Phase 2/3 base model
  What to add: 5-minute calibration session per patient → LoRA fine-tune
  Clinical value: Accounts for individual electrode placement variation
```
