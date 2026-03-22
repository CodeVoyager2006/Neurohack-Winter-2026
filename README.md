# NeuroRehab — EEG-Driven Facial Projection for Stroke Recovery

A real-time Brain-Computer Interface that detects voluntary eye-movement intent from EEG signals and projects it onto a patient's stroke-affected facial side using a live webcam overlay.

Built at the **SURGE Neurotech Hackathon 2026** (Real-Time BCI stream).

---

## What It Does

Stroke patients often lose motor control on one side of the face. The healthy side still works, but the affected side cannot mirror it. This system:

1. **Reads live EEG** from an OpenBCI Cyton board (4 frontal/temporal channels)
2. **Classifies the patient's intent** — neutral, looking up, or looking down — using a trained EEG Transformer model running at ~3 Hz
3. **Projects the movement** onto the patient's stroke-affected side via a webcam overlay, creating a real-time visual feedback loop that reinforces neuropathway recovery through neuroplasticity

The affected side overlay is **position-locked** — it ignores all uncontrolled muscle activity from the healthy side and only moves when the EEG model fires a confident prediction.

```
  Patient thinks "look up"
         │
         ▼
  EEG signal detected (AF7/AF8/TP9/TP10)
         │
         ▼
  Transformer model → "looking_up" (87% confidence)
         │
         ▼
  Webcam overlay raises the eyebrow/eye landmarks
  on the affected side in real time
```

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        HARDWARE LAYER                            │
│                                                                  │
│   OpenBCI Cyton (8-ch)          USB Webcam                       │
│   ch5→AF7  ch6→AF8              MediaPipe FaceLandmarker         │
│   ch7→TP9  ch8→TP10             478-point face mesh              │
└────────────────┬─────────────────────────┬───────────────────────┘
                 │                         │
                 ▼                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                       FLASK SERVER  :5050                        │
│                                                                  │
│   /api/eeg-predict          /api/process-frame                   │
│   ─────────────────         ───────────────────                  │
│   BrainFlow stream          MediaPipe landmark                   │
│   64-sample window          extraction per frame                 │
│   13 features/channel       (eyebrow, eye, iris,                 │
│   EEG Transformer           mouth corners)                       │
│   3-window smoothing                                             │
└────────────────┬─────────────────────────┬───────────────────────┘
                 │                         │
                 ▼                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                     BROWSER UI  (app.js)                         │
│                                                                  │
│   EEG poll every 300ms → setExpression()                         │
│   Camera poll every 80ms → calibration + head tracking           │
│   Canvas overlay: position-locked to calibration snapshot        │
│   Expression offsets applied on top (raise/lower landmarks)      │
└──────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
Neurohack-Winter-2026/
│
├── Stroke_rehab/                   # EEG model — training & inference
│   ├── train_eeg_transformer.py    # Full training pipeline (run this first)
│   ├── preprocess_jaiden.py        # Raw CSV → windowed feature dataset
│   ├── model/                      # Saved model artifacts (committed)
│   │   ├── eeg_transformer.pt      # Trained PyTorch weights
│   │   ├── scaler.pkl              # Fitted StandardScaler
│   │   └── model_config.pkl        # Architecture + class config
│   ├── backend/                    # Data capture app (Tkinter GUI)
│   │   ├── main.py                 # Entry point
│   │   ├── openbci_stream.py       # BrainFlow Cyton streaming
│   │   ├── camera_tracker.py       # MediaPipe eye tracking
│   │   └── data_recorder.py        # Multi-stream CSV recorder
│   └── README.md                   # Detailed model documentation
│
├── real-time-bci-stream/
│   └── facial_mapping/             # Live demo application
│       ├── server.py               # Flask entry point  ← START HERE
│       ├── routes/
│       │   ├── eeg_predict.py      # Live Cyton inference endpoint
│       │   └── landmarks.py        # MediaPipe frame endpoint
│       ├── static/
│       │   ├── js/app.js           # Main UI controller
│       │   ├── js/renderer.js      # Canvas drawing
│       │   ├── js/expressions.js   # Expression offset maths
│       │   └── css/style.css
│       ├── templates/index.html
│       └── requirements.txt        # All Python dependencies
│
├── resources/
│   ├── jaiden_master.csv           # Primary training dataset (windowed)
│   ├── accurate_jaiden.csv         # Raw EEG capture from target subject
│   └── final_bci_master.csv        # Earlier multi-session dataset
│
└── getting-setup/                  # Environment setup guides
```

---

## Hardware Requirements

| Component | Spec | Notes |
|---|---|---|
| OpenBCI Cyton board | 8-channel, 250 Hz | Channels 5–8 used for EEG |
| USB RF dongle | FTDI-based (VID 0403:6015) | Ships with Cyton |
| EEG electrodes | Channels 5–8: AF7, AF8, TP9, TP10 | 10-20 system positions |
| Webcam | Any USB/built-in (720p+) | For face overlay |
| OS | Windows 10/11 | COM port required for Cyton |
| Python | 3.10 – 3.13 | 3.13 confirmed working |

### Electrode Placement

```
  Cyton channel → electrode position:
  ch5 → AF7   (left frontal, above left eyebrow)
  ch6 → AF8   (right frontal, above right eyebrow)
  ch7 → TP9   (behind left ear)
  ch8 → TP10  (behind right ear)
```

---

## Quick Start

### 1. Clone and install dependencies

```bash
git clone https://github.com/CodeVoyager2006/Neurohack-Winter-2026.git
cd Neurohack-Winter-2026

cd real-time-bci-stream/facial_mapping
pip install -r requirements.txt
```

### 2. Run the server (as Administrator on Windows)

The Cyton USB dongle requires elevated permissions on Windows.

**Right-click your terminal → "Run as administrator"**, then:

```bash
cd real-time-bci-stream/facial_mapping
python server.py
```

You should see:
```
[EEG] Model loaded  |  classes: ['neutral', 'looking_up', 'looking_down']
[EEG] Cyton connected on COM7  |  EXG rows: [1, 2, 3, 4, 5, 6, 7, 8]
 * Running on http://127.0.0.1:5050
```

### 3. Open the application

Navigate to `http://127.0.0.1:5050` in your browser.

```
  Step 1 — Choose affected side (left or right)
  Step 2 — Hold still during calibration (~1 second)
  Step 3 — EEG auto mode enables automatically on first signal
  Step 4 — Patient performs eye movements → overlay responds
```

### 4. If the board isn't detected automatically

Use the **port dropdown + Connect button** in the EEG panel on the right side of the interface. COM ports with USB Serial (FTDI) are pre-starred for easy identification.

---

## Model Details

### Classes
| Label | Meaning | Frontend key |
|---|---|---|
| `neutral` | Resting gaze | `neutral` |
| `looking_up` | Upward voluntary gaze | `lookup` |
| `looking_down` | Downward voluntary gaze | `lookdown` |

### Feature extraction (per 300ms window, per channel)
- **5 bandpowers** (Welch): theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz), gamma (30–80 Hz), EMG (80–200 Hz)
- **8 statistics**: mean, std, RMS, MAV, peak-to-peak, kurtosis, skewness, zero-crossing rate

Total input: 4 channels × 13 features = **52 features**

### Architecture
Channel-wise Transformer: each EEG channel is projected to a 64-dim token, then 4 Transformer encoder layers (8 attention heads) model inter-channel relationships. Global average pool → FC → softmax(3).

- Parameters: ~85,000
- Inference latency: < 5 ms on CPU
- Training macro-F1: **0.75** (5-fold stratified CV on Jaiden dataset)

### Prediction smoothing
3-window majority vote deque + 0.40 confidence floor fallback to neutral.

### Retraining the model

If you capture new data with the recorder (`Stroke_rehab/backend/main.py`):

```bash
cd Stroke_rehab

# Preprocess raw CSV → windowed features
python preprocess_jaiden.py

# Retrain with cross-validation + save deployment model
python train_eeg_transformer.py --final
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Access is denied` on COM port | Not running as admin | Right-click terminal → Run as administrator |
| Board shows disconnected in UI | Board plugged in after server started | Use Connect button in EEG panel |
| Model not loaded on new machine | `.pt`/`.pkl` files were gitignored | `git pull` — model files are now committed |
| `UNABLE_TO_OPEN_PORT_ERROR` | Another app holds the port | Close OpenBCI GUI or other serial apps |
| Overlay doesn't move | Auto mode not enabled | Enabled automatically on first EEG signal; or click "Auto: ON" |
| `ModuleNotFoundError: brainflow` | Missing dependency | `pip install -r requirements.txt` |
| No face detected badge | Poor lighting / camera angle | Ensure face is centred and well-lit |
| COM port not listed | FTDI driver not installed | Install from ftdichip.com or use OpenBCI drivers |

---

## Team

Built at SURGE Neurotech Hackathon 2026 — Real-Time BCI stream.
