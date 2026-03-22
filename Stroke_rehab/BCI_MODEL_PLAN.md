# BCI Model Plan — Stroke Rehabilitation Motor Intent Decoder

## Overview
Train an EEG Transformer classifier on pre-extracted windowed features from a 4-channel Muse
headset to decode motor/visual intent. At inference time, the model ingests a 250 ms EEG window
and predicts which action the patient is attempting, driving the visual feedback loop.

---

## 1. Dataset Summary

| Property | Value |
|---|---|
| File | `resources/final_bci_master.csv` |
| Rows | 38,346 (~250 ms sliding windows, 44 ms step) |
| EEG Channels | TP9, AF7, AF8, TP10 (Muse headset, 256 Hz) |
| Classes | `rest` (0) · `left_blink` (1) · `right_blink` (2) |
| Class balance | 35,706 · 1,310 · 1,330 — **severely imbalanced** |
| Sessions | LeftEye · RightEye · MixedBlink (~12,782 windows each) |

---

## 2. Channel Role & Neurophysiological Rationale

| Channel | Location | Role in this Task |
|---|---|---|
| **AF7** | Left prefrontal | Left hemisphere motor planning; left voluntary eye movement |
| **AF8** | Right prefrontal | Right hemisphere motor planning; right voluntary eye movement |
| **TP9** | Left temporal (behind left ear) | Proximal to left visual cortex; left EOG-EMG artifact = blink signal |
| **TP10** | Right temporal (behind right ear) | Proximal to right visual cortex; right EOG-EMG artifact = blink signal |

### Extension to Motor Imagery (Stroke Rehab)
When upgrading to a full motor-imagery headset (OpenBCI with C3/C4/CP5/CP6):
- **C3** (left motor cortex): alpha/beta ERD during right-limb motor imagery
- **C4** (right motor cortex): alpha/beta ERD during left-limb motor imagery
- Muse TP9/TP10 partially pick up motor planning signals via volume conduction but are not ideal.
  The AF7/AF8 prefrontal channels carry motor-preparation potentials (Bereitschaftspotential).

---

## 3. Feature Selection

### Frequency Bands Used

| Band | Range | Why It Matters |
|---|---|---|
| **Theta** | 4–8 Hz | Working memory, attention, motor preparation |
| **Alpha** | 8–13 Hz | **Event-Related Desynchronization (ERD)** — drops during voluntary movement / motor imagery |
| **Beta** | 13–30 Hz | **ERD pre-movement, ERS post-movement** — best marker for motor intent |
| **Gamma** | 30–100 Hz | Fine motor coordination, sensory binding |
| **EMG band** | 100–200 Hz | Actual muscle activation signal (facial EMG via blink) |
| ~~Delta~~ | ~~0.5–4 Hz~~ | **Excluded** — always 0 in dataset (window too short for reliable delta estimate) |

### Statistical Features Used (per channel)

| Feature | Description |
|---|---|
| `mean` | DC offset / baseline shift |
| `std` | Signal variability |
| `rms` | Signal power |
| `mav` | Mean absolute value |
| `peak2peak` | Peak-to-peak amplitude |
| `kurtosis` | Impulsiveness (spike artifacts) |
| `skewness` | Signal asymmetry |
| `zcr` | Zero-crossing rate (frequency proxy) |
| `iemg` | Integrated EMG energy |

**Total features per window: 4 channels × 13 features = 52 input features**
Plus 4 raw EEG scalar values = **56 features total fed to model**

### Feature Matrix Shape
```
X : [N_windows, N_channels=4, N_features=13]   → transformer input
y : [N_windows]                                 → class label (0/1/2)
```

---

## 4. Model Architecture — EEG Channel Transformer

```
Input [B, 4 channels, 13 features]
        ↓
Linear Projection → [B, 4, d_model=64]
        ↓
+ Learnable Channel Positional Embedding
        ↓
Transformer Encoder (4 layers, 8 heads, dropout=0.3)
  → Self-attention across channels (learn which channel combinations matter)
        ↓
Global Average Pool over channel dimension → [B, d_model]
        ↓
LayerNorm → Dropout
        ↓
FC(64 → 32) → GELU → Dropout
        ↓
FC(32 → 3) → Softmax
        ↓
Predicted Class (rest / left_blink / right_blink)
```

**Why Transformer over CNN/LSTM?**
- Attention over channels directly models inter-hemispheric coherence (e.g., AF7↔AF8 synchrony)
- No fixed spatial filter assumption — learns which channel pair matters per class
- Lightweight enough for real-time CPU inference (<5 ms per window)

---

## 5. Class Imbalance Strategy

- **Inverse-frequency class weights** in CrossEntropyLoss
  - rest: weight = 1.0
  - left_blink: weight ≈ 27×
  - right_blink: weight ≈ 27×
- Evaluate with **macro-F1** and **per-class recall**, not accuracy

---

## 6. Validation Strategy

- **Session-based leave-one-out cross-validation** (3 folds: LeftEye / RightEye / MixedBlink)
- Train on 2 sessions, validate on 1 — avoids data leakage from overlapping sliding windows
- Final model trained on all 3 sessions for deployment

---

## 7. Training Pipeline

```
1. Load CSV → drop NaNs → encode action_left/action_right as binary
2. Build feature matrix [N, 4, 13]
3. Normalize per-feature with StandardScaler (fit on train, apply to val/test)
4. Compute class weights
5. Session-based CV:
     for held_out_session in [LeftEye, RightEye, MixedBlink]:
         train / val / test split
         instantiate EEGTransformer
         Adam optimizer + CosineAnnealingLR
         train with early stopping (patience=10 on val macro-F1)
         evaluate → log metrics
6. Save best model weights + scaler → model/eeg_transformer.pt + model/scaler.pkl
```

---

## 8. Real-Time Inference Integration

```python
# In openbci_stream.py / Stroke_rehab pipeline:
while streaming:
    window = buffer.get_latest_250ms()         # [4 channels × 64 samples @ 256Hz]
    features = extract_features(window)         # same pipeline as training → [4, 13]
    features_scaled = scaler.transform(...)
    logits = model(tensor(features_scaled))
    action = argmax(logits)                     # 0=rest, 1=left_blink, 2=right_blink
    visual_feedback.trigger(action)
```

---

## 9. Extension Roadmap for Full Stroke Motor Imagery

| Phase | What to Add |
|---|---|
| Phase 1 (current) | Eye blink intent detection (left/right voluntary blink as control signal) |
| Phase 2 | Motor imagery labels (hand open/close, arm raise) — requires OpenBCI with C3/C4 |
| Phase 3 | Combined EEG + surface EMG (sEMG) fusion — compare intended vs. actual muscle activation |
| Phase 4 | Personalized fine-tuning per patient (transfer learning from Phase 1/2 base model) |
