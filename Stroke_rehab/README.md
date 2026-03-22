# Stroke Rehab BCI — Data Capture Tool

A Python desktop application (Windows) that simultaneously captures three live input streams — webcam video, EEG, and EMG — synchronises them by timestamp, and exports the combined data to a single CSV file.

---

## Setup

### Requirements

- Python 3.10 or newer
- OpenBCI Cyton board + USB dongle
- Webcam (built-in or USB)

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to run

```bash
python main.py
```

The application window opens immediately. The webcam feed starts on launch.

---

## Usage walkthrough

1. **Connect OpenBCI**
   - Select the COM port from the dropdown (click **Refresh** if the port is not listed).
   - Click **Connect** — the board session starts and data begins streaming.

2. **Select eye(s) to track**
   - Choose **Left**, **Right**, or **Both** from the radio buttons.

3. **Calibrate**
   - Look straight ahead at the camera.
   - Click **Calibrate (3 s)** and hold still for 3 seconds.
   - The computed thresholds appear below the button.

4. **Override thresholds (optional)**
   - Enter custom Upper and Lower pixel-distance values in the manual fields.
   - Select the eye (Left / Right) and click **Apply**.

5. **Record**
   - Click **Start Recording** to begin logging all streams.
   - The row counter updates in real time.
   - Click **Stop & Export CSV** to stop recording and choose a save location.

---

## Channel assignment (OpenBCI Cyton, 8 channels)

| Board channel | Label                  | Placement              |
|---------------|------------------------|------------------------|
| 1             | EMG_1                  | Facial EMG electrode 1 |
| 2             | EMG_2                  | Facial EMG electrode 2 |
| 3             | EMG_3                  | Facial EMG electrode 3 |
| 4             | EMG_4                  | Facial EMG electrode 4 |
| 5             | EEG_Frontal_1          | Frontal lobe           |
| 6             | EEG_Frontal_2          | Frontal lobe           |
| 7             | EEG_Behind_Left_Ear    | Left mastoid / temporal|
| 8             | EEG_Behind_Right_Ear   | Right mastoid / temporal|

---

## CSV column descriptions

| Column                | Type   | Description                                                       |
|-----------------------|--------|-------------------------------------------------------------------|
| `timestamp`           | float  | Wall-clock time (seconds since Unix epoch, `time.time()`)         |
| `eye_distance_left`   | float  | Eyebrow-to-eyelid pixel distance, left eye (NaN if not selected)  |
| `eye_distance_right`  | float  | Eyebrow-to-eyelid pixel distance, right eye (NaN if not selected) |
| `action_left`         | str    | Gaze classification for left eye: `LOOKING_UP`, `LOOKING_DOWN`, or `NEUTRAL` |
| `action_right`        | str    | Gaze classification for right eye                                 |
| `EMG_1`–`EMG_4`       | float  | Raw facial EMG values in microvolts                               |
| `EEG_Frontal_1`–`EEG_Behind_Right_Ear` | float | Raw EEG values in microvolts                    |

---

## Synchronisation approach

The camera (~30 fps) and OpenBCI Cyton (250 Hz) operate at different rates. This tool uses a **last-known-value (zero-order hold)** strategy:

- **OpenBCI samples drive the row cadence** at its native 250 Hz.
- The camera thread updates a small in-memory "latest camera state" record at ~30 fps.
- Each OpenBCI sample row snapshots the most recent camera state and combines the two into one CSV row.
- Consecutive OpenBCI rows share the same camera values until the next camera frame arrives (up to ~33 ms of temporal error — acceptable for gesture classification where movements unfold over hundreds of milliseconds).

This approach avoids interpolation artefacts in classification labels, requires no post-processing alignment, and produces a clean 250 Hz CSV with no missing rows.

---

## Project structure

```
Stroke_rehab/
├── main.py              # Entry point — wires modules together
├── camera_tracker.py    # MediaPipe face-mesh eye/eyebrow tracking
├── openbci_stream.py    # BrainFlow Cyton EEG/EMG acquisition
├── data_recorder.py     # Thread-safe multi-stream data recorder
├── gui.py               # Tkinter control panel
├── requirements.txt     # Python dependencies
└── README.md
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| "Camera not found" error | Wrong camera index | Edit `CameraTracker(camera_index=N)` in `main.py` |
| COM port not listed | Dongle not plugged in / driver missing | Install FTDI or CP210x driver |
| "BrainFlow error while connecting" | Wrong COM port or board off | Verify port in Device Manager; power-cycle Cyton |
| CSV has all-NaN camera columns | Camera failed during recording | Check webcam connection and relaunch |
