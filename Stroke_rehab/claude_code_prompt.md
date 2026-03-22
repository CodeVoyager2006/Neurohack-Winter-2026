# Prompt for Claude Code

## Important: Use Agents

**This is a large, multi-component project. Use subagents to build each module independently so you do not exhaust the context window.** Break the work into the following agent tasks:

1. **Agent 1 — Project scaffold**: Create the folder structure, `requirements.txt`, and a `main.py` entry point that imports and wires together all modules.
2. **Agent 2 — Camera module** (`camera_tracker.py`): MediaPipe face-mesh eye/eyebrow tracking, distance calculation, action classification, and live video overlay.
3. **Agent 3 — OpenBCI module** (`openbci_stream.py`): Cyton board connection via USB dongle, raw EEG/EMG data acquisition, and channel labeling.
4. **Agent 4 — GUI module** (`gui.py`): Tkinter-based control panel with recording controls, eye selection, threshold override, and embedded video feed.
5. **Agent 5 — Data recorder module** (`data_recorder.py`): Thread-safe time-stamped data collection from all three streams, synchronization, and CSV export.
6. **Agent 6 — Integration and testing**: Wire all modules together in `main.py`, add error handling, and verify the full pipeline runs end-to-end.

After each agent completes, briefly verify the module works before moving to the next.

---

## Project Description

Build a **Python desktop application** (Windows) that simultaneously captures three live input streams — webcam video, EEG, and EMG — synchronizes them by timestamp, and exports the combined data to a single CSV file. The GUI should be simple and minimalistic using tkinter.

---

## Module Specifications

### 1. Camera Tracking (`camera_tracker.py`)

- **Library**: MediaPipe Face Mesh (NOT face detection — use the 468-landmark mesh).
- **What to track**: For each selected eye, map one landmark on the **eyebrow** and one on the **upper eyelid/eye** and compute the vertical pixel distance between them in each frame.
- **Eye selection**: Provide a dropdown or radio button in the GUI to select **Left Eye**, **Right Eye**, or **Both Eyes**. When "Both" is selected, compute and log distances for each eye independently.
- **Action classification based on distance**:
  - `LOOKING_UP` — distance is significantly above the neutral baseline
  - `LOOKING_DOWN` — distance is significantly below the neutral baseline
  - `NEUTRAL` — distance is within the neutral range
- **Auto-calibration**:
  - On startup (or on a "Calibrate" button press), run a 3-5 second calibration phase where the user looks straight ahead. Compute the mean and standard deviation of the distance during this window. Use mean ± 1.5×std (or similar) as the threshold boundaries.
  - Display the computed thresholds in the GUI after calibration.
- **Manual threshold override (backup)**:
  - Provide input fields in the GUI where the user can manually enter upper and lower thresholds. These override the auto-calibrated values when set.
- **Live video feed**:
  - Display the webcam feed in the GUI with the tracked landmarks overlaid (draw circles on the eyebrow and eye points, and a line showing the measured distance).
  - Display the current action label (`LOOKING_UP`, `LOOKING_DOWN`, `NEUTRAL`) on the video feed.

### 2. OpenBCI EEG/EMG Stream (`openbci_stream.py`)

- **Board**: OpenBCI **Cyton** (8 channels).
- **Connection**: USB dongle (serial port). Auto-detect the COM port if possible, otherwise provide a dropdown in the GUI to select it.
- **Channel assignment** (8 channels total):
  - Channels 1-4: **EMG** (label as `EMG_1`, `EMG_2`, `EMG_3`, `EMG_4`)
  - Channels 5-8: **EEG** (label as `EEG_Frontal_1`, `EEG_Frontal_2`, `EEG_Behind_Left_Ear`, `EEG_Behind_Right_Ear`)
  - Note: 2 of the 4 EEG channels are placed behind the ears. The other 2 are frontal. The labels above reflect placement.
- **Data acquisition**:
  - Stream raw microvolt values from all 8 channels.
  - Use the `brainflow` library (BrainFlow SDK) for OpenBCI communication — it is well-maintained and handles Cyton boards reliably.
- **Real-time signal processing (stub for future)**:
  - Create placeholder functions for: bandpass filtering, notch filter (50/60 Hz), FFT power spectrum, and band power extraction (delta, theta, alpha, beta, gamma).
  - These should be clearly marked with `# TODO: Enable for real-time processing in future development` comments.
  - Do NOT call these during recording for now — just log raw values.

### 3. GUI (`gui.py`)

- **Framework**: tkinter (keep it simple and minimalistic).
- **Layout**:
  - **Left panel**: Live webcam feed with landmark overlay (use a tkinter Canvas or Label with PIL/ImageTk).
  - **Right panel**: Controls stacked vertically:
    - COM port selector (dropdown, auto-populated if possible)
    - Eye selection (Left / Right / Both)
    - "Calibrate" button — runs the auto-calibration routine
    - Manual threshold input fields (upper and lower) with an "Apply" button
    - Display area showing current calibrated thresholds and current distance value
    - **"Start Recording"** button — begins logging all streams with timestamps
    - **"Stop Recording & Export"** button — stops logging and saves to CSV
    - Status bar at the bottom showing connection status and recording state
- **Behavior**:
  - The webcam feed and OpenBCI stream should start on launch (or on a "Connect" button).
  - Recording only begins when the user clicks "Start Recording".
  - On "Stop Recording & Export", prompt a file-save dialog for the CSV location.

### 4. Data Recorder (`data_recorder.py`)

- **Thread-safe** data collection: each stream (camera, EEG/EMG) pushes data into a shared, synchronized buffer.
- **Timestamp**: Use `time.time()` or `datetime` for a common high-resolution timestamp across all streams. All data rows should be aligned to the same time reference.
- **CSV output format** — one row per sample, columns:
  ```
  timestamp, eye_distance_left, eye_distance_right, action_left, action_right, EMG_1, EMG_2, EMG_3, EMG_4, EEG_Frontal_1, EEG_Frontal_2, EEG_Behind_Left_Ear, EEG_Behind_Right_Ear
  ```
  - If only one eye is selected, the unused eye columns should be `NaN` or empty.
  - Since camera and OpenBCI sample at different rates, use **nearest-timestamp matching** or interpolation to align rows. Document which approach you chose and why.
- **Synchronization strategy**: Clearly comment how you handle the different sampling rates (camera ~30fps vs OpenBCI Cyton at 250Hz). A practical approach: log OpenBCI data at its native rate and attach the most recent camera action/distance to each OpenBCI sample row.

---

## Technical Requirements

- **Python 3.10+**
- **Dependencies** (include in `requirements.txt`):
  - `mediapipe`
  - `opencv-python`
  - `brainflow`
  - `numpy`
  - `pandas`
  - `Pillow`
  - Any other required packages
- **Platform**: Windows 10/11
- **Error handling**: Gracefully handle camera not found, OpenBCI board not connected, and serial port access errors. Show user-friendly error messages in the GUI status bar.
- **Code quality**: Type hints, docstrings on every function, and clear inline comments explaining non-obvious logic.

---

## Deliverables

1. Complete project folder with all modules
2. `requirements.txt`
3. `README.md` with:
   - Setup instructions
   - How to run the program
   - Channel assignment reference
   - CSV column descriptions
   - Notes on the synchronization approach
