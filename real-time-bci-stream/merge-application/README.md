# merge-application — Unified BCI Launcher

A single `main.py` entry point that sits at the root of the project and
lets you launch either sub-application from one minimal dark window.

---

## Repository layout (relevant parts)

```
real-time-bci-stream/
│
├── facial_mapping/              ← Live rehab feedback (Flask + browser)
│   ├── server.py                   entry point  →  http://localhost:5050
│   ├── app.py
│   ├── routes/landmarks.py
│   ├── static/
│   └── templates/
│
├── merge-application/
│   └── main.py                  ← THIS FILE — unified launcher
│
└── cyton_setup_instructions.md

Stroke_rehab/
├── backend/                     ← Data capture (Python / tkinter)
│   ├── main.py                     entry point
│   ├── gui.py
│   ├── camera_tracker.py
│   ├── openbci_stream.py
│   └── data_recorder.py
├── frontend/                    ← Browser version of data capture (optional)
│   ├── index.html
│   ├── style.css
│   └── app.js
└── requirements.txt
```

---

## How to run

```bash
cd real-time-bci-stream/merge-application
python main.py
```

The launcher window opens. Click either tile to start the corresponding app.

---

## Applications

### DATA CAPTURE  (`Stroke_rehab/backend/main.py`)
- Python desktop app (tkinter)
- Connects to OpenBCI Cyton via USB dongle
- Tracks eye landmarks with MediaPipe
- Records EEG + EMG + camera data → exports CSV
- Runs in its own window; launcher stays open

### FACIAL MAPPING  (`real-time-bci-stream/facial_mapping/server.py`)
- Flask web server on `http://localhost:5050`
- Opens your default browser automatically
- Live webcam feed with MediaPipe landmark overlay
- Blacks out one side of the face and mirrors the other
- Expression buttons (neutral / raise eyebrow / knit / look up / look down)

---

## Dependencies

Install each app's requirements separately:

```bash
# Data capture
pip install -r ../../Stroke_rehab/requirements.txt

# Facial mapping
pip install -r ../facial_mapping/requirements.txt
```

The launcher itself only requires the Python standard library + `tkinter`
(included with all standard Python distributions).

---

## Notes

- Both apps run as **separate subprocesses** — their event loops never conflict.
- The launcher stays alive while an app is running so you can stop it or
  switch to the other app without restarting.
- The **Stop running app** button sends `SIGTERM` to the subprocess and waits
  up to 4 seconds before force-killing it.
- If a subprocess exits on its own (e.g. user closes the app window), the
  launcher detects this within ~800 ms and updates the status bar.