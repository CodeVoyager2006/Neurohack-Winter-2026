# merge-application — Stroke Rehab BCI (Unified Launcher)

Merges the **data-capture** and **live-inference** pipelines into a single
entry point with a minimal dark-themed launcher.

---

## Folder structure

```
merge-application/
├── main.py              ← entry point — shows launcher, then routes to chosen app
├── launcher.py          ← two-button mode selector window
├── ui_shared.py         ← shared palette, fonts, and theme helpers
├── app_record.py        ← Record mode GUI (replaces the original gui.py)
├── app_inference.py     ← Inference mode GUI (live overlay, no CSV)
│
│   ── unchanged from original project ──
├── camera_tracker.py
├── openbci_stream.py
├── data_recorder.py
└── requirements.txt
```

---

## How to run

```bash
cd merge-application
pip install -r requirements.txt
python main.py
```

The launcher opens and asks you to pick a mode.

---

## Modes

### RECORD
Identical feature set to the original `gui.py`:
- Connect OpenBCI Cyton via COM port
- Select eye(s), calibrate EAR thresholds
- Manual threshold override
- Start / Stop recording → export CSV at 250 Hz (last-known-value sync)

### INFERENCE
Live stream window — no CSV, no recording controls:
- Same camera EAR tracking and calibration
- Large real-time action label (LOOKING_UP / NEUTRAL / LOOKING_DOWN)
- Mini bar chart showing all 8 EEG/EMG channel values in µV
- `# TODO: plug in trained model` marker in `app_inference.py` for future
  replacement of the rule-based labels with a classifier

---

## Architecture decisions

| Decision | Rationale |
|---|---|
| Single `main.py` → `Launcher` → chosen `App` | Only one Tk root alive at a time — avoids multi-root Tkinter issues |
| `ui_shared.py` palette + theme | Both windows share identical colours/fonts without duplication |
| `app_record.py` / `app_inference.py` separate | Each mode has a different control set; keeping them separate avoids if/else bloat in a single GUI class |
| Original subsystem modules untouched | Zero risk of regression in the core data pipeline |