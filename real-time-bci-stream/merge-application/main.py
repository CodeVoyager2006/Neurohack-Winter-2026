"""
main.py — Entry point for the Stroke Rehab BCI merged application.

Presents a minimal launcher that lets the user choose between two modes:

    1. RECORD mode  — Capture + label session data (EEG/EMG + camera) and
                      export to CSV for offline training.

    2. INFERENCE mode — Stream live data with real-time eye-action labels
                        projected as a visual overlay.  No CSV export.

Usage:
    python main.py
"""

from camera_tracker import CameraTracker
from openbci_stream import OpenBCIStream
from data_recorder import DataRecorder
from launcher import Launcher


def main() -> None:
    """Show the launcher, then start the selected application mode.

    The Launcher window is destroyed before the chosen App window opens so
    only one Tk root is alive at a time (avoids Tkinter multi-root issues).
    """
    launcher = Launcher()
    mode = launcher.run()          # blocks until the user picks a mode

    if mode is None:
        return                     # user closed the launcher

    camera_tracker = CameraTracker()
    openbci_stream = OpenBCIStream()

    if mode == "record":
        from app_record import RecordApp
        data_recorder = DataRecorder()
        app = RecordApp(
            camera_tracker=camera_tracker,
            openbci_stream=openbci_stream,
            data_recorder=data_recorder,
        )
    else:  # mode == "inference"
        from app_inference import InferenceApp
        app = InferenceApp(
            camera_tracker=camera_tracker,
            openbci_stream=openbci_stream,
        )

    app.run()


if __name__ == "__main__":
    main()