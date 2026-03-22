"""
main.py — Entry point for the Stroke Rehab BCI application.

This module wires together the four core subsystems:
  - CameraTracker: captures and processes webcam frames via MediaPipe
  - OpenBCIStream: reads raw EEG/EMG data from an OpenBCI board via BrainFlow
  - DataRecorder: persists session data to disk
  - App: drives the Tkinter GUI and coordinates all subsystems

Usage:
    python main.py
"""

from camera_tracker import CameraTracker
from openbci_stream import OpenBCIStream
from data_recorder import DataRecorder
from gui import App


def main() -> None:
    """Instantiate all subsystems and launch the application.

    Creates one instance each of CameraTracker, OpenBCIStream, and
    DataRecorder, then passes them as dependencies to App before
    calling app.run() to enter the main event loop.

    Returns:
        None
    """
    camera_tracker: CameraTracker = CameraTracker()
    openbci_stream: OpenBCIStream = OpenBCIStream()
    data_recorder: DataRecorder = DataRecorder()

    app: App = App(
        camera_tracker=camera_tracker,
        openbci_stream=openbci_stream,
        data_recorder=data_recorder,
    )
    app.run()


if __name__ == "__main__":
    main()
