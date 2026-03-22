"""
test_camera.py
--------------
Standalone camera test — no OpenBCI required.
Run with:  python test_camera.py
"""

import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from camera_tracker import CameraTracker

_VIDEO_W = 640
_VIDEO_H = 480
_POLL_MS = 33  # ~30 fps


class CameraTestApp:
    def __init__(self) -> None:
        self._tracker = CameraTracker(camera_index=0, eye_selection="both")

        self._root = tk.Tk()
        self._root.title("Camera Tracker — Test")
        self._root.resizable(False, False)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        # --- Video label ---
        self._video_label = tk.Label(self._root, bg="black",
                                     width=_VIDEO_W, height=_VIDEO_H)
        self._video_label.grid(row=0, column=0, columnspan=3, padx=8, pady=8)

        # --- Controls row ---
        ttk.Label(self._root, text="Eye:").grid(row=1, column=0, padx=(8,2))
        self._eye_var = tk.StringVar(value="both")
        eye_cb = ttk.Combobox(self._root, textvariable=self._eye_var,
                              values=["left", "right", "both"], width=6, state="readonly")
        eye_cb.grid(row=1, column=1, padx=(0, 8))
        eye_cb.bind("<<ComboboxSelected>>",
                    lambda _: self._tracker.set_eye_selection(self._eye_var.get()))

        self._calib_btn = ttk.Button(self._root, text="Calibrate (3 s)",
                                     command=self._calibrate)
        self._calib_btn.grid(row=1, column=2, padx=(0, 8))

        # --- Status bar ---
        self._status_var = tk.StringVar(value="Starting camera...")
        ttk.Label(self._root, textvariable=self._status_var,
                  relief="sunken", anchor="w", padding=(4, 2)).grid(
            row=2, column=0, columnspan=3, sticky="ew", padx=8, pady=(0, 8))

        # --- Start camera ---
        if not self._tracker.start():
            self._status_var.set("ERROR: Could not open camera.")
            return

        self._status_var.set("Camera running. Press 'Calibrate' to auto-set thresholds.")
        self._root.after(_POLL_MS, self._poll)

    def _poll(self) -> None:
        self._tracker.process_frame()
        data = self._tracker.get_current_data()

        frame = data["frame"]
        if frame is not None:
            frame_resized = cv2.resize(frame, (_VIDEO_W, _VIDEO_H))
            rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            self._video_label.configure(image=img)
            self._video_label._img = img  # keep reference

        if not self._tracker.is_calibrating():
            t = self._tracker.get_thresholds()
            l_d = data["left_distance"]
            r_d = data["right_distance"]
            l_str = f"{l_d:.1f}" if l_d is not None else "—"
            r_str = f"{r_d:.1f}" if r_d is not None else "—"
            self._status_var.set(
                f"L dist={l_str}px [{data['action_left']}]  |  "
                f"R dist={r_str}px [{data['action_right']}]  |  "
                f"Thresholds L({t['left']['lower']:.1f}–{t['left']['upper']:.1f}) "
                f"R({t['right']['lower']:.1f}–{t['right']['upper']:.1f})"
            )
        else:
            self._status_var.set("CALIBRATING — look straight ahead...")

        self._root.after(_POLL_MS, self._poll)

    def _calibrate(self) -> None:
        self._calib_btn.configure(state="disabled")
        self._tracker.start_calibration(duration_seconds=3.0)

        def _wait():
            if self._tracker.is_calibrating():
                self._root.after(200, _wait)
            else:
                self._calib_btn.configure(state="normal")

        self._root.after(200, _wait)

    def _on_close(self) -> None:
        self._tracker.stop()
        self._root.destroy()

    def run(self) -> None:
        self._root.mainloop()


if __name__ == "__main__":
    CameraTestApp().run()
