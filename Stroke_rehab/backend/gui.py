"""
gui.py
======
Tkinter-based control panel for the Stroke Rehab BCI data capture application.

Layout
------
Left panel  : Live webcam feed with MediaPipe landmark overlay.
Right panel : Connection controls, eye selection, calibration, threshold
              overrides, recording controls, and a status bar.

Dependency wiring
-----------------
The ``App`` class receives fully-constructed ``CameraTracker``,
``OpenBCIStream``, and ``DataRecorder`` instances via its constructor
(dependency-injection pattern) so that each subsystem can be tested
independently without the GUI.
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import cv2
from PIL import Image, ImageTk

from camera_tracker import CameraTracker
from data_recorder import DataRecorder
from openbci_stream import OpenBCIStream

# ---------------------------------------------------------------------------
# Layout constants — tweak these to adjust the look
# ---------------------------------------------------------------------------
_VIDEO_WIDTH: int = 480       # pixels — webcam display width
_VIDEO_HEIGHT: int = 360      # pixels — webcam display height
_POLL_CAMERA_MS: int = 33     # ~30 fps camera refresh
_POLL_OPENBCI_MS: int = 4     # ~250 Hz OpenBCI poll
_POLL_STATUS_MS: int = 500    # status bar refresh


class App:
    """Main application window wiring camera, OpenBCI, and recorder together.

    Parameters
    ----------
    camera_tracker:
        Fully constructed :class:`~camera_tracker.CameraTracker` instance.
    openbci_stream:
        Fully constructed :class:`~openbci_stream.OpenBCIStream` instance.
    data_recorder:
        Fully constructed :class:`~data_recorder.DataRecorder` instance.
    """

    def __init__(
        self,
        camera_tracker: CameraTracker,
        openbci_stream: OpenBCIStream,
        data_recorder: DataRecorder,
    ) -> None:
        self._tracker = camera_tracker
        self._openbci = openbci_stream
        self._recorder = data_recorder

        # ------------------------------------------------------------------
        # Root window
        # ------------------------------------------------------------------
        self._root = tk.Tk()
        self._root.title("Stroke Rehab BCI — Data Capture")
        self._root.resizable(False, False)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        # ------------------------------------------------------------------
        # Build UI
        # ------------------------------------------------------------------
        self._build_ui()

        # ------------------------------------------------------------------
        # Start camera
        # ------------------------------------------------------------------
        self._camera_ok: bool = self._tracker.start()
        if not self._camera_ok:
            self._set_status("ERROR: Camera not found — check camera index.", error=True)

        # ------------------------------------------------------------------
        # Populate COM port list
        # ------------------------------------------------------------------
        self._refresh_ports()

        # ------------------------------------------------------------------
        # Schedule recurring update callbacks
        # ------------------------------------------------------------------
        self._root.after(_POLL_CAMERA_MS, self._poll_camera)
        self._root.after(_POLL_OPENBCI_MS, self._poll_openbci)
        self._root.after(_POLL_STATUS_MS, self._poll_status)

    # ---------------------------------------------------------------------- #
    # UI construction                                                          #
    # ---------------------------------------------------------------------- #

    def _build_ui(self) -> None:
        """Build and grid all widgets into the root window."""
        # Outer container
        outer = ttk.Frame(self._root, padding=8)
        outer.grid(row=0, column=0, sticky="nsew")

        # ---------- Left panel: video feed --------------------------------
        left = ttk.LabelFrame(outer, text="Camera Feed", padding=4)
        left.grid(row=0, column=0, padx=(0, 8), pady=(0, 4), sticky="nsew")

        self._video_label = tk.Label(
            left,
            width=_VIDEO_WIDTH,
            height=_VIDEO_HEIGHT,
            bg="black",
        )
        self._video_label.pack()

        # ---------- Right panel: controls ---------------------------------
        right = ttk.Frame(outer, padding=4)
        right.grid(row=0, column=1, sticky="nsew")

        row = 0  # running row counter for right panel

        # ---- OpenBCI connection ------------------------------------------
        conn_frame = ttk.LabelFrame(right, text="OpenBCI Connection", padding=6)
        conn_frame.grid(row=row, column=0, sticky="ew", pady=(0, 6))
        row += 1

        ttk.Label(conn_frame, text="COM Port:").grid(row=0, column=0, sticky="w")
        self._port_var = tk.StringVar()
        self._port_combo = ttk.Combobox(
            conn_frame, textvariable=self._port_var, width=12, state="readonly"
        )
        self._port_combo.grid(row=0, column=1, padx=(4, 0), sticky="w")

        ttk.Button(conn_frame, text="Refresh", command=self._refresh_ports).grid(
            row=0, column=2, padx=(4, 0)
        )
        self._connect_btn = ttk.Button(
            conn_frame, text="Connect", command=self._toggle_openbci
        )
        self._connect_btn.grid(row=0, column=3, padx=(4, 0))

        # ---- Eye selection -----------------------------------------------
        eye_frame = ttk.LabelFrame(right, text="Eye Selection", padding=6)
        eye_frame.grid(row=row, column=0, sticky="ew", pady=(0, 6))
        row += 1

        self._eye_var = tk.StringVar(value="both")
        for col, (text, val) in enumerate(
            [("Left", "left"), ("Right", "right"), ("Both", "both")]
        ):
            ttk.Radiobutton(
                eye_frame,
                text=text,
                variable=self._eye_var,
                value=val,
                command=self._on_eye_change,
            ).grid(row=0, column=col, padx=4)

        # ---- Calibration -------------------------------------------------
        calib_frame = ttk.LabelFrame(right, text="Auto-Calibration", padding=6)
        calib_frame.grid(row=row, column=0, sticky="ew", pady=(0, 6))
        row += 1

        self._calib_btn = ttk.Button(
            calib_frame,
            text="Calibrate (3 s) — look straight ahead",
            command=self._start_calibration,
        )
        self._calib_btn.grid(row=0, column=0, columnspan=2, sticky="ew")

        self._calib_result_var = tk.StringVar(value="Thresholds: not calibrated")
        ttk.Label(calib_frame, textvariable=self._calib_result_var, foreground="gray").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )

        # ---- Manual threshold override -----------------------------------
        thresh_frame = ttk.LabelFrame(right, text="Manual Threshold Override", padding=6)
        thresh_frame.grid(row=row, column=0, sticky="ew", pady=(0, 6))
        row += 1

        ttk.Label(thresh_frame, text="Eye:").grid(row=0, column=0, sticky="w")
        self._thresh_eye_var = tk.StringVar(value="left")
        ttk.Combobox(
            thresh_frame,
            textvariable=self._thresh_eye_var,
            values=["left", "right"],
            width=6,
            state="readonly",
        ).grid(row=0, column=1, padx=(4, 0), sticky="w")

        ttk.Label(thresh_frame, text="Upper:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self._upper_entry = ttk.Entry(thresh_frame, width=8)
        self._upper_entry.grid(row=1, column=1, padx=(4, 0), pady=(4, 0), sticky="w")

        ttk.Label(thresh_frame, text="Lower:").grid(row=2, column=0, sticky="w", pady=(2, 0))
        self._lower_entry = ttk.Entry(thresh_frame, width=8)
        self._lower_entry.grid(row=2, column=1, padx=(4, 0), pady=(2, 0), sticky="w")

        ttk.Button(thresh_frame, text="Apply", command=self._apply_thresholds).grid(
            row=3, column=0, columnspan=2, pady=(6, 0), sticky="ew"
        )

        # ---- Live readout ------------------------------------------------
        readout_frame = ttk.LabelFrame(right, text="Live Readout", padding=6)
        readout_frame.grid(row=row, column=0, sticky="ew", pady=(0, 6))
        row += 1

        self._dist_var = tk.StringVar(value="Distance — L: —  R: —")
        self._action_var = tk.StringVar(value="Action — L: NEUTRAL  R: NEUTRAL")
        ttk.Label(readout_frame, textvariable=self._dist_var).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(readout_frame, textvariable=self._action_var).grid(
            row=1, column=0, sticky="w"
        )

        # ---- Recording controls ------------------------------------------
        rec_frame = ttk.LabelFrame(right, text="Recording", padding=6)
        rec_frame.grid(row=row, column=0, sticky="ew", pady=(0, 6))
        row += 1

        self._start_rec_btn = ttk.Button(
            rec_frame, text="Start Recording", command=self._start_recording
        )
        self._start_rec_btn.grid(row=0, column=0, padx=(0, 4), sticky="ew")

        self._stop_rec_btn = ttk.Button(
            rec_frame,
            text="Stop & Export CSV",
            command=self._stop_recording,
            state="disabled",
        )
        self._stop_rec_btn.grid(row=0, column=1, sticky="ew")

        self._row_count_var = tk.StringVar(value="Rows: 0")
        ttk.Label(rec_frame, textvariable=self._row_count_var, foreground="gray").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )

        # ---- Status bar --------------------------------------------------
        self._status_var = tk.StringVar(value="Ready.")
        status_bar = ttk.Label(
            outer,
            textvariable=self._status_var,
            relief="sunken",
            anchor="w",
            padding=(4, 2),
        )
        status_bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        self._status_label = status_bar

    # ---------------------------------------------------------------------- #
    # Camera polling                                                           #
    # ---------------------------------------------------------------------- #

    def _poll_camera(self) -> None:
        """Read one camera frame, push data to recorder, update video label.

        Scheduled via :meth:`tkinter.Misc.after` at ~30 fps.
        """
        if self._camera_ok:
            self._tracker.process_frame()
            data = self._tracker.get_current_data()

            # Push camera data to recorder regardless of recording state
            # (DataRecorder ignores pushes when not recording)
            self._recorder.push_camera_data(
                left_distance=data["left_distance"],
                right_distance=data["right_distance"],
                action_left=data["action_left"],
                action_right=data["action_right"],
            )

            # Update video label
            frame = data["frame"]
            if frame is not None:
                # Resize to display dimensions
                frame_resized = cv2.resize(frame, (_VIDEO_WIDTH, _VIDEO_HEIGHT))
                # Convert BGR → RGB for PIL
                rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                img = ImageTk.PhotoImage(image=Image.fromarray(rgb))
                self._video_label.configure(image=img)
                # Keep a reference to prevent garbage collection
                self._video_label._img = img  # type: ignore[attr-defined]

            # Update live readout
            l_dist = data["left_distance"]
            r_dist = data["right_distance"]
            l_str = f"{l_dist:.1f}" if l_dist is not None else "—"
            r_str = f"{r_dist:.1f}" if r_dist is not None else "—"
            self._dist_var.set(f"Distance — L: {l_str}px  R: {r_str}px")
            self._action_var.set(
                f"Action — L: {data['action_left']}  R: {data['action_right']}"
            )

            # Refresh calibration thresholds display
            if not self._tracker.is_calibrating():
                thresholds = self._tracker.get_thresholds()
                self._calib_result_var.set(
                    f"L: upper={thresholds['left']['upper']:.1f}  "
                    f"lower={thresholds['left']['lower']:.1f}  |  "
                    f"R: upper={thresholds['right']['upper']:.1f}  "
                    f"lower={thresholds['right']['lower']:.1f}"
                )

        self._root.after(_POLL_CAMERA_MS, self._poll_camera)

    # ---------------------------------------------------------------------- #
    # OpenBCI polling                                                          #
    # ---------------------------------------------------------------------- #

    def _poll_openbci(self) -> None:
        """Pull the latest OpenBCI sample and push it to the recorder.

        Scheduled via :meth:`tkinter.Misc.after` at ~250 Hz.
        """
        if self._openbci.is_connected():
            sample = self._openbci.get_latest_sample()
            if sample is not None and self._recorder.is_recording():
                self._recorder.push_openbci_sample(sample)
                # Update row count display every poll
                self._row_count_var.set(f"Rows: {self._recorder.get_row_count()}")

        self._root.after(_POLL_OPENBCI_MS, self._poll_openbci)

    # ---------------------------------------------------------------------- #
    # Status bar polling                                                       #
    # ---------------------------------------------------------------------- #

    def _poll_status(self) -> None:
        """Refresh the status bar with connection and recording state.

        Scheduled via :meth:`tkinter.Misc.after` at 2 Hz.
        """
        cam_str = "Camera: OK" if self._camera_ok else "Camera: ERROR"
        bci_str = "OpenBCI: connected" if self._openbci.is_connected() else "OpenBCI: disconnected"
        rec_str = "Recording..." if self._recorder.is_recording() else "Idle"

        if self._tracker.is_calibrating():
            rec_str = "CALIBRATING..."

        self._set_status(f"{cam_str}  |  {bci_str}  |  {rec_str}")
        self._root.after(_POLL_STATUS_MS, self._poll_status)

    # ---------------------------------------------------------------------- #
    # Control handlers                                                         #
    # ---------------------------------------------------------------------- #

    def _refresh_ports(self) -> None:
        """Re-enumerate serial ports and populate the COM port dropdown."""
        ports = self._openbci.get_available_ports()
        self._port_combo["values"] = ports
        if ports:
            self._port_combo.current(0)
        else:
            self._port_var.set("")

    def _toggle_openbci(self) -> None:
        """Connect or disconnect from the OpenBCI board."""
        if self._openbci.is_connected():
            # Disconnect
            self._openbci.stop_stream()
            self._openbci.disconnect()
            self._connect_btn.configure(text="Connect")
            self._set_status("OpenBCI disconnected.")
        else:
            # Connect
            port = self._port_var.get()
            if not port:
                messagebox.showwarning("No Port", "Please select a COM port first.")
                return
            if not self._openbci.connect(port):
                self._set_status(
                    f"OpenBCI connection failed: {self._openbci.last_error}", error=True
                )
                return
            try:
                self._openbci.start_stream()
            except RuntimeError as exc:
                self._set_status(f"Stream start failed: {exc}", error=True)
                return
            self._connect_btn.configure(text="Disconnect")
            self._set_status(f"OpenBCI connected on {port}, streaming.")

    def _on_eye_change(self) -> None:
        """Apply the new eye selection to the camera tracker."""
        selection = self._eye_var.get()
        self._tracker.set_eye_selection(selection)

    def _start_calibration(self) -> None:
        """Trigger a 3-second auto-calibration run."""
        if self._tracker.is_calibrating():
            messagebox.showinfo("Calibrating", "Calibration is already running.")
            return
        self._calib_btn.configure(state="disabled")
        self._calib_result_var.set("Calibrating — look straight ahead...")

        def _on_done() -> None:
            """Re-enable button once calibration finishes (called from camera poll)."""
            # Polling the calibration flag via the camera loop avoids
            # cross-thread Tkinter calls.
            if self._tracker.is_calibrating():
                self._root.after(200, _on_done)
            else:
                self._calib_btn.configure(state="normal")

        self._tracker.start_calibration(duration_seconds=3.0)
        self._root.after(200, _on_done)

    def _apply_thresholds(self) -> None:
        """Read manual threshold fields and apply them to the camera tracker."""
        eye = self._thresh_eye_var.get()
        try:
            upper = float(self._upper_entry.get())
            lower = float(self._lower_entry.get())
        except ValueError:
            messagebox.showerror(
                "Invalid Input", "Upper and Lower thresholds must be numbers."
            )
            return
        try:
            self._tracker.set_manual_thresholds(eye=eye, upper=upper, lower=lower)
            self._set_status(
                f"Manual thresholds applied — {eye}: upper={upper:.1f} lower={lower:.1f}"
            )
        except ValueError as exc:
            messagebox.showerror("Invalid Thresholds", str(exc))

    def _start_recording(self) -> None:
        """Begin recording data from all active streams."""
        if self._recorder.is_recording():
            return
        self._recorder.start_recording()
        self._start_rec_btn.configure(state="disabled")
        self._stop_rec_btn.configure(state="normal")
        self._set_status("Recording started.")

    def _stop_recording(self) -> None:
        """Stop recording and prompt the user for a CSV save location."""
        if not self._recorder.is_recording():
            return
        self._recorder.stop_recording()
        self._start_rec_btn.configure(state="normal")
        self._stop_rec_btn.configure(state="disabled")

        row_count = self._recorder.get_row_count()
        if row_count == 0:
            messagebox.showinfo("No Data", "Recording stopped but no rows were captured.")
            self._set_status("Recording stopped — no data.")
            return

        # Prompt for save path
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save recording as…",
        )
        if not filepath:
            self._set_status(f"Recording stopped — {row_count} rows not saved (cancelled).")
            return

        success = self._recorder.export_csv(filepath)
        if success:
            self._set_status(f"Saved {row_count} rows to {filepath}")
            messagebox.showinfo("Saved", f"Recording saved:\n{filepath}\n({row_count} rows)")
        else:
            self._set_status("ERROR: Failed to save CSV — check console.", error=True)
            messagebox.showerror("Save Failed", "Could not write CSV. See console for details.")

    # ---------------------------------------------------------------------- #
    # Status helper                                                            #
    # ---------------------------------------------------------------------- #

    def _set_status(self, message: str, error: bool = False) -> None:
        """Update the status bar text.

        Parameters
        ----------
        message:
            Text to display.
        error:
            When ``True``, colour the status bar red to draw attention.
        """
        self._status_var.set(message)
        self._status_label.configure(foreground="red" if error else "black")

    # ---------------------------------------------------------------------- #
    # Application lifecycle                                                    #
    # ---------------------------------------------------------------------- #

    def _on_close(self) -> None:
        """Gracefully shut down all subsystems before destroying the window."""
        # Stop recording if active (don't export — just discard)
        if self._recorder.is_recording():
            self._recorder.stop_recording()

        # Stop OpenBCI stream
        if self._openbci.is_connected():
            self._openbci.stop_stream()
            self._openbci.disconnect()

        # Release camera
        self._tracker.stop()

        self._root.destroy()

    def run(self) -> None:
        """Enter the Tkinter main event loop.

        Blocks until the window is closed.
        """
        self._root.mainloop()
