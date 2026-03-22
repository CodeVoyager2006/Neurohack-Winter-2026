"""
main.py — Unified BCI Launcher
================================
Single entry point for the Stroke Rehab BCI project.
Presents a minimal dark Tkinter launcher window with two options:

  1. DATA CAPTURE  →  launches Stroke_rehab/backend/main.py
                      (Python/tkinter — EEG/EMG + camera recording → CSV)

  2. FACIAL MAPPING →  launches real-time-bci-stream/facial_mapping/server.py
                       (Flask server + opens browser — live rehab feedback)

Each app runs as a separate subprocess so their event loops never conflict.
The launcher stays open so the user can switch between apps or kill a running
one without restarting the launcher.

Usage
-----
    python main.py

Requirements
------------
Both sub-applications must have their own dependencies installed.
See Stroke_rehab/requirements.txt and real-time-bci-stream/facial_mapping/requirements.txt.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tkinter as tk
from typing import Optional


# ---------------------------------------------------------------------------
# Resolve paths relative to this file's location
# ---------------------------------------------------------------------------
_HERE        = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT   = os.path.abspath(os.path.join(_HERE, "..", ".."))

# Path to the data-capture app entry point
_STROKE_REHAB_MAIN = os.path.join(_REPO_ROOT, "Stroke_rehab", "backend", "main.py")

# Path to the facial-mapping Flask server
_FACIAL_MAPPING_SERVER = os.path.join(
    _REPO_ROOT, "real-time-bci-stream", "facial_mapping", "server.py"
)


# ---------------------------------------------------------------------------
# Palette — minimal dark clinical aesthetic
# ---------------------------------------------------------------------------
BG       = "#0d0d0f"
SURFACE  = "#16161a"
BORDER   = "#2a2a30"
FG       = "#e8e8ec"
FG_DIM   = "#6b6b7a"
ACCENT_A = "#81c784"   # green  — data capture
ACCENT_B = "#4fc3f7"   # blue   — facial mapping
DANGER   = "#ef5350"   # red    — kill process
FONT     = "Courier New"


# ---------------------------------------------------------------------------
# Launcher window
# ---------------------------------------------------------------------------

class Launcher:
    """
    Minimal two-button launcher.

    Each button spawns the corresponding app as a subprocess.
    A running process is tracked so it can be terminated before
    launching a second one, and its status is shown in the launcher.
    """

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._active_label: Optional[str] = None

        self._root = tk.Tk()
        self._root.title("BCI Stroke Rehab — Launcher")
        self._root.resizable(False, False)
        self._root.configure(bg=BG)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build()
        self._centre()

        # Poll subprocess status every 800 ms to update the status bar
        self._root.after(800, self._poll_proc)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build(self) -> None:
        pad_x = 44

        # ── Title ──────────────────────────────────────────────────────
        tk.Label(
            self._root, text="BCI  Stroke  Rehab",
            font=(FONT, 20, "bold"), fg=ACCENT_B, bg=BG,
        ).pack(pady=(36, 4), padx=pad_x)

        tk.Label(
            self._root, text="select an application to launch",
            font=(FONT, 10), fg=FG_DIM, bg=BG,
        ).pack(pady=(0, 28), padx=pad_x)

        tk.Frame(self._root, bg=BORDER, height=1).pack(fill="x", padx=pad_x)

        # ── App buttons ────────────────────────────────────────────────
        btn_area = tk.Frame(self._root, bg=BG)
        btn_area.pack(padx=pad_x, pady=28, fill="x")

        self._btn_capture = self._make_app_tile(
            btn_area,
            label      = "DATA CAPTURE",
            sublabel   = "EEG / EMG + camera  →  CSV export",
            detail     = "Stroke_rehab / backend / main.py",
            color      = ACCENT_A,
            on_click   = self._launch_capture,
        )
        self._btn_capture.pack(fill="x", pady=(0, 10))

        self._btn_mapping = self._make_app_tile(
            btn_area,
            label      = "FACIAL MAPPING",
            sublabel   = "Live rehab feedback  →  browser overlay",
            detail     = "facial_mapping / server.py  →  localhost:5050",
            color      = ACCENT_B,
            on_click   = self._launch_mapping,
        )
        self._btn_mapping.pack(fill="x")

        # ── Kill button ────────────────────────────────────────────────
        self._kill_btn = tk.Frame(
            self._root, bg=BG,
            highlightthickness=1, highlightbackground=BORDER,
            cursor="hand2",
        )
        self._kill_btn.pack(padx=pad_x, fill="x", pady=(0, 8))

        self._kill_inner = tk.Label(
            self._kill_btn, text="■  Stop running app",
            font=(FONT, 10), fg=FG_DIM, bg=SURFACE,
            padx=16, pady=10,
        )
        self._kill_inner.pack(fill="x")

        for w in (self._kill_btn, self._kill_inner):
            w.bind("<Button-1>", lambda _e: self._kill_proc())
            w.bind("<Enter>",    lambda _e: self._kill_btn.configure(highlightbackground=DANGER))
            w.bind("<Leave>",    lambda _e: self._kill_btn.configure(highlightbackground=BORDER))

        # ── Status bar ─────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="No app running.")
        tk.Label(
            self._root, textvariable=self._status_var,
            font=(FONT, 9), fg=FG_DIM, bg=SURFACE,
            anchor="w", padx=12, pady=6,
        ).pack(fill="x", pady=(4, 16), padx=pad_x)

        # ── Footer ─────────────────────────────────────────────────────
        tk.Label(
            self._root,
            text="OpenBCI Cyton · MediaPipe · BrainFlow · Flask",
            font=(FONT, 8), fg=FG_DIM, bg=BG,
        ).pack(pady=(0, 20))

    def _make_app_tile(
        self,
        parent: tk.Widget,
        label: str,
        sublabel: str,
        detail: str,
        color: str,
        on_click,
    ) -> tk.Frame:
        """Return a clickable card tile for one application."""
        frame = tk.Frame(
            parent, bg=SURFACE, cursor="hand2",
            highlightthickness=1, highlightbackground=BORDER,
        )

        tk.Label(frame, text=label, font=(FONT, 13, "bold"),
                 fg=color, bg=SURFACE).pack(anchor="w", padx=14, pady=(12, 2))
        tk.Label(frame, text=sublabel, font=(FONT, 9),
                 fg=FG, bg=SURFACE).pack(anchor="w", padx=14)
        tk.Label(frame, text=detail, font=(FONT, 8),
                 fg=FG_DIM, bg=SURFACE).pack(anchor="w", padx=14, pady=(2, 12))

        def _click(_e=None):
            on_click()

        def _enter(_e=None):
            frame.configure(highlightbackground=color)

        def _leave(_e=None):
            frame.configure(highlightbackground=BORDER)

        frame.bind("<Map>", lambda _e: _rebind(frame, _click, _enter, _leave))
        for w in (frame,):
            w.bind("<Button-1>", _click)
            w.bind("<Enter>", _enter)
            w.bind("<Leave>", _leave)

        return frame

    # ------------------------------------------------------------------
    # Launch helpers
    # ------------------------------------------------------------------

    def _launch(self, script_path: str, label: str) -> None:
        """Kill any running app then spawn *script_path* as a subprocess."""
        if not os.path.exists(script_path):
            self._set_status(
                f"ERROR: cannot find  {script_path}", error=True
            )
            return

        self._kill_proc(silent=True)

        self._proc = subprocess.Popen(
            [sys.executable, script_path],
            cwd=os.path.dirname(script_path),
        )
        self._active_label = label
        self._set_status(f"● {label} running  (PID {self._proc.pid})")

    def _launch_capture(self) -> None:
        self._launch(_STROKE_REHAB_MAIN, "DATA CAPTURE")

    def _launch_mapping(self) -> None:
        self._launch(_FACIAL_MAPPING_SERVER, "FACIAL MAPPING")

    def _kill_proc(self, silent: bool = False) -> None:
        """Terminate the running subprocess if one exists."""
        if self._proc is None:
            if not silent:
                self._set_status("No app is currently running.")
            return

        if self._proc.poll() is None:          # still alive
            self._proc.terminate()
            try:
                self._proc.wait(timeout=4)
            except subprocess.TimeoutExpired:
                self._proc.kill()

        label = self._active_label or "app"
        self._proc         = None
        self._active_label = None

        if not silent:
            self._set_status(f"{label} stopped.")

    # ------------------------------------------------------------------
    # Status polling
    # ------------------------------------------------------------------

    def _poll_proc(self) -> None:
        """Check if the subprocess has exited and update the status bar."""
        if self._proc is not None and self._proc.poll() is not None:
            label = self._active_label or "app"
            code  = self._proc.returncode
            self._proc         = None
            self._active_label = None
            self._set_status(
                f"{label} exited (code {code}).",
                error=(code != 0),
            )
        self._root.after(800, self._poll_proc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_status(self, msg: str, error: bool = False) -> None:
        self._status_var.set(msg)

    def _centre(self) -> None:
        self._root.update_idletasks()
        w  = self._root.winfo_width()
        h  = self._root.winfo_height()
        sw = self._root.winfo_screenwidth()
        sh = self._root.winfo_screenheight()
        self._root.geometry(f"+{(sw - w) // 2}+{(sh - h) // 2}")

    def _on_close(self) -> None:
        self._kill_proc(silent=True)
        self._root.destroy()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Enter the Tkinter main loop."""
        self._root.mainloop()


# ---------------------------------------------------------------------------
# Helper used inside _make_app_tile after child widgets exist
# ---------------------------------------------------------------------------

def _rebind(frame, click, enter, leave) -> None:
    for w in frame.winfo_children():
        w.bind("<Button-1>", click)
        w.bind("<Enter>",    enter)
        w.bind("<Leave>",    leave)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    Launcher().run()