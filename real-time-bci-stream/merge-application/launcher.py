"""
launcher.py
===========
Minimal launcher window.  Asks the user to choose between Record mode and
Inference mode before the full application starts.

The window is intentionally sparse — dark background, two large buttons,
nothing else.  It is destroyed (not hidden) once a choice is made so that
the chosen App window can own the Tk main-loop cleanly.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Optional


# Palette — shared with the two app windows for visual consistency.
BG       = "#0d0d0f"
SURFACE  = "#16161a"
BORDER   = "#2a2a30"
FG       = "#e8e8ec"
FG_DIM   = "#6b6b7a"
ACCENT   = "#4fc3f7"    # cool blue — neural / medical feel
ACCENT2  = "#81c784"    # green — "go / record"


class Launcher:
    """Simple two-button mode selector.

    Call :meth:`run` to show the window.  It blocks until the user clicks a
    mode button or closes the window, then returns the selected mode string
    (``"record"`` | ``"inference"``) or ``None`` if the window was dismissed.
    """

    def __init__(self) -> None:
        self._root = tk.Tk()
        self._root.title("BCI — Stroke Rehab")
        self._root.resizable(False, False)
        self._root.configure(bg=BG)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._choice: Optional[str] = None
        self._build()

        # Centre on screen
        self._root.update_idletasks()
        w = self._root.winfo_width()
        h = self._root.winfo_height()
        sw = self._root.winfo_screenwidth()
        sh = self._root.winfo_screenheight()
        self._root.geometry(f"+{(sw - w) // 2}+{(sh - h) // 2}")

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build(self) -> None:
        pad = dict(padx=40, pady=10)

        # ── Title ──────────────────────────────────────────────────────
        tk.Label(
            self._root,
            text="BCI Stroke Rehab",
            font=("Courier New", 22, "bold"),
            fg=ACCENT,
            bg=BG,
        ).pack(pady=(36, 4), **{k: v for k, v in pad.items() if k == "padx"})

        tk.Label(
            self._root,
            text="Select application mode",
            font=("Courier New", 11),
            fg=FG_DIM,
            bg=BG,
        ).pack(pady=(0, 32), **{k: v for k, v in pad.items() if k == "padx"})

        # ── Divider ────────────────────────────────────────────────────
        tk.Frame(self._root, bg=BORDER, height=1).pack(fill="x", padx=40)

        # ── Mode buttons ───────────────────────────────────────────────
        btn_frame = tk.Frame(self._root, bg=BG)
        btn_frame.pack(padx=40, pady=32, fill="x")

        self._make_mode_btn(
            btn_frame,
            label="RECORD",
            sublabel="Capture EEG/EMG + camera → export CSV",
            color=ACCENT2,
            mode="record",
        ).pack(fill="x", pady=(0, 12))

        self._make_mode_btn(
            btn_frame,
            label="INFERENCE",
            sublabel="Live stream with real-time action overlay",
            color=ACCENT,
            mode="inference",
        ).pack(fill="x")

        # ── Footer ─────────────────────────────────────────────────────
        tk.Label(
            self._root,
            text="OpenBCI Cyton · MediaPipe · BrainFlow",
            font=("Courier New", 9),
            fg=FG_DIM,
            bg=BG,
        ).pack(pady=(24, 28))

    def _make_mode_btn(
        self,
        parent: tk.Widget,
        label: str,
        sublabel: str,
        color: str,
        mode: str,
    ) -> tk.Frame:
        """Return a framed composite button for a single mode."""
        frame = tk.Frame(parent, bg=SURFACE, cursor="hand2",
                         highlightthickness=1, highlightbackground=BORDER)

        tk.Label(frame, text=label, font=("Courier New", 14, "bold"),
                 fg=color, bg=SURFACE).pack(anchor="w", padx=16, pady=(14, 2))
        tk.Label(frame, text=sublabel, font=("Courier New", 9),
                 fg=FG_DIM, bg=SURFACE).pack(anchor="w", padx=16, pady=(0, 14))

        # Bind entire frame and children to the click handler
        def _click(_event=None, m=mode):
            self._choice = m
            self._root.quit()

        def _enter(_event=None):
            frame.configure(highlightbackground=color)

        def _leave(_event=None):
            frame.configure(highlightbackground=BORDER)

        for widget in (frame, *frame.winfo_children()):
            widget.bind("<Button-1>", _click)
            widget.bind("<Enter>", _enter)
            widget.bind("<Leave>", _leave)

        # Re-bind after children are packed (children not yet created here,
        # so we schedule a late bind)
        frame.bind("<Map>", lambda _e: self._rebind_children(frame, _click, _enter, _leave))

        return frame

    @staticmethod
    def _rebind_children(frame, click, enter, leave) -> None:
        for widget in frame.winfo_children():
            widget.bind("<Button-1>", click)
            widget.bind("<Enter>", enter)
            widget.bind("<Leave>", leave)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _on_close(self) -> None:
        self._choice = None
        self._root.quit()

    def run(self) -> Optional[str]:
        """Block until the user makes a choice.  Returns the mode string or None."""
        self._root.mainloop()
        self._root.destroy()
        return self._choice