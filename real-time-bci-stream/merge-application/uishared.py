"""
ui_shared.py
============
Shared palette, font constants, and a base mixin used by both
app_record.py and app_inference.py.

Keeping these in one place ensures the two windows look identical
even though they live in separate modules.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

# ── Palette ────────────────────────────────────────────────────────────────
BG       = "#0d0d0f"
SURFACE  = "#16161a"
SURFACE2 = "#1e1e24"
BORDER   = "#2a2a30"
FG       = "#e8e8ec"
FG_DIM   = "#6b6b7a"
ACCENT   = "#4fc3f7"   # blue
ACCENT2  = "#81c784"   # green — record / active
DANGER   = "#ef5350"   # red — stop / error
WARN     = "#ffb74d"   # amber — calibrating

# ── Fonts ──────────────────────────────────────────────────────────────────
FONT_MONO   = "Courier New"
FONT_TITLE  = ("Courier New", 13, "bold")
FONT_BODY   = ("Courier New", 10)
FONT_SMALL  = ("Courier New", 9)
FONT_STATUS = ("Courier New", 9)

# ── Layout ─────────────────────────────────────────────────────────────────
VIDEO_W = 480
VIDEO_H = 360

POLL_CAMERA_MS  = 33    # ~30 fps
POLL_OPENBCI_MS = 4     # ~250 Hz
POLL_STATUS_MS  = 500


def apply_dark_theme(root: tk.Tk) -> None:
    """Apply a minimal dark ttk theme to *root*."""
    style = ttk.Style(root)
    style.theme_use("clam")

    style.configure(".",
        background=BG,
        foreground=FG,
        fieldbackground=SURFACE2,
        selectbackground=ACCENT,
        selectforeground=BG,
        font=FONT_BODY,
        relief="flat",
        borderwidth=0,
    )
    style.configure("TLabel",   background=BG,      foreground=FG)
    style.configure("TFrame",   background=BG)
    style.configure("TLabelframe", background=BG,   foreground=FG_DIM,
                    bordercolor=BORDER, relief="solid", borderwidth=1)
    style.configure("TLabelframe.Label", background=BG, foreground=FG_DIM,
                    font=FONT_SMALL)
    style.configure("TButton",  background=SURFACE2, foreground=FG,
                    borderwidth=1, relief="solid")
    style.map("TButton",
        background=[("active", SURFACE), ("pressed", BORDER)],
        foreground=[("active", ACCENT)],
    )
    style.configure("Accent.TButton", foreground=ACCENT, bordercolor=ACCENT)
    style.configure("Danger.TButton", foreground=DANGER, bordercolor=DANGER)
    style.configure("Success.TButton", foreground=ACCENT2, bordercolor=ACCENT2)
    style.configure("TCombobox", fieldbackground=SURFACE2, foreground=FG,
                    selectbackground=SURFACE2, selectforeground=FG)
    style.configure("TEntry",    fieldbackground=SURFACE2, foreground=FG,
                    insertcolor=FG)
    style.configure("TRadiobutton", background=BG, foreground=FG)
    style.map("TRadiobutton", background=[("active", BG)])
    style.configure("Status.TLabel", background=SURFACE, foreground=FG_DIM,
                    font=FONT_STATUS, relief="flat", padding=(8, 4))
    style.configure("Error.Status.TLabel",  foreground=DANGER)
    style.configure("Record.Status.TLabel", foreground=ACCENT2)


def section(parent: tk.Widget, title: str) -> ttk.LabelFrame:
    """Return a consistently styled LabelFrame for a control section."""
    return ttk.LabelFrame(parent, text=title, padding=8)


def label(parent: tk.Widget, text: str, dim: bool = False) -> ttk.Label:
    fg = FG_DIM if dim else FG
    return ttk.Label(parent, text=text, foreground=fg)


def mono_label(parent: tk.Widget, textvariable: tk.StringVar,
               color: str = FG) -> ttk.Label:
    return ttk.Label(parent, textvariable=textvariable,
                     foreground=color, font=FONT_BODY)