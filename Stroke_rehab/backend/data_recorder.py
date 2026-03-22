"""
data_recorder.py
================
Thread-safe data recorder for a multi-stream BCI application that synchronizes
a camera stream (~30 fps) and an OpenBCI Cyton EEG/EMG stream (250 Hz) and
exports the combined data to CSV.

Synchronization strategy
-------------------------
The two streams run at very different rates (250 Hz vs ~30 Hz), so a
sample-aligned merge is not practical at record time.  Instead this module
uses a *last-known-value* approach:

1.  OpenBCI samples are stored at their native 250 Hz rate.  Each sample row
    carries a wall-clock timestamp produced by ``time.time()`` at the moment
    ``push_openbci_sample`` is called.

2.  The camera thread calls ``push_camera_data`` whenever it produces a new
    frame (~30 Hz).  This call only updates a small in-memory "latest camera
    state" record; it does *not* append a row to the buffer.

3.  Every call to ``push_openbci_sample`` snapshots the current camera state
    and writes it alongside the OpenBCI channels into a single buffer row.
    Because camera frames arrive roughly every 33 ms and OpenBCI samples
    arrive every 4 ms, consecutive OpenBCI rows will share the same camera
    values until the next camera frame arrives — this is equivalent to a
    zero-order hold / forward-fill.

Rationale
~~~~~~~~~
* Avoids interpolation artifacts in classification labels.
* The temporal error introduced is at most one camera frame period (~33 ms),
  which is acceptable for rehab gesture classification where movements unfold
  over hundreds of milliseconds.
* A single writer thread per stream means the two locks never need to be held
  simultaneously, preventing deadlocks.

CSV columns
-----------
timestamp              : float  — wall-clock time (seconds since Unix epoch)
ear_left      : float  — inter-feature eye distance, left eye camera
                                  (NaN when left eye not selected)
ear_right     : float  — inter-feature eye distance, right eye camera
                                  (NaN when right eye not selected)
action_left            : str    — action label from the left eye/camera stream
action_right           : str    — action label from the right eye/camera stream
EMG_1                  : float  — OpenBCI channel 1  (facial EMG)
EMG_2                  : float  — OpenBCI channel 2  (facial EMG)
EMG_3                  : float  — OpenBCI channel 3  (facial EMG)
EMG_4                  : float  — OpenBCI channel 4  (facial EMG)
EEG_Frontal_1          : float  — OpenBCI channel 5  (frontal EEG)
EEG_Frontal_2          : float  — OpenBCI channel 6  (frontal EEG)
EEG_Behind_Left_Ear    : float  — OpenBCI channel 7  (temporal/mastoid EEG)
EEG_Behind_Right_Ear   : float  — OpenBCI channel 8  (temporal/mastoid EEG)
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Column order that will be used in the exported CSV.
# ---------------------------------------------------------------------------
_CSV_COLUMNS: list[str] = [
    "timestamp",
    "ear_left",
    "ear_right",
    "action_left",
    "action_right",
    "EMG_1",
    "EMG_2",
    "EMG_3",
    "EMG_4",
    "EEG_Frontal_1",
    "EEG_Frontal_2",
    "EEG_Behind_Left_Ear",
    "EEG_Behind_Right_Ear",
]


class DataRecorder:
    """Thread-safe recorder that merges camera and OpenBCI Cyton data streams.

    Typical usage
    -------------
    ::

        recorder = DataRecorder()
        recorder.start_recording()

        # In camera thread:
        recorder.push_camera_data(left_distance=0.42, right_distance=None,
                                   action_left="blink", action_right="idle")

        # In OpenBCI thread (called at 250 Hz by the board callback):
        recorder.push_openbci_sample({
            "EMG_1": 1.2, "EMG_2": -0.3, "EMG_3": 0.8, "EMG_4": 0.1,
            "EEG_Frontal_1": 5.4, "EEG_Frontal_2": -2.1,
            "EEG_Behind_Left_Ear": 3.3, "EEG_Behind_Right_Ear": -1.0,
        })

        recorder.stop_recording()
        recorder.export_csv("session_001.csv")

    Thread safety
    -------------
    Two separate :class:`threading.Lock` objects are used:

    * ``_camera_lock``  — protects ``_latest_camera_state``.
    * ``_buffer_lock``  — protects ``_buffer`` and ``_recording``.

    The camera lock is acquired briefly inside ``push_openbci_sample`` only to
    *read* the latest camera snapshot; neither lock is ever held while the
    other is being acquired, so no deadlock is possible.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        """Initialise an idle (not recording) DataRecorder with an empty buffer."""

        # --- synchronisation primitives -----------------------------------
        self._camera_lock: threading.Lock = threading.Lock()
        self._buffer_lock: threading.Lock = threading.Lock()

        # --- recording state ----------------------------------------------
        self._recording: bool = False

        # --- row storage --------------------------------------------------
        # Each element is a dict whose keys match _CSV_COLUMNS.
        self._buffer: deque[dict] = deque()

        # --- latest camera state (zero-order hold) ------------------------
        # Updated by push_camera_data; consumed (read-only) by
        # push_openbci_sample.
        self._latest_camera_state: dict = {
            "ear_left": float("nan"),
            "ear_right": float("nan"),
            "action_left": "",
            "action_right": "",
        }

    # ------------------------------------------------------------------
    # Recording control
    # ------------------------------------------------------------------

    def start_recording(self) -> None:
        """Clear the buffer and begin recording.

        Any previously buffered rows are discarded.  After this call,
        ``push_openbci_sample`` will start appending rows.
        """
        with self._buffer_lock:
            self._buffer.clear()
            self._recording = True

    def stop_recording(self) -> None:
        """Stop recording.

        Buffered rows are preserved so that ``export_csv`` can still be called
        after this method returns.
        """
        with self._buffer_lock:
            self._recording = False

    def is_recording(self) -> bool:
        """Return ``True`` if the recorder is currently active.

        Returns
        -------
        bool
            Current recording state.
        """
        with self._buffer_lock:
            return self._recording

    # ------------------------------------------------------------------
    # Data ingestion — camera thread
    # ------------------------------------------------------------------

    def push_camera_data(
        self,
        left_distance: Optional[float],
        right_distance: Optional[float],
        action_left: str,
        action_right: str,
    ) -> None:
        """Update the latest camera state (called from the camera thread, ~30 Hz).

        This method does *not* append a row to the buffer.  It simply stores
        the most recent camera values so that the next OpenBCI sample can pick
        them up.  Unused eye channels should be passed as ``None``; they will
        be stored as ``float('nan')`` and written as ``NaN`` in the CSV.

        Parameters
        ----------
        left_distance:
            Normalised eye-landmark distance for the left eye, or ``None`` if
            the left eye camera is not active.
        right_distance:
            Normalised eye-landmark distance for the right eye, or ``None`` if
            the right eye camera is not active.
        action_left:
            Action label produced by the left eye/camera classifier
            (e.g. ``"blink"``, ``"idle"``).
        action_right:
            Action label produced by the right eye/camera classifier.
        """
        # Convert None → NaN so downstream code always deals with floats.
        left_val: float = float("nan") if left_distance is None else float(left_distance)
        right_val: float = float("nan") if right_distance is None else float(right_distance)

        with self._camera_lock:
            self._latest_camera_state["ear_left"] = left_val
            self._latest_camera_state["ear_right"] = right_val
            self._latest_camera_state["action_left"] = action_left
            self._latest_camera_state["action_right"] = action_right

    # ------------------------------------------------------------------
    # Data ingestion — OpenBCI thread
    # ------------------------------------------------------------------

    def push_openbci_sample(self, sample: dict) -> None:
        """Append one merged row to the buffer (called from the OpenBCI thread, 250 Hz).

        The row combines:

        * A wall-clock timestamp (``time.time()``).
        * A snapshot of the current camera state (zero-order hold).
        * The eight OpenBCI channel values supplied in *sample*.

        The method is a no-op when ``is_recording()`` is ``False``.

        Parameters
        ----------
        sample:
            Dictionary containing the eight OpenBCI channel values.  Expected
            keys: ``EMG_1``, ``EMG_2``, ``EMG_3``, ``EMG_4``,
            ``EEG_Frontal_1``, ``EEG_Frontal_2``, ``EEG_Behind_Left_Ear``,
            ``EEG_Behind_Right_Ear``.  Missing keys will be stored as ``NaN``.
        """
        # Capture wall-clock time as early as possible to minimise jitter.
        ts: float = time.time()

        # Fast check — avoids lock acquisition overhead when not recording.
        # The authoritative check under the lock follows below.
        if not self._recording:
            return

        # --- snapshot camera state (brief lock) ---------------------------
        with self._camera_lock:
            # Shallow copy is sufficient; all values are immutable scalars/strings.
            camera_snapshot = self._latest_camera_state.copy()

        # --- build the row ------------------------------------------------
        row: dict = {
            "timestamp": ts,
            # Camera fields from zero-order hold snapshot
            "ear_left": camera_snapshot["ear_left"],
            "ear_right": camera_snapshot["ear_right"],
            "action_left": camera_snapshot["action_left"],
            "action_right": camera_snapshot["action_right"],
            # OpenBCI channels — fall back to NaN for any missing key
            "EMG_1": float(sample.get("EMG_1", float("nan"))),
            "EMG_2": float(sample.get("EMG_2", float("nan"))),
            "EMG_3": float(sample.get("EMG_3", float("nan"))),
            "EMG_4": float(sample.get("EMG_4", float("nan"))),
            "EEG_Frontal_1": float(sample.get("EEG_Frontal_1", float("nan"))),
            "EEG_Frontal_2": float(sample.get("EEG_Frontal_2", float("nan"))),
            "EEG_Behind_Left_Ear": float(sample.get("EEG_Behind_Left_Ear", float("nan"))),
            "EEG_Behind_Right_Ear": float(sample.get("EEG_Behind_Right_Ear", float("nan"))),
        }

        # --- append under buffer lock -------------------------------------
        with self._buffer_lock:
            # Re-check recording flag under the lock (avoids a TOCTOU race
            # between the fast check above and this append).
            if self._recording:
                self._buffer.append(row)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_csv(self, filepath: str) -> bool:
        """Write buffered rows to a CSV file using pandas.

        The CSV will have a header row followed by one data row per OpenBCI
        sample that was recorded.  Column order follows ``_CSV_COLUMNS``.

        If the buffer is empty the file is still created (header only) and
        the method returns ``True``.

        Parameters
        ----------
        filepath:
            Destination path for the CSV file (e.g. ``"session_001.csv"``).

        Returns
        -------
        bool
            ``True`` on success, ``False`` if an exception was raised (the
            exception message is printed to stderr via ``print``).
        """
        # Take a snapshot of the buffer so the lock is held as briefly as
        # possible and so pandas work happens outside the critical section.
        with self._buffer_lock:
            rows_snapshot = list(self._buffer)

        try:
            df = pd.DataFrame(rows_snapshot, columns=_CSV_COLUMNS)
            df.to_csv(filepath, index=False)
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[DataRecorder] export_csv failed: {exc}")
            return False

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_row_count(self) -> int:
        """Return the number of rows currently held in the buffer.

        Returns
        -------
        int
            Number of buffered OpenBCI sample rows.
        """
        with self._buffer_lock:
            return len(self._buffer)
