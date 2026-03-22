"""
openbci_stream.py
=================
Streams EEG/EMG data from an OpenBCI Cyton board via USB dongle using the
BrainFlow library.

Channel assignment (8 channels, 1-indexed as labelled on the Cyton board):
    Channels 1-4  → EMG : ["EMG_1", "EMG_2", "EMG_3", "EMG_4"]
    Channels 5-8  → EEG : ["EEG_Frontal_1", "EEG_Frontal_2",
                            "EEG_Behind_Left_Ear", "EEG_Behind_Right_Ear"]

All voltage values are raw microvolts as delivered by BrainFlow.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import serial.tools.list_ports  # pyserial is a brainflow dependency

from brainflow.board_shim import BoardShim, BoardIds, BrainFlowInputParams
from brainflow.data_filter import DataFilter  # noqa: F401  (imported for callers)
from brainflow.exit_codes import BrainFlowError

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Human-readable labels for all 8 acquisition channels, in board order.
CHANNEL_LABELS: list[str] = [
    "EMG_1",
    "EMG_2",
    "EMG_3",
    "EMG_4",
    "EEG_Frontal_1",
    "EEG_Frontal_2",
    "EEG_Behind_Left_Ear",
    "EEG_Behind_Right_Ear",
]

#: Number of EXG channels used.
NUM_CHANNELS: int = len(CHANNEL_LABELS)

#: Nominal sampling rate of the Cyton board (Hz).
CYTON_SAMPLE_RATE: int = 250

#: Number of samples fetched from the ring buffer on each background-thread
#: iteration.  Larger values reduce CPU overhead; smaller values reduce latency.
_FETCH_SIZE: int = 10


# ---------------------------------------------------------------------------
# Stub signal-processing helpers
# (NOT called during recording — provided for future real-time processing)
# ---------------------------------------------------------------------------

def bandpass_filter(
    data: list[float],
    lowcut: float,
    highcut: float,
    fs: float,
) -> list[float]:
    """Apply a bandpass filter to a 1-D signal.

    Args:
        data:    Raw voltage samples (microvolts).
        lowcut:  Lower cutoff frequency in Hz.
        highcut: Upper cutoff frequency in Hz.
        fs:      Sampling frequency in Hz.

    Returns:
        Filtered signal as a list of floats.

    # TODO: Enable for real-time processing in future development
    """
    raise NotImplementedError(
        "bandpass_filter is a stub and is not active during recording."
    )


def notch_filter(
    data: list[float],
    freq: float,
    fs: float,
) -> list[float]:
    """Apply a notch (band-stop) filter to suppress a single frequency.

    Typically used at 50 Hz or 60 Hz to remove power-line interference.

    Args:
        data: Raw voltage samples (microvolts).
        freq: Centre frequency to suppress in Hz (e.g. 50 or 60).
        fs:   Sampling frequency in Hz.

    Returns:
        Filtered signal as a list of floats.

    # TODO: Enable for real-time processing in future development
    """
    raise NotImplementedError(
        "notch_filter is a stub and is not active during recording."
    )


def compute_fft(
    data: list[float],
    fs: float,
) -> tuple[list[float], list[float]]:
    """Compute the single-sided amplitude spectrum of a signal via FFT.

    Args:
        data: Time-domain voltage samples (microvolts).
        fs:   Sampling frequency in Hz.

    Returns:
        A tuple of (frequencies_hz, amplitudes) both as lists of floats.

    # TODO: Enable for real-time processing in future development
    """
    raise NotImplementedError(
        "compute_fft is a stub and is not active during recording."
    )


def extract_band_power(
    data: list[float],
    fs: float,
) -> dict[str, float]:
    """Compute absolute band power for canonical EEG frequency bands.

    Bands computed:
        - delta : 0.5 –  4 Hz
        - theta :  4  –  8 Hz
        - alpha :  8  – 13 Hz
        - beta  : 13  – 30 Hz
        - gamma : 30  – 100 Hz

    Args:
        data: Time-domain voltage samples (microvolts).
        fs:   Sampling frequency in Hz.

    Returns:
        Dictionary mapping band name to absolute power value (µV²).

    # TODO: Enable for real-time processing in future development
    """
    raise NotImplementedError(
        "extract_band_power is a stub and is not active during recording."
    )


# ---------------------------------------------------------------------------
# OpenBCIStream
# ---------------------------------------------------------------------------

class OpenBCIStream:
    """Manages a streaming connection to an OpenBCI Cyton board.

    The class wraps BrainFlow's :class:`~brainflow.board_shim.BoardShim` and
    provides a simple interface for:

    * Discovering available serial (COM) ports.
    * Connecting / disconnecting from the Cyton dongle.
    * Starting / stopping a background acquisition thread.
    * Retrieving the most recent sample as a labelled dictionary.

    Attributes:
        last_error (str): Human-readable description of the most recent error.
            Empty string when no error has occurred since the last successful
            operation.

    Example usage::

        stream = OpenBCIStream()
        ports = stream.get_available_ports()
        if stream.connect(ports[0]):
            stream.start_stream()
            time.sleep(1)
            sample = stream.get_latest_sample()
            print(sample)
            stream.stop_stream()
            stream.disconnect()
    """

    def __init__(self, com_port: str = "") -> None:
        """Initialise the stream manager.

        The board is *not* connected during ``__init__``; call
        :meth:`connect` explicitly.

        Args:
            com_port: Optional COM port string (e.g. ``"COM3"`` on Windows or
                ``"/dev/ttyUSB0"`` on Linux).  If left empty the port must be
                supplied when calling :meth:`connect`.
        """
        # Public error state — updated whenever an exception is caught.
        self.last_error: str = ""

        # BrainFlow objects — None until connect() succeeds.
        self._board: Optional[BoardShim] = None
        self._params: Optional[BrainFlowInputParams] = None

        # Store the port supplied at construction time (may be overridden in
        # connect()).
        self._com_port: str = com_port

        # Background acquisition thread state.
        self._stream_thread: Optional[threading.Thread] = None
        self._streaming: bool = False

        # Thread-safe buffer for the latest complete sample.
        self._buffer_lock: threading.Lock = threading.Lock()
        self._latest_sample: Optional[dict] = None

        # Retrieve EXG channel indices from BrainFlow for the Cyton board.
        # This is done once at construction time so it does not block the
        # acquisition loop.
        self._exg_channels: list[int] = BoardShim.get_exg_channels(
            BoardIds.CYTON_BOARD.value
        )
        self._timestamp_channel: int = BoardShim.get_timestamp_channel(
            BoardIds.CYTON_BOARD.value
        )

        # Enable BrainFlow's internal logger at WARN level to reduce noise
        # while still surfacing important messages.
        BoardShim.disable_board_logger()

    # ------------------------------------------------------------------
    # Port discovery
    # ------------------------------------------------------------------

    def get_available_ports(self) -> list[str]:
        """Return a list of serial port names detected on this machine.

        Uses :mod:`serial.tools.list_ports` (shipped with *pyserial*, which is
        a BrainFlow dependency) to enumerate all available ports.

        Returns:
            List of port name strings, e.g. ``["COM3", "COM7"]`` on Windows or
            ``["/dev/ttyUSB0", "/dev/ttyACM0"]`` on Linux/macOS.  An empty
            list is returned if no ports are found or if the enumeration fails.
        """
        try:
            ports = serial.tools.list_ports.comports()
            port_names: list[str] = [p.device for p in ports]
            return port_names
        except Exception as exc:  # pragma: no cover
            self.last_error = f"Port enumeration failed: {exc}"
            return []

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self, com_port: str) -> bool:
        """Connect to the OpenBCI Cyton board on the specified COM port.

        Prepares and sessions a :class:`~brainflow.board_shim.BoardShim` for
        the Cyton board.  The streaming session is *not* started here — call
        :meth:`start_stream` after a successful ``connect``.

        Args:
            com_port: Serial port string identifying the USB dongle
                (e.g. ``"COM3"`` on Windows, ``"/dev/ttyUSB0"`` on Linux).

        Returns:
            ``True`` on success, ``False`` if the connection failed.  On
            failure, a description is stored in :attr:`last_error`.
        """
        # Refuse to double-connect without an explicit disconnect first.
        if self._board is not None and self._board.is_prepared():
            self.last_error = (
                "Board is already connected.  Call disconnect() first."
            )
            return False

        if not com_port:
            self.last_error = (
                "No COM port specified.  Pass a port string to connect()."
            )
            return False

        self._com_port = com_port

        # Build BrainFlow input parameters.
        params = BrainFlowInputParams()
        params.serial_port = com_port

        board = BoardShim(BoardIds.CYTON_BOARD.value, params)

        try:
            board.prepare_session()
        except BrainFlowError as exc:
            self.last_error = (
                f"BrainFlow error while connecting to {com_port}: {exc}"
            )
            self._board = None
            return False
        except Exception as exc:  # pragma: no cover
            self.last_error = (
                f"Unexpected error while connecting to {com_port}: {exc}"
            )
            self._board = None
            return False

        # Store references only after prepare_session() succeeds.
        self._params = params
        self._board = board
        self.last_error = ""
        return True

    def disconnect(self) -> None:
        """Stop streaming (if active) and release the board session.

        Safe to call even if the board is not currently connected.
        """
        # Ensure the acquisition thread is stopped before releasing the board.
        if self._streaming:
            self.stop_stream()

        if self._board is not None:
            try:
                if self._board.is_prepared():
                    self._board.release_session()
            except BrainFlowError as exc:
                self.last_error = f"BrainFlow error during disconnect: {exc}"
            except Exception as exc:  # pragma: no cover
                self.last_error = f"Unexpected error during disconnect: {exc}"
            finally:
                self._board = None
                self._params = None

    def is_connected(self) -> bool:
        """Return ``True`` if the board session is prepared and ready.

        Returns:
            ``True`` when the board session is active, ``False`` otherwise.
        """
        if self._board is None:
            return False
        try:
            return self._board.is_prepared()
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Streaming control
    # ------------------------------------------------------------------

    def start_stream(self) -> None:
        """Begin data acquisition from the Cyton board.

        Instructs BrainFlow to start streaming and spawns a daemon background
        thread that continuously reads new samples into an internal buffer.

        Raises:
            RuntimeError: If the board is not connected or streaming has
                already been started.
        """
        if not self.is_connected():
            raise RuntimeError(
                "Cannot start stream: board is not connected.  "
                "Call connect() first."
            )
        if self._streaming:
            raise RuntimeError(
                "Stream is already running.  Call stop_stream() first."
            )

        try:
            # The ring buffer on the board side holds 45 seconds of data by
            # default; no need to change it here.
            self._board.start_stream()
        except BrainFlowError as exc:
            self.last_error = f"BrainFlow error starting stream: {exc}"
            raise RuntimeError(self.last_error) from exc

        self._streaming = True
        self.last_error = ""

        # Spawn background acquisition thread.
        self._stream_thread = threading.Thread(
            target=self._acquisition_loop,
            name="OpenBCI-AcquisitionThread",
            daemon=True,  # Thread will not prevent process exit.
        )
        self._stream_thread.start()

    def stop_stream(self) -> None:
        """Stop data acquisition and join the background thread.

        Safe to call even if the stream is not currently running.
        """
        if not self._streaming:
            return

        # Signal the acquisition loop to exit.
        self._streaming = False

        # Wait for the thread to finish its current iteration.
        if self._stream_thread is not None:
            self._stream_thread.join(timeout=5.0)
            self._stream_thread = None

        # Tell the board to stop producing data.
        if self._board is not None:
            try:
                self._board.stop_stream()
            except BrainFlowError as exc:
                self.last_error = f"BrainFlow error stopping stream: {exc}"
            except Exception as exc:  # pragma: no cover
                self.last_error = f"Unexpected error stopping stream: {exc}"

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def get_latest_sample(self) -> Optional[dict]:
        """Return the most recently acquired sample as a labelled dictionary.

        The dictionary contains one key per channel (using the strings from
        :attr:`CHANNEL_LABELS`) plus a ``"timestamp"`` key containing the
        Unix epoch timestamp supplied by BrainFlow.

        Returns:
            A dictionary of the form::

                {
                    "EMG_1":                <float µV>,
                    "EMG_2":                <float µV>,
                    "EMG_3":                <float µV>,
                    "EMG_4":                <float µV>,
                    "EEG_Frontal_1":        <float µV>,
                    "EEG_Frontal_2":        <float µV>,
                    "EEG_Behind_Left_Ear":  <float µV>,
                    "EEG_Behind_Right_Ear": <float µV>,
                    "timestamp":            <float seconds since epoch>,
                }

            Returns ``None`` if the stream has not been started or no data has
            arrived yet.
        """
        if not self._streaming:
            return None

        with self._buffer_lock:
            # Return a shallow copy so the caller cannot mutate the buffer.
            return dict(self._latest_sample) if self._latest_sample else None

    def get_channel_labels(self) -> list[str]:
        """Return the ordered list of channel label strings.

        Returns:
            List of 8 strings in board order:
            ``["EMG_1", "EMG_2", "EMG_3", "EMG_4",
            "EEG_Frontal_1", "EEG_Frontal_2",
            "EEG_Behind_Left_Ear", "EEG_Behind_Right_Ear"]``
        """
        return list(CHANNEL_LABELS)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _acquisition_loop(self) -> None:
        """Background thread: continuously pull samples from the BrainFlow
        ring buffer and store the latest one in :attr:`_latest_sample`.

        This method is intended to be run exclusively via
        :meth:`start_stream` and must not be called directly.
        """
        while self._streaming:
            try:
                # get_board_data() drains the internal ring buffer and returns
                # a 2-D NumPy array shaped (num_board_channels, num_samples).
                data = self._board.get_board_data(_FETCH_SIZE)
            except BrainFlowError as exc:
                self.last_error = f"BrainFlow error in acquisition loop: {exc}"
                # Do not crash the thread — log and retry next iteration.
                time.sleep(0.01)
                continue
            except Exception as exc:  # pragma: no cover
                self.last_error = f"Unexpected error in acquisition loop: {exc}"
                time.sleep(0.01)
                continue

            # data.shape[1] is the number of new samples received.
            if data.shape[1] == 0:
                # Board buffer was empty — yield briefly and retry.
                time.sleep(0.004)  # ~4 ms ≈ 1 sample at 250 Hz
                continue

            # Extract only the 8 EXG channels we care about.
            # _exg_channels contains the row indices for all EXG channels on
            # the Cyton board (indices 0-7 in the 8-channel configuration).
            # We take only the first NUM_CHANNELS to match our label list.
            exg_rows = self._exg_channels[:NUM_CHANNELS]

            # Use the last column (most recent sample) from each EXG row.
            latest_values: list[float] = [
                float(data[row, -1]) for row in exg_rows
            ]

            # Retrieve the timestamp for the most recent sample.
            timestamp: float = float(data[self._timestamp_channel, -1])

            # Build the labelled sample dictionary.
            sample: dict = {
                label: value
                for label, value in zip(CHANNEL_LABELS, latest_values)
            }
            sample["timestamp"] = timestamp

            # Store atomically under the lock so get_latest_sample() always
            # sees a consistent snapshot.
            with self._buffer_lock:
                self._latest_sample = sample

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "connected" if self.is_connected() else "disconnected"
        streaming = ", streaming" if self._streaming else ""
        port = self._com_port or "—"
        return (
            f"OpenBCIStream(port={port!r}, status={status}{streaming})"
        )
