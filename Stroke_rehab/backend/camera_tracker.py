"""
camera_tracker.py
-----------------
MediaPipe Face Mesh eye/eyebrow tracking module for stroke rehabilitation BCI data capture.

Uses the MediaPipe Tasks API (mediapipe >= 0.10) with FaceLandmarker in VIDEO mode.
The model file 'face_landmarker.task' is downloaded automatically on first run if not present.

Tracks vertical pixel distance between eyebrow and upper eyelid landmarks to classify
gaze direction as LOOKING_UP, LOOKING_DOWN, or NEUTRAL. Supports auto-calibration and
manual threshold override. Thread-safe for concurrent frame processing and calibration.
"""

import os
import threading
import time
import urllib.request
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ---------------------------------------------------------------------------
# Model file — auto-downloaded alongside this script on first run
# ---------------------------------------------------------------------------
_MODEL_FILENAME = "face_landmarker.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

def _ensure_model(model_path: str) -> None:
    """Download the face landmarker model file if it is not already present.

    Parameters
    ----------
    model_path:
        Absolute path where the model should be stored.
    """
    if not os.path.exists(model_path):
        print(f"[CameraTracker] Downloading FaceLandmarker model to {model_path} ...")
        urllib.request.urlretrieve(_MODEL_URL, model_path)
        print(f"[CameraTracker] Model downloaded ({os.path.getsize(model_path)} bytes).")


# ---------------------------------------------------------------------------
# MediaPipe landmark indices — Eye Aspect Ratio (EAR) system
#
# EAR = mean(v1, v2, v3) / eye_width
#   v1..v3 = vertical distances between 3 matched upper/lower eyelid pairs
#   eye_width = horizontal corner-to-corner distance (normalization)
#
# Because the brow directly pulls the upper lid up or down:
#   High EAR → eye wide open → brow raised   → LOOKING_UP
#   Low  EAR → eye narrowed  → brow lowered  → LOOKING_DOWN
#
# Left eye  upper/lower pairs: (160,144) (159,145) (158,153)  corners: 33, 133
# Right eye upper/lower pairs: (387,373) (386,374) (385,380)  corners: 263, 362
# ---------------------------------------------------------------------------
_L_EAR_UPPER: tuple[int, ...] = (160, 159, 158)
_L_EAR_LOWER: tuple[int, ...] = (144, 145, 153)
_L_EYE_CORNERS: tuple[int, int] = (33, 133)
_R_EAR_UPPER: tuple[int, ...] = (387, 386, 385)
_R_EAR_LOWER: tuple[int, ...] = (373, 374, 380)
_R_EYE_CORNERS: tuple[int, int] = (263, 362)

# Action label constants
LOOKING_UP: str = "LOOKING_UP"
LOOKING_DOWN: str = "LOOKING_DOWN"
NEUTRAL: str = "NEUTRAL"


class CameraTracker:
    """
    MediaPipe FaceLandmarker-based Eye Aspect Ratio (EAR) tracker for BCI stroke rehabilitation.

    Computes EAR = mean(v1,v2,v3) / eye_width where v1..v3 are the vertical distances
    between three matched upper/lower eyelid pairs and eye_width is the corner-to-corner
    horizontal span.  Because the brow directly controls how open the eye is, EAR captures
    both raising (eye widens → high EAR) and lowering (eye narrows → low EAR) with far
    better dynamic range than raw brow position.  The ratio is scale-invariant.

    Auto-calibration collects neutral EAR values over a short window, then sets
    upper_threshold = mean + 2.0 * std  and  lower_threshold = mean - 2.0 * std.

    All shared state is protected by a threading.Lock so that calibration threads and
    the main capture loop can coexist safely.

    Parameters
    ----------
    camera_index : int
        OpenCV camera index (default 0).
    eye_selection : str
        Which eye(s) to track: "left", "right", or "both" (default "both").
    model_path : str | None
        Path to the face_landmarker.task model file. If None, defaults to a file
        named 'face_landmarker.task' in the same directory as this module.
    """

    def __init__(
        self,
        camera_index: int = 0,
        eye_selection: str = "both",
        model_path: Optional[str] = None,
    ) -> None:
        # ------------------------------------------------------------------ #
        # Model path                                                           #
        # ------------------------------------------------------------------ #
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _MODEL_FILENAME)
        _ensure_model(model_path)

        # ------------------------------------------------------------------ #
        # Camera handle                                                        #
        # ------------------------------------------------------------------ #
        self._camera_index: int = camera_index
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_timestamp_ms: int = 0  # monotonically increasing timestamp for Tasks API

        # ------------------------------------------------------------------ #
        # MediaPipe FaceLandmarker (Tasks API, VIDEO mode)                    #
        # ------------------------------------------------------------------ #
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self._face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)

        # ------------------------------------------------------------------ #
        # Eye selection                                                        #
        # ------------------------------------------------------------------ #
        self._valid_selections = {"left", "right", "both"}
        self._eye_selection: str = eye_selection.lower()
        if self._eye_selection not in self._valid_selections:
            raise ValueError(
                f"eye_selection must be one of {self._valid_selections}, got '{eye_selection}'"
            )

        # ------------------------------------------------------------------ #
        # Thresholds (per eye) — defaults until calibration or manual override #
        # ------------------------------------------------------------------ #
        # Thresholds are EAR values (typical neutral ~0.25-0.30):
        #   EAR > upper → LOOKING_UP  (eye wide / brow raised)
        #   EAR < lower → LOOKING_DOWN (eye narrow / brow lowered)
        self._upper_threshold: dict[str, float] = {"left": 0.32, "right": 0.32}
        self._lower_threshold: dict[str, float] = {"left": 0.20, "right": 0.20}

        # ------------------------------------------------------------------ #
        # Calibration state                                                    #
        # ------------------------------------------------------------------ #
        self._calibrating: bool = False
        self._calib_duration: float = 3.0
        self._calib_thread: Optional[threading.Thread] = None
        self._calib_samples: dict[str, list[float]] = {"left": [], "right": []}

        # ------------------------------------------------------------------ #
        # Current processed data                                               #
        # ------------------------------------------------------------------ #
        self._left_distance: Optional[float] = None
        self._right_distance: Optional[float] = None
        self._action_left: str = NEUTRAL
        self._action_right: str = NEUTRAL
        self._current_frame: Optional[np.ndarray] = None

        # ------------------------------------------------------------------ #
        # Thread safety                                                        #
        # ------------------------------------------------------------------ #
        self._lock: threading.Lock = threading.Lock()

    # ---------------------------------------------------------------------- #
    # Public API — lifecycle                                                   #
    # ---------------------------------------------------------------------- #

    def start(self) -> bool:
        """Open the camera capture device.

        Returns
        -------
        bool
            True if the camera was opened successfully, False otherwise.
        """
        # Use DirectShow backend on Windows — avoids MSMF frame-read errors
        self._cap = cv2.VideoCapture(self._camera_index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            print(f"[CameraTracker] ERROR: Could not open camera at index {self._camera_index}.")
            self._cap = None
            return False

        # On Windows the camera driver needs a few frames to warm up before
        # it starts delivering valid data. Drain up to 30 frames (≈1 s) until
        # we get a successful read.
        for _ in range(30):
            ret, _ = self._cap.read()
            if ret:
                break
            time.sleep(0.033)
        else:
            print(f"[CameraTracker] ERROR: Camera opened but could not read frames.")
            self._cap.release()
            self._cap = None
            return False

        print(f"[CameraTracker] Camera {self._camera_index} opened successfully.")
        return True

    def stop(self) -> None:
        """Release the camera and clean up MediaPipe resources.

        Safe to call even if the camera was never successfully opened.
        """
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._face_landmarker.close()
        print("[CameraTracker] Camera released.")

    # ---------------------------------------------------------------------- #
    # Public API — calibration                                                 #
    # ---------------------------------------------------------------------- #

    def start_calibration(self, duration_seconds: float = 3.0) -> None:
        """Begin auto-calibration in a background thread.

        Collects eyebrow-to-eyelid distances for *duration_seconds* seconds.
        At the end, thresholds are set to mean ± 1.5 × std.

        Parameters
        ----------
        duration_seconds : float
            How long (in seconds) to collect calibration samples (default 3.0).
        """
        with self._lock:
            if self._calibrating:
                print("[CameraTracker] Calibration already in progress.")
                return
            self._calib_duration = duration_seconds
            self._calib_samples = {"left": [], "right": []}
            self._calibrating = True

        self._calib_thread = threading.Thread(
            target=self._calibration_worker,
            daemon=True,
            name="CameraTracker-Calibration",
        )
        self._calib_thread.start()
        print(f"[CameraTracker] Calibration started for {duration_seconds:.1f} s.")

    def is_calibrating(self) -> bool:
        """Return True while the background calibration thread is active."""
        with self._lock:
            return self._calibrating

    # ---------------------------------------------------------------------- #
    # Public API — thresholds                                                  #
    # ---------------------------------------------------------------------- #

    def get_thresholds(self) -> dict:
        """Return the current threshold values for both eyes.

        Returns
        -------
        dict
            ``{"left": {"upper": float, "lower": float},
               "right": {"upper": float, "lower": float}}``
        """
        with self._lock:
            return {
                "left": {
                    "upper": self._upper_threshold["left"],
                    "lower": self._lower_threshold["left"],
                },
                "right": {
                    "upper": self._upper_threshold["right"],
                    "lower": self._lower_threshold["right"],
                },
            }

    def set_manual_thresholds(self, eye: str, upper: float, lower: float) -> None:
        """Override thresholds manually for a specific eye.

        Parameters
        ----------
        eye : str
            "left" or "right".
        upper : float
            Upper threshold (pixels). Distances above this → LOOKING_UP.
        lower : float
            Lower threshold (pixels). Distances below this → LOOKING_DOWN.

        Raises
        ------
        ValueError
            If *eye* is not "left" or "right", or if upper <= lower.
        """
        eye = eye.lower()
        if eye not in {"left", "right"}:
            raise ValueError(f"eye must be 'left' or 'right', got '{eye}'")
        if upper <= lower:
            raise ValueError(f"upper ({upper}) must be greater than lower ({lower})")
        with self._lock:
            self._upper_threshold[eye] = upper
            self._lower_threshold[eye] = lower
        print(f"[CameraTracker] Manual thresholds set for {eye} eye: upper={upper:.2f}, lower={lower:.2f}")

    def set_eye_selection(self, selection: str) -> None:
        """Change which eye(s) to track at runtime.

        Parameters
        ----------
        selection : str
            "left", "right", or "both".
        """
        selection = selection.lower()
        if selection not in self._valid_selections:
            raise ValueError(f"selection must be one of {self._valid_selections}, got '{selection}'")
        with self._lock:
            self._eye_selection = selection
        print(f"[CameraTracker] Eye selection set to '{selection}'.")

    # ---------------------------------------------------------------------- #
    # Public API — data access                                                 #
    # ---------------------------------------------------------------------- #

    def get_current_data(self) -> dict:
        """Return the most recently processed data snapshot.

        Returns
        -------
        dict
            Keys: ``left_distance``, ``right_distance``, ``action_left``,
            ``action_right``, ``frame`` (annotated BGR ndarray | None).
        """
        with self._lock:
            return {
                "left_distance": self._left_distance,
                "right_distance": self._right_distance,
                "action_left": self._action_left,
                "action_right": self._action_right,
                "frame": self._current_frame.copy() if self._current_frame is not None else None,
            }

    # ---------------------------------------------------------------------- #
    # Public API — frame processing                                            #
    # ---------------------------------------------------------------------- #

    def process_frame(self) -> bool:
        """Read one frame, run FaceLandmarker, annotate, classify, update state.

        Returns
        -------
        bool
            False if camera unavailable or frame read failed; True on success.
        """
        if self._cap is None or not self._cap.isOpened():
            return False

        ret, frame = self._cap.read()
        if not ret or frame is None:
            print("[CameraTracker] Failed to read frame from camera.")
            return False

        # Mirror for natural display
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # MediaPipe Tasks API requires an mp.Image wrapping an RGB array
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Timestamp must be strictly increasing (milliseconds)
        self._frame_timestamp_ms += 33  # ~30 fps
        detection_result = self._face_landmarker.detect_for_video(
            mp_image, self._frame_timestamp_ms
        )

        left_dist: Optional[float] = None
        right_dist: Optional[float] = None

        with self._lock:
            eye_sel = self._eye_selection
            upper_thresh = dict(self._upper_threshold)
            lower_thresh = dict(self._lower_threshold)
            calibrating = self._calibrating

        if detection_result.face_landmarks:
            # face_landmarks is a list of landmark lists; take the first face
            landmarks = detection_result.face_landmarks[0]

            def _y(idx: int) -> float:
                return landmarks[idx].y * h

            def _x(idx: int) -> float:
                return landmarks[idx].x * w

            def _pt(idx: int) -> tuple[int, int]:
                return int(_x(idx)), int(_y(idx))

            def _compute_ear(upper_ids: tuple[int, ...],
                             lower_ids: tuple[int, ...],
                             corner_ids: tuple[int, int]) -> Optional[float]:
                """Return EAR = mean vertical openness / eye width, or None if degenerate."""
                vert_sum = sum(abs(_y(u) - _y(l))
                               for u, l in zip(upper_ids, lower_ids))
                mean_vert = vert_sum / len(upper_ids)
                dx = _x(corner_ids[0]) - _x(corner_ids[1])
                dy = _y(corner_ids[0]) - _y(corner_ids[1])
                eye_w = (dx * dx + dy * dy) ** 0.5
                if eye_w < 5:
                    return None
                return mean_vert / eye_w

            def _draw_ear(upper_ids: tuple[int, ...],
                          lower_ids: tuple[int, ...],
                          corner_ids: tuple[int, int]) -> None:
                """Draw EAR landmarks on frame."""
                for u_idx, l_idx in zip(upper_ids, lower_ids):
                    cv2.circle(frame, _pt(u_idx), 3, (0, 255, 136), -1)   # green = upper lid
                    cv2.circle(frame, _pt(l_idx), 3, (68, 68, 255), -1)   # red   = lower lid
                    cv2.line(frame, _pt(u_idx), _pt(l_idx),
                             (200, 200, 200), 1, cv2.LINE_AA)
                cv2.circle(frame, _pt(corner_ids[0]), 3, (255, 153, 85), -1)  # corner dots
                cv2.circle(frame, _pt(corner_ids[1]), 3, (255, 153, 85), -1)
                cv2.line(frame, _pt(corner_ids[0]), _pt(corner_ids[1]),
                         (100, 153, 255), 1, cv2.LINE_AA)

            if eye_sel in {"left", "both"}:
                left_dist = _compute_ear(_L_EAR_UPPER, _L_EAR_LOWER, _L_EYE_CORNERS)
                if left_dist is not None:
                    _draw_ear(_L_EAR_UPPER, _L_EAR_LOWER, _L_EYE_CORNERS)

            if eye_sel in {"right", "both"}:
                right_dist = _compute_ear(_R_EAR_UPPER, _R_EAR_LOWER, _R_EYE_CORNERS)
                if right_dist is not None:
                    _draw_ear(_R_EAR_UPPER, _R_EAR_LOWER, _R_EYE_CORNERS)

            # Feed calibration samples while calibration is active
            if calibrating:
                with self._lock:
                    if left_dist is not None:
                        self._calib_samples["left"].append(left_dist)
                    if right_dist is not None:
                        self._calib_samples["right"].append(right_dist)

        # Classify actions
        action_left = self._classify(left_dist, upper_thresh["left"], lower_thresh["left"])
        action_right = self._classify(right_dist, upper_thresh["right"], lower_thresh["right"])

        # Draw HUD overlay
        self._draw_overlay(frame, left_dist, right_dist, action_left, action_right, eye_sel, calibrating)

        # Update shared state
        with self._lock:
            self._left_distance = left_dist
            self._right_distance = right_dist
            self._action_left = action_left
            self._action_right = action_right
            self._current_frame = frame

        return True

    # ---------------------------------------------------------------------- #
    # Private helpers                                                          #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _classify(ratio: Optional[float], upper: float, lower: float) -> str:
        """Classify an EAR value as LOOKING_UP, LOOKING_DOWN, or NEUTRAL.

        EAR > upper → LOOKING_UP   (eye wide / brow raised)
        EAR < lower → LOOKING_DOWN (eye narrow / brow lowered)
        Returns NEUTRAL when ratio is None (face not detected).
        """
        if ratio is None:
            return NEUTRAL
        if ratio > upper:
            return LOOKING_UP
        if ratio < lower:
            return LOOKING_DOWN
        return NEUTRAL

    def _calibration_worker(self) -> None:
        """Background thread: collect samples then compute per-eye thresholds."""
        end_time = time.monotonic() + self._calib_duration
        while time.monotonic() < end_time:
            time.sleep(0.01)  # samples arrive via process_frame()

        with self._lock:
            for eye in ("left", "right"):
                samples = self._calib_samples[eye]
                if len(samples) >= 2:
                    arr = np.array(samples, dtype=float)
                    mean = float(np.mean(arr))
                    std = float(np.std(arr))
                    self._upper_threshold[eye] = min(0.99, mean + 2.0 * std)
                    self._lower_threshold[eye] = max(0.01, mean - 2.0 * std)
                    print(
                        f"[CameraTracker] Calibration complete ({eye}): "
                        f"mean={mean:.2f}, std={std:.2f}, "
                        f"upper={self._upper_threshold[eye]:.2f}, "
                        f"lower={self._lower_threshold[eye]:.2f} "
                        f"({len(samples)} samples)"
                    )
                else:
                    print(
                        f"[CameraTracker] WARNING: Not enough calibration samples "
                        f"for {eye} eye ({len(samples)} collected); thresholds unchanged."
                    )
            self._calibrating = False

    def _draw_overlay(
        self,
        frame: np.ndarray,
        left_dist: Optional[float],
        right_dist: Optional[float],
        action_left: str,
        action_right: str,
        eye_sel: str,
        calibrating: bool,
    ) -> None:
        """Draw HUD text onto *frame* in-place."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_pos = 30

        colour_map = {
            LOOKING_UP: (0, 255, 0),
            LOOKING_DOWN: (0, 0, 255),
            NEUTRAL: (255, 255, 255),
        }

        if eye_sel in {"left", "both"}:
            ratio_str = f"{left_dist:.3f}" if left_dist is not None else "N/A"
            label = f"LEFT  EAR={ratio_str}  {action_left}"
            cv2.putText(frame, label, (10, y_pos), font, 0.6,
                        colour_map.get(action_left, (255, 255, 255)), 2, cv2.LINE_AA)
            y_pos += 30

        if eye_sel in {"right", "both"}:
            ratio_str = f"{right_dist:.3f}" if right_dist is not None else "N/A"
            label = f"RIGHT EAR={ratio_str}  {action_right}"
            cv2.putText(frame, label, (10, y_pos), font, 0.6,
                        colour_map.get(action_right, (255, 255, 255)), 2, cv2.LINE_AA)
            y_pos += 30

        cv2.putText(frame, f"Eye: {eye_sel.upper()}", (10, y_pos),
                    font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        if calibrating:
            banner = "  CALIBRATING...  "
            text_size, _ = cv2.getTextSize(banner, font, 0.8, 2)
            bx = (frame.shape[1] - text_size[0]) // 2
            by = frame.shape[0] - 20
            overlay = frame.copy()
            cv2.rectangle(overlay, (bx - 8, by - text_size[1] - 8),
                          (bx + text_size[0] + 8, by + 8), (0, 140, 255), -1)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
            cv2.putText(frame, banner, (bx, by), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Quick smoke-test / demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tracker = CameraTracker(camera_index=0, eye_selection="both")

    if not tracker.start():
        raise SystemExit("Could not open camera.")

    print("Press 'c' to calibrate, 'q' to quit.")
    cv2.namedWindow("CameraTracker", cv2.WINDOW_NORMAL)

    try:
        while True:
            if not tracker.process_frame():
                break
            data = tracker.get_current_data()
            if data["frame"] is not None:
                cv2.imshow("CameraTracker", data["frame"])
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                tracker.start_calibration(duration_seconds=3.0)
    finally:
        tracker.stop()
        cv2.destroyAllWindows()
