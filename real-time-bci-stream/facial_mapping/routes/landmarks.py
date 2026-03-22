"""
routes/landmarks.py  –  Landmark Detection Endpoint
====================================================
Receives a base-64 encoded JPEG/PNG frame from the browser webcam,
runs MediaPipe FaceLandmarker (Tasks API – Python 3.13 compatible),
and returns the subset of landmarks needed by the frontend.

Why the Tasks API?
------------------
The legacy `mp.solutions.face_mesh` API was removed from MediaPipe
after version 0.10.9 and is no longer available on Python 3.13.
The replacement is `mediapipe.tasks.python.vision.FaceLandmarker`,
which requires a `.task` model file. This module downloads that file
automatically on first run (~29 MB, stored next to this file).

POST /api/process-frame
-----------------------
Request JSON
    {
        "image": "<base-64 data-URL string>",
        "side":  "left" | "right"        // side the user wants to MIRROR
    }

Response JSON (success)
    {
        "success": true,
        "landmarks": {
            // All values are normalised [0-1] relative to the frame size
            "eyebrow": [{"x": 0.4, "y": 0.3}, ...],   // 5 points
            "eye":     [{"x": 0.4, "y": 0.35}, ...],  // 6 points (eye contour)
            "iris":    {"x": 0.42, "y": 0.37},         // iris centre
            "mouth":   [{"x": 0.4, "y": 0.6}, ...]    // 4 corner points
        },
        "imageWidth":  640,
        "imageHeight": 480
    }

Response JSON (failure)
    { "success": false, "error": "<reason>" }

MediaPipe FaceLandmarker returns 478 landmarks (same indices as the
legacy FaceMesh when output_face_blendshapes=False).

Landmark indices used
---------------------
Left  eyebrow : 70, 63, 105, 66, 107      (inner --> outer)
Right eyebrow : 300, 293, 334, 296, 336
Left  eye     : 33, 160, 158, 133, 153, 144
Right eye     : 263, 387, 385, 362, 380, 373
Left  iris    : 468  (centre)
Right iris    : 473
Mouth corners : 61 (left), 291 (right), 0 (top), 17 (bottom)
"""

import base64
import io
import os
import urllib.request
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from flask import Blueprint, request, jsonify
from PIL import Image


# -- Blueprint ----------------------------------------------------------------
landmarks_bp = Blueprint("landmarks", __name__)


# -- Model bootstrap ----------------------------------------------------------

# Store the model file next to this source file so it travels with the project
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


def _ensure_model() -> str:
    """
    Download the FaceLandmarker model file if it is not already present.
    Returns the local path to the model file.

    The file is ~29 MB and is only downloaded once.
    """
    if not os.path.exists(_MODEL_PATH):
        print(
            f"[FaceMap] Downloading FaceLandmarker model (~29 MB) --> {_MODEL_PATH}\n"
            "          This happens only once on first run."
        )
        try:
            urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
            print("[FaceMap] Model downloaded successfully.")
        except Exception as exc:
            raise RuntimeError(
                f"Could not download the MediaPipe face landmarker model.\n"
                f"URL : {_MODEL_URL}\n"
                f"Err : {exc}\n\n"
                "If you are offline, manually download the file from the URL above\n"
                f"and save it to:  {_MODEL_PATH}"
            ) from exc
    return _MODEL_PATH


# -- MediaPipe FaceLandmarker (initialise once, reuse across requests) --------

def _build_landmarker() -> mp_vision.FaceLandmarker:
    """
    Construct and return a FaceLandmarker configured for IMAGE mode.
    IMAGE mode processes each frame independently (no state between calls),
    which is safe for the multi-threaded Flask request model.
    """
    model_path   = _ensure_model()
    base_options = mp_tasks.BaseOptions(model_asset_path=model_path)

    options = mp_vision.FaceLandmarkerOptions(
        base_options                          = base_options,
        running_mode                          = mp_vision.RunningMode.IMAGE,
        num_faces                             = 1,
        min_face_detection_confidence         = 0.5,
        min_face_presence_confidence          = 0.5,
        min_tracking_confidence               = 0.5,
        output_face_blendshapes               = False,
        output_facial_transformation_matrixes = False,
    )

    return mp_vision.FaceLandmarker.create_from_options(options)


# Module-level singleton -- created once when Flask imports this file
_face_landmarker: mp_vision.FaceLandmarker = _build_landmarker()


# -- Landmark index maps ------------------------------------------------------
# Ordered inner-corner --> outer-corner so expressions.js can weight the
# inner point more heavily for knit / look-up / look-down animations.

_INDICES = {
    "left": {
        "eyebrow": [70, 63, 105, 66, 107],
        "eye":     [33, 160, 158, 133, 153, 144],
        "iris":    468,
        "mouth":   [61, 291, 0, 17],
    },
    "right": {
        "eyebrow": [300, 293, 334, 296, 336],
        "eye":     [263, 387, 385, 362, 380, 373],
        "iris":    473,
        "mouth":   [61, 291, 0, 17],   # mouth corners are shared landmarks
    },
}


# -- Helpers ------------------------------------------------------------------

def _decode_image(data_url: str) -> np.ndarray:
    """
    Convert a base-64 data-URL (image/jpeg or image/png) into an
    RGB numpy array suitable for MediaPipe.

    Parameters
    ----------
    data_url : str
        Either a full data-URL ("data:image/jpeg;base64,....") or a
        plain base-64 string.

    Returns
    -------
    np.ndarray  shape (H, W, 3), dtype uint8, channel order RGB
    """
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    image_bytes = base64.b64decode(data_url)
    pil_image   = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(pil_image)


def _extract_landmarks(detection_result, indices: dict) -> dict:
    """
    Pull the required landmark positions from a FaceLandmarkerResult.

    The Tasks API returns normalised NormalizedLandmark objects with
    {x, y, z} fields.  We discard z and return plain {x, y} dicts so
    the frontend has a simple, stable contract.

    Parameters
    ----------
    detection_result : FaceLandmarkerResult
    indices : dict
        { eyebrow: [int], eye: [int], iris: int, mouth: [int] }

    Returns
    -------
    dict with keys eyebrow, eye, iris, mouth
    """
    # face_landmarks is a list-of-faces; index 0 = first (only) detected face
    # Each face is a list of 478 NormalizedLandmark objects
    lm = detection_result.face_landmarks[0]

    def point(idx: int) -> dict:
        return {"x": float(lm[idx].x), "y": float(lm[idx].y)}

    return {
        "eyebrow": [point(i) for i in indices["eyebrow"]],
        "eye":     [point(i) for i in indices["eye"]],
        "iris":    point(indices["iris"]),
        "mouth":   [point(i) for i in indices["mouth"]],
    }


# -- Route --------------------------------------------------------------------

@landmarks_bp.route("/api/process-frame", methods=["POST"])
def process_frame():
    """
    Main endpoint called by the frontend every ~80 ms.

    Steps:
      1. Validate the JSON payload
      2. Decode the base-64 webcam frame to an RGB numpy array
      3. Wrap in a MediaPipe Image and run FaceLandmarker
      4. Extract and return the landmark subset for the requested side
    """
    payload = request.get_json(silent=True)

    # -- Input validation -----------------------------------------------------
    if not payload:
        return jsonify({"success": False, "error": "No JSON body received"}), 400

    image_data = payload.get("image", "")
    side       = payload.get("side", "right").lower()

    if not image_data:
        return jsonify({"success": False, "error": "Missing 'image' field"}), 400

    if side not in ("left", "right"):
        return jsonify({"success": False, "error": "side must be 'left' or 'right'"}), 400

    # -- Decode frame ---------------------------------------------------------
    try:
        rgb_array = _decode_image(image_data)
    except Exception as exc:
        return jsonify({"success": False, "error": f"Image decode failed: {exc}"}), 400

    img_h, img_w = rgb_array.shape[:2]

    # -- Run FaceLandmarker ---------------------------------------------------
    # mp.Image wraps the numpy array without copying data
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_array)

    try:
        result = _face_landmarker.detect(mp_image)
    except Exception as exc:
        return jsonify({"success": False, "error": f"Detection failed: {exc}"}), 500

    # No face found -- return success=False so the frontend shows the badge
    if not result.face_landmarks:
        return jsonify({"success": False, "error": "No face detected"}), 200

    # -- Extract landmarks ----------------------------------------------------
    try:
        landmarks = _extract_landmarks(result, _INDICES[side])
    except Exception as exc:
        return jsonify({"success": False, "error": f"Landmark extraction failed: {exc}"}), 500

    return jsonify({
        "success":     True,
        "landmarks":   landmarks,
        "imageWidth":  img_w,
        "imageHeight": img_h,
    })