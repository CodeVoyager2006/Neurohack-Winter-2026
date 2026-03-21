"""
routes/landmarks.py  –  Landmark Detection Endpoint
====================================================
Receives a base-64 encoded JPEG/PNG frame from the browser webcam,
runs MediaPipe Face Mesh, and returns the subset of landmarks needed
by the frontend to draw the facial overlay.

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

MediaPipe landmark indices used
--------------------------------
Left eyebrow  : 70, 63, 105, 66, 107   (inner → outer)
Right eyebrow : 300, 293, 334, 296, 336
Left eye      : 33, 160, 158, 133, 153, 144
Right eye     : 263, 387, 385, 362, 380, 373
Left iris     : 468  (centre – requires refine_landmarks=True)
Right iris    : 473
Left mouth    : 61, 291, 0, 17         (left corner, right corner, top, bottom)
Right mouth   : same four points are shared/mirrored by design
"""

import base64
import io
import numpy as np
import cv2
import mediapipe as mp
from flask import Blueprint, request, jsonify
from PIL import Image

# ── Blueprint ────────────────────────────────────────────────────────────────
landmarks_bp = Blueprint("landmarks", __name__)

# ── MediaPipe setup (initialise once, reuse across requests) ─────────────────
_mp_face_mesh = mp.solutions.face_mesh
_face_mesh = _mp_face_mesh.FaceMesh(
    static_image_mode=False,       # streaming mode → faster for repeated calls
    max_num_faces=1,
    refine_landmarks=True,         # enables iris landmarks (468, 473)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ── Landmark index maps ───────────────────────────────────────────────────────
# Each list is ordered inner-corner → outer-corner so the frontend
# can infer which end is "inner" for knit/look animations.
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


# ── Helper ────────────────────────────────────────────────────────────────────

def _decode_image(data_url: str) -> np.ndarray:
    """
    Convert a base-64 data-URL (image/jpeg or image/png) to an
    OpenCV BGR numpy array.
    """
    # Strip the "data:image/...;base64," prefix if present
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    image_bytes = base64.b64decode(data_url)
    pil_image   = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    bgr_array   = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return bgr_array


def _extract_landmarks(results, indices: dict, img_h: int, img_w: int) -> dict:
    """
    Pull the required landmark positions from a MediaPipe result object.

    Parameters
    ----------
    results : MediaPipe FaceMesh results
    indices : dict with keys eyebrow, eye, iris, mouth → index or list of indices
    img_h, img_w : frame dimensions (used to normalise to [0-1])

    Returns
    -------
    dict with the same keys, values are {"x": float, "y": float} dicts
    (lists for multi-point features, single dict for iris).
    """
    lm = results.multi_face_landmarks[0].landmark

    def point(idx):
        return {"x": lm[idx].x, "y": lm[idx].y}

    return {
        "eyebrow": [point(i) for i in indices["eyebrow"]],
        "eye":     [point(i) for i in indices["eye"]],
        "iris":    point(indices["iris"]),
        "mouth":   [point(i) for i in indices["mouth"]],
    }


# ── Route ─────────────────────────────────────────────────────────────────────

@landmarks_bp.route("/api/process-frame", methods=["POST"])
def process_frame():
    """
    Main endpoint called by the frontend every animation frame.
    Decodes the webcam image, runs face mesh, and returns landmark data.
    """
    payload = request.get_json(silent=True)

    # ── Validate input ────────────────────────────────────────────────────
    if not payload:
        return jsonify({"success": False, "error": "No JSON body received"}), 400

    image_data = payload.get("image", "")
    side       = payload.get("side", "right").lower()

    if not image_data:
        return jsonify({"success": False, "error": "Missing 'image' field"}), 400

    if side not in ("left", "right"):
        return jsonify({"success": False, "error": "side must be 'left' or 'right'"}), 400

    # ── Decode & detect ───────────────────────────────────────────────────
    try:
        bgr_frame = _decode_image(image_data)
    except Exception as exc:
        return jsonify({"success": False, "error": f"Image decode failed: {exc}"}), 400

    img_h, img_w = bgr_frame.shape[:2]

    # MediaPipe expects RGB
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    results   = _face_mesh.process(rgb_frame)

    if not results.multi_face_landmarks:
        return jsonify({"success": False, "error": "No face detected"}), 200

    # ── Extract landmarks for the requested side ──────────────────────────
    try:
        landmarks = _extract_landmarks(
            results,
            _INDICES[side],
            img_h,
            img_w,
        )
    except Exception as exc:
        return jsonify({"success": False, "error": f"Landmark extraction failed: {exc}"}), 500

    return jsonify({
        "success":     True,
        "landmarks":   landmarks,
        "imageWidth":  img_w,
        "imageHeight": img_h,
    })