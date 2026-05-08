"""
web/backend/services/pose_service.py
=====================================
Singleton service that owns the YOLO11s-Pose model and exposes one clean
method: run_inference(frame) -> list of 17 (x, y, conf) tuples.

WHY a singleton service?
    Loading YOLO11s takes ~1-2 seconds and ~200 MB of RAM.  Loading it once at
    startup and reusing for every WebSocket frame keeps latency under 30 ms.

WHY a separate service module?
    The FastAPI route (ws_routes.py) should handle HTTP/WS concerns only.
    Keeping model I/O here means the route stays thin and this service is
    independently testable (you can import and call it without a server).
"""

import sys
from pathlib import Path

# Allow importing from project root when running as a module
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# MODEL — loaded once at import time
# ---------------------------------------------------------------------------
_MODEL_PATH = ROOT / "models" / "yolo11s-pose.pt"
_model: YOLO = None   # lazy-loaded on first call


def _get_model() -> YOLO:
    """Lazy singleton loader — safe to call from any thread."""
    global _model
    if _model is None:
        print(f"[PoseService] Loading YOLO model from {_MODEL_PATH} ...")
        _model = YOLO(str(_MODEL_PATH))
        print("[PoseService] Model loaded.")
    return _model


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def run_inference(frame: np.ndarray, conf_threshold: float = 0.5) -> list:
    """
    Run YOLO11s-Pose on one BGR frame and return keypoints for the first
    detected person.

    Parameters
    ----------
    frame          : BGR numpy array from cv2 / decoded JPEG
    conf_threshold : discard keypoints below this confidence

    Returns
    -------
    List of 17 dicts:
        [{
            "x": float,         # pixel x (relative to frame width)
            "y": float,         # pixel y (relative to frame height)
            "xn": float,        # normalised x in [0, 1]
            "yn": float,        # normalised y in [0, 1]
            "conf": float,      # YOLO11s keypoint confidence
            "visible": bool,    # conf >= conf_threshold
        }, ...]
    or [] if no person detected.

    WHY return both pixel and normalised coordinates?
        The backend uses pixel coords for angle math (needs real distances).
        The frontend uses normalised coords to scale to any canvas size.
    """
    model  = _get_model()
    h, w   = frame.shape[:2]
    result = model(frame, verbose=False, conf=0.5)
    kp_data = result[0].keypoints

    if kp_data is None or len(kp_data.data) == 0:
        return []

    # Only process the first detected person for now
    kpts = kp_data.data[0].cpu().numpy()   # shape (17, 3): x, y, conf

    keypoints = []
    for i in range(17):
        x, y, conf = float(kpts[i][0]), float(kpts[i][1]), float(kpts[i][2])
        keypoints.append({
            "x":       x,
            "y":       y,
            "xn":      x / w,   # normalised for frontend canvas scaling
            "yn":      y / h,
            "conf":    round(conf, 3),
            "visible": conf >= conf_threshold,
        })

    return keypoints
