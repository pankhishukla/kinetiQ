"""
web/backend/routes/video_routes.py
====================================
REST endpoint for offline video analysis.

FLOW
────
POST /analyze-video  (multipart file upload)
  1. Save to a temp file
  2. Open with cv2.VideoCapture
  3. Process every Nth frame through the same YOLO11s → angle → form pipeline
  4. Return a JSON array of per-frame results (keypoints + joints + rep count)

The frontend plays the video locally and steps through the result array in sync
with video.currentTime, so no streaming is required.

WHY not WebSocket for video analysis?
    WebSocket is pull-based (browser sends frames).  For uploaded files the
    backend owns the video — it can decode frames far faster than real-time,
    so a single blocking HTTP request with a JSON response is simpler and
    avoids synchronisation complexity.
"""

import sys, os, time, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from web.backend.services.pose_service    import run_inference
from web.backend.services.angle_service  import compute_angles_and_feedback, update_rep_counter
from src.form_evaluator                  import RepCounter

router = APIRouter()

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# Process 1 out of every STRIDE frames to keep response size reasonable.
# At 30 fps camera, STRIDE=3 → ~10 analysis frames/second → smooth playback.
FRAME_STRIDE   = 3

# Hard cap: never return more than this many frames (prevents huge responses
# for long videos — ~5 minutes at 10 fps = 3000 frames).
MAX_FRAMES     = 3000

# YOLO confidence passed to pose_service (same as webcam pipeline).
INFER_CONF     = 0.35


# ---------------------------------------------------------------------------
# ENDPOINT
# ---------------------------------------------------------------------------

@router.post("/analyze-video")
async def analyze_video(
    file:     UploadFile = File(...),
    exercise: str        = Form(default="squat"),
):
    """
    Accept a video file and return per-frame pose analysis.

    Returns
    -------
    JSON object:
    {
      "exercise": "squat",
      "fps":       30.0,          # original video FPS
      "stride":    3,             # frames skipped between analyses
      "total_frames": 900,        # total frames in video
      "results": [
        {
          "frame_idx":  0,        # original frame index in video
          "time_s":     0.0,      # timestamp in seconds
          "detected":   true,
          "keypoints":  [...],    # 17-point list (same format as webcam WS)
          "joints":     {...},    # per-joint feedback
          "overall":    "correct",
          "issues":     0,
          "reps":       0,
        },
        ...
      ]
    }
    """

    # --- Validate MIME type (loose check) ---
    if file.content_type and not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video.")

    # Validate exercise name
    VALID = {"bicep_curl","squat","lateral_raise","push_up","lunge","plank"}
    if exercise not in VALID:
        raise HTTPException(status_code=400, detail=f"Unknown exercise: {exercise}")

    # --- Save upload to a temp file ---
    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)

        # --- Open with OpenCV ---
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=422, detail="Cannot decode video file.")

        video_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        rep_counter = RepCounter(exercise)
        results     = []
        frame_idx   = 0

        print(f"[VideoAnalysis] Processing '{file.filename}' | "
              f"{total_frames} frames @ {video_fps:.1f} fps | exercise={exercise}")

        t_start = time.perf_counter()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Only analyse every FRAME_STRIDE-th frame
            if frame_idx % FRAME_STRIDE == 0:
                time_s = frame_idx / video_fps

                keypoints = run_inference(frame, conf_threshold=INFER_CONF)

                if not keypoints:
                    results.append({
                        "frame_idx": frame_idx,
                        "time_s":    round(time_s, 4),
                        "detected":  False,
                        "keypoints": [],
                        "joints":    {},
                        "overall":   "unknown",
                        "issues":    0,
                        "reps":      rep_counter.count,
                    })
                else:
                    feedback = compute_angles_and_feedback(keypoints, exercise)
                    reps     = update_rep_counter(rep_counter, feedback["joints"])
                    results.append({
                        "frame_idx": frame_idx,
                        "time_s":    round(time_s, 4),
                        "detected":  True,
                        "keypoints": keypoints,
                        "joints":    feedback["joints"],
                        "overall":   feedback["overall"],
                        "issues":    feedback["issues"],
                        "reps":      reps,
                    })

                if len(results) >= MAX_FRAMES:
                    print(f"[VideoAnalysis] Hit MAX_FRAMES={MAX_FRAMES} cap, stopping early.")
                    break

            frame_idx += 1

        cap.release()
        elapsed = time.perf_counter() - t_start
        print(f"[VideoAnalysis] Done: {len(results)} analysis frames in {elapsed:.1f}s")

        return JSONResponse({
            "exercise":     exercise,
            "fps":          round(video_fps, 3),
            "stride":       FRAME_STRIDE,
            "total_frames": total_frames,
            "results":      results,
        })

    finally:
        # Always clean up the temp file
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
