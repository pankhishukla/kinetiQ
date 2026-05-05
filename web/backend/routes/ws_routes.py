"""
web/backend/routes/ws_routes.py
================================
WebSocket endpoint that is the heart of the real-time pipeline.

HOW THE FRAME PIPELINE WORKS
─────────────────────────────
Browser                          Backend (this file)
  │                                    │
  │── connect ws://localhost:8000/ws ──►│  handshake
  │                                    │
  │── binary JPEG frame ──────────────►│  receive_bytes()
  │                                    │  cv2.imdecode()  → BGR frame
  │                                    │  pose_service    → 17 keypoints
  │                                    │  angle_service   → angles + feedback
  │                                    │  rep_counter     → count
  │◄── JSON response ─────────────────│  send_json()
  │                                    │
  │  (repeat every ~30ms)             │
  │── text "EXERCISE:squat" ──────────►│  switch exercise, reset counter
  │── text "RESET" ───────────────────►│  reset rep counter
  │── disconnect ──────────────────────►│  cleanup

WHY WebSocket instead of REST polling?
    REST: browser asks → server answers → browser waits → repeat.
    At 30 fps that's 30 HTTP round-trips/second, each with TCP overhead.
    WebSocket: one persistent TCP connection, data flows both ways without
    the HTTP header overhead.  This halves latency at typical frame rates.

WHY binary frames (not base64)?
    base64 adds ~33% overhead.  A 640×480 JPEG at quality=70 is ~30 KB.
    base64 inflates it to ~40 KB.  At 30 fps that's 300 KB/s extra on
    localhost — not critical, but binary is strictly faster.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

import asyncio
import cv2
import numpy as np
import json
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from web.backend.services.pose_service import run_inference
from web.backend.services.angle_service import compute_angles_and_feedback, update_rep_counter

import importlib
sys.path.insert(0, str(ROOT))
from src.form_evaluator import RepCounter

router = APIRouter()

# Valid exercise names (must match src/angle_engine.py ANGLE_DEFINITIONS keys)
VALID_EXERCISES = {
    "bicep_curl", "squat", "lateral_raise",
    "push_up", "shoulder_press", "lunge", "plank",
}


async def frame_processor(frame_queue: asyncio.Queue, websocket: WebSocket, state: dict):
    """
    Pulls frames from the queue and processes them. Runs as a separate
    background task. This is the "consumer" part of the pipeline.
    """
    while True:
        try:
            raw_bytes = await frame_queue.get()
            if raw_bytes is None:  # Sentinel for shutdown
                break

            # Decode JPEG → BGR numpy array
            nparr = np.frombuffer(raw_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                await websocket.send_json({"type": "error", "msg": "Bad frame"})
                continue

            t0 = time.perf_counter()

            # --- INFERENCE PIPELINE ---
            keypoints = run_inference(frame)

            if not keypoints:
                await websocket.send_json({
                    "type": "frame", "detected": False, "exercise": state["exercise"],
                    "reps": state["rep_counter"].count, "keypoints": [], "joints": {},
                    "overall": "unknown", "issues": 0, "fps": state["fps"],
                })
                continue

            feedback = compute_angles_and_feedback(keypoints, state["exercise"])
            reps = update_rep_counter(state["rep_counter"], feedback["joints"])

            t1 = time.perf_counter()
            inf_ms = round((t1 - t0) * 1000, 1)

            state["frame_count"] += 1
            elapsed = time.time() - state["t_start"]
            state["fps"] = round(state["frame_count"] / elapsed, 1) if elapsed > 0 else 0

            # --- SEND JSON RESPONSE ---
            await websocket.send_json({
                "type": "frame", "detected": True, "exercise": state["exercise"],
                "reps": reps, "keypoints": keypoints, "joints": feedback["joints"],
                "overall": feedback["overall"], "issues": feedback["issues"],
                "fps": state["fps"], "inf_ms": inf_ms,
            })

        except Exception as e:
            print(f"[Processor] Error: {e}")
            # Don't crash the whole processor for one bad frame
            continue


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Single WebSocket endpoint serving one browser client. This is the "producer"
    part of the pipeline, which receives frames and puts them on the queue.
    """
    await websocket.accept()
    print("[WS] Client connected")

    # State dictionary to be shared with the processor task
    state = {
        "exercise": "squat",
        "rep_counter": RepCounter("squat"),
        "frame_count": 0,
        "t_start": time.time(),
        "fps": 0,
    }

    # Queue with a max size of 1: acts as a buffer for one pending frame,
    # preventing a large backlog while avoiding dropping too many frames.
    frame_queue = asyncio.Queue(maxsize=1)
    processor_task = asyncio.create_task(frame_processor(frame_queue, websocket, state))

    try:
        while True:
            message = await websocket.receive()

            if "text" in message:
                text = message["text"].strip().upper()
                if text.startswith("EXERCISE:"):
                    exercise = text.split(":", 1)[1].lower()
                    if exercise in VALID_EXERCISES:
                        state["exercise"] = exercise
                        state["rep_counter"] = RepCounter(exercise)
                        print(f"[WS] Exercise -> {state['exercise']}")
                    await websocket.send_json({"type": "ack", "exercise": state["exercise"]})
                elif text == "RESET":
                    state["rep_counter"].reset()
                    await websocket.send_json({"type": "ack", "reps": 0})
                continue

            if "bytes" in message:
                try:
                    # Non-blocking put: if queue is full, drop the frame.
                    frame_queue.put_nowait(message["bytes"])
                except asyncio.QueueFull:
                    # This is expected if the frontend is faster than the backend.
                    # Silently drop the frame to prevent lag.
                    pass

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")
        try:
            await websocket.send_json({"type": "error", "msg": str(e)})
        except Exception:
            pass
    finally:
        # Cleanly shut down the processor task
        if not processor_task.done():
            await frame_queue.put(None)  # Send sentinel to stop the loop
            processor_task.cancel()
            try:
                await processor_task
            except asyncio.CancelledError:
                print("[WS] Processor task cancelled.")
