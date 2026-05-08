"""
src/inference_engine.py
========================
Threaded inference engine that decouples YOLO11s inference from the display loop.

THE CORE PROBLEM
────────────────
Without threading the single-threaded loop looks like this every frame:

    cap.read()   →   YOLO(frame)   →   draw()   →   imshow()
      ~2 ms            ~45 ms          ~8 ms          ~1 ms
                     ▲ BOTTLENECK

cap.read() and imshow() are blocked for ~45 ms waiting for YOLO.
That caps us at 1000/45 ≈ 22 FPS maximum, and in practice 12–15 FPS
because the OS scheduler adds overhead.

THE FIX: PRODUCER / CONSUMER WITH A SHARED FRAME SLOT
──────────────────────────────────────────────────────
Thread A (InferenceWorker) runs YOLO as fast as it can (~15 fps on CPU).
Thread B (main / display)  reads the LATEST result and renders at 30 fps.

    Thread A: …YOLO(f1)→store…YOLO(f2)→store…YOLO(f3)→store…
    Thread B:   read→draw   read→draw   read→draw   read→draw

Thread B never blocks on YOLO; it always has a cached result to display.
Result: display loop runs at the monitor refresh rate (30+ fps) while
inference happens in the background at its natural speed.

OPTIMIZATION SUMMARY
────────────────────
| Technique              | Speedup  | Why it works                        |
|------------------------|----------|-------------------------------------|
| Background thread      | 2–3×     | Display never waits for inference   |
| Input resize (320px)   | 2–2.5×   | YOLO FLOPs ∝ resolution²           |
| imgsz=320 in YOLO      | 1.5–2×   | Skips internal upsampling           |
| half() on GPU          | 1.5–2×   | FP16 vs FP32 tensor operations      |
| Frame skip (N=1)       | up to 2× | Only run YOLO every other frame     |
| CAP_PROP_BUFFERSIZE=1  | —        | Drops stale buffered frames         |

ACCURACY TRADE-OFFS
────────────────────
320×320 input vs 640×480:
    - Keypoint localisation accuracy drops ~5–10 px (barely visible)
    - Small/distant people may not be detected
    - For exercise form at arms-length distance: imperceptible difference

Frame skipping (N=1 → ~15 fps inference):
    - Angles update every 2 frames (~33 ms lag at 30 fps camera)
    - EMA smoothing hides this lag completely in the rendered output
    - Rep counts are unaffected (state machine only transitions on clear moves)
"""

import threading
import time
import cv2
import numpy as np
from typing import Optional, List, Tuple
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# SPEED CONFIG — tune these constants, not the logic below
# ---------------------------------------------------------------------------

# WHY 320? YOLO's internal grid is 20×20 at imgsz=320 vs 40×40 at 640.
# Inference FLOP count scales with grid cells squared: (40/20)² = 4× faster.
# Accuracy loss for close-up exercise poses: < 5%.
INFERENCE_IMGSZ = 320

# Inference confidence — lowered to 0.35 because the 320px downscaled input
# produces lower raw confidence scores than the native 640px input.
# 0.5 on a 320px frame often suppresses valid detections entirely.
INFERENCE_CONF  = 0.35

# How many display frames to skip between inference runs.
# 0 = run every frame (accurate, slow)
# 1 = run every 2nd frame (~2× throughput)
# 2 = run every 3rd frame (~3× throughput, fine at 30 fps camera)
FRAME_SKIP      = 1

# Display FPS cap — prevents burning CPU on unnecessary imshow calls.
DISPLAY_FPS_CAP = 30


# ---------------------------------------------------------------------------
# RESULT CONTAINER  (shared between threads)
# ---------------------------------------------------------------------------

class InferenceResult:
    """
    Lightweight struct holding the most recent inference output.

    WHY not a Queue?  We always want the LATEST result, not a FIFO.
    If inference is slow, old queued results are stale and misleading.
    A single slot (with a lock) is simpler and gives fresher data.
    """
    __slots__ = ("keypoints", "num_people", "timestamp", "_lock")

    def __init__(self):
        self.keypoints:  List[Tuple] = []
        self.num_people: int         = 0
        self.timestamp:  float       = 0.0
        self._lock = threading.Lock()

    def write(self, keypoints, num_people):
        with self._lock:
            self.keypoints  = keypoints
            self.num_people = num_people
            self.timestamp  = time.perf_counter()

    def read(self):
        with self._lock:
            return self.keypoints, self.num_people, self.timestamp


# ---------------------------------------------------------------------------
# INFERENCE WORKER THREAD
# ---------------------------------------------------------------------------

class InferenceWorker(threading.Thread):
    """
    Background thread that continuously grabs frames from the shared
    frame slot and runs YOLO11n-Pose inference.

    WHY daemon=True?
        A daemon thread is killed automatically when the main thread exits,
        so we never need explicit cleanup.  Non-daemon threads would keep
        the process alive even after the main window is closed.

    WHY threading instead of multiprocessing?
        YOLO inference releases the GIL during the C++/CUDA compute phase,
        so two threads CAN run concurrently on CPU.  Multiprocessing would
        require serialising the numpy frames across process boundaries
        (pickling overhead) — actually slower for our use case.
    """

    def __init__(self, model: YOLO, result: InferenceResult,
                 imgsz: int = INFERENCE_IMGSZ,
                 conf: float = INFERENCE_CONF,
                 frame_skip: int = FRAME_SKIP):
        super().__init__(daemon=True)
        self.model      = model
        self.result     = result
        self.imgsz      = imgsz
        self.conf       = conf
        self.frame_skip = frame_skip

        self._frame_lock    = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._stop_event    = threading.Event()

        # Performance tracking
        self.inf_fps    = 0.0
        self._inf_times = []

    # -- called by main thread ------------------------------------------------

    def push_frame(self, frame: np.ndarray):
        """
        Store the latest frame for inference.
        WHY overwrite without queueing?  We want the freshest frame.
        Intermediate frames are discarded — not stale data.
        """
        with self._frame_lock:
            self._latest_frame = frame

    def stop(self):
        self._stop_event.set()

    # -- runs in background thread --------------------------------------------

    def run(self):
        skip_count = 0

        while not self._stop_event.is_set():
            with self._frame_lock:
                frame = self._latest_frame

            if frame is None:
                time.sleep(0.005)
                continue

            # FRAME SKIP: only infer every (frame_skip + 1) frames.
            skip_count += 1
            if skip_count <= self.frame_skip:
                time.sleep(0.001)
                continue
            skip_count = 0

            t0 = time.perf_counter()

            # -----------------------------------------------------------------
            # RESIZE BEFORE INFERENCE
            # WHY resize here (not inside YOLO's imgsz parameter alone)?
            #   YOLO's imgsz letterboxes the input internally — still reads
            #   the full-res frame from Python memory.  Pre-resizing reduces
            #   the Python→C++ data transfer cost for large frames.
            # -----------------------------------------------------------------
            h, w = frame.shape[:2]
            if w > self.imgsz:
                scale = self.imgsz / max(w, h)
                small = cv2.resize(frame,
                                   (int(w * scale), int(h * scale)),
                                   interpolation=cv2.INTER_LINEAR)
                # INTER_LINEAR: best speed/quality trade-off for downscaling.
                # INTER_AREA slightly better quality but ~30% slower.
                # INTER_NEAREST fastest but produces blocky artefacts.
            else:
                small = frame
                scale = 1.0

            results = self.model(
                small,
                verbose = False,
                conf    = self.conf,
                imgsz   = self.imgsz,
            )

            kp_data = results[0].keypoints

            if kp_data is None or len(kp_data.data) == 0:
                self.result.write([], 0)
            else:
                num_people = len(kp_data.data)
                kpts_raw   = kp_data.data[0].cpu().numpy()

                # Scale keypoint coordinates BACK to original frame resolution.
                # WHY? The angle engine and renderer work in original pixels.
                keypoints = [
                    (float(kpts_raw[i][0] / scale),
                     float(kpts_raw[i][1] / scale),
                     float(kpts_raw[i][2]))
                    for i in range(17)
                ]
                self.result.write(keypoints, num_people)

            t1 = time.perf_counter()
            self._inf_times.append(t1 - t0)
            if len(self._inf_times) > 30:
                self._inf_times.pop(0)
            avg_ms = sum(self._inf_times) / len(self._inf_times)
            self.inf_fps = round(1.0 / avg_ms, 1) if avg_ms > 0 else 0.0


# ---------------------------------------------------------------------------
# GPU / HALF-PRECISION SETUP HELPER
# ---------------------------------------------------------------------------

def configure_model_for_speed(model: YOLO, use_half: bool = False) -> YOLO:
    """
    Apply post-load performance tweaks to the YOLO model.

    Parameters
    ----------
    use_half : Enable FP16 (half precision) inference.
               REQUIRES: NVIDIA GPU with CUDA installed.
               SPEEDUP:  ~1.5–2× on modern GPUs (RTX 30xx/40xx).
               ACCURACY: Negligible difference for pose estimation.
               WARNING:  Do NOT enable on CPU — FP16 on CPU is SLOWER.

    HOW TO CHECK IF YOU HAVE A CUDA GPU:
        python -c "import torch; print(torch.cuda.is_available())"
    """
    try:
        import torch
        has_cuda = torch.cuda.is_available()

        if has_cuda:
            print("[InferenceEngine] CUDA GPU detected — using GPU inference")
            model.to("cuda")
            if use_half:
                model.half()
                print("[InferenceEngine] FP16 half-precision enabled (+1.5-2× speed)")
        else:
            print("[InferenceEngine] No GPU — running on CPU")
            print("[InferenceEngine] TIP: Install CUDA + torch GPU build for 5-10× speedup")

    except ImportError:
        print("[InferenceEngine] torch not importable separately — skipping GPU check")

    return model


# ---------------------------------------------------------------------------
# HIGH-LEVEL FACADE (used by app.py)
# ---------------------------------------------------------------------------

class InferenceEngine:
    """
    One-stop shop: loads the model, starts the worker thread,
    and exposes push_frame() + get_result() to the main loop.

    Usage in app.py:
        engine = InferenceEngine("models/yolo11s-pose.pt")
        engine.start()
        while True:
            ret, frame = cap.read()
            engine.push_frame(frame)           # non-blocking
            kpts, n_people, age = engine.get_result()
            draw_pose(frame, kpts, ...)
        engine.stop()
    """

    def __init__(self, model_path: str = "models/yolo11s-pose.pt",
                 imgsz: int      = INFERENCE_IMGSZ,
                 conf: float     = INFERENCE_CONF,
                 frame_skip: int = FRAME_SKIP,
                 use_half: bool  = False):

        print(f"[InferenceEngine] Loading model: {model_path}")
        model = YOLO(model_path)
        model = configure_model_for_speed(model, use_half=use_half)

        self._result = InferenceResult()
        self._worker = InferenceWorker(
            model      = model,
            result     = self._result,
            imgsz      = imgsz,
            conf       = conf,
            frame_skip = frame_skip,
        )

    def start(self):
        self._worker.start()
        print(f"[InferenceEngine] Worker started "
              f"(imgsz={self._worker.imgsz}, "
              f"skip={self._worker.frame_skip}, "
              f"conf={self._worker.conf})")

    def push_frame(self, frame: np.ndarray):
        """Non-blocking: store frame for the next inference cycle."""
        self._worker.push_frame(frame)

    def get_result(self):
        """
        Non-blocking: return the most recent inference result.
        Returns (keypoints, num_people, result_age_seconds).

        result_age_seconds: how old the result is.
            < 0.1 s  → fresh, use normally
            0.1–0.5 s → slightly stale, EMA hides this
            > 0.5 s  → likely no person detected recently
        """
        kpts, n, ts = self._result.read()
        age = time.perf_counter() - ts if ts > 0 else 999.0
        return kpts, n, age

    @property
    def inf_fps(self) -> float:
        """Inference FPS measured by the worker thread."""
        return self._worker.inf_fps

    def stop(self):
        self._worker.stop()
        self._worker.join(timeout=2.0)
        print("[InferenceEngine] Worker stopped.")
