"""
src/keypoint_smoother.py
========================
Confidence-weighted temporal smoother for YOLO pose keypoints.

WHY this exists:
    YOLO11s-Pose outputs raw keypoints every frame.  When a joint is briefly
    occluded, motion-blurred, or at the edge of the frame, the confidence
    score drops but YOLO still outputs a position — often a jittery, incorrect
    one.  Without smoothing this causes:
      - False "incorrect" posture alerts on good reps
      - Joint dots flickering red/green rapidly
      - Angle calculations corrupted by one bad frame

HOW it works (three tiers):
    HIGH   (conf ≥ HIGH_CONF=0.65)  → pass through normally, update EMA
    MEDIUM (conf ≥ MED_CONF=0.35)   → blend with EMA, reduce weight
    LOW    (conf <  MED_CONF=0.35)  → substitute last stable EMA for up to
                                       MAX_GHOST_FRAMES frames, mark as "ghost"

The smoother is stateful — instantiate ONE per session/video, not per frame.
"""

import numpy as np

# ---------------------------------------------------------------------------
# CONFIDENCE TIERS
# ---------------------------------------------------------------------------
HIGH_CONF       = 0.65   # fully trust; update EMA aggressively
MED_CONF        = 0.35   # blend with history; mark "medium"
MAX_GHOST_FRAMES = 6     # how long to carry a low-conf joint before giving up

# EMA blend factor α: higher = faster response, lower = smoother
# α=0.4 means 40% new value, 60% history each frame.
EMA_ALPHA_HIGH  = 0.45
EMA_ALPHA_MED   = 0.20   # slower update for shaky medium-conf detections


class KeypointSmoother:
    """
    Stateful per-session keypoint smoother.

    Usage
    -----
    smoother = KeypointSmoother()
    for frame in video:
        keypoints = run_inference(frame)          # raw YOLO output (list of dicts)
        smoothed  = smoother.update(keypoints)    # returns same format but stabilised
        feedback  = compute_angles_and_feedback(smoothed, exercise, phase)
    """

    def __init__(self):
        # EMA state: {kp_index: [x_ema, y_ema]}
        self._ema: dict[int, list] = {}
        # Last known HIGH-confidence position for ghost substitution
        self._last_stable: dict[int, list] = {}   # {idx: [x, y]}
        # Ghost frame counter: {idx: int} — incremented each frame the kp is LOW
        self._ghost_count: dict[int, int] = {}
        # Per-keypoint confidence tier from last update: {idx: "high"|"medium"|"low"}
        self._conf_tier: dict[int, str] = {}

    # -----------------------------------------------------------------------
    # PUBLIC
    # -----------------------------------------------------------------------

    def update(self, keypoints: list) -> list:
        """
        Smooth a list of 17 keypoint dicts (from pose_service.run_inference).

        Returns a new list with the same dict structure but with:
          - x, y replaced by EMA-stabilised positions
          - conf unchanged (original YOLO confidence)
          - conf_tier added: "high" | "medium" | "low"
          - ghosted: bool — True if using a previous-frame position
        """
        if not keypoints:
            return keypoints

        smoothed = []
        for idx, kp in enumerate(keypoints):
            x_raw  = kp["x"]
            y_raw  = kp["y"]
            conf   = kp["conf"]

            tier, x_out, y_out, ghosted = self._smooth_kp(idx, x_raw, y_raw, conf)

            new_kp = dict(kp)   # shallow copy — preserve xn, yn, visible, etc.
            new_kp["x"]          = x_out
            new_kp["y"]          = y_out
            new_kp["conf_tier"]  = tier
            new_kp["ghosted"]    = ghosted
            smoothed.append(new_kp)

        return smoothed

    def reset(self):
        """Clear all state (call when switching exercises or resetting a session)."""
        self._ema.clear()
        self._last_stable.clear()
        self._ghost_count.clear()
        self._conf_tier.clear()

    # -----------------------------------------------------------------------
    # PRIVATE
    # -----------------------------------------------------------------------

    def _smooth_kp(self, idx: int, x: float, y: float, conf: float):
        """
        Apply EMA smoothing to a single keypoint.

        Returns (tier, x_out, y_out, ghosted)
        """
        ghosted = False

        if conf >= HIGH_CONF:
            # ----- HIGH confidence -----
            tier = "high"
            self._ghost_count[idx] = 0

            if idx in self._ema:
                alpha = EMA_ALPHA_HIGH
                x_ema = alpha * x + (1 - alpha) * self._ema[idx][0]
                y_ema = alpha * y + (1 - alpha) * self._ema[idx][1]
            else:
                x_ema, y_ema = x, y

            self._ema[idx]         = [x_ema, y_ema]
            self._last_stable[idx] = [x_ema, y_ema]
            return tier, x_ema, y_ema, ghosted

        elif conf >= MED_CONF:
            # ----- MEDIUM confidence — blend slowly -----
            tier = "medium"
            self._ghost_count[idx] = 0

            if idx in self._ema:
                alpha = EMA_ALPHA_MED
                x_ema = alpha * x + (1 - alpha) * self._ema[idx][0]
                y_ema = alpha * y + (1 - alpha) * self._ema[idx][1]
            else:
                x_ema, y_ema = x, y

            self._ema[idx] = [x_ema, y_ema]
            # Don't update _last_stable — medium conf isn't reliable enough
            return tier, x_ema, y_ema, ghosted

        else:
            # ----- LOW confidence -----
            tier = "low"
            self._ghost_count[idx] = self._ghost_count.get(idx, 0) + 1

            if self._ghost_count[idx] <= MAX_GHOST_FRAMES and idx in self._last_stable:
                # Substitute last known stable position (ghost frame)
                x_out, y_out = self._last_stable[idx]
                ghosted = True
                return tier, x_out, y_out, ghosted
            else:
                # Exhausted ghost budget — return raw (will produce angle=None downstream)
                return tier, x, y, ghosted
