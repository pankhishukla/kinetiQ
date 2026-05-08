"""
src/renderer.py
===============
Dedicated pose visualization module.  All drawing logic lives here so that
app.py stays clean and every visual tweak has exactly ONE place to edit.

DESIGN PHILOSOPHY
-----------------
1. Color-coded anatomy : The eye instantly groups body parts by color, making
   it faster to spot which limb has bad form — you don't need to read a label.

2. Layered rendering order :
       background frame
         └─ glow / shadow blur  (depth cue, improves dark-background contrast)
             └─ skeleton lines  (structural)
                 └─ joint circles (focal points on top)
   Painting in this order means joints always appear "in front of" bones.

3. EMA keypoint smoothing :  Raw YOLOv8 keypoints jitter ±3–8 px even on a
   still person.  Applying EMA here (before drawing) eliminates the flicker
   WITHOUT delaying angle math — angles are computed from raw keypoints, the
   smoothing is purely cosmetic.

4. Confidence-based opacity :  A joint seen clearly (conf ≈ 1.0) gets a solid
   bright dot; an occluded joint (conf ≈ 0.5) gets a faded one.  This lets
   you immediately know how much to trust any given keypoint.

5. Trail / ghost effect :  Storing the last N keypoint positions and drawing
   faded copies gives velocity cues at zero computational cost beyond a deque.

6. Glow effect :  We draw an oversized blurred circle, then the sharp circle
   on top.  OpenCV has no native "glow" primitive, so the blur simulates it.
   We use a temporary layer + addWeighted to composite without modifying the
   background pixel values destructively.
"""

import cv2
import numpy as np
from collections import deque
from typing import List, Tuple, Optional, Dict

# ---------------------------------------------------------------------------
# SECTION 1 — COLOR PALETTE
# ---------------------------------------------------------------------------
# WHY BGR not RGB?  OpenCV uses BGR natively.  Storing constants in BGR means
# we never have to flip channels, which would add invisible bugs if anyone
# passed them to cv2 functions expecting RGB.

class Colors:
    # Keypoint groups — body-part semantics at a glance
    FACE        = (200, 200, 200)   # subtle light grey — face landmarks matter least
    LEFT_BODY   = (255, 100,  30)   # vibrant orange-blue (BGR: blue=255)
    RIGHT_BODY  = ( 30, 100, 255)   # vibrant red-orange (BGR: red=255)
    CENTER_BODY = (200, 255, 100)   # yellow-green for midline (pelvis, nose-bridge)

    # Skeleton line groups
    LEFT_BONE   = (220,  80,  30)   # cool blue for left-side limbs
    RIGHT_BONE  = ( 30,  80, 220)   # warm red for right-side limbs
    TORSO_BONE  = (100, 200, 100)   # neutral green for torso connections

    # Form feedback colors (match form_evaluator.py)
    CORRECT     = (  0, 220,   0)   # green
    INCORRECT   = (  0,  30, 220)   # red
    UNKNOWN     = (150, 150, 150)   # grey

    # Glow  — slightly de-saturated version of the base color mixed with white
    @staticmethod
    def glow(bgr: Tuple[int,int,int], factor: float = 0.55) -> Tuple[int,int,int]:
        """
        Return a lighter, washed-out version of *bgr* for glow rendering.
        WHY not just pure white? A white glow looks generic; a tinted glow
        keeps the color identity of each body-part group visible.
        """
        b, g, r = bgr
        gb = int(b + (255 - b) * factor)
        gg = int(g + (255 - g) * factor)
        gr = int(r + (255 - r) * factor)
        return (gb, gg, gr)


# ---------------------------------------------------------------------------
# SECTION 2 — KEYPOINT GROUP ASSIGNMENTS
# ---------------------------------------------------------------------------
# Assign each of the 17 COCO keypoints to a body-part group.
# Index → group string → color looked up at draw time.
#
# WHY a lookup table instead of if/elif chains?
#   Tables are O(1), readable, and easy to change for one joint without
#   accidentally breaking the logic for another.

KP_GROUP = [
    "face",   # 0  nose
    "face",   # 1  left_eye
    "face",   # 2  right_eye
    "face",   # 3  left_ear
    "face",   # 4  right_ear
    "left",   # 5  left_shoulder
    "right",  # 6  right_shoulder
    "left",   # 7  left_elbow
    "right",  # 8  right_elbow
    "left",   # 9  left_wrist
    "right",  # 10 right_wrist
    "left",   # 11 left_hip
    "right",  # 12 right_hip
    "left",   # 13 left_knee
    "right",  # 14 right_knee
    "left",   # 15 left_ankle
    "right",  # 16 right_ankle
]

_GROUP_COLOR = {
    "face":   Colors.FACE,
    "left":   Colors.LEFT_BODY,
    "right":  Colors.RIGHT_BODY,
    "center": Colors.CENTER_BODY,
}

# Each skeleton connection also has a side
# (a_idx, b_idx) → side string
_BONE_SIDE = {
    (5,  6):  "torso",   # shoulder bar
    (5,  7):  "left",    # L upper arm
    (7,  9):  "left",    # L forearm
    (6,  8):  "right",   # R upper arm
    (8, 10):  "right",   # R forearm
    (5, 11):  "left",    # L torso
    (6, 12):  "right",   # R torso
    (11,12):  "torso",   # pelvis bar
    (11,13):  "left",    # L thigh
    (13,15):  "left",    # L shin
    (12,14):  "right",   # R thigh
    (14,16):  "right",   # R shin
}

_BONE_COLOR = {
    "left":  Colors.LEFT_BONE,
    "right": Colors.RIGHT_BONE,
    "torso": Colors.TORSO_BONE,
}


# ---------------------------------------------------------------------------
# SECTION 3 — ACTIVE JOINTS PER EXERCISE
# ---------------------------------------------------------------------------
# These are the COCO indices of the joints that drive form evaluation for each
# exercise.  The renderer highlights them with a larger pulsing ring.
#
# WHY visual highlighting?
#   When doing a bicep curl the user naturally watches their arm, but our
#   form system monitors the elbow angle.  Highlighting the elbow immediately
#   trains the user to pay attention to the right joint.

ACTIVE_JOINTS = {
    "bicep_curl":     [7, 8, 9, 10],          # elbows + wrists
    "squat":          [11, 12, 13, 14],        # hips + knees
    "lateral_raise":  [5, 6, 7, 8],           # shoulders + elbows
    "push_up":        [7, 8, 11, 12],          # elbows + hips
    "shoulder_press": [5, 6, 7, 8],           # shoulders + elbows
    "lunge":          [11, 12, 13, 14, 15, 16],# hips + knees + ankles
    "plank":          [11, 12, 13, 14, 15, 16],# full lower chain
}


# ---------------------------------------------------------------------------
# SECTION 4 — EMA KEYPOINT SMOOTHER
# ---------------------------------------------------------------------------
# WHY smooth keypoints for DISPLAY but not for angles?
#   Angle math needs the most up-to-date position for responsive rep counting.
#   Display can tolerate 1–2 frames of lag to look smooth.  Separating these
#   two concerns gives us both accuracy AND visual stability.

class KeypointSmoother:
    """
    Per-person EMA smoother for 17 (x, y) keypoint positions.

    alpha : blending weight for new observation.
            Lower = smoother but more lag.  0.4 is a good real-time balance.
    """
    def __init__(self, n_keypoints: int = 17, alpha: float = 0.40):
        self.alpha  = alpha
        self.smooth = None   # shape (17, 2) once initialised

    def update(self, keypoints: List[Tuple[float, float, float]]) -> List[Tuple]:
        """
        Return smoothed (x, y, conf) tuples.
        Only x, y are smoothed; confidence is passed through unchanged because
        it's a per-frame model output, not a measurement to be averaged.
        """
        raw_xy = np.array([[kp[0], kp[1]] for kp in keypoints], dtype=np.float32)
        confs  = [kp[2] for kp in keypoints]

        if self.smooth is None:
            self.smooth = raw_xy.copy()
        else:
            # EMA: new = alpha * raw + (1-alpha) * previous
            self.smooth = self.alpha * raw_xy + (1 - self.alpha) * self.smooth

        return [(self.smooth[i, 0], self.smooth[i, 1], confs[i])
                for i in range(len(keypoints))]

    def reset(self):
        self.smooth = None


# ---------------------------------------------------------------------------
# SECTION 5 — TRAIL BUFFER
# ---------------------------------------------------------------------------

class TrailBuffer:
    """
    Stores the last *max_frames* keypoint arrays for ghost / trail rendering.

    WHY a deque?  deque(maxlen=N) auto-discards the oldest frame when a new
    one is appended — O(1) vs list.pop(0) which is O(N).
    """
    def __init__(self, max_frames: int = 5):
        self.buffer = deque(maxlen=max_frames)

    def push(self, keypoints: List[Tuple]):
        self.buffer.appendleft(keypoints)   # newest at index 0

    def frames(self) -> List[List[Tuple]]:
        return list(self.buffer)

    def reset(self):
        self.buffer.clear()


# ---------------------------------------------------------------------------
# SECTION 6 — CORE DRAWING PRIMITIVES
# ---------------------------------------------------------------------------

def _draw_glow_circle(frame: np.ndarray,
                      center: Tuple[int, int],
                      radius: int,
                      color: Tuple[int, int, int],
                      intensity: float = 0.6) -> None:
    """
    Simulate a soft glow around a circle by drawing a blurred, larger disc
    on a temporary layer and blending it into the frame.

    WHY a temporary layer?
        Drawing the glow directly on the frame would corrupt the pixel values
        of adjacent joints/bones.  A temp layer lets us composite cleanly.

    Parameters
    ----------
    intensity : 0.0–1.0  weight of the glow layer blended into frame
    """
    # Only allocate the tiny bounding region, not the entire frame
    glow_r  = radius * 3
    x0 = max(0, center[0] - glow_r)
    y0 = max(0, center[1] - glow_r)
    x1 = min(frame.shape[1], center[0] + glow_r)
    y1 = min(frame.shape[0], center[1] + glow_r)
    if x1 <= x0 or y1 <= y0:
        return

    roi      = frame[y0:y1, x0:x1]
    glow_img = np.zeros_like(roi)
    lc       = (center[0] - x0, center[1] - y0)

    glow_color = Colors.glow(color, 0.5)
    cv2.circle(glow_img, lc, glow_r, glow_color, -1, cv2.LINE_AA)

    # Gaussian blur creates the soft falloff — kernel size must be odd
    ksize = glow_r * 2 + 1
    glow_img = cv2.GaussianBlur(glow_img, (ksize, ksize), 0)

    frame[y0:y1, x0:x1] = cv2.addWeighted(glow_img, intensity, roi, 1.0, 0)


def _draw_aa_circle(frame: np.ndarray,
                    center: Tuple[int, int],
                    radius: int,
                    color: Tuple[int, int, int],
                    thickness: int = -1) -> None:
    """
    Draw a premium aesthetic keypoint with a neon style.
    Features an outer halo, a dark spacer, a vibrant main body, and a bright white core.
    """
    if radius <= 0:
        return

    # 1. Subtle Outer Halo
    cv2.circle(frame, center, radius + 4, color, 1, cv2.LINE_AA)
    
    # 2. Dark spacer ring to separate the halo from the body
    cv2.circle(frame, center, radius + 2, (20, 20, 20), 2, cv2.LINE_AA)
    
    # 3. Main vibrant body
    cv2.circle(frame, center, radius, color, thickness, cv2.LINE_AA)
    
    # 4. Bright inner core (Neon effect)
    if radius >= 4:
        # Mix the base color heavily with white
        core_color = (int(color[0]*0.3 + 255*0.7), 
                      int(color[1]*0.3 + 255*0.7), 
                      int(color[2]*0.3 + 255*0.7))
        cv2.circle(frame, center, max(1, int(radius * 0.4)), core_color, -1, cv2.LINE_AA)


def _draw_aa_line(frame: np.ndarray,
                  pt1: Tuple[int, int],
                  pt2: Tuple[int, int],
                  color: Tuple[int, int, int],
                  thickness: int,
                  shadow: bool = True) -> None:
    """
    Draw a premium neon-tube bone connection.
    Consists of a thick translucent-looking outer aura, a vibrant main colored band,
    and a bright white/tinted inner core.
    """
    if shadow:
        # Deep shadow offset for a 3D lift off the background
        cv2.line(frame, (pt1[0]+2, pt1[1]+2), (pt2[0]+2, pt2[1]+2),
                 (10, 10, 10), thickness + 4, cv2.LINE_AA)
                 
    # 1. Outer Glow Aura (Thickest, darkest version of the color)
    aura_color = (int(color[0]*0.4), int(color[1]*0.4), int(color[2]*0.4))
    cv2.line(frame, pt1, pt2, aura_color, thickness + 6, cv2.LINE_AA)
    
    # 2. Main color body
    cv2.line(frame, pt1, pt2, color, thickness + 2, cv2.LINE_AA)
    
    # 3. Bright inner core (Neon effect)
    core_color = (int(color[0]*0.4 + 255*0.6), 
                  int(color[1]*0.4 + 255*0.6), 
                  int(color[2]*0.4 + 255*0.6))
    cv2.line(frame, pt1, pt2, core_color, max(1, thickness - 1), cv2.LINE_AA)


# ---------------------------------------------------------------------------
# SECTION 7 — SKELETON RENDERER (main public function)
# ---------------------------------------------------------------------------

def draw_pose(frame: np.ndarray,
              keypoints: List[Tuple[float, float, float]],
              conf_threshold: float = 0.5,
              exercise_name: str    = "",
              eval_colors: Dict[int, Tuple] = None,
              skeleton_connections: List[Tuple[int,int]] = None,
              line_thickness: int   = 3,
              joint_radius: int     = 8,
              show_glow: bool       = True,
              trail_kpts: List[List[Tuple]] = None) -> np.ndarray:
    """
    Full premium pose overlay.

    Parameters
    ----------
    keypoints           : 17 (x, y, conf) — already EMA-smoothed
    conf_threshold      : skip joints below this confidence
    exercise_name       : used to look up active (highlighted) joints
    eval_colors         : {kp_index: (B,G,R)} override from form evaluator
                          so incorrect joints flash red, correct flash green
    skeleton_connections: list of (a, b) index pairs to connect
    line_thickness      : bone line width in pixels
    joint_radius        : base radius for joint circles
    show_glow           : whether to render the soft glow effect
    trail_kpts          : list of older keypoint arrays for ghost effect

    Returns
    -------
    frame  (modified in-place)
    """
    from src.pose_extractor import SKELETON_CONNECTIONS as DEFAULT_CONNECTIONS
    connections = skeleton_connections or DEFAULT_CONNECTIONS
    active_set  = set(ACTIVE_JOINTS.get(exercise_name, []))

    # ------------------------------------------------------------------
    # PASS 1 — TRAIL  (oldest frame first → naturally gets overdrawn)
    # ------------------------------------------------------------------
    # WHY draw ghosts before the current frame?  Painter's algorithm:
    # whatever is drawn first is occluded by whatever is drawn after.
    # The ghost fades into the background; the current pose appears solid.
    if trail_kpts:
        n = len(trail_kpts)
        for t_idx, trail in enumerate(reversed(trail_kpts)):   # oldest first
            alpha = 0.06 + 0.06 * (n - t_idx - 1)             # 0.06 → 0.36
            for (a, b) in connections:
                xa, ya, ca = trail[a]
                xb, yb, cb = trail[b]
                if ca < conf_threshold or cb < conf_threshold:
                    continue
                side  = _BONE_SIDE.get((a, b), _BONE_SIDE.get((b, a), "torso"))
                color = _BONE_COLOR[side]
                # Ghost line on temp layer
                overlay = frame.copy()
                cv2.line(overlay, (int(xa), int(ya)), (int(xb), int(yb)),
                         color, max(1, line_thickness - 1), cv2.LINE_AA)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # ------------------------------------------------------------------
    # PASS 2 — BONES
    # ------------------------------------------------------------------
    # WHY draw bones before joints?  Joints (circles) sit on top and cover
    # the raw line endpoints, giving the impression of rounded bone caps.
    for (a, b) in connections:
        xa, ya, ca = keypoints[a]
        xb, yb, cb = keypoints[b]
        if ca < conf_threshold or cb < conf_threshold:
            continue

        side  = _BONE_SIDE.get((a, b), _BONE_SIDE.get((b, a), "torso"))
        color = _BONE_COLOR[side]

        _draw_aa_line(frame,
                      (int(xa), int(ya)),
                      (int(xb), int(yb)),
                      color, line_thickness, shadow=True)

    # ------------------------------------------------------------------
    # PASS 3 — JOINTS
    # ------------------------------------------------------------------
    for idx, (x, y, conf) in enumerate(keypoints):
        if conf < conf_threshold:
            continue

        group = KP_GROUP[idx] if idx < len(KP_GROUP) else "center"

        # Form evaluator can override color (correct/incorrect)
        base_color = (eval_colors or {}).get(idx, _GROUP_COLOR.get(group, Colors.FACE))

        # Scale opacity with confidence:
        # conf=1.0 → bright base_color; conf=0.5 → 55% brightness
        # WHY? A faded dot signals "low confidence — treat with scepticism"
        opacity_factor = 0.55 + 0.45 * conf
        scaled = tuple(int(c * opacity_factor) for c in base_color)

        # Active joints are larger + have a pulsing ring
        is_active = idx in active_set
        r = joint_radius + (3 if is_active else 0)

        # Glow pass (optional, skip for face landmarks to reduce clutter)
        if show_glow and group != "face":
            _draw_glow_circle(frame, (int(x), int(y)), r, base_color,
                              intensity=0.45)

        # Active joint extra ring — a bright outline to catch the eye
        if is_active:
            cv2.circle(frame, (int(x), int(y)), r + 5,
                       Colors.glow(base_color, 0.7), 2, cv2.LINE_AA)

        # Main joint circle
        _draw_aa_circle(frame, (int(x), int(y)), r, scaled)

    return frame


# ---------------------------------------------------------------------------
# SECTION 8 — CONFIDENCE LABEL HELPER
# ---------------------------------------------------------------------------

def draw_confidence_label(frame: np.ndarray,
                           keypoints: List[Tuple],
                           conf_threshold: float = 0.5) -> np.ndarray:
    """
    Draw tiny confidence % next to each visible joint.
    Only enabled in debug mode — too noisy for normal use.
    """
    for idx, (x, y, conf) in enumerate(keypoints):
        if conf < conf_threshold:
            continue
        label = f"{conf:.0%}"
        cv2.putText(frame, label, (int(x) + 10, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                    (220, 220, 100), 1, cv2.LINE_AA)
    return frame
