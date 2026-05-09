"""
src/angle_engine.py
===================
Converts raw (x, y, confidence) keypoints into meaningful joint angles.

WHY angles instead of raw coordinates?
    Raw pixel positions change with camera distance and body position in frame.
    Angles are invariant to these factors — a squat at 90 degrees looks the
    same whether you're 1 m or 3 m from the camera.  This makes them a
    reliable, generalisable signal for form assessment.

FORMULA (dot-product angle):
    Given three points A, B, C — angle at vertex B:
        BA = A - B,  BC = C - B
        cos(theta) = (BA . BC) / (|BA| * |BC|)
        theta = arccos(cos(theta))  [degrees]
"""

import numpy as np
import cv2

from src.pose_extractor import KEYPOINT_NAMES, CONF_THRESHOLD

# ---------------------------------------------------------------------------
# SECTION 1 — ANGLE DEFINITIONS PER EXERCISE
# ---------------------------------------------------------------------------
# Each entry names three keypoints (A, B, C) where B is the joint vertex.
# We use KEYPOINT_NAMES.index() so the definition is readable, not magic ints.

ANGLE_DEFINITIONS = {

    # -- Bicep Curl: elbow flexion ------------------------------------------
    # Straight arm ~170 deg, fully curled ~40 deg.
    "bicep_curl": {
        "left_elbow":  {"points": (KEYPOINT_NAMES.index("left_shoulder"),
                                   KEYPOINT_NAMES.index("left_elbow"),
                                   KEYPOINT_NAMES.index("left_wrist")),
                        "display_name": "L Flexion"},
        "right_elbow": {"points": (KEYPOINT_NAMES.index("right_shoulder"),
                                   KEYPOINT_NAMES.index("right_elbow"),
                                   KEYPOINT_NAMES.index("right_wrist")),
                        "display_name": "R Flexion"},
        "left_shoulder": {"points": (KEYPOINT_NAMES.index("left_hip"),
                                     KEYPOINT_NAMES.index("left_shoulder"),
                                     KEYPOINT_NAMES.index("left_elbow")),
                          "display_name": "L Elbow Drift"},
        "right_shoulder": {"points": (KEYPOINT_NAMES.index("right_hip"),
                                      KEYPOINT_NAMES.index("right_shoulder"),
                                      KEYPOINT_NAMES.index("right_elbow")),
                           "display_name": "R Elbow Drift"},
        "left_hip":    {"points": ("vertical",
                                   KEYPOINT_NAMES.index("left_hip"),
                                   KEYPOINT_NAMES.index("left_shoulder")),
                        "display_name": "L Spine Tilt"},
        "right_hip":   {"points": ("vertical",
                                   KEYPOINT_NAMES.index("right_hip"),
                                   KEYPOINT_NAMES.index("right_shoulder")),
                        "display_name": "R Spine Tilt"},
    },

    # -- Squat: knee depth + torso angle ------------------------------------
    # Standing ~170 deg, parallel squat ~90 deg, deep squat ~60 deg.
    "squat": {
        "left_knee":  {"points": (KEYPOINT_NAMES.index("left_hip"),
                                  KEYPOINT_NAMES.index("left_knee"),
                                  KEYPOINT_NAMES.index("left_ankle")),
                       "display_name": "L Knee"},
        "right_knee": {"points": (KEYPOINT_NAMES.index("right_hip"),
                                  KEYPOINT_NAMES.index("right_knee"),
                                  KEYPOINT_NAMES.index("right_ankle")),
                       "display_name": "R Knee"},
        "left_hip":   {"points": (KEYPOINT_NAMES.index("left_shoulder"),
                                  KEYPOINT_NAMES.index("left_hip"),
                                  KEYPOINT_NAMES.index("left_knee")),
                       "display_name": "L Hip"},
        "right_hip":  {"points": (KEYPOINT_NAMES.index("right_shoulder"),
                                  KEYPOINT_NAMES.index("right_hip"),
                                  KEYPOINT_NAMES.index("right_knee")),
                       "display_name": "R Hip"},
    },

    # -- Lateral Raise: shoulder abduction ----------------------------------
    "lateral_raise": {
        "left_shoulder":  {"points": (KEYPOINT_NAMES.index("left_hip"),
                                      KEYPOINT_NAMES.index("left_shoulder"),
                                      KEYPOINT_NAMES.index("left_elbow")),
                           "display_name": "L Shoulder"},
        "right_shoulder": {"points": (KEYPOINT_NAMES.index("right_hip"),
                                      KEYPOINT_NAMES.index("right_shoulder"),
                                      KEYPOINT_NAMES.index("right_elbow")),
                           "display_name": "R Shoulder"},
        "left_hip":    {"points": ("vertical",
                                   KEYPOINT_NAMES.index("left_hip"),
                                   KEYPOINT_NAMES.index("left_shoulder")),
                        "display_name": "L Torso Tilt"},
        "right_hip":   {"points": ("vertical",
                                   KEYPOINT_NAMES.index("right_hip"),
                                   KEYPOINT_NAMES.index("right_shoulder")),
                        "display_name": "R Torso Tilt"},
        "left_elbow":  {"points": (KEYPOINT_NAMES.index("left_shoulder"),
                                   KEYPOINT_NAMES.index("left_elbow"),
                                   KEYPOINT_NAMES.index("left_wrist")),
                        "display_name": "L Arm Bend"},
        "right_elbow": {"points": (KEYPOINT_NAMES.index("right_shoulder"),
                                   KEYPOINT_NAMES.index("right_elbow"),
                                   KEYPOINT_NAMES.index("right_wrist")),
                        "display_name": "R Arm Bend"},
    },

    # -- Push-Up: elbow flexion + body alignment ----------------------------
    "push_up": {
        "left_elbow":  {"points": (KEYPOINT_NAMES.index("left_shoulder"),
                                   KEYPOINT_NAMES.index("left_elbow"),
                                   KEYPOINT_NAMES.index("left_wrist")),
                        "display_name": "L Elbow"},
        "right_elbow": {"points": (KEYPOINT_NAMES.index("right_shoulder"),
                                   KEYPOINT_NAMES.index("right_elbow"),
                                   KEYPOINT_NAMES.index("right_wrist")),
                        "display_name": "R Elbow"},
        "left_hip":    {"points": (KEYPOINT_NAMES.index("left_shoulder"),
                                   KEYPOINT_NAMES.index("left_hip"),
                                   KEYPOINT_NAMES.index("left_ankle")),
                        "display_name": "L Body Align"},
        "right_hip":   {"points": (KEYPOINT_NAMES.index("right_shoulder"),
                                   KEYPOINT_NAMES.index("right_hip"),
                                   KEYPOINT_NAMES.index("right_ankle")),
                        "display_name": "R Body Align"},
        "left_shoulder": {"points": (KEYPOINT_NAMES.index("left_hip"),
                                     KEYPOINT_NAMES.index("left_shoulder"),
                                     KEYPOINT_NAMES.index("left_elbow")),
                          "display_name": "L Elbow Flare"},
        "right_shoulder": {"points": (KEYPOINT_NAMES.index("right_hip"),
                                      KEYPOINT_NAMES.index("right_shoulder"),
                                      KEYPOINT_NAMES.index("right_elbow")),
                           "display_name": "R Elbow Flare"},
        "shoulder_level": {"points": ("down",
                                      KEYPOINT_NAMES.index("left_shoulder"),
                                      KEYPOINT_NAMES.index("right_shoulder")),
                           "display_name": "Shoulder Level"},
        "left_hand_placement": {"points": ("down",
                                           KEYPOINT_NAMES.index("left_shoulder"),
                                           KEYPOINT_NAMES.index("left_wrist")),
                                "display_name": "L Hand Align"},
        "right_hand_placement": {"points": ("down",
                                            KEYPOINT_NAMES.index("right_shoulder"),
                                            KEYPOINT_NAMES.index("right_wrist")),
                                 "display_name": "R Hand Align"},
        "left_forearm_align": {"points": ("down",
                                          KEYPOINT_NAMES.index("left_elbow"),
                                          KEYPOINT_NAMES.index("left_wrist")),
                               "display_name": "L Forearm Vert"},
        "right_forearm_align": {"points": ("down",
                                           KEYPOINT_NAMES.index("right_elbow"),
                                           KEYPOINT_NAMES.index("right_wrist")),
                                "display_name": "R Forearm Vert"},
    },



    # -- Lunge: front knee + back knee + torso -----------------------------
    "lunge": {
        "left_knee":  {"points": (KEYPOINT_NAMES.index("left_hip"),
                                  KEYPOINT_NAMES.index("left_knee"),
                                  KEYPOINT_NAMES.index("left_ankle")),
                       "display_name": "Front Knee"},
        "right_knee": {"points": (KEYPOINT_NAMES.index("right_hip"),
                                  KEYPOINT_NAMES.index("right_knee"),
                                  KEYPOINT_NAMES.index("right_ankle")),
                       "display_name": "Back Knee"},
        "left_hip":   {"points": ("vertical",
                                  KEYPOINT_NAMES.index("left_hip"),
                                  KEYPOINT_NAMES.index("left_shoulder")),
                       "display_name": "L Torso Tilt"},
        "right_hip":  {"points": ("vertical",
                                  KEYPOINT_NAMES.index("right_hip"),
                                  KEYPOINT_NAMES.index("right_shoulder")),
                       "display_name": "R Torso Tilt"},
        "left_shin":  {"points": ("vertical",
                                  KEYPOINT_NAMES.index("left_ankle"),
                                  KEYPOINT_NAMES.index("left_knee")),
                       "display_name": "L Shin Align"},
        "right_shin": {"points": ("vertical",
                                  KEYPOINT_NAMES.index("right_ankle"),
                                  KEYPOINT_NAMES.index("right_knee")),
                       "display_name": "R Shin Align"},
    },

    # -- Plank: full body alignment ----------------------------------------
    # WHY shoulder-hip-ankle? It captures hip sag (the most common error).
    "plank": {
        "left_hip":   {"points": (KEYPOINT_NAMES.index("left_shoulder"),
                                  KEYPOINT_NAMES.index("left_hip"),
                                  KEYPOINT_NAMES.index("left_ankle")),
                       "display_name": "L Alignment"},
        "right_hip":  {"points": (KEYPOINT_NAMES.index("right_shoulder"),
                                  KEYPOINT_NAMES.index("right_hip"),
                                  KEYPOINT_NAMES.index("right_ankle")),
                       "display_name": "R Alignment"},
        "left_knee":  {"points": (KEYPOINT_NAMES.index("left_hip"),
                                  KEYPOINT_NAMES.index("left_knee"),
                                  KEYPOINT_NAMES.index("left_ankle")),
                       "display_name": "L Leg"},
        "right_knee": {"points": (KEYPOINT_NAMES.index("right_hip"),
                                  KEYPOINT_NAMES.index("right_knee"),
                                  KEYPOINT_NAMES.index("right_ankle")),
                       "display_name": "R Leg"},
        "left_shoulder": {"points": (KEYPOINT_NAMES.index("left_ear"),
                                     KEYPOINT_NAMES.index("left_shoulder"),
                                     KEYPOINT_NAMES.index("left_hip")),
                          "display_name": "L Neck Align"},
        "right_shoulder": {"points": (KEYPOINT_NAMES.index("right_ear"),
                                      KEYPOINT_NAMES.index("right_shoulder"),
                                      KEYPOINT_NAMES.index("right_hip")),
                           "display_name": "R Neck Align"},
        "left_arm_align": {"points": ("down",
                                      KEYPOINT_NAMES.index("left_shoulder"),
                                      KEYPOINT_NAMES.index("left_elbow")),
                           "display_name": "L Arm Align"},
        "right_arm_align": {"points": ("down",
                                       KEYPOINT_NAMES.index("right_shoulder"),
                                       KEYPOINT_NAMES.index("right_elbow")),
                            "display_name": "R Arm Align"},
    },
}


# ---------------------------------------------------------------------------
# SECTION 2 — CORE ANGLE FUNCTION
# ---------------------------------------------------------------------------

def calculate_angle(ax, ay, bx, by, cx, cy):
    """
    Angle at vertex B formed by points A-B-C, returned in degrees [0, 180].
    Returns None if either arm vector has zero length (coincident points).
    """
    BA = np.array([ax - bx, ay - by], dtype=np.float64)
    BC = np.array([cx - bx, cy - by], dtype=np.float64)
    mag_BA, mag_BC = np.linalg.norm(BA), np.linalg.norm(BC)
    if mag_BA < 1e-6 or mag_BC < 1e-6:
        return None
    cos_a = np.clip(np.dot(BA, BC) / (mag_BA * mag_BC), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


# ---------------------------------------------------------------------------
# SECTION 3 — EXTRACT ALL ANGLES FOR ONE EXERCISE FRAME
# ---------------------------------------------------------------------------

def get_exercise_angles(keypoints_list, exercise_name):
    """
    Given one person's 17 keypoints and an exercise name, returns a dict of
    all relevant joint angles.

    Parameters
    ----------
    keypoints_list : list of 17 (x, y, conf) tuples
    exercise_name  : str — must be a key in ANGLE_DEFINITIONS

    Returns
    -------
    dict  { joint_name: {"angle": float|None,
                          "display_name": str,
                          "vertex_xy": (x, y)} }
    """
    if exercise_name not in ANGLE_DEFINITIONS:
        raise ValueError(
            f"Unknown exercise: '{exercise_name}'. "
            f"Valid choices: {list(ANGLE_DEFINITIONS.keys())}"
        )

    definitions = ANGLE_DEFINITIONS[exercise_name]
    result = {}

    for joint_name, joint_def in definitions.items():
        idx_a, idx_b, idx_c = joint_def["points"]
        xb, yb, cb = keypoints_list[idx_b]

        def get_point(idx_or_str, vx, vy):
            if idx_or_str == "vertical":
                return vx, vy - 100, 1.0
            if idx_or_str == "down":
                return vx, vy + 100, 1.0
            return keypoints_list[idx_or_str]

        xa, ya, ca = get_point(idx_a, xb, yb)
        xc, yc, cc = get_point(idx_c, xb, yb)

        # Use a lower cut-off (0.25) so ghost-frame / medium-conf keypoints
        # still produce an angle. The form_evaluator uses min_conf to decide
        # how much to trust that angle.
        _HARD_CUTOFF = 0.25
        if ca < _HARD_CUTOFF or cb < _HARD_CUTOFF or cc < _HARD_CUTOFF:
            angle = None
        else:
            angle = calculate_angle(xa, ya, xb, yb, xc, yc)

        min_conf = min(
            ca if ca <= 1.0 else 1.0,
            cb if cb <= 1.0 else 1.0,
            cc if cc <= 1.0 else 1.0,
        )

        result[joint_name] = {
            "angle":        angle,
            "display_name": joint_def["display_name"],
            "vertex_xy":    (xb, yb),
            # Confidence metadata — used by form_evaluator for weighted scoring
            "conf_a":   ca,
            "conf_b":   cb,
            "conf_c":   cc,
            "min_conf": min_conf,
        }

    return result


# ---------------------------------------------------------------------------
# SECTION 4 — DRAW ANGLE LABELS ON FRAME
# ---------------------------------------------------------------------------

def draw_angle_labels(frame, angles_dict, color=(255, 255, 0)):
    """Overlay angle values next to each joint on the video frame."""
    for joint_name, data in angles_dict.items():
        angle = data["angle"]
        vx, vy = data["vertex_xy"]
        label  = data["display_name"]

        text = (f"{label}: {angle:.1f} deg"
                if angle is not None else f"{label}: N/A")
        text_color = color if angle is not None else (128, 128, 128)

        tx, ty = int(vx) + 10, int(vy) - 10
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (tx-2, ty-th-2), (tx+tw+2, ty+2), (0, 0, 0), -1)
        cv2.putText(frame, text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1, cv2.LINE_AA)

    return frame
