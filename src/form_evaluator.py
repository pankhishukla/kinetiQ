"""
src/form_evaluator.py
=====================
Rule-based form assessment + rep counting for all 7 exercises.

WHY rule-based (not ML) for real-time feedback?
    Rules are instant (no inference latency), interpretable (we can print
    exactly which joint is wrong), and tuned to real data via Phase 5
    calibration.  ML anomaly detection is layered on TOP as a second pass
    in the training pipeline — rules remain the primary real-time signal.
"""

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# SECTION 1 — FORM RULES DATABASE
# ---------------------------------------------------------------------------
# Structure per joint:
#   "min" / "max"  : angle range that constitutes CORRECT form (degrees)
#   "cue_low"      : feedback shown when angle < min (joint too compressed)
#   "cue_high"     : feedback shown when angle > max (joint too extended)
#   "cue_good"     : feedback shown when angle is within [min, max]
#
# WHY data-driven mins/maxes?
#   Values marked with dataset stats were calibrated from real COCO annotations
#   in Phase 5 (5_calibrate.py / tools/calibrate_thresholds.py).
#   Others are conservative clinical/coaching values.

FORM_RULES = {

    "bicep_curl": {
        "left_elbow": {
            "weight": 1.0,
            "phases": {
                "up": {"min": 130, "max": 180, "cue_low": "", "cue_high": "Straighten arm", "cue_good": "L Elbow: Ready"},
                "down": {"min": 20, "max": 80, "cue_low": "Don't overcurl", "cue_high": "Curl higher", "cue_good": "L Elbow: Good curl"},
                "transition": {"min": 20, "max": 180, "cue_low": "Don't overcurl", "cue_high": "Straighten arm", "cue_good": "L Elbow: Moving"}
            }
        },
        "right_elbow": {
            "weight": 1.0,
            "phases": {
                "up": {"min": 130, "max": 180, "cue_low": "", "cue_high": "Straighten arm", "cue_good": "R Elbow: Ready"},
                "down": {"min": 20, "max": 80, "cue_low": "Don't overcurl", "cue_high": "Curl higher", "cue_good": "R Elbow: Good curl"},
                "transition": {"min": 20, "max": 180, "cue_low": "Don't overcurl", "cue_high": "Straighten arm", "cue_good": "R Elbow: Moving"}
            }
        },
        "left_shoulder": {
            "weight": 2.0, "min": 0, "max": 25,
            "cue_low":  "", "cue_high": "Keep elbows tucked", "cue_good": "L Elbow: Tucked",
        },
        "right_shoulder": {
            "weight": 2.0, "min": 0, "max": 25,
            "cue_low":  "", "cue_high": "Keep elbows tucked", "cue_good": "R Elbow: Tucked",
        },
        "left_hip": {
            "weight": 2.0, "min": 0, "max": 15,
            "cue_low":  "", "cue_high": "Don't lean back", "cue_good": "Spine: Straight",
        },
        "right_hip": {
            "weight": 2.0, "min": 0, "max": 15,
            "cue_low":  "", "cue_high": "Don't lean back", "cue_good": "Spine: Straight",
        },
    },

    "squat": {
        "left_knee": {
            "weight": 2.0,
            "phases": {
                "up": {"min": 145, "max": 180, "cue_low": "", "cue_high": "Stand straight", "cue_good": "L Knee: Ready"},
                "down": {"min": 60, "max": 110, "cue_low": "Too deep", "cue_high": "Squat deeper", "cue_good": "L Knee: Good depth"},
                "transition": {"min": 60, "max": 180, "cue_low": "Too deep", "cue_high": "Stand straight", "cue_good": "L Knee: Moving"}
            }
        },
        "right_knee": {
            "weight": 2.0,
            "phases": {
                "up": {"min": 145, "max": 180, "cue_low": "", "cue_high": "Stand straight", "cue_good": "R Knee: Ready"},
                "down": {"min": 60, "max": 110, "cue_low": "Too deep", "cue_high": "Squat deeper", "cue_good": "R Knee: Good depth"},
                "transition": {"min": 60, "max": 180, "cue_low": "Too deep", "cue_high": "Stand straight", "cue_good": "R Knee: Moving"}
            }
        },
        "left_hip": {
            "weight": 2.0, "min": 55, "max": 140,
            "cue_low":  "Lean back — chest up!", "cue_high": "Hinge at hip", "cue_good": "L hip: Good",
        },
        "right_hip": {
            "weight": 2.0, "min": 55, "max": 140,
            "cue_low":  "Lean back — chest up!", "cue_high": "Hinge at hip", "cue_good": "R hip: Good",
        },
    },

    "lateral_raise": {
        "left_shoulder": {
            "weight": 2.0,
            "phases": {
                "down": {"min": 0, "max": 50, "cue_low": "", "cue_high": "Lower arm", "cue_good": "L Shoulder: Ready"},
                "up": {"min": 60, "max": 120, "cue_low": "Raise arm higher", "cue_high": "Too high", "cue_good": "L Shoulder: Good height"},
                "transition": {"min": 0, "max": 120, "cue_low": "", "cue_high": "Too high", "cue_good": "L Shoulder: Moving"}
            }
        },
        "right_shoulder": {
            "weight": 2.0,
            "phases": {
                "down": {"min": 0, "max": 50, "cue_low": "", "cue_high": "Lower arm", "cue_good": "R Shoulder: Ready"},
                "up": {"min": 60, "max": 120, "cue_low": "Raise arm higher", "cue_high": "Too high", "cue_good": "R Shoulder: Good height"},
                "transition": {"min": 0, "max": 120, "cue_low": "", "cue_high": "Too high", "cue_good": "R Shoulder: Moving"}
            }
        },
        "left_hip": {
            "weight": 2.0, "min": 0, "max": 15,
            "cue_low":  "", "cue_high": "Don't swing", "cue_good": "L Torso: Stable",
        },
        "right_hip": {
            "weight": 2.0, "min": 0, "max": 15,
            "cue_low":  "", "cue_high": "Don't swing", "cue_good": "R Torso: Stable",
        },
        "left_elbow": {
            "weight": 1.0, "min": 130, "max": 175,
            "cue_low":  "Don't bend elbows too much", "cue_high": "Keep slight bend", "cue_good": "L Elbow: Good bend",
        },
        "right_elbow": {
            "weight": 1.0, "min": 130, "max": 175,
            "cue_low":  "Don't bend elbows too much", "cue_high": "Keep slight bend", "cue_good": "R Elbow: Good bend",
        },
    },

    "push_up": {
        "left_elbow": {
            "weight": 2.0,
            "phases": {
                "up": {"min": 140, "max": 180, "cue_low": "Don't lock out", "cue_high": "Straighten arms", "cue_good": "L Elbow: Ready"},
                "down": {"min": 65, "max": 100, "cue_low": "Don't go too low", "cue_high": "Lower your chest", "cue_good": "L Elbow: Good depth"},
                "transition": {"min": 65, "max": 180, "cue_low": "Don't go too low", "cue_high": "Straighten arms", "cue_good": "L Elbow: Moving"}
            }
        },
        "right_elbow": {
            "weight": 2.0,
            "phases": {
                "up": {"min": 140, "max": 180, "cue_low": "Don't lock out", "cue_high": "Straighten arms", "cue_good": "R Elbow: Ready"},
                "down": {"min": 65, "max": 100, "cue_low": "Don't go too low", "cue_high": "Lower your chest", "cue_good": "R Elbow: Good depth"},
                "transition": {"min": 65, "max": 180, "cue_low": "Don't go too low", "cue_high": "Straighten arms", "cue_good": "R Elbow: Moving"}
            }
        },
        "left_hip": {
            "weight": 3.0, "min": 155, "max": 195,
            "cue_low":  "Keep body straight (hips sagging or too high)", "cue_high": "", "cue_good": "L Body: Good",
        },
        "right_hip": {
            "weight": 3.0, "min": 155, "max": 195,
            "cue_low":  "Keep body straight (hips sagging or too high)", "cue_high": "", "cue_good": "R Body: Good",
        },
        "left_shoulder": {
            "weight": 1.0, "min": 0, "max": 85,
            "cue_low":  "", "cue_high": "Keep elbows tucked", "cue_good": "L Elbow: Tucked",
        },
        "right_shoulder": {
            "weight": 1.0, "min": 0, "max": 85,
            "cue_low":  "", "cue_high": "Keep elbows tucked", "cue_good": "R Elbow: Tucked",
        },
        "shoulder_level": {
            "weight": 2.0, "min": 75, "max": 105,
            "cue_low":  "Shoulders tilted left", "cue_high": "Shoulders tilted right", "cue_good": "Shoulders: Level",
        },
        "left_hand_placement": {
            "weight": 1.5,
            "phases": {
                "up": {"min": 0, "max": 25, "cue_low": "Hands too far back", "cue_high": "Hands too far forward", "cue_good": "L Hand: Under shoulder"},
                "down": {"min": 0, "max": 45, "cue_low": "Hands too far back", "cue_high": "Hands too far forward", "cue_good": "L Hand: Good"},
                "transition": {"min": 0, "max": 45, "cue_low": "", "cue_high": "", "cue_good": "L Hand: Moving"}
            }
        },
        "right_hand_placement": {
            "weight": 1.5,
            "phases": {
                "up": {"min": 0, "max": 25, "cue_low": "Hands too far back", "cue_high": "Hands too far forward", "cue_good": "R Hand: Under shoulder"},
                "down": {"min": 0, "max": 45, "cue_low": "Hands too far back", "cue_high": "Hands too far forward", "cue_good": "R Hand: Good"},
                "transition": {"min": 0, "max": 45, "cue_low": "", "cue_high": "", "cue_good": "R Hand: Moving"}
            }
        },
        "left_forearm_align": {
            "weight": 2.0, "min": 0, "max": 15,
            "cue_low": "Forearms not vertical (hands too wide/close)", "cue_high": "Forearms not vertical (hands too wide/close)", "cue_good": "L Forearm: Vertical",
        },
        "right_forearm_align": {
            "weight": 2.0, "min": 0, "max": 15,
            "cue_low": "Forearms not vertical (hands too wide/close)", "cue_high": "Forearms not vertical (hands too wide/close)", "cue_good": "R Forearm: Vertical",
        },
    },

    "lunge": {
        "left_knee": {
            "weight": 2.0,
            "phases": {
                "up": {"min": 140, "max": 180, "cue_low": "Don't lock knee", "cue_high": "Straighten leg", "cue_good": "L Knee: Good"},
                "down": {"min": 70, "max": 130, "cue_low": "Too deep", "cue_high": "Lunge deeper", "cue_good": "L Knee: Good depth"},
                "transition": {"min": 70, "max": 180, "cue_low": "Too deep", "cue_high": "Straighten leg", "cue_good": "L Knee: Moving"}
            }
        },
        "right_knee": {
            "weight": 2.0,
            "phases": {
                "up": {"min": 140, "max": 180, "cue_low": "Don't lock knee", "cue_high": "Straighten leg", "cue_good": "R Knee: Good"},
                "down": {"min": 70, "max": 130, "cue_low": "Back knee low", "cue_high": "Drop back knee", "cue_good": "R Knee: Good pos"},
                "transition": {"min": 70, "max": 180, "cue_low": "Too deep", "cue_high": "Straighten leg", "cue_good": "R Knee: Moving"}
            }
        },
        "left_hip": {
            "weight": 2.0, "min": 0, "max": 20,
            "cue_low":  "", "cue_high": "Torso leaning forward", "cue_good": "L Torso: Upright",
        },
        "right_hip": {
            "weight": 2.0, "min": 0, "max": 20,
            "cue_low":  "", "cue_high": "Torso leaning forward", "cue_good": "R Torso: Upright",
        },
        "left_shin": {
            "weight": 2.0, "min": 0, "max": 25,
            "cue_low": "", "cue_high": "Knee too far forward", "cue_good": "L Shin: Vertical"
        },
        "right_shin": {
            "weight": 2.0, "min": 0, "max": 25,
            "cue_low": "", "cue_high": "Knee too far forward", "cue_good": "R Shin: Vertical"
        },
    },

    "plank": {
        "left_hip": {
            "weight": 3.0, "min": 155, "max": 195,
            "cue_low":  "Keep body straight (hips sagging or too high)", "cue_high": "", "cue_good": "L Body: Aligned",
        },
        "right_hip": {
            "weight": 3.0, "min": 155, "max": 195,
            "cue_low":  "Keep body straight (hips sagging or too high)", "cue_high": "", "cue_good": "R Body: Aligned",
        },
        "left_arm_align": {
            "weight": 2.0, "min": 0, "max": 25,
            "cue_low": "Elbows too far back", "cue_high": "Elbows too far forward", "cue_good": "L Arm: Vertical"
        },
        "right_arm_align": {
            "weight": 2.0, "min": 0, "max": 25,
            "cue_low": "Elbows too far back", "cue_high": "Elbows too far forward", "cue_good": "R Arm: Vertical"
        },
        "left_knee": {
            "weight": 1.0, "min": 160, "max": 210,
            "cue_low":  "Straighten leg", "cue_high": "Keep leg straight", "cue_good": "L Leg: Straight",
        },
        "right_knee": {
            "weight": 1.0, "min": 160, "max": 210,
            "cue_low":  "Straighten leg", "cue_high": "Keep leg straight", "cue_good": "R Leg: Straight",
        },
        "left_shoulder": {
            "weight": 0.5, "min": 140, "max": 180,
            "cue_low":  "Neck neutral", "cue_high": "Neck neutral", "cue_good": "L Neck: Neutral",
        },
        "right_shoulder": {
            "weight": 0.5, "min": 140, "max": 180,
            "cue_low":  "Neck neutral", "cue_high": "Neck neutral", "cue_good": "R Neck: Neutral",
        },
    },
}


# ---------------------------------------------------------------------------
# SECTION 2 — REP COUNTER CONFIGURATION
# ---------------------------------------------------------------------------
# WHY a two-threshold state machine?
#   A single threshold causes double-counting when the angle hovers near it.
#   Two thresholds (down_threshold < up_threshold) create a "dead band" that
#   requires a full committed movement before the counter advances.

REP_CONFIG = {
    "bicep_curl":     {"primary_joint": "right_elbow",   "down_threshold": 70,   "up_threshold": 140, "count_on": "up"},
    "squat":          {"primary_joint": "right_knee",    "down_threshold": 100,  "up_threshold": 155, "count_on": "up"},
    "lateral_raise":  {"primary_joint": "right_shoulder","down_threshold": 40,   "up_threshold": 80,  "count_on": "up"},
    "push_up":        {"primary_joint": "right_elbow",   "down_threshold": 90,   "up_threshold": 150, "count_on": "up"},
    "lunge":          {"primary_joint": "left_knee",     "down_threshold": 100,  "up_threshold": 155, "count_on": "up"},
    "plank":          {"primary_joint": None,            "down_threshold": None, "up_threshold": None,"count_on": None},
}


# ---------------------------------------------------------------------------
# SECTION 2b — CRITICAL JOINTS (can invalidate a rep cycle)
# ---------------------------------------------------------------------------
# Only joints listed here will block rep counting if they are incorrect.
# Advisory joints (hand placement, forearm alignment etc.) show red feedback
# but do NOT discard the rep.

CRITICAL_JOINTS = {
    "bicep_curl":    {"left_elbow", "right_elbow"},
    "squat":         {"left_knee", "right_knee", "left_hip", "right_hip"},
    "lateral_raise": {"left_shoulder", "right_shoulder"},
    "push_up":       {"left_elbow", "right_elbow", "left_hip", "right_hip"},
    "lunge":         {"left_knee", "right_knee"},
    "plank":         set(),   # static hold — no rep blocking needed
}


# ---------------------------------------------------------------------------
# SECTION 3 — MOVEMENT STATE TRACKER
# ---------------------------------------------------------------------------

class RepCounter:
    """
    Phase-aware state machine for counting exercise repetitions.
    Tracks 'up', 'down', and 'transition' phases.

    Rep counting is based purely on angular range of motion (down_threshold /
    up_threshold). Form feedback (red joints, cues) is displayed separately
    and does NOT gate rep counting — the state machine already guarantees a
    full range of motion was completed which validates the rep.
    """

    def __init__(self, exercise_name):
        self.exercise_name = exercise_name
        self.count = 0
        self.phase = "up"       # assume starting in extended position
        self.last_state = "up"
        cfg = REP_CONFIG.get(exercise_name, {})
        self.primary_joint  = cfg.get("primary_joint")
        self.down_threshold = cfg.get("down_threshold")
        self.up_threshold   = cfg.get("up_threshold")
        self.count_on       = cfg.get("count_on")

    def update(self, angles_dict, form_evaluations=None):
        """Feed the latest angles; returns current rep count.
        form_evaluations is accepted for API compatibility but no longer gates rep counting.
        """
        if self.primary_joint is None:
            self.phase = "static"
            return self.count           # plank / hold exercise

        joint_data = angles_dict.get(self.primary_joint)
        if joint_data is None:
            return self.count
        angle = joint_data.get("angle")
        if angle is None:
            return self.count

        # State machine — count fires when a full movement cycle is detected
        if angle > self.up_threshold:
            if self.last_state == "down" and self.count_on == "up":
                self.count += 1
            self.phase = "up"
            self.last_state = "up"
        elif angle < self.down_threshold:
            if self.last_state == "up" and self.count_on == "down":
                self.count += 1
            self.phase = "down"
            self.last_state = "down"
        else:
            self.phase = "transition"

        return self.count

    def reset(self):
        self.count = 0
        self.phase = "up"
        self.last_state = "up"


# ---------------------------------------------------------------------------
# SECTION 4 — FEEDBACK COLORS
# ---------------------------------------------------------------------------
COLOR_CORRECT   = (  0, 220,   0)   # BGR green
COLOR_INCORRECT = (  0,   0, 220)   # BGR red
COLOR_UNKNOWN   = (150, 150, 150)   # BGR grey
COLOR_TEXT_GOOD = (  0, 220,   0)   # BGR green
COLOR_TEXT_BAD  = (  0,   0, 220)   # BGR red


# ---------------------------------------------------------------------------
# SECTION 5 — FORM EVALUATOR
# ---------------------------------------------------------------------------

def evaluate_form(angles_dict, exercise_name, current_phase="up"):
    """
    Compare each joint's current angle against its rule thresholds,
    taking into account the current phase of the movement.

    Returns
    -------
    dict  { joint_name: {"status": str,  "cue": str,
                          "angle": float, "color": tuple,
                          "vertex_xy": tuple, "display_name": str} }
    """
    rules = FORM_RULES.get(exercise_name, {})
    evaluations = {}

    for joint_name, angle_data in angles_dict.items():
        angle     = angle_data["angle"]
        vertex_xy = angle_data["vertex_xy"]
        disp_name = angle_data["display_name"]

        if joint_name not in rules:
            evaluations[joint_name] = {
                "status": "unknown", "cue": "",
                "angle": angle, "color": COLOR_UNKNOWN,
                "vertex_xy": vertex_xy, "display_name": disp_name,
            }
            continue

        rule = rules[joint_name]

        if angle is None:
            evaluations[joint_name] = {
                "status": "unknown",
                "cue":    f"{disp_name}: Can't see joint clearly",
                "angle":  None, "color": COLOR_UNKNOWN,
                "vertex_xy": vertex_xy, "display_name": disp_name,
            }
            continue

        if "phases" in rule:
            phase_rule = rule["phases"].get(current_phase)
            if not phase_rule:
                # Fallback to the first phase definition if current_phase not found
                phase_rule = list(rule["phases"].values())[0]
            r_min = phase_rule.get("min", 0)
            r_max = phase_rule.get("max", 180)
            cue_low = phase_rule.get("cue_low", "")
            cue_high = phase_rule.get("cue_high", "")
            cue_good = phase_rule.get("cue_good", "")
        else:
            r_min = rule.get("min", 0)
            r_max = rule.get("max", 180)
            cue_low = rule.get("cue_low", "")
            cue_high = rule.get("cue_high", "")
            cue_good = rule.get("cue_good", "")

        if angle < r_min:
            status, cue, color = "incorrect", cue_low, COLOR_INCORRECT
        elif angle > r_max:
            status, cue, color = "incorrect", cue_high, COLOR_INCORRECT
        else:
            status, cue, color = "correct", cue_good, COLOR_CORRECT

        evaluations[joint_name] = {
            "status": status, "cue": cue, "angle": angle,
            "color": color, "vertex_xy": vertex_xy, "display_name": disp_name,
        }


    return evaluations


# ---------------------------------------------------------------------------
# PUSH-UP SPECIFIC: Coordinate-space hand placement checks
# ---------------------------------------------------------------------------
# These checks cannot be done with pure angles because they require comparing
# the RELATIVE X (horizontal) positions of wrists vs shoulders in image space.

_PUSHUP_HAND_CACHE = {}   # stores raw keypoints from last push_up frame

def _inject_pushup_coord_checks(keypoints_list, evaluations, current_phase):
    """
    Adds / overrides evaluations for hand placement using raw pixel coordinates.
    Checks:
      1. Hands too far forward  — wrist is significantly in front of shoulder (x-axis)
      2. Hands too wide         — wrist is significantly wider than shoulder span
      3. Hands too close/narrow — wrists much closer together than shoulders
      4. Shoulder ahead of wrist — shoulder has drifted far forward of wrist
      5. Uneven L/R arm placement — left and right wrist not symmetrically placed
    """
    from src.pose_extractor import KEYPOINT_NAMES, CONF_THRESHOLD

    def kp(name):
        idx = KEYPOINT_NAMES.index(name)
        x, y, c = keypoints_list[idx]
        return (x, y, c)

    ls_x, ls_y, ls_c = kp("left_shoulder")
    rs_x, rs_y, rs_c = kp("right_shoulder")
    lw_x, lw_y, lw_c = kp("left_wrist")
    rw_x, rw_y, rw_c = kp("right_wrist")
    le_x, le_y, le_c = kp("left_elbow")
    re_x, re_y, re_c = kp("right_elbow")

    if any(c < CONF_THRESHOLD for c in [ls_c, rs_c, lw_c, rw_c]):
        return  # not enough confidence to make these checks

    # Shoulder span — the natural reference width
    shoulder_span = abs(rs_x - ls_x)
    if shoulder_span < 1:
        return  # degenerate frame

    # --- 1. Hands too far forward (wrist x ahead of shoulder x in image space) ---
    # In a side-view camera, "forward" means lower x (left side of frame for left arm)
    # We allow up to 30% of shoulder span as tolerance
    fwd_tolerance = 0.30 * shoulder_span

    # Left hand too far forward: left wrist x is notably less than left shoulder x
    l_fwd_offset = ls_x - lw_x  # positive = wrist is to the left (forward) of shoulder
    if l_fwd_offset > fwd_tolerance:
        evaluations["left_hand_placement"] = {
            "status": "incorrect", "cue": "L hand too far forward",
            "angle": None, "color": COLOR_INCORRECT,
            "vertex_xy": (lw_x, lw_y), "display_name": "L Hand Align",
        }
    # Right hand too far forward: right wrist x is notably greater than right shoulder x
    r_fwd_offset = rw_x - rs_x
    if r_fwd_offset > fwd_tolerance:
        evaluations["right_hand_placement"] = {
            "status": "incorrect", "cue": "R hand too far forward",
            "angle": None, "color": COLOR_INCORRECT,
            "vertex_xy": (rw_x, rw_y), "display_name": "R Hand Align",
        }

    # --- 2 & 3. Hands too wide or too narrow (wrist span vs shoulder span) ---
    wrist_span = abs(rw_x - lw_x)
    width_ratio = wrist_span / shoulder_span

    # Too wide: wrists more than 1.2× shoulder width apart (was 1.6×)
    if width_ratio > 1.2:
        for side, wx, wy, disp, key in [
            ("L", lw_x, lw_y, "L Hand Align", "left_hand_placement"),
            ("R", rw_x, rw_y, "R Hand Align", "right_hand_placement"),
        ]:
            if evaluations.get(key, {}).get("status") != "incorrect":
                evaluations[key] = {
                    "status": "incorrect", "cue": "Hands too wide — bring them in",
                    "angle": None, "color": COLOR_INCORRECT,
                    "vertex_xy": (wx, wy), "display_name": disp,
                }

    # Too narrow: wrists less than 0.6× shoulder width apart (was 0.5×)
    elif width_ratio < 0.6:
        for side, wx, wy, disp, key in [
            ("L", lw_x, lw_y, "L Hand Align", "left_hand_placement"),
            ("R", rw_x, rw_y, "R Hand Align", "right_hand_placement"),
        ]:
            if evaluations.get(key, {}).get("status") != "incorrect":
                evaluations[key] = {
                    "status": "incorrect", "cue": "Hands too close — widen them",
                    "angle": None, "color": COLOR_INCORRECT,
                    "vertex_xy": (wx, wy), "display_name": disp,
                }

    # --- 4. Shoulder joint far ahead of wrists (shoulders drifted forward) ---
    # This catches when the torso slides forward over the hands
    l_shoulder_fwd = lw_x - ls_x  # positive = shoulder is behind wrist (correct)
    r_shoulder_fwd = rs_x - rw_x
    shld_fwd_tolerance = 0.35 * shoulder_span

    if l_shoulder_fwd < -shld_fwd_tolerance:  # shoulder is ahead of wrist
        existing = evaluations.get("left_shoulder", {})
        if existing.get("status") != "incorrect":
            evaluations["left_shoulder"] = {
                "status": "incorrect", "cue": "L shoulder too far forward — shift back",
                "angle": None, "color": COLOR_INCORRECT,
                "vertex_xy": (ls_x, ls_y), "display_name": "L Elbow Flare",
            }
    if r_shoulder_fwd < -shld_fwd_tolerance:
        existing = evaluations.get("right_shoulder", {})
        if existing.get("status") != "incorrect":
            evaluations["right_shoulder"] = {
                "status": "incorrect", "cue": "R shoulder too far forward — shift back",
                "angle": None, "color": COLOR_INCORRECT,
                "vertex_xy": (rs_x, rs_y), "display_name": "R Elbow Flare",
            }

    # --- 5. Uneven left/right arm placement ---
    # Compare each wrist's offset from its shoulder to detect asymmetry
    l_offset_x = ls_x - lw_x   # how far left wrist deviates from left shoulder (x)
    r_offset_x = rw_x - rs_x   # how far right wrist deviates from right shoulder (x)
    asymmetry = abs(l_offset_x - r_offset_x)
    asymmetry_tolerance = 0.25 * shoulder_span

    if asymmetry > asymmetry_tolerance:
        # Flag the more deviated side
        if abs(l_offset_x) > abs(r_offset_x):
            key, wx, wy, disp = "left_hand_placement", lw_x, lw_y, "L Hand Align"
            cue = "Uneven arm placement — L hand misaligned"
        else:
            key, wx, wy, disp = "right_hand_placement", rw_x, rw_y, "R Hand Align"
            cue = "Uneven arm placement — R hand misaligned"
        if evaluations.get(key, {}).get("status") != "incorrect":
            evaluations[key] = {
                "status": "incorrect", "cue": cue,
                "angle": None, "color": COLOR_INCORRECT,
                "vertex_xy": (wx, wy), "display_name": disp,
            }


def calculate_posture_score(evaluations, exercise_name):
    """
    Computes a weighted posture score (0-100) and returns (score, overall_status).
    Uses the "weight" defined in FORM_RULES.
    """
    rules = FORM_RULES.get(exercise_name, {})
    total_weight = 0.0
    earned_weight = 0.0

    valid_evals = [ev for name, ev in evaluations.items() if ev["status"] != "unknown" and name in rules]

    if not valid_evals:
        return 0.0, "unknown"

    for joint_name, ev in evaluations.items():
        if ev["status"] == "unknown" or joint_name not in rules:
            continue
        
        weight = rules[joint_name].get("weight", 1.0)
        total_weight += weight
        if ev["status"] == "correct":
            earned_weight += weight

    if total_weight == 0:
        return 0.0, "unknown"

    score = (earned_weight / total_weight) * 100.0
    
    if score >= 85.0:
        overall_status = "excellent"
    elif score >= 60.0:
        overall_status = "good"
    else:
        overall_status = "poor"
        
    return round(score, 1), overall_status


# ---------------------------------------------------------------------------
# SECTION 6 — FEEDBACK OVERLAY DRAWING
# ---------------------------------------------------------------------------

def draw_feedback_overlay(frame, evaluations, rep_count=0, exercise_name=""):
    """
    Draws:
      - Colored circles at each measured joint
      - Angle value labels
      - Right-side cue panel
      - Rep counter box
      - Bottom banner (FORM: CORRECT / ISSUES)
    """
    h, w = frame.shape[:2]

    # Colored joint circles + angle text
    for joint_name, eval_data in evaluations.items():
        vx, vy = eval_data["vertex_xy"]
        color  = eval_data["color"]
        angle  = eval_data["angle"]
        cv2.circle(frame, (int(vx), int(vy)), 14, color, 3)
        text = f"{angle:.0f} deg" if angle is not None else "N/A"
        tx, ty = int(vx) + 16, int(vy) - 8
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (tx-2, ty-th-2), (tx+tw+2, ty+2), (0, 0, 0), -1)
        cv2.putText(frame, text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Right-side cue panel
    panel_x = w - 290
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x - 10, 55),
                  (w - 5, 60 + len(evaluations) * 38 + 20), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, "FORM FEEDBACK", (panel_x, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    y_cursor = 105
    for joint_name, eval_data in evaluations.items():
        cue    = eval_data["cue"]
        status = eval_data["status"]
        if not cue:
            continue
        text_color = (COLOR_TEXT_GOOD if status == "correct" else
                      ((160, 160, 160) if status == "unknown" else COLOR_TEXT_BAD))
        if len(cue) > 26:
            split_at = cue.rfind(' ', 0, 26)
            line1, line2 = cue[:split_at], cue[split_at + 1:]
        else:
            line1, line2 = cue, None
        cv2.putText(frame, line1, (panel_x, y_cursor),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, text_color, 1, cv2.LINE_AA)
        y_cursor += 17
        if line2:
            cv2.putText(frame, f"  {line2}", (panel_x, y_cursor),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, text_color, 1, cv2.LINE_AA)
            y_cursor += 17
        y_cursor += 5

    # Rep counter box
    rep_text = "HOLD" if exercise_name == "plank" else str(rep_count)
    cv2.rectangle(frame, (8, 92), (115, 145), (30, 30, 30), -1)
    cv2.rectangle(frame, (8, 92), (115, 145), (255, 200, 0), 2)
    cv2.putText(frame, "REPS", (14, 112),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, rep_text, (18, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3, cv2.LINE_AA)

    # Bottom banner
    all_statuses = [e["status"] for e in evaluations.values()
                    if e["status"] != "unknown"]
    if not all_statuses:
        banner_text, banner_color = "Position yourself in frame", (180, 180, 180)
    elif all(s == "correct" for s in all_statuses):
        banner_text, banner_color = "FORM: CORRECT", COLOR_CORRECT
    else:
        n = sum(1 for s in all_statuses if s == "incorrect")
        banner_text = f"FORM: {n} ISSUE(S) DETECTED"
        banner_color = COLOR_INCORRECT

    cv2.rectangle(frame, (0, h - 45), (w, h), (0, 0, 0), -1)
    ts, _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(frame, banner_text, ((w - ts[0]) // 2, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, banner_color, 2, cv2.LINE_AA)

    return frame
