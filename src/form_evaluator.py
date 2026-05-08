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

    # Bicep Curl — flag if arm stays too straight (never curled, >155) or hyperflexed (<20)
    "bicep_curl": {
        "left_elbow": {
            "min": 20, "max": 155,
            "cue_low":  "Don't lock out — keep a slight bend",
            "cue_high": "Curl higher — bring wrist to shoulder",
            "cue_good": "Left arm: Great curl!",
        },
        "right_elbow": {
            "min": 20, "max": 155,
            "cue_low":  "Don't lock out — keep a slight bend",
            "cue_high": "Curl higher — bring wrist to shoulder",
            "cue_good": "Right arm: Great curl!",
        },
        "left_shoulder": {
            "min": 0, "max": 25,
            "cue_low":  "",
            "cue_high": "Keep elbows tucked to your sides",
            "cue_good": "L Elbow: Tucked",
        },
        "right_shoulder": {
            "min": 0, "max": 25,
            "cue_low":  "",
            "cue_high": "Keep elbows tucked to your sides",
            "cue_good": "R Elbow: Tucked",
        },
        "left_hip": {
            "min": 0, "max": 15,
            "cue_low":  "",
            "cue_high": "Don't lean back — keep spine vertical",
            "cue_good": "Spine: Straight",
        },
        "right_hip": {
            "min": 0, "max": 15,
            "cue_low":  "",
            "cue_high": "Don't lean back — keep spine vertical",
            "cue_good": "Spine: Straight",
        },
    },

    # Squat — standing (~175) is INCORRECT (>140). Correct range: 60-140 (squatting)
    "squat": {
        "left_knee": {
            "min": 60, "max": 140,
            "cue_low":  "Too deep — rise slightly",
            "cue_high": "Squat deeper — bend your knees",
            "cue_good": "Left knee: Good depth",
        },
        "right_knee": {
            "min": 60, "max": 140,
            "cue_low":  "Too deep — rise slightly",
            "cue_high": "Squat deeper — bend your knees",
            "cue_good": "Right knee: Good depth",
        },
        "left_hip": {
            "min": 55, "max": 140,
            "cue_low":  "Lean back — chest up!",
            "cue_high": "Hinge at hip — sit back more",
            "cue_good": "Left hip: Good posture",
        },
        "right_hip": {
            "min": 55, "max": 140,
            "cue_low":  "Lean back — chest up!",
            "cue_high": "Hinge at hip — sit back more",
            "cue_good": "Right hip: Good posture",
        },
    },

    # Lateral Raise — arms at sides (~15) is INCORRECT (<60). Correct: 60-120
    "lateral_raise": {
        "left_shoulder": {
            "min": 60, "max": 120,
            "cue_low":  "Raise left arm to shoulder height",
            "cue_high": "Lower left arm — too high",
            "cue_good": "Left shoulder: Good height",
        },
        "right_shoulder": {
            "min": 60, "max": 120,
            "cue_low":  "Raise right arm to shoulder height",
            "cue_high": "Lower right arm — too high",
            "cue_good": "Right shoulder: Good height",
        },
    },

    # Push-Up — arms fully straight (>165) is INCORRECT. Correct: 65-165
    "push_up": {
        "left_elbow": {
            "min": 65, "max": 165,
            "cue_low":  "Don't go too low — protect shoulders",
            "cue_high": "Lower your chest — bend elbows more",
            "cue_good": "L Elbow: Good depth",
        },
        "right_elbow": {
            "min": 65, "max": 165,
            "cue_low":  "Don't go too low — protect shoulders",
            "cue_high": "Lower your chest — bend elbows more",
            "cue_good": "R Elbow: Good depth",
        },
        "left_hip": {
            "min": 155, "max": 195,
            "cue_low":  "Hips sagging — engage your core!",
            "cue_high": "Hips too high — lower them",
            "cue_good": "Body alignment: Good",
        },
    },

    # Lunge — standing is INCORRECT (>130). Correct: 70-130 (lunge position)
    "lunge": {
        "left_knee": {
            "min": 70, "max": 130,
            "cue_low":  "Don't let knee go past toes",
            "cue_high": "Lunge deeper — bend front knee to 90",
            "cue_good": "Front knee: Good angle",
        },
        "right_knee": {
            "min": 70, "max": 130,
            "cue_low":  "Back knee too close to ground",
            "cue_high": "Drop back knee lower",
            "cue_good": "Back knee: Good position",
        },
        "left_hip": {
            "min": 65, "max": 150,
            "cue_low":  "Stand taller — chest up",
            "cue_high": "Hinge forward more at the hip",
            "cue_good": "Torso: Upright",
        },
    },

    # Plank — body must be straight (155-195). Sag or pike = INCORRECT
    "plank": {
        "left_hip": {
            "min": 155, "max": 195,
            "cue_low":  "Hips sagging — lift them up",
            "cue_high": "Hips too high — lower them",
            "cue_good": "L Body: Aligned",
        },
        "right_hip": {
            "min": 155, "max": 195,
            "cue_low":  "Hips sagging — lift them up",
            "cue_high": "Hips too high — lower them",
            "cue_good": "R Body: Aligned",
        },
        "left_knee": {
            "min": 160, "max": 210,
            "cue_low":  "Straighten your left leg",
            "cue_high": "Keep left leg straight",
            "cue_good": "L Leg: Straight",
        },
        "right_knee": {
            "min": 160, "max": 210,
            "cue_low":  "Straighten your right leg",
            "cue_high": "Keep right leg straight",
            "cue_good": "R Leg: Straight",
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
# SECTION 3 — REP COUNTER CLASS
# ---------------------------------------------------------------------------

class RepCounter:
    """
    Two-state machine (up / down) for counting exercise repetitions.

    WHY two states?
        "Up" and "Down" represent the two ends of a rep.  The counter only
        increments on a confirmed state TRANSITION, not on every frame.
        This is immune to jitter — even if the angle fluctuates near the
        threshold it won't double-count.
    """

    def __init__(self, exercise_name):
        self.exercise_name = exercise_name
        self.count = 0
        self.state = "up"       # assume starting in extended position
        cfg = REP_CONFIG.get(exercise_name, {})
        self.primary_joint  = cfg.get("primary_joint")
        self.down_threshold = cfg.get("down_threshold")
        self.up_threshold   = cfg.get("up_threshold")
        self.count_on       = cfg.get("count_on")

    def update(self, angles_dict):
        """Feed the latest smoothed angles; returns current rep count."""
        if self.primary_joint is None:
            return self.count           # plank / hold exercise

        joint_data = angles_dict.get(self.primary_joint)
        if joint_data is None:
            return self.count
        angle = joint_data.get("angle")
        if angle is None:
            return self.count

        # State transitions with hysteresis
        if self.state == "up" and angle < self.down_threshold:
            self.state = "down"
        elif self.state == "down" and angle > self.up_threshold:
            self.state = "up"
            if self.count_on == "up":
                self.count += 1

        return self.count

    def reset(self):
        self.count = 0
        self.state = "up"


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

def evaluate_form(angles_dict, exercise_name):
    """
    Compare each joint's current angle against its rule thresholds.

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

        if angle < rule["min"]:
            status, cue, color = "incorrect", rule["cue_low"],  COLOR_INCORRECT
        elif angle > rule["max"]:
            status, cue, color = "incorrect", rule["cue_high"], COLOR_INCORRECT
        else:
            status, cue, color = "correct",   rule["cue_good"], COLOR_CORRECT

        evaluations[joint_name] = {
            "status": status, "cue": cue, "angle": angle,
            "color": color, "vertex_xy": vertex_xy, "display_name": disp_name,
        }

    return evaluations


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
