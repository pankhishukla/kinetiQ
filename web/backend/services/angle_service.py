"""
web/backend/services/angle_service.py
=======================================
Wraps src/angle_engine.py + src/form_evaluator.py for the web API layer.

WHY a thin wrapper instead of calling src directly from the route?
    Routes should speak HTTP/JSON — they should not know about numpy arrays,
    OpenCV BGR tuples, or Python-side angle dicts.  This service translates
    between "Python internal types" and "JSON-serialisable types" in one place.

TRANSLATION that happens here:
    src angle dict   : {"right_knee": {"angle": 94.3, "vertex_xy": (320.1, 210.5), ...}}
    JSON sent to UI  : {"right_knee": {"angle": 94.3, "x": 320.1, "y": 210.5,
                                        "status": "correct", "cue": "...", "color": "#00dc00"}}
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.angle_engine import get_exercise_angles
from src.form_evaluator import evaluate_form, REP_CONFIG, RepCounter, calculate_posture_score, _inject_pushup_coord_checks
from src.keypoint_smoother import KeypointSmoother


# ---------------------------------------------------------------------------
# COLOR CONVERSION  BGR → hex
# ---------------------------------------------------------------------------

def _bgr_to_hex(bgr: tuple) -> str:
    """
    Convert an OpenCV BGR tuple to a CSS hex color string.
    WHY? JavaScript canvas uses CSS colors; Python uses BGR tuples.
    """
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


# ---------------------------------------------------------------------------
# KEYPOINT LIST REBUILD
# ---------------------------------------------------------------------------

def _kp_list(keypoints_json: list) -> list:
    """
    Rebuild the (x, y, conf) tuple list from the JSON dicts returned by
    pose_service.run_inference().  Needed because src/angle_engine.py
    expects a list of tuples, not dicts.
    """
    return [(kp["x"], kp["y"], kp["conf"]) for kp in keypoints_json]


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def compute_angles_and_feedback(keypoints_json: list,
                                  exercise: str,
                                  current_phase: str = "up",
                                  smoother: KeypointSmoother = None) -> dict:
    """
    Given the keypoint JSON from pose_service, an exercise name, and the current phase, compute
    joint angles and form feedback.

    Parameters
    ----------
    keypoints_json : list of keypoint dicts from pose_service
    exercise       : exercise name string
    current_phase  : rep phase ("up" | "down" | "transition")
    smoother       : optional KeypointSmoother instance for this session.
                     If provided, raw keypoints are stabilised before angle math.

    Returns
    -------
    dict ready to be sent as JSON to the frontend:
    {
      "joints": {
        "right_knee": {
          "angle":    94.3,
          "x":        320.1,    # vertex pixel x
          "y":        210.5,    # vertex pixel y
          "status":   "correct" | "incorrect" | "uncertain" | "unknown",
          "cue":      "Right knee: Good depth",
          "color":    "#00dc00",
          "display":  "R Knee",
          "min_conf": 0.82,
        },
        ...
      },
      "overall": "excellent" | "good" | "poor" | "unknown",
      "score":   92.5,      # posture score 0-100
      "issues":  2          # count of incorrect joints
    }
    """
    # Apply temporal smoothing if a smoother is provided
    if smoother is not None:
        keypoints_json = smoother.update(keypoints_json)

    kp_list = _kp_list(keypoints_json)

    try:
        raw_angles = get_exercise_angles(kp_list, exercise)
        evaluations = evaluate_form(raw_angles, exercise, current_phase)
        # Inject coordinate-space hand placement checks for push-ups
        if exercise == "push_up":
            _inject_pushup_coord_checks(kp_list, evaluations, current_phase)
        score, overall = calculate_posture_score(evaluations, exercise)
    except ValueError:
        return {"joints": {}, "overall": "unknown", "score": 0.0, "issues": 0}

    joints = {}
    for joint_name, ev in evaluations.items():
        vx, vy = ev["vertex_xy"]
        joints[joint_name] = {
            "angle":    round(ev["angle"], 1) if ev["angle"] is not None else None,
            "x":        round(vx, 1),
            "y":        round(vy, 1),
            "status":   ev["status"],
            "cue":      ev["cue"],
            "color":    _bgr_to_hex(ev["color"]),
            "display":  ev["display_name"],
            "min_conf": round(ev.get("min_conf", 1.0), 3),
        }

    issues = sum(1 for j in joints.values() if j["status"] == "incorrect")

    return {"joints": joints, "overall": overall, "score": score, "issues": issues}


def update_rep_counter(rep_counter: RepCounter,
                       joints: dict) -> int:
    """
    Feed the latest joint angles and status back into the rep counter.
    Returns the current rep count.
    """
    angle_dict = {
        name: {"angle": data["angle"]}
        for name, data in joints.items()
    }
    return rep_counter.update(angle_dict, form_evaluations=joints)
