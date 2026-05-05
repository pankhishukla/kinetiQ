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
from src.form_evaluator import evaluate_form, REP_CONFIG, RepCounter


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
                                  exercise: str) -> dict:
    """
    Given the keypoint JSON from pose_service and an exercise name, compute
    joint angles and form feedback.

    Returns
    -------
    dict ready to be sent as JSON to the frontend:
    {
      "joints": {
        "right_knee": {
          "angle":    94.3,
          "x":        320.1,    # vertex pixel x
          "y":        210.5,    # vertex pixel y
          "status":   "correct",
          "cue":      "Right knee: Good depth",
          "color":    "#00dc00",
          "display":  "R Knee"
        },
        ...
      },
      "overall": "correct" | "incorrect" | "unknown",
      "issues":  2          # count of incorrect joints
    }
    """
    kp_list = _kp_list(keypoints_json)

    try:
        raw_angles = get_exercise_angles(kp_list, exercise)
        evaluations = evaluate_form(raw_angles, exercise)
    except ValueError:
        return {"joints": {}, "overall": "unknown", "issues": 0}

    joints = {}
    for joint_name, ev in evaluations.items():
        vx, vy = ev["vertex_xy"]
        joints[joint_name] = {
            "angle":   round(ev["angle"], 1) if ev["angle"] is not None else None,
            "x":       round(vx, 1),
            "y":       round(vy, 1),
            "status":  ev["status"],
            "cue":     ev["cue"],
            "color":   _bgr_to_hex(ev["color"]),
            "display": ev["display_name"],
        }

    # Overall form status
    statuses = [j["status"] for j in joints.values() if j["status"] != "unknown"]
    if not statuses:
        overall, issues = "unknown", 0
    elif all(s == "correct" for s in statuses):
        overall, issues = "correct", 0
    else:
        issues  = sum(1 for s in statuses if s == "incorrect")
        overall = "incorrect"

    return {"joints": joints, "overall": overall, "issues": issues}


def update_rep_counter(rep_counter: RepCounter,
                       joints: dict) -> int:
    """
    Feed the latest joint angles back into the rep counter.
    Returns the current rep count.

    WHY pass joints (already-evaluated dict) not raw angles?
        The rep counter only needs {joint_name: {"angle": val}} —
        which is exactly the shape of the joints dict from above.
    """
    angle_dict = {
        name: {"angle": data["angle"]}
        for name, data in joints.items()
    }
    return rep_counter.update(angle_dict)
