"""
tools/calibrate_thresholds.py
==============================
Parses the COCO-annotated Dataset, computes real joint angle statistics,
and outputs data-calibrated thresholds for FORM_RULES in src/form_evaluator.py.

Also trains one Isolation Forest anomaly detector per exercise and saves all
models to models/phase5_models.pkl.

Run with:   python tools/calibrate_thresholds.py

WHY keep this as a separate tool (not part of app.py)?
    Calibration is an offline batch process — it only needs to run once after
    the dataset is updated.  Keeping it out of the live app keeps the app fast
    and ensures the calibration logic is independently testable.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict
from sklearn.ensemble import IsolationForest

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR    = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "Dataset"
TRAIN_JSON  = DATASET_DIR / "train" / "_annotations.coco.json"
TEST_JSON   = DATASET_DIR / "test"  / "_annotations.coco.json"
OUTPUT_PKL  = BASE_DIR / "models" / "phase5_models.pkl"


# ---------------------------------------------------------------------------
# KEYPOINT NAME NORMALISER
# ---------------------------------------------------------------------------
# WHY needed? The Roboflow COCO export contains typos like "right_soulder"
# and hyphen-separated names like "left-ankle".  We canonicalise all names
# before using them so the rest of the code is typo-free.

def norm_kp(name):
    return (name.replace("-", "_")
                .replace("right_soulder", "right_shoulder")
                .replace("left_wrsit",    "left_wrist"))


# ---------------------------------------------------------------------------
# ANGLE MATH  (identical to src/angle_engine.py — duplicated intentionally
# so this tool has ZERO dependency on the live-app src package)
# ---------------------------------------------------------------------------

def angle_at_b(ax, ay, bx, by, cx, cy):
    BA = np.array([ax - bx, ay - by], dtype=np.float64)
    BC = np.array([cx - bx, cy - by], dtype=np.float64)
    mBA, mBC = np.linalg.norm(BA), np.linalg.norm(BC)
    if mBA < 1e-6 or mBC < 1e-6:
        return None
    cos_a = np.clip(np.dot(BA, BC) / (mBA * mBC), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


# ---------------------------------------------------------------------------
# ANGLE DEFINITIONS per exercise
# ---------------------------------------------------------------------------
# These mirror the dataset's exercise categories.  Only angles computable
# from the 12-keypoint custom skeleton are listed here.

EXERCISE_ANGLES = {
    "Squats": {
        "right_knee": ("right_hip",      "right_knee",  "right_ankle"),
        "left_knee":  ("left_hip",       "left_knee",   "left_ankle"),
        "right_hip":  ("right_shoulder", "right_hip",   "right_knee"),
    },
    "Lunges": {
        "right_knee": ("right_hip",      "right_knee",  "right_ankle"),
        "left_knee":  ("left_hip",       "left_knee",   "left_ankle"),
        "right_hip":  ("right_shoulder", "right_hip",   "right_knee"),
    },
    "Push-up": {
        "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
        "left_elbow":  ("left_shoulder",  "left_elbow",  "left_wrist"),
        "right_hip":   ("right_shoulder", "right_hip",   "right_ankle"),
    },
    "Sit-ups": {
        "right_hip": ("right_shoulder", "right_hip", "right_knee"),
        "left_hip":  ("left_shoulder",  "left_hip",  "left_knee"),
    },
    "JumpingJacks": {
        "right_shoulder": ("right_hip", "right_shoulder", "right_elbow"),
        "left_shoulder":  ("left_hip",  "left_shoulder",  "left_elbow"),
    },
}


def cat_to_exercise(cat_name):
    """Map a COCO category name to one of our exercise keys."""
    cn = cat_name.replace("-", "").replace(" ", "").lower()
    for ex in EXERCISE_ANGLES:
        if ex.lower().replace("-", "").replace(" ", "") in cn:
            return ex
    return None


# ---------------------------------------------------------------------------
# COCO JSON LOADER
# ---------------------------------------------------------------------------

def load_coco(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    cat_info = {}
    for cat in data["categories"]:
        kp_map = {norm_kp(kp): i for i, kp in enumerate(cat.get("keypoints", []))}
        ex    = cat_to_exercise(cat["name"])
        phase = "End" if "end" in cat["name"].lower() else "Start"
        cat_info[cat["id"]] = {
            "name": cat["name"], "kp_map": kp_map,
            "exercise": ex, "phase": phase,
        }
    return data["annotations"], cat_info


# ---------------------------------------------------------------------------
# ANGLE EXTRACTOR (per annotation)
# ---------------------------------------------------------------------------

def extract_angles(ann, cat_info):
    cat = cat_info.get(ann["category_id"])
    if cat is None or cat["exercise"] is None:
        return None
    ex, phase, kp_map = cat["exercise"], cat["phase"], cat["kp_map"]
    kpts = ann["keypoints"]

    def get_kp(name):
        idx = kp_map.get(name)
        if idx is None:
            return None
        return kpts[idx * 3], kpts[idx * 3 + 1], kpts[idx * 3 + 2]

    angles = {}
    for angle_name, (a_part, b_part, c_part) in EXERCISE_ANGLES[ex].items():
        pa, pb, pc = get_kp(a_part), get_kp(b_part), get_kp(c_part)
        if pa is None or pb is None or pc is None:
            continue
        if pa[2] == 0 or pb[2] == 0 or pc[2] == 0:   # not labeled
            continue
        ang = angle_at_b(pa[0], pa[1], pb[0], pb[1], pc[0], pc[1])
        if ang is not None:
            angles[angle_name] = ang

    return {"exercise": ex, "phase": phase, "angles": angles}


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    # -- Load all annotations --
    records = []
    for json_path in [TRAIN_JSON, TEST_JSON]:
        if not json_path.exists():
            print(f"[WARN] Not found: {json_path}")
            continue
        anns, cat_info = load_coco(json_path)
        for ann in anns:
            r = extract_angles(ann, cat_info)
            if r and r["angles"]:
                records.append(r)

    print(f"\n[INFO] Total valid annotations: {len(records)}")

    # -- Group angles by exercise + phase --
    groups = defaultdict(lambda: defaultdict(list))
    for r in records:
        for angle_name, val in r["angles"].items():
            groups[r["exercise"]][(r["phase"], angle_name)].append(val)

    # -- Compute calibrated thresholds --
    print("\n" + "=" * 70)
    print("  CALIBRATED THRESHOLDS  (from real dataset)")
    print("=" * 70)

    calibrated_rules = {}

    for ex, angle_groups in sorted(groups.items()):
        print(f"\n>>>  {ex}")
        calibrated_rules[ex] = {}

        joint_data = defaultdict(lambda: {"Start": [], "End": []})
        for (phase, angle_name), vals in angle_groups.items():
            joint_data[angle_name][phase] = vals

        for joint, phases in joint_data.items():
            sv = np.array(phases["Start"]) if phases["Start"] else None
            ev = np.array(phases["End"])   if phases["End"]   else None

            s_mean = round(float(sv.mean()), 1) if sv is not None and len(sv) > 2 else None
            s_p95  = float(np.percentile(sv, 95)) if sv is not None and len(sv) > 2 else None
            e_mean = round(float(ev.mean()), 1) if ev is not None and len(ev) > 2 else None
            e_p5   = float(np.percentile(ev, 5)) if ev is not None and len(ev) > 2 else None

            if e_p5 is not None and s_p95 is not None:
                thr_min, thr_max = round(e_p5, 1), round(s_p95, 1)
            elif e_p5 is not None:
                thr_min, thr_max = round(e_p5, 1), round(e_p5 + 60, 1)
            elif s_p95 is not None:
                thr_max = round(s_p95, 1)
                thr_min = max(0, round(s_p95 - 60, 1))
            else:
                continue

            n_s, n_e = len(phases["Start"]), len(phases["End"])
            print(f"   {joint:20s}  "
                  f"Start n={n_s:3d} mean={s_mean}  "
                  f"End n={n_e:3d} mean={e_mean}  "
                  f"  --> [{thr_min} deg, {thr_max} deg]")

            calibrated_rules[ex][joint] = {
                "min": thr_min, "max": thr_max,
                "start_mean": s_mean, "end_mean": e_mean,
                "n_start": n_s, "n_end": n_e,
            }

    # -- Train Isolation Forest per exercise --
    print("\n" + "=" * 70)
    print("  ANOMALY DETECTORS  (Isolation Forest per exercise)")
    print("=" * 70)

    anomaly_models = {}
    joint_order    = {}

    for ex, angle_groups in sorted(groups.items()):
        joint_data = defaultdict(lambda: {"Start": [], "End": []})
        for (phase, angle_name), vals in angle_groups.items():
            joint_data[angle_name][phase] = vals

        joints = sorted(joint_data.keys())
        joint_order[ex] = joints

        all_samples = defaultdict(dict)
        for (phase, angle_name), vals in angle_groups.items():
            for i, v in enumerate(vals):
                all_samples[(phase, i)][angle_name] = v

        X = [[row[j] for j in joints]
             for row in all_samples.values()
             if all(j in row for j in joints)]

        if len(X) < 5:
            print(f"   {ex}: too few complete samples ({len(X)}) - skipped")
            continue

        X = np.array(X)
        model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
        model.fit(X)
        anomaly_models[ex] = model
        print(f"   {ex}: trained on {len(X)} samples  features={joints}")

    # -- Save --
    payload = {
        "calibrated_rules": calibrated_rules,
        "anomaly_models":   anomaly_models,
        "joint_order":      joint_order,
    }
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(payload, f)
    print(f"\n[INFO] Saved -> {OUTPUT_PKL}")

    # -- Print paste-ready snippet for form_evaluator.py --
    EX_NAME_MAP = {
        "Squats":       "squat",
        "Lunges":       "lunge",
        "Push-up":      "push_up",
        "Sit-ups":      "sit_up",
        "JumpingJacks": "jumping_jacks",
    }
    print("\n" + "=" * 70)
    print("  PASTE INTO src/form_evaluator.py  ->  FORM_RULES")
    print("=" * 70)
    for ex, joints in calibrated_rules.items():
        sys_name = EX_NAME_MAP.get(ex, ex.lower())
        print(f'\n    "{sys_name}": {{')
        for joint, s in joints.items():
            print(f'        "{joint}": {{"min": {s["min"]}, "max": {s["max"]}}},')
            print(f'          # Start mean={s["start_mean"]} deg (n={s["n_start"]})'
                  f', End mean={s["end_mean"]} deg (n={s["n_end"]})')
        print(f'    }},')


if __name__ == "__main__":
    main()
