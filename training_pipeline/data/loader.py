"""
training_pipeline/data/loader.py
=================================
Loads the COCO-annotated exercise dataset and yields clean records.

WHY COCO format?
    Our dataset was annotated via Roboflow and exported as COCO JSON, which is
    the most widely supported annotation format.  Each JSON file contains:
        - "images"      : file metadata (id, filename, size)
        - "categories"  : class definitions WITH keypoint schemas per exercise
        - "annotations" : bounding boxes + 12 keypoints per person instance

HOW the 12-keypoint schema works:
    Unlike the standard COCO 17-keypoint format used by YOLOv8, this dataset
    uses a CUSTOM 12-keypoint skeleton (no face/head points).  Crucially, the
    KEYPOINT ORDER differs per category — "right_knee" might be index 4 in
    Squats but index 5 in Lunges.  We normalise this by building a name→index
    map per category so downstream code is order-independent.
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional

from training_pipeline.config import Config

cfg = Config()


# ---------------------------------------------------------------------------
# KEYPOINT NAME NORMALISER
# ---------------------------------------------------------------------------
def _norm_kp(name: str) -> str:
    """Canonicalise keypoint names — fix typos from the Roboflow export."""
    return (name.replace("-", "_")
                .replace("right_soulder", "right_shoulder")
                .replace("left_wrsit",    "left_wrist"))


# ---------------------------------------------------------------------------
# EXERCISE NAME MAPPER
# ---------------------------------------------------------------------------
# Maps verbose COCO category names → short internal keys.
_EXERCISE_MAP = {
    "squats":       "Squats",
    "lunges":       "Lunges",
    "pushup":       "Push-up",
    "situps":       "Sit-ups",
    "jumpingjacks": "JumpingJacks",
}

def _cat_to_exercise(cat_name: str) -> Optional[str]:
    """Return the canonical exercise key for a COCO category name."""
    cn = cat_name.replace("-", "").replace(" ", "").replace("_", "").lower()
    for key, val in _EXERCISE_MAP.items():
        if key in cn:
            return val
    return None   # unknown category (e.g. generic "Humans")


# ---------------------------------------------------------------------------
# SINGLE JSON LOADER
# ---------------------------------------------------------------------------

def _load_json(json_path: Path) -> tuple:
    """
    Parse one COCO JSON file.

    Returns
    -------
    annotations : list of raw annotation dicts
    cat_info    : dict { cat_id: {name, exercise, phase, kp_map} }
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    cat_info = {}
    for cat in data.get("categories", []):
        # Build name→index map for this category's unique keypoint ordering
        kp_map = {_norm_kp(kp): i
                  for i, kp in enumerate(cat.get("keypoints", []))}
        exercise = _cat_to_exercise(cat["name"])
        # "End" category = bottom of rep, "Start" = top of rep
        phase = "End" if "end" in cat["name"].lower() else "Start"
        cat_info[cat["id"]] = {
            "name":     cat["name"],
            "exercise": exercise,
            "phase":    phase,
            "kp_map":   kp_map,
        }

    return data.get("annotations", []), cat_info


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def load_dataset(split: str = "both") -> List[Dict]:
    """
    Load all annotations from the dataset.

    Parameters
    ----------
    split : "train" | "test" | "both"
        Which COCO JSON files to read.

    Returns
    -------
    List of record dicts:
        {
          "exercise"   : str   — e.g. "Squats"
          "phase"      : str   — "Start" or "End"
          "keypoints"  : dict  — { body_part_name: (x, y, visibility) }
          "bbox"       : list  — [x, y, w, h]
          "image_id"   : int
        }

    WHY return named keypoints (not flat arrays)?
        Downstream feature extraction needs to look up specific joints by
        name (e.g. "right_knee") regardless of their position in the array.
        Named dicts make that lookup O(1) and prevent off-by-one errors.
    """
    paths = {
        "train": [cfg.TRAIN_JSON],
        "test":  [cfg.TEST_JSON],
        "both":  [cfg.TRAIN_JSON, cfg.TEST_JSON],
    }.get(split, [cfg.TRAIN_JSON, cfg.TEST_JSON])

    records = []

    for json_path in paths:
        if not json_path.exists():
            print(f"[WARN] Dataset file not found: {json_path}")
            continue

        annotations, cat_info = _load_json(json_path)
        n_skipped = 0

        for ann in annotations:
            cat = cat_info.get(ann["category_id"])
            if cat is None or cat["exercise"] is None:
                n_skipped += 1
                continue   # unknown category

            kp_map  = cat["kp_map"]
            flat_kp = ann.get("keypoints", [])

            # Convert flat [x,y,v, x,y,v, ...] to {name: (x, y, v)}
            named_kp = {}
            for part_name, idx in kp_map.items():
                x = flat_kp[idx * 3]
                y = flat_kp[idx * 3 + 1]
                v = flat_kp[idx * 3 + 2]   # visibility: 0=no label, 1=occluded, 2=visible
                named_kp[part_name] = (x, y, v)

            records.append({
                "exercise":  cat["exercise"],
                "phase":     cat["phase"],
                "keypoints": named_kp,
                "bbox":      ann.get("bbox", []),
                "image_id":  ann.get("image_id"),
            })

        if cfg.VERBOSE:
            print(f"[Loader] {json_path.name}: "
                  f"{len(annotations)} annotations loaded, "
                  f"{n_skipped} skipped (unknown category)")

    print(f"[Loader] Total usable records: {len(records)}")
    return records


def get_label_distribution(records: List[Dict]) -> Dict:
    """
    Return a count of records per (exercise, phase) pair.
    Useful for spotting class imbalance before training.
    """
    counts = defaultdict(int)
    for r in records:
        counts[(r["exercise"], r["phase"])] += 1
    return dict(counts)
