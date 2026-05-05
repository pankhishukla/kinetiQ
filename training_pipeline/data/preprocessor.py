"""
training_pipeline/data/preprocessor.py
========================================
Cleans, normalises, and splits the raw records from loader.py.

WHY preprocessing matters:
    Raw keypoint coordinates are in pixel space of 640x640 images.
    If we trained on these directly, the model would learn "knee at pixel 400"
    rather than "knee at 60% of image height" — it would fail on any other
    resolution.  Normalisation removes this dependency.

    Visibility filtering is equally critical: including v=0 (unlabelled)
    or v=1 (occluded) points introduces random noise into the feature space.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import train_test_split

from training_pipeline.config import Config

cfg = Config()


# ---------------------------------------------------------------------------
# VISIBILITY FILTER
# ---------------------------------------------------------------------------

def filter_by_visibility(keypoints: Dict, min_visibility: int = 1) -> Dict:
    """
    Return only keypoints with visibility >= min_visibility.

    Visibility codes:
        0 = not labelled (position is a guess — discard)
        1 = labelled but occluded (use with caution)
        2 = fully visible (most reliable)

    WHY keep v=1?
        In many exercise images the wrist or ankle is partially hidden.
        Discarding all v=1 points would leave us with too few joints for
        meaningful angle computation, especially for push-ups and planks.
    """
    return {
        name: (x, y, v)
        for name, (x, y, v) in keypoints.items()
        if v >= min_visibility
    }


# ---------------------------------------------------------------------------
# BOUNDING-BOX NORMALISATION
# ---------------------------------------------------------------------------

def normalise_keypoints(keypoints: Dict, bbox: List[float]) -> Dict:
    """
    Normalise pixel coordinates to [0, 1] using the person's bounding box.

    WHY bounding-box normalisation (not image-size normalisation)?
        Using image dimensions still encodes WHERE in the frame the person
        stands.  Bounding-box normalisation centres on the person, making
        features position-independent: a squat in the top-left corner looks
        the same as one in the bottom-right.

    Parameters
    ----------
    keypoints : {name: (x, y, v)}
    bbox      : [x_min, y_min, width, height]  (COCO format)

    Returns
    -------
    Normalised keypoints with the same structure.
    """
    if not bbox or bbox[2] == 0 or bbox[3] == 0:
        return keypoints   # can't normalise without valid bbox

    bx, by, bw, bh = bbox
    normalised = {}
    for name, (x, y, v) in keypoints.items():
        nx = (x - bx) / bw   # [0, 1] within bounding box width
        ny = (y - by) / bh   # [0, 1] within bounding box height
        normalised[name] = (nx, ny, v)
    return normalised


# ---------------------------------------------------------------------------
# LABEL ENCODER
# ---------------------------------------------------------------------------

def build_label_maps(records: List[Dict]) -> Tuple[Dict, Dict]:
    """
    Build integer ↔ string mappings for the (exercise, phase) label.

    WHY encode labels as integers?
        Most ML libraries (scikit-learn, PyTorch) require integer class labels.
        We keep the reverse map so we can decode predictions back to strings
        for display.

    Returns
    -------
    label_to_int : {"Squats_Start": 0, "Squats_End": 1, ...}
    int_to_label : {0: "Squats_Start", 1: "Squats_End", ...}
    """
    unique_labels = sorted(
        set(f"{r['exercise']}_{r['phase']}" for r in records)
    )
    label_to_int = {lbl: i for i, lbl in enumerate(unique_labels)}
    int_to_label = {i: lbl for lbl, i in label_to_int.items()}
    return label_to_int, int_to_label


# ---------------------------------------------------------------------------
# TRAIN / VALIDATION SPLIT
# ---------------------------------------------------------------------------

def train_val_split(records: List[Dict],
                    label_to_int: Dict,
                    val_fraction: float = None,
                    random_seed: int = None
                    ) -> Tuple[List[Dict], List[Dict]]:
    """
    Stratified split preserving class proportions in both sets.

    WHY stratified?
        If one exercise has far fewer samples than others, a random split
        might leave it entirely in the training set — the validation metrics
        would then be meaningless for that class.

    Parameters
    ----------
    records       : all preprocessed records
    label_to_int  : mapping from build_label_maps()
    val_fraction  : fraction for validation (default: cfg.VAL_SPLIT)
    random_seed   : for reproducibility (default: cfg.RANDOM_SEED)

    Returns
    -------
    (train_records, val_records)
    """
    val_fraction = val_fraction or cfg.VAL_SPLIT
    random_seed  = random_seed  or cfg.RANDOM_SEED

    labels = [label_to_int[f"{r['exercise']}_{r['phase']}"] for r in records]

    train_idx, val_idx = train_test_split(
        range(len(records)),
        test_size=val_fraction,
        stratify=labels,
        random_state=random_seed,
    )

    train_records = [records[i] for i in train_idx]
    val_records   = [records[i] for i in val_idx]

    if cfg.VERBOSE:
        print(f"[Preprocessor] Train: {len(train_records)} | Val: {len(val_records)}")

    return train_records, val_records


# ---------------------------------------------------------------------------
# FULL PREPROCESSING PIPELINE (convenience wrapper)
# ---------------------------------------------------------------------------

def preprocess(records: List[Dict],
               min_visibility: int = 1,
               normalise: bool = True) -> List[Dict]:
    """
    Apply visibility filtering + normalisation to every record in-place.

    Returns the cleaned records (records with NO valid keypoints are dropped).
    """
    cleaned = []
    for r in records:
        kp = filter_by_visibility(r["keypoints"], min_visibility)
        if normalise and r.get("bbox"):
            kp = normalise_keypoints(kp, r["bbox"])
        if len(kp) < 4:
            # Need at least 4 visible joints to compute meaningful angles
            continue
        cleaned.append({**r, "keypoints": kp})

    dropped = len(records) - len(cleaned)
    if cfg.VERBOSE and dropped > 0:
        print(f"[Preprocessor] Dropped {dropped} records (too few visible joints)")

    return cleaned
