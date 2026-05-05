"""
training_pipeline/features/extractor.py
=========================================
Converts preprocessed keypoint records into fixed-length feature vectors.

WHY joint angles as features (not raw coordinates)?
    Raw (x, y) coordinates change with:
        - Camera distance (zoom)
        - Body position in frame (left/right/centre)
        - Person height
    Joint angles are invariant to ALL of these: a squat at 90 degrees
    looks the same regardless of where the person stands or how tall they are.
    This makes the model generalisable across users and camera setups.

FEATURE VECTOR STRUCTURE:
    Each record produces a 1-D numpy array of joint angles (in degrees).
    The ordering is defined by ANGLE_FEATURES in config.py.
    Missing joints (not visible, not in skeleton) are filled with NaN
    and optionally imputed.

FORMULA:
    For three points A (proximal), B (vertex/joint), C (distal):
        angle = arccos( (BA . BC) / (|BA| * |BC|) )  [degrees]
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from training_pipeline.config import Config

cfg = Config()

# ---------------------------------------------------------------------------
# ANGLE DEFINITIONS
# ---------------------------------------------------------------------------
# Maps a feature name to the three body-part names forming the angle.
# Only body parts present in the dataset's 12-keypoint skeleton are used.

ANGLE_DEFS: Dict[str, Tuple[str, str, str]] = {
    # Knee angles: detect squat / lunge depth
    "right_knee":     ("right_hip",      "right_knee",     "right_ankle"),
    "left_knee":      ("left_hip",       "left_knee",      "left_ankle"),

    # Hip angles: detect torso lean / sit-up depth
    "right_hip":      ("right_shoulder", "right_hip",      "right_knee"),
    "left_hip":       ("left_shoulder",  "left_hip",       "left_knee"),

    # Elbow angles: detect push-up / bicep curl depth
    "right_elbow":    ("right_shoulder", "right_elbow",    "right_wrist"),
    "left_elbow":     ("left_shoulder",  "left_elbow",     "left_wrist"),

    # Shoulder angles: detect arm raise / jumping jack position
    "right_shoulder": ("right_hip",      "right_shoulder", "right_elbow"),
    "left_shoulder":  ("left_hip",       "left_shoulder",  "left_elbow"),

    # Body alignment: shoulder-hip-ankle straight line (plank / push-up)
    "right_body_align": ("right_shoulder", "right_hip",    "right_ankle"),
    "left_body_align":  ("left_shoulder",  "left_hip",     "left_ankle"),
}


# ---------------------------------------------------------------------------
# CORE ANGLE COMPUTATION
# ---------------------------------------------------------------------------

def _angle_at_vertex(ax: float, ay: float,
                     bx: float, by: float,
                     cx: float, cy: float) -> Optional[float]:
    """
    Compute the angle at B formed by A-B-C.
    Returns None if either vector has zero length (degenerate/coincident pts).
    """
    BA = np.array([ax - bx, ay - by], dtype=np.float64)
    BC = np.array([cx - bx, cy - by], dtype=np.float64)
    mag_BA = np.linalg.norm(BA)
    mag_BC = np.linalg.norm(BC)
    if mag_BA < 1e-6 or mag_BC < 1e-6:
        return None
    cos_a = np.clip(np.dot(BA, BC) / (mag_BA * mag_BC), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


# ---------------------------------------------------------------------------
# PER-RECORD FEATURE EXTRACTION
# ---------------------------------------------------------------------------

def extract_angles(keypoints: Dict[str, Tuple],
                   angle_names: List[str] = None) -> Dict[str, Optional[float]]:
    """
    Compute joint angles from a named keypoint dict.

    Parameters
    ----------
    keypoints   : {part_name: (x, y, visibility)}  from loader.py
    angle_names : which angles to compute (defaults to cfg.ANGLE_FEATURES)

    Returns
    -------
    {angle_name: float_degrees | None}
        None means one or more of the three required joints was missing.
    """
    if angle_names is None:
        angle_names = cfg.ANGLE_FEATURES

    result = {}
    for name in angle_names:
        if name not in ANGLE_DEFS:
            result[name] = None
            continue

        part_a, part_b, part_c = ANGLE_DEFS[name]
        kp_a = keypoints.get(part_a)
        kp_b = keypoints.get(part_b)
        kp_c = keypoints.get(part_c)

        if kp_a is None or kp_b is None or kp_c is None:
            result[name] = None
            continue

        ax, ay, _ = kp_a
        bx, by, _ = kp_b
        cx, cy, _ = kp_c
        result[name] = _angle_at_vertex(ax, ay, bx, by, cx, cy)

    return result


# ---------------------------------------------------------------------------
# BATCH FEATURE MATRIX BUILDER
# ---------------------------------------------------------------------------

def build_feature_matrix(
    records: List[Dict],
    angle_names: List[str] = None,
    fill_missing: float = -1.0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Convert a list of preprocessed records into a numpy feature matrix.

    WHY fill_missing = -1?
        -1 is outside the valid angle range [0, 180], so models can learn
        to treat it as a "joint not visible" sentinel.  Alternatively set
        fill_missing=np.nan and use an imputer (see preprocessor).

    Parameters
    ----------
    records      : list of dicts from preprocessor.preprocess()
    angle_names  : feature names to extract (default: cfg.ANGLE_FEATURES)
    fill_missing : value to substitute for None angles

    Returns
    -------
    X         : ndarray shape (N, F) — feature matrix
    y_labels  : ndarray shape (N,)  — raw label strings e.g. "Squats_Start"
    feat_names: list of F feature names (same order as X columns)
    """
    if angle_names is None:
        angle_names = cfg.ANGLE_FEATURES

    rows   = []
    labels = []

    for r in records:
        angles = extract_angles(r["keypoints"], angle_names)
        row    = [
            (angles[name] if angles[name] is not None else fill_missing)
            for name in angle_names
        ]
        rows.append(row)
        labels.append(f"{r['exercise']}_{r['phase']}")

    X        = np.array(rows,  dtype=np.float32)
    y_labels = np.array(labels, dtype=object)

    if cfg.VERBOSE:
        missing_frac = (X == fill_missing).mean()
        print(f"[Features] Matrix shape: {X.shape}  "
              f"Missing values: {missing_frac:.1%}")

    return X, y_labels, angle_names


# ---------------------------------------------------------------------------
# FEATURE NORMALISATION
# ---------------------------------------------------------------------------

def normalise_features(X_train: np.ndarray,
                       X_val: np.ndarray = None
                       ) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
    """
    Z-score normalise features using TRAINING set statistics only.

    WHY only training stats?
        Using the full dataset (including validation) to compute mean/std
        would constitute data leakage — the model would indirectly see val
        statistics during training.  We fit on train, then APPLY those
        same mean/std to val (and later to live inference).

    Returns
    -------
    X_train_norm, X_val_norm (or None), {"mean": ..., "std": ...}
    """
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0)
    std[std < 1e-6] = 1.0   # prevent division by zero for constant features

    X_train_norm = (X_train - mean) / std
    X_val_norm   = ((X_val - mean) / std) if X_val is not None else None

    return X_train_norm, X_val_norm, {"mean": mean, "std": std}
