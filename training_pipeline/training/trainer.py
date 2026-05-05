"""
training_pipeline/training/trainer.py
======================================
Orchestrates the full training pipeline: data -> features -> models -> save.

IMPORTANT: This file defines the STRUCTURE only.
    The actual model fitting calls are commented out as placeholders.
    Run this file to verify the pipeline data flow without training.

WHY structure first?
    Writing the full pipeline structure before training serves two purposes:
    1. It validates that data flows correctly through every module
    2. It documents exactly what training will do, making it reviewable

TO ENABLE TRAINING:
    Uncomment the sections marked "# --- TRAIN ---" below.
"""

import pickle
import numpy as np
from pathlib import Path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training_pipeline.config import Config
from training_pipeline.data.loader import load_dataset, get_label_distribution
from training_pipeline.data.preprocessor import (
    preprocess, build_label_maps, train_val_split,
)
from training_pipeline.features.extractor import (
    build_feature_matrix, normalise_features,
)
from training_pipeline.models.anomaly_model import AnomalyModel
from training_pipeline.models.classifier_model import ExerciseClassifier
from training_pipeline.training.evaluator import evaluate_classifier, evaluate_anomaly

cfg = Config()


# ---------------------------------------------------------------------------
# STEP 1 — DATA LOADING
# ---------------------------------------------------------------------------

def step_load_data():
    """Load + preprocess the full dataset."""
    print("\n[Step 1] Loading dataset...")

    # Load all COCO annotations from train + test splits
    records = load_dataset(split="both")

    # Show class distribution — spot imbalances early
    dist = get_label_distribution(records)
    print("[Step 1] Label distribution:")
    for label, count in sorted(dist.items()):
        print(f"         {label[0]:20s} {label[1]:6s}  -> {count:4d} samples")

    # Clean: filter low-confidence keypoints + normalise coordinates
    records = preprocess(records, min_visibility=1, normalise=True)

    return records


# ---------------------------------------------------------------------------
# STEP 2 — FEATURE EXTRACTION
# ---------------------------------------------------------------------------

def step_extract_features(records):
    """Convert keypoint records into angle feature matrices."""
    print("\n[Step 2] Extracting features...")

    # Build integer label mapping (needed for classifier training)
    label_to_int, int_to_label = build_label_maps(records)
    print(f"[Step 2] Classes: {list(label_to_int.keys())}")

    # Stratified train/val split BEFORE feature extraction
    # WHY before? Ensures no data leakage in normalisation stats.
    train_records, val_records = train_val_split(records, label_to_int)

    # Convert records to numpy feature matrices
    X_train, y_train_str, feat_names = build_feature_matrix(train_records)
    X_val,   y_val_str,   _          = build_feature_matrix(val_records)

    # Integer labels for the classifier
    y_train_int = np.array([label_to_int[lbl] for lbl in y_train_str])
    y_val_int   = np.array([label_to_int[lbl] for lbl in y_val_str])

    # Normalise (fit on train, apply to val)
    X_train_norm, X_val_norm, norm_params = normalise_features(X_train, X_val)

    print(f"[Step 2] Feature matrix: train={X_train_norm.shape}, "
          f"val={X_val_norm.shape}")
    print(f"[Step 2] Features: {feat_names}")

    return {
        "X_train":      X_train_norm,
        "X_val":        X_val_norm,
        "y_train_int":  y_train_int,
        "y_val_int":    y_val_int,
        "y_train_str":  y_train_str,
        "y_val_str":    y_val_str,
        "feat_names":   feat_names,
        "label_to_int": label_to_int,
        "int_to_label": int_to_label,
        "norm_params":  norm_params,
    }


# ---------------------------------------------------------------------------
# STEP 3 — ANOMALY MODEL TRAINING
# ---------------------------------------------------------------------------

def step_train_anomaly(data: dict) -> AnomalyModel:
    """
    Train the Isolation Forest anomaly detector.

    WHY train on ALL data (not just train split)?
        For anomaly detection we want to model the full distribution of
        correct exercise poses.  The train/val split is more relevant for
        the classifier where generalisation matters per class.
    """
    print("\n[Step 3] Training anomaly model...")

    model = AnomalyModel()

    model.train(
        X             = data["X_train"],
        feature_names = data["feat_names"],
        norm_params   = data["norm_params"],
    )

    return model


# ---------------------------------------------------------------------------
# STEP 4 — CLASSIFIER TRAINING
# ---------------------------------------------------------------------------

def step_train_classifier(data: dict) -> ExerciseClassifier:
    """Train the exercise / phase classifier."""
    print("\n[Step 4] Training exercise classifier...")

    clf = ExerciseClassifier()

    clf.train(
        X            = data["X_train"],
        y_int        = data["y_train_int"],
        label_to_int = data["label_to_int"],
        int_to_label = data["int_to_label"],
        feature_names= data["feat_names"],
    )

    return clf


# ---------------------------------------------------------------------------
# STEP 5 — EVALUATION
# ---------------------------------------------------------------------------

def step_evaluate(clf: ExerciseClassifier,
                  anomaly: AnomalyModel,
                  data: dict) -> None:
    """Run evaluation metrics on validation set."""
    print("\n[Step 5] Evaluating models...")
    # Evaluation functions are no-ops when models are untrained.
    evaluate_classifier(clf, data["X_val"], data["y_val_str"])
    evaluate_anomaly(anomaly, data["X_val"])


# ---------------------------------------------------------------------------
# STEP 6 — SAVE
# ---------------------------------------------------------------------------

def step_save(clf: ExerciseClassifier,
              anomaly: AnomalyModel,
              norm_params: dict) -> None:
    """Persist trained models and normalisation params."""
    print("\n[Step 6] Saving models...")
    clf.save()
    anomaly.save()

    # Also save normalisation params so live inference can replicate
    # the same scaling that was applied at training time.
    norm_path = cfg.MODELS_DIR / "norm_params.pkl"
    with open(norm_path, "wb") as f:
        pickle.dump(norm_params, f)
    print(f"[Step 6] Norm params saved -> {norm_path}")


# ---------------------------------------------------------------------------
# MAIN PIPELINE RUNNER
# ---------------------------------------------------------------------------

def run_pipeline():
    print("=" * 60)
    print("  EXERCISE FORM TRAINING PIPELINE")
    print("=" * 60)

    records = step_load_data()
    data    = step_extract_features(records)
    anomaly = step_train_anomaly(data)
    clf     = step_train_classifier(data)
    step_evaluate(clf, anomaly, data)
    step_save(clf, anomaly, data["norm_params"])

    print("\n[Pipeline] Training complete. Models saved to models/ directory.")


if __name__ == "__main__":
    run_pipeline()
