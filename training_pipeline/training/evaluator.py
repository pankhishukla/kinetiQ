"""
training_pipeline/training/evaluator.py
=========================================
Metrics and evaluation helpers for classifier and anomaly model.

WHY a separate evaluator module?
    Mixing evaluation logic into the training loop makes it hard to:
    - Rerun evaluation on a saved model without retraining
    - Add new metrics later without touching training code
    Keeping it separate follows the Single Responsibility principle.
"""

import numpy as np
from typing import Optional
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from training_pipeline.config import Config

cfg = Config()


# ---------------------------------------------------------------------------
# CLASSIFIER EVALUATION
# ---------------------------------------------------------------------------

def evaluate_classifier(clf, X_val: np.ndarray, y_val_str: np.ndarray) -> dict:
    """
    Evaluate the exercise classifier on the validation set.

    Returns
    -------
    dict with accuracy, per-class precision/recall/f1
    """
    if clf._model is None:
        print("[Evaluator] Classifier not trained — skipping evaluation.")
        return {}

    y_pred = clf.predict(X_val)
    acc    = accuracy_score(y_val_str, y_pred)

    print(f"\n[Classifier Eval] Accuracy: {acc:.3f}")
    print(classification_report(y_val_str, y_pred))

    # WHY confusion matrix?
    #   It shows which exercise classes are confused with each other.
    #   E.g. if Squats_Start is often predicted as Lunges_Start, those
    #   two starting positions look similar in angle space — useful to know.
    cm = confusion_matrix(y_val_str, y_pred, labels=sorted(set(y_val_str)))
    if cfg.VERBOSE:
        print("[Classifier Eval] Confusion matrix (rows=true, cols=pred):")
        print(cm)

    # Feature importance (which joints matter most for classification)
    fi = clf.feature_importance()
    if fi:
        print("\n[Classifier Eval] Top features:")
        for name, score in fi[:5]:
            print(f"    {name:25s}  importance={score:.4f}")

    return {"accuracy": acc, "confusion_matrix": cm}


# ---------------------------------------------------------------------------
# ANOMALY MODEL EVALUATION
# ---------------------------------------------------------------------------

def evaluate_anomaly(anomaly_model, X_val: np.ndarray) -> dict:
    """
    Evaluate the anomaly detector on validation data.

    Since we don't have ground-truth anomaly labels, we report:
    - Score distribution statistics (useful for threshold tuning)
    - Fraction of val set flagged as anomalies at current threshold

    In practice, you would show a test user some deliberately bad form
    and check that scores drop below cfg.ANOMALY_SCORE_THRESHOLD.
    """
    if anomaly_model._model is None:
        print("[Evaluator] Anomaly model not trained — skipping evaluation.")
        return {}

    scores  = anomaly_model.score(X_val)
    flagged = (scores < cfg.ANOMALY_SCORE_THRESHOLD).mean()

    print(f"\n[Anomaly Eval] Score stats:")
    print(f"    mean={scores.mean():.4f}  std={scores.std():.4f}")
    print(f"    min={scores.min():.4f}  max={scores.max():.4f}")
    print(f"    Fraction flagged as anomaly: {flagged:.1%}  "
          f"(threshold={cfg.ANOMALY_SCORE_THRESHOLD})")

    # WHY report flagged fraction?
    #   If nearly 0% are flagged, the threshold is too strict.
    #   If > 30% are flagged on CORRECT-form validation data, the model is
    #   over-sensitive — adjust cfg.ANOMALY_CONTAMINATION or the threshold.

    return {"score_mean": scores.mean(), "flagged_fraction": flagged}


# ---------------------------------------------------------------------------
# THRESHOLD TUNER  (helper for anomaly model calibration)
# ---------------------------------------------------------------------------

def find_optimal_threshold(scores_correct: np.ndarray,
                            scores_incorrect: np.ndarray,
                            n_candidates: int = 100) -> float:
    """
    Given anomaly scores for correct-form and incorrect-form samples,
    find the threshold that maximises balanced accuracy.

    WHY needed?
        The default Isolation Forest threshold (0.0) is a rough heuristic.
        Using labelled good/bad examples (even a small held-out set) to
        tune the threshold significantly improves real-world precision.

    Parameters
    ----------
    scores_correct   : anomaly scores for known-good poses
    scores_incorrect : anomaly scores for known-bad poses
    n_candidates     : number of threshold values to try

    Returns
    -------
    best_threshold : float
    """
    all_scores = np.concatenate([scores_correct, scores_incorrect])
    candidates = np.linspace(all_scores.min(), all_scores.max(), n_candidates)
    best_thr, best_acc = candidates[0], 0.0

    y_true = np.array(
        [1] * len(scores_correct) + [-1] * len(scores_incorrect)
    )

    for thr in candidates:
        y_pred = np.where(all_scores >= thr, 1, -1)
        # Balanced accuracy: average of sensitivity and specificity
        tp = ((y_pred == 1)  & (y_true == 1)).sum()
        tn = ((y_pred == -1) & (y_true == -1)).sum()
        fp = ((y_pred == 1)  & (y_true == -1)).sum()
        fn = ((y_pred == -1) & (y_true == 1)).sum()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        bal_acc = (sens + spec) / 2.0
        if bal_acc > best_acc:
            best_acc, best_thr = bal_acc, thr

    print(f"[ThresholdTuner] Best threshold: {best_thr:.4f}  "
          f"balanced_acc={best_acc:.3f}")
    return best_thr
