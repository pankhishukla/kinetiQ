"""
training_pipeline/models/classifier_model.py
=============================================
Random Forest classifier to identify the current exercise and rep phase.

WHY a classifier (separate from anomaly detection)?
    The anomaly model answers "is this pose NORMAL for the current exercise?"
    The classifier answers "WHICH exercise is the user doing?"

    If integrated into app.py, this enables AUTO-DETECTION of the exercise
    being performed, removing the need for the user to press 1-7 to select it.

WHY Random Forest?
    - Handles the mixed continuous + bounded feature space well
    - Provides feature importance (tells us WHICH joints matter most)
    - No hyperparameter tuning needed for a first working version
    - Easily swappable for XGBoost, SVM, or a small MLP later

HOW TO SWAP:
    Replace RandomForestClassifier with any sklearn-compatible estimator.
    The rest of the pipeline (feature extraction, evaluation) requires only
    .fit(X, y) / .predict(X) / .predict_proba(X) — standard sklearn API.
"""

import numpy as np
import pickle
from typing import Optional, List
from sklearn.ensemble import RandomForestClassifier

from training_pipeline.config import Config

cfg = Config()


class ExerciseClassifier:
    """
    Multi-class classifier predicting (exercise, phase) from joint angles.

    Outputs e.g. "Squats_Start", "Push-up_End", "Lunges_Start" etc.

    Usage (structure only):
        clf = ExerciseClassifier()
        clf.train(X_train, y_train)
        y_pred = clf.predict(X_val)
        clf.save()
    """

    def __init__(self,
                 n_estimators: int = None,
                 max_depth: int    = None,
                 random_seed: int  = None):
        self.n_estimators = n_estimators or cfg.CLASSIFIER_N_ESTIMATORS
        self.max_depth    = max_depth    or cfg.CLASSIFIER_MAX_DEPTH
        self.random_seed  = random_seed  or cfg.RANDOM_SEED

        self._model: Optional[RandomForestClassifier] = None
        self.label_to_int: Optional[dict] = None
        self.int_to_label: Optional[dict] = None
        self.feature_names: Optional[List[str]] = None

    # -----------------------------------------------------------------------
    # TRAINING
    # -----------------------------------------------------------------------

    def train(self, X: np.ndarray,
              y_int: np.ndarray,
              label_to_int: dict,
              int_to_label: dict,
              feature_names: List[str] = None) -> None:
        """
        Fit the Random Forest classifier.

        Parameters
        ----------
        X            : (N, F) feature matrix
        y_int        : (N,) integer class labels from preprocessor.build_label_maps()
        label_to_int : {"Squats_Start": 0, ...}
        int_to_label : {0: "Squats_Start", ...}
        feature_names: column names of X (for feature importance display)
        """
        # --- PLACEHOLDER: uncomment when ready to train ---
        self._model = RandomForestClassifier(
            n_estimators = self.n_estimators,
            max_depth    = self.max_depth,
            random_state = self.random_seed,
            n_jobs       = -1,   # use all CPU cores
        )
        self._model.fit(X, y_int)
        self.label_to_int = label_to_int
        self.int_to_label = int_to_label
        self.feature_names = feature_names
        print(f"[Classifier] Trained on {X.shape[0]} samples, "
              f"{len(label_to_int)} classes.")

    # -----------------------------------------------------------------------
    # INFERENCE
    # -----------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted label strings for each row of X."""
        if self._model is None:
            raise RuntimeError("Classifier not trained.")
        y_int = self._model.predict(X)
        return np.array([self.int_to_label[i] for i in y_int])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities — useful for confidence thresholding."""
        if self._model is None:
            raise RuntimeError("Classifier not trained.")
        return self._model.predict_proba(X)

    def predict_single(self, angle_vector: np.ndarray) -> dict:
        """
        Classify a single live pose.

        Returns
        -------
        {"exercise": str, "phase": str, "confidence": float}
        """
        X = angle_vector.reshape(1, -1)
        proba = self.predict_proba(X)[0]
        pred_int = int(np.argmax(proba))
        label = self.int_to_label[pred_int]
        exercise, phase = label.rsplit("_", 1)
        return {
            "exercise":   exercise,
            "phase":      phase,
            "confidence": float(proba[pred_int]),
        }

    def feature_importance(self) -> Optional[List[tuple]]:
        """
        Return sorted list of (feature_name, importance_score).
        Useful for understanding which joints are most diagnostic.
        """
        if self._model is None or self.feature_names is None:
            return None
        importances = self._model.feature_importances_
        return sorted(
            zip(self.feature_names, importances),
            key=lambda x: x[1], reverse=True
        )

    # -----------------------------------------------------------------------
    # PERSISTENCE
    # -----------------------------------------------------------------------

    def save(self, path=None) -> None:
        path = path or cfg.CLASSIFIER_MODEL_PATH
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[Classifier] Saved -> {path}")

    @classmethod
    def load(cls, path=None) -> "ExerciseClassifier":
        path = path or cfg.CLASSIFIER_MODEL_PATH
        with open(path, "rb") as f:
            clf = pickle.load(f)
        print(f"[Classifier] Loaded <- {path}")
        return clf
