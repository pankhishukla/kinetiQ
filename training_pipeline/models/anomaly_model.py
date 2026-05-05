"""
training_pipeline/models/anomaly_model.py
==========================================
Isolation Forest wrapper for detecting "bad form" poses.

WHY Isolation Forest for form detection?
    We do NOT have labelled "bad form" examples — the dataset only shows
    correct exercise postures.  Isolation Forest is an UNSUPERVISED anomaly
    detector: it learns what "normal" looks like, then flags anything that
    deviates from that.  This means:
        - No need to manually label incorrect poses
        - Generalises to novel form errors we haven't seen before
        - Fast inference (< 1 ms per frame on CPU)

    When a user's knee angle during a squat is 30 degrees (very unusual
    compared to training data), the forest produces a low anomaly score
    and we flag it as potential bad form.

HOW TO SWAP ARCHITECTURES:
    Replace IsolationForest with any estimator that exposes
    .fit(X) and .decision_function(X) or .score_samples(X).
    One-Class SVM is a common alternative for smaller datasets.
"""

import numpy as np
import pickle
from typing import Optional, List
from sklearn.ensemble import IsolationForest

from training_pipeline.config import Config

cfg = Config()


class AnomalyModel:
    """
    Thin wrapper around IsolationForest that adds save/load and
    a human-readable predict interface.

    Usage (structure only — do NOT call .train() yet):
        model = AnomalyModel()
        model.train(X_train)                     # fit Isolation Forest
        scores = model.score(X_val)              # anomaly scores
        predictions = model.predict(X_val)       # 1=normal, -1=anomaly
        model.save()                             # persist to disk
    """

    def __init__(self,
                 n_estimators: int  = None,
                 contamination: float = None,
                 random_seed: int   = None):
        """
        Parameters
        ----------
        n_estimators  : number of trees (default from cfg)
        contamination : expected fraction of outliers (default from cfg)
        random_seed   : for reproducibility (default from cfg)
        """
        self.n_estimators  = n_estimators  or cfg.ANOMALY_N_ESTIMATORS
        self.contamination = contamination or cfg.ANOMALY_CONTAMINATION
        self.random_seed   = random_seed   or cfg.RANDOM_SEED

        # The underlying sklearn model — created fresh at train time
        self._model: Optional[IsolationForest] = None

        # Normalisation params stored alongside the model so inference
        # uses the same scale as training
        self.norm_params: Optional[dict] = None
        self.feature_names: Optional[List[str]] = None

    # -----------------------------------------------------------------------
    # TRAINING
    # -----------------------------------------------------------------------

    def train(self, X: np.ndarray,
              feature_names: List[str] = None,
              norm_params: dict = None) -> None:
        """
        Fit the Isolation Forest on normal-form training data.

        IMPORTANT: Do NOT call this during the demo.  Training is an
        offline step.  The fitted model is saved to disk and loaded at
        inference time.

        Parameters
        ----------
        X             : (N, F) feature matrix — ONLY correct-form samples
        feature_names : list of F angle names (for interpretability)
        norm_params   : dict from extractor.normalise_features()
        """
        # --- PLACEHOLDER: uncomment when ready to train ---
        self._model = IsolationForest(
            n_estimators  = self.n_estimators,
            contamination = self.contamination,
            random_state  = self.random_seed,
        )
        self._model.fit(X)
        self.feature_names = feature_names
        self.norm_params   = norm_params
        print(f"[AnomalyModel] Trained on {X.shape[0]} samples, "
              f"{X.shape[1]} features.")

    # -----------------------------------------------------------------------
    # INFERENCE
    # -----------------------------------------------------------------------

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Return anomaly scores for each sample.

        Higher score = more normal.  Threshold defined in cfg.ANOMALY_SCORE_THRESHOLD.
        Scores typically range from -0.5 (very anomalous) to +0.5 (very normal).
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call .train() first.")
        return self._model.score_samples(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return 1 (normal/correct form) or -1 (anomaly/bad form) per sample.
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call .train() first.")
        return self._model.predict(X)

    def predict_single(self, angle_vector: np.ndarray) -> dict:
        """
        Score a single pose for live inference.

        Parameters
        ----------
        angle_vector : 1-D array of joint angles (same order as training)

        Returns
        -------
        {"score": float, "is_anomaly": bool, "label": str}
        """
        X = angle_vector.reshape(1, -1)
        score = float(self.score(X)[0])
        is_anomaly = score < cfg.ANOMALY_SCORE_THRESHOLD
        return {
            "score":      score,
            "is_anomaly": is_anomaly,
            "label":      "BAD FORM" if is_anomaly else "OK",
        }

    # -----------------------------------------------------------------------
    # PERSISTENCE
    # -----------------------------------------------------------------------

    def save(self, path=None) -> None:
        """Serialise the fitted model to disk."""
        path = path or cfg.ANOMALY_MODEL_PATH
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[AnomalyModel] Saved -> {path}")

    @classmethod
    def load(cls, path=None) -> "AnomalyModel":
        """Load a previously saved model from disk."""
        path = path or cfg.ANOMALY_MODEL_PATH
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"[AnomalyModel] Loaded <- {path}")
        return model
