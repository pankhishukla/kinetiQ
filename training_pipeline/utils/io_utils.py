"""
training_pipeline/utils/io_utils.py
=====================================
Save and load helpers for models, features, and normalisation parameters.

WHY centralise I/O here?
    Pickle format, file paths, and versioning are cross-cutting concerns.
    Centralising them means:
    - One place to add model versioning later (e.g. timestamp suffixes)
    - One place to switch from pickle to joblib/onnx/safetensors
    - Consistent error messages across all save/load calls
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Any

from training_pipeline.config import Config

cfg = Config()


def save_pickle(obj: Any, path: Path, label: str = "Object") -> None:
    """Serialise any Python object to disk using pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    size_kb = path.stat().st_size / 1024
    print(f"[IO] {label} saved -> {path}  ({size_kb:.1f} KB)")


def load_pickle(path: Path, label: str = "Object") -> Any:
    """Load a pickled object from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"[IO] {label} not found at {path}")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print(f"[IO] {label} loaded <- {path}")
    return obj


def save_features_cache(X: np.ndarray,
                        y: np.ndarray,
                        feat_names: list,
                        norm_params: dict) -> None:
    """
    Cache the feature matrix to disk to avoid re-extracting on every run.

    WHY cache?
        Feature extraction over ~1500 images + JSON parsing takes ~5–10 s.
        After the first run, loading the cache is < 0.1 s.
        Only re-run extraction when the dataset or feature definitions change.
    """
    payload = {
        "X": X, "y": y,
        "feature_names": feat_names,
        "norm_params":   norm_params,
    }
    save_pickle(payload, cfg.FEATURES_CACHE_PATH, "Features cache")


def load_features_cache() -> dict:
    """Load a previously saved feature matrix cache."""
    return load_pickle(cfg.FEATURES_CACHE_PATH, "Features cache")


def cache_exists() -> bool:
    """Return True if a feature cache file exists on disk."""
    return cfg.FEATURES_CACHE_PATH.exists()
