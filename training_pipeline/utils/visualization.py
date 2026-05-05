"""
training_pipeline/utils/visualization.py
==========================================
Plotting utilities for exploring the feature space and model behaviour.

WHY visualise?
    Angle distributions reveal whether the dataset is balanced, whether
    features separate classes, and whether normalisation is working.
    PCA scatter plots show whether the model has learned a meaningful
    embedding of the exercise space.

All functions save plots to disk (non-interactive) so they work on headless
servers as well as local machines.

NOTE: Requires matplotlib and optionally seaborn.
      Install with: pip install matplotlib seaborn
"""

from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend — safe on all systems
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[Visualization] matplotlib not installed — plots disabled.")

from training_pipeline.config import Config

cfg = Config()
PLOTS_DIR = cfg.BASE_DIR / "plots"


def _ensure_plots_dir():
    PLOTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# ANGLE DISTRIBUTION PLOTS
# ---------------------------------------------------------------------------

def plot_angle_distributions(X: np.ndarray,
                              y_labels: np.ndarray,
                              feat_names: List[str],
                              save_path: Path = None) -> None:
    """
    Plot histograms of each joint angle, colour-coded by exercise class.

    WHY useful?
        Overlapping distributions indicate the feature doesn't discriminate
        well between those classes.  Well-separated distributions mean the
        angle is a strong signal for classification.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("[Visualization] Skipping — matplotlib not available.")
        return

    _ensure_plots_dir()
    save_path = save_path or PLOTS_DIR / "angle_distributions.png"

    unique_labels = sorted(set(y_labels))
    n_features    = len(feat_names)
    n_cols        = 4
    n_rows        = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    axes = axes.flatten()

    for fi, feat in enumerate(feat_names):
        ax = axes[fi]
        for lbl in unique_labels:
            mask = y_labels == lbl
            vals = X[mask, fi]
            # Filter out fill values (-1 = missing)
            vals = vals[vals >= 0]
            ax.hist(vals, bins=20, alpha=0.5, label=lbl, density=True)
        ax.set_title(feat, fontsize=9)
        ax.set_xlabel("Angle (norm.)")
        ax.set_ylabel("Density")

    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=7)
    fig.suptitle("Joint Angle Distributions by Exercise Class", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved -> {save_path}")


# ---------------------------------------------------------------------------
# PCA SCATTER PLOT
# ---------------------------------------------------------------------------

def plot_pca_scatter(X: np.ndarray,
                     y_labels: np.ndarray,
                     save_path: Path = None) -> None:
    """
    Project features to 2-D via PCA and scatter plot by class.

    WHY PCA?
        With 8+ angle features we can't directly visualise the feature space.
        PCA reduces to 2-D while preserving maximum variance, giving an
        intuitive sense of how well-separated the exercise classes are.
        If they cluster cleanly in 2-D, the classifier will work well.
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("[Visualization] sklearn not available for PCA.")
        return

    _ensure_plots_dir()
    save_path = save_path or PLOTS_DIR / "pca_scatter.png"

    # Drop missing-value rows before PCA
    valid_mask = (X >= 0).all(axis=1)
    X_valid    = X[valid_mask]
    y_valid    = y_labels[valid_mask]

    pca    = PCA(n_components=2, random_state=cfg.RANDOM_SEED)
    X_2d   = pca.fit_transform(X_valid)
    var_ex = pca.explained_variance_ratio_

    unique_labels = sorted(set(y_valid))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    fig, ax = plt.subplots(figsize=(10, 7))
    for lbl, col in zip(unique_labels, colors):
        mask = y_valid == lbl
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   label=lbl, color=col, alpha=0.6, s=20)

    ax.set_xlabel(f"PC1 ({var_ex[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({var_ex[1]:.1%} variance)")
    ax.set_title("PCA of Joint Angle Features")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved -> {save_path}")


# ---------------------------------------------------------------------------
# ANOMALY SCORE HISTOGRAM
# ---------------------------------------------------------------------------

def plot_anomaly_scores(scores_correct: np.ndarray,
                         scores_incorrect: Optional[np.ndarray] = None,
                         threshold: float = None,
                         save_path: Path = None) -> None:
    """
    Plot the distribution of anomaly scores for correct (and optionally
    incorrect) form poses, with the decision threshold marked.
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    _ensure_plots_dir()
    save_path = save_path or PLOTS_DIR / "anomaly_scores.png"
    threshold = threshold or cfg.ANOMALY_SCORE_THRESHOLD

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores_correct, bins=30, alpha=0.7, color="green", label="Correct form")
    if scores_incorrect is not None:
        ax.hist(scores_incorrect, bins=30, alpha=0.7, color="red", label="Bad form")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=2,
               label=f"Threshold ({threshold})")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Count")
    ax.set_title("Anomaly Score Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved -> {save_path}")
