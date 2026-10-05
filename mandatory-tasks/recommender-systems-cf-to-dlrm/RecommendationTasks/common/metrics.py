"""Evaluation metrics and reliability visualization tools for CTR binary classification models."""

from typing import Dict, Tuple, Optional
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) over n_bins equal-width probability intervals."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper) if i > 0 else (y_prob >= bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * (np.sum(in_bin) / total_samples)

    return float(ece)


def find_best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Sweeps probability thresholds to find the threshold maximizing F1 score on validation set."""
    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1, best_thresh = -1.0, 0.5

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(t)

    return best_thresh


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Computes full suite of classification, ranking, calibration, and threshold-based metrics."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    eps = 1e-15
    y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
    preds = (y_prob >= threshold).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob_clipped)),
        "accuracy_0.5": float(accuracy_score(y_true, (y_prob >= 0.5).astype(int))),
        "threshold_used": float(threshold),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "ece": float(compute_ece(y_true, y_prob)),
        "pos_rate_baseline": float(np.mean(y_true)),
    }


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    model_name: str = "Model",
    save_path: Optional[pathlib.Path] = None,
) -> None:
    """Plots a calibration reliability diagram comparing predicted confidence against empirical accuracy."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2.0
    bin_accs = []
    bin_confs = []

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper) if i > 0 else (y_prob >= bin_lower) & (y_prob <= bin_upper)
        if np.sum(in_bin) > 0:
            bin_accs.append(np.mean(y_true[in_bin]))
            bin_confs.append(np.mean(y_prob[in_bin]))
        else:
            bin_accs.append(np.nan)
            bin_confs.append(bin_centers[i])

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.plot(bin_confs, bin_accs, "s-", color="crimson", label=f"{model_name}")
    plt.xlabel("Mean Predicted Probability (Confidence)")
    plt.ylabel("Empirical Positive Proportion (Accuracy)")
    plt.title(f"Reliability Diagram — {model_name}")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()
