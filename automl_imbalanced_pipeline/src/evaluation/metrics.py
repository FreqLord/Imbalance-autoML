"""Evaluation metrics tailored for extreme class imbalance and probability calibration.

Includes:
- Average Precision (AP / PR-AUC)
- Brier Skill Score (BSS) against uninformative prevalence prior baseline
- Adaptive Expected Calibration Error (quantiled / equal-frequency binning)
- Precision @ top-k capacity budget
"""

from typing import Dict, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss


def calculate_average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculates PR-AUC (Average Precision)."""
    return float(average_precision_score(y_true, y_prob))


def brier_skill_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Computes Brier Skill Score against the uninformative prior baseline.

    BSS = 1 - (BS_model / BS_reference)
    where BS_reference = brier_score_loss(y_true, [pi]*N) = pi * (1 - pi)
    """
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)

    bs_model = brier_score_loss(y_true_arr, y_prob_arr)
    base_rate = float(np.mean(y_true_arr))
    bs_ref = base_rate * (1.0 - base_rate)

    if bs_ref <= 0:
        return 0.0
    return float(1.0 - (bs_model / bs_ref))


def adaptive_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Computes ECE using equal-frequency (quantile) adaptive binning.

    Avoids empty bin distortions in sparse extreme probability ranges.
    """
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    n_samples = len(y_true_arr)

    if n_samples == 0:
        return 0.0

    # Quantile bin edges
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(y_prob_arr, quantiles)
    bin_edges[-1] += 1e-8

    ece = 0.0
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob_arr >= low) & (y_prob_arr < high)
        bin_count = np.sum(mask)

        if bin_count > 0:
            bin_acc = float(np.mean(y_true_arr[mask]))
            bin_conf = float(np.mean(y_prob_arr[mask]))
            ece += (bin_count / n_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def precision_at_top_k(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
    """Computes precision among the top-k highest scoring predictions (alert budget)."""
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)

    k_eff = min(k, len(y_prob_arr))
    if k_eff <= 0:
        return 0.0

    top_idx = np.argsort(y_prob_arr)[::-1][:k_eff]
    return float(np.mean(y_true_arr[top_idx]))


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    top_k: int = 50,
) -> Dict[str, float]:
    """Generates complete dictionary of imbalanced evaluation metrics."""
    return {
        "average_precision": calculate_average_precision(y_true, y_prob),
        "brier_skill_score": brier_skill_score(y_true, y_prob),
        "adaptive_ece": adaptive_expected_calibration_error(y_true, y_prob),
        "precision_at_top_k": precision_at_top_k(y_true, y_prob, k=top_k),
        "prevalence": float(np.mean(y_true)),
    }
