"""Evaluation metrics for extreme class imbalance: BSS, adaptive ECE, and Precision @ top-k budget."""

from typing import Tuple
import numpy as np
from sklearn.metrics import brier_score_loss, average_precision_score


def brier_skill_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Computes Brier Skill Score against the uninformative prior baseline:

    BSS = 1 - (BS_model / BS_reference)
    where BS_reference is the Brier score of predicting the base rate.
    """
    bs_model = brier_score_loss(y_true, y_prob)
    base_rate = np.mean(y_true)
    bs_ref = brier_score_loss(y_true, np.full_like(y_prob, base_rate))
    if bs_ref == 0:
        return 0.0
    return float(1.0 - (bs_model / bs_ref))


def expected_calibration_error_adaptive(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Computes Expected Calibration Error (ECE) using equal-frequency (quantile/adaptive) binning."""
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(y_prob, quantiles)
    bin_edges[-1] += 1e-8

    ece = 0.0
    total_samples = len(y_prob)

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        bin_size = np.sum(mask)
        if bin_size > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (bin_size / total_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)


def precision_at_top_k(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
    """Calculates Precision among the top-k highest predicted probability cases (budget constraint)."""
    top_indices = np.argsort(y_prob)[::-1][:k]
    return float(np.mean(y_true[top_indices]))
