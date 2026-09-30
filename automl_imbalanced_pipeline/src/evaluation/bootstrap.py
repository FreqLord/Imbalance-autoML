"""Bootstrap estimation for statistical confidence intervals on holdout and OOF predictions."""

from typing import Callable, Tuple
import numpy as np


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred_or_prob: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstraps: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Computes point estimate and empirical bootstrap percentile confidence interval (low, high)."""
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    point_est = metric_fn(y_true, y_pred_or_prob)

    bootstrap_scores = []
    for _ in range(n_bootstraps):
        boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
        # Check stratification safeguard
        if np.sum(y_true[boot_idx]) == 0:
            continue
        score = metric_fn(y_true[boot_idx], y_pred_or_prob[boot_idx])
        bootstrap_scores.append(score)

    alpha = (1.0 - ci_level) / 2.0
    lower = float(np.percentile(bootstrap_scores, 100 * alpha))
    upper = float(np.percentile(bootstrap_scores, 100 * (1.0 - alpha)))
    return point_est, lower, upper
