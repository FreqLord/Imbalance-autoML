"""Bootstrap estimation for statistical confidence intervals on holdout/OOF predictions."""

from typing import Callable, Dict, Tuple, Union
import numpy as np


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstraps: int = 1000,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """Computes point estimate and empirical bootstrap percentile confidence interval (point, lower, upper)."""
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    n = len(y_true_arr)

    point_est = float(metric_fn(y_true_arr, y_prob_arr))
    rng = np.random.default_rng(random_state)
    boot_scores = []

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        # Minority presence safeguard
        if np.sum(y_true_arr[idx] == 1) == 0:
            continue
        try:
            score = metric_fn(y_true_arr[idx], y_prob_arr[idx])
            boot_scores.append(score)
        except Exception:
            continue

    if not boot_scores:
        return point_est, point_est, point_est

    alpha = (1.0 - ci_level) / 2.0
    lower = float(np.percentile(boot_scores, 100.0 * alpha))
    upper = float(np.percentile(boot_scores, 100.0 * (1.0 - alpha)))
    return point_est, lower, upper


def compute_all_bootstrap_cis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metrics_dict: Dict[str, Callable[[np.ndarray, np.ndarray], float]],
    n_bootstraps: int = 1000,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Calculates bootstrap confidence intervals across a dictionary of evaluation metrics."""
    results = {}
    for name, fn in metrics_dict.items():
        pt, lo, hi = compute_bootstrap_ci(
            y_true,
            y_prob,
            metric_fn=fn,
            n_bootstraps=n_bootstraps,
            ci_level=ci_level,
            random_state=random_state,
        )
        results[name] = {"point": pt, "lower": lo, "upper": hi}
    return results
