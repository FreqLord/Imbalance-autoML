"""Decision thresholding via closed-form cost optimization and TunedThresholdClassifierCV."""

from typing import Optional
from sklearn.model_selection import TunedThresholdClassifierCV


def calculate_closed_form_threshold(
    c_fp: float,
    c_fn: float,
    c_tp: float = 0.0,
    c_tn: float = 0.0,
) -> float:
    """Computes the Bayes-optimal decision threshold from the cost matrix:

    t* = (c_FP - c_TN) / ((c_FP - c_TN) + (c_FN - c_TP))
    """
    numerator = c_fp - c_tn
    denominator = (c_fp - c_tn) + (c_fn - c_tp)
    if denominator <= 0:
        raise ValueError("Cost matrix yields non-positive denominator.")
    return float(numerator / denominator)


def build_tuned_threshold_classifier(
    estimator,
    scoring: str = "f1",
    cv: int = 5,
) -> TunedThresholdClassifierCV:
    """Cross-fits decision threshold optimizer to prevent threshold overfitting on test sets."""
    return TunedThresholdClassifierCV(
        estimator=estimator,
        scoring=scoring,
        cv=cv,
    )
