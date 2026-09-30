"""Decision thresholding: Closed-form cost matrix optimization and cross-fitted metric tuning.

Formulas:
- Closed-form Bayes Optimal Decision Threshold:
  t* = (c_FP - c_TN) / ((c_FP - c_TN) + (c_FN - c_TP))
  Preconditions: c_FP > c_TN and c_FN > c_TP.
- Fork B: Cross-fitted F1/MCC threshold optimization via TunedThresholdClassifierCV.
"""

from typing import Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import TunedThresholdClassifierCV


def calculate_closed_form_threshold(
    c_fp: float,
    c_fn: float,
    c_tp: float = 0.0,
    c_tn: float = 0.0,
) -> float:
    """Calculates Bayes-optimal decision threshold minimizing expected business loss.

    Preconditions:
    - Cost of False Positive > Cost of True Negative (c_FP > c_TN)
    - Cost of False Negative > Cost of True Positive (c_FN > c_TP)
    """
    if c_fp <= c_tn:
        raise ValueError(f"Precondition violated: c_FP ({c_fp}) must be strictly greater than c_TN ({c_tn}).")
    if c_fn <= c_tp:
        raise ValueError(f"Precondition violated: c_FN ({c_fn}) must be strictly greater than c_TP ({c_tp}).")

    num = c_fp - c_tn
    denom = (c_fp - c_tn) + (c_fn - c_tp)
    return float(num / denom)


class ThresholdedClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper applying a fixed decision threshold to probability estimates."""

    def __init__(self, estimator: Any, threshold: float = 0.5):
        self.estimator = estimator
        self.threshold = threshold

    def fit(self, X: Any, y: Any = None):
        if not hasattr(self.estimator, "predict_proba"):
            self.estimator.fit(X, y)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.estimator.predict_proba(X)

    def predict(self, X: Any) -> np.ndarray:
        probs = self.predict_proba(X)
        pos_prob = probs[:, 1] if probs.ndim == 2 else probs
        return (pos_prob >= self.threshold).astype(int)


def tune_metric_threshold(
    calibrated_estimator: Any,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    scoring: str = "f1",
    cv: int = 5,
) -> TunedThresholdClassifierCV:
    """Cross-fits decision threshold optimizer over calibration data folds to prevent threshold leakage."""
    tuner = TunedThresholdClassifierCV(
        estimator=calibrated_estimator,
        scoring=scoring,
        cv=cv,
        thresholds=100,
    )
    tuner.fit(X_cal, y_cal)
    return tuner
