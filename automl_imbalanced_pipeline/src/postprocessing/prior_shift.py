"""Prior shift adjustment: Bayes odds-reweighting for deployment prevalence divergence.

Mathematical Formula:
Odds_cal = p_cal / (1 - p_cal)
Odds_ratio = (pi_deploy / (1 - pi_deploy)) / (pi_cal / (1 - pi_cal))
Odds_deploy = Odds_cal * Odds_ratio
p_deploy = Odds_deploy / (1 + Odds_deploy)
"""

from typing import Any, Optional, Union
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def adjust_odds_for_prior_shift(
    probabilities: np.ndarray,
    prevalence_cal: float,
    prevalence_deploy: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """Applies odds-reweighting to convert calibration-space probabilities to deployment-space probabilities."""
    p_clipped = np.clip(probabilities, eps, 1.0 - eps)
    odds_cal = p_clipped / (1.0 - p_clipped)

    ratio = (prevalence_deploy / (1.0 - prevalence_deploy)) / (
        prevalence_cal / (1.0 - prevalence_cal)
    )
    odds_deploy = odds_cal * ratio
    return odds_deploy / (1.0 + odds_deploy)


class PriorShiftAdjustedClassifier(BaseEstimator, ClassifierMixin):
    """Classifier wrapper dynamically reweighting output probabilities for known target deployment prevalence."""

    def __init__(
        self,
        base_estimator: Any,
        prevalence_cal: float,
        prevalence_deploy: float,
        threshold: float = 0.5,
    ):
        self.base_estimator = base_estimator
        self.prevalence_cal = prevalence_cal
        self.prevalence_deploy = prevalence_deploy
        self.threshold = threshold

    def fit(self, X: Any, y: Any = None):
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        raw_probs = self.base_estimator.predict_proba(X)
        if raw_probs.ndim == 2:
            pos_prob = raw_probs[:, 1]
            adj_pos = adjust_odds_for_prior_shift(
                pos_prob,
                prevalence_cal=self.prevalence_cal,
                prevalence_deploy=self.prevalence_deploy,
            )
            adj_neg = 1.0 - adj_pos
            return np.column_stack([adj_neg, adj_pos])
        else:
            return adjust_odds_for_prior_shift(
                raw_probs,
                prevalence_cal=self.prevalence_cal,
                prevalence_deploy=self.prevalence_deploy,
            )

    def predict(self, X: Any) -> np.ndarray:
        probs = self.predict_proba(X)
        pos = probs[:, 1] if probs.ndim == 2 else probs
        return (pos >= self.threshold).astype(int)
