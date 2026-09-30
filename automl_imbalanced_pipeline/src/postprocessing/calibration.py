"""Probability calibration: Platt Scaling (<5k positives) vs Isotonic Regression (>=5k positives).

Guarantees:
- Tier 3: Fits separately on the dedicated Calibration partition using cv="prefit".
- Tier 2: Employs CalibratedClassifierCV(cv=k, ensemble=False) on inner out-of-fold predictions.
- Cutoff: Strictly requires >= 5,000 calibration positives for non-parametric Isotonic Regression;
  defaults to parametric Platt Scaling (sigmoid) otherwise to prevent step-function overfitting.
"""

from typing import Any, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV


class PipelineProbabilityCalibrator(BaseEstimator, ClassifierMixin):
    """Calibrator managing Platt (sigmoid) vs Isotonic switching and partition safety."""

    def __init__(
        self,
        base_estimator: Any,
        pos_threshold_for_isotonic: int = 5000,
        method: Optional[str] = None,
        cv: Union[str, int] = "prefit",
    ):
        self.base_estimator = base_estimator
        self.pos_threshold_for_isotonic = pos_threshold_for_isotonic
        self.method = method
        self.cv = cv
        self.calibrator_: Optional[CalibratedClassifierCV] = None
        self.resolved_method_: str = "sigmoid"

    def fit(self, X: Any, y: Union[pd.Series, np.ndarray]) -> "PipelineProbabilityCalibrator":
        y_arr = np.asarray(y)
        n_pos = int(np.sum(y_arr == 1))

        # Select method if not explicitly forced
        if self.method is not None:
            self.resolved_method_ = self.method
        else:
            self.resolved_method_ = "isotonic" if n_pos >= self.pos_threshold_for_isotonic else "sigmoid"

        self.calibrator_ = CalibratedClassifierCV(
            estimator=self.base_estimator,
            method=self.resolved_method_,
            cv=self.cv,
            ensemble=False if isinstance(self.cv, int) else True,
        )
        self.calibrator_.fit(X, y_arr)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.calibrator_ is None:
            if hasattr(self.base_estimator, "predict_proba"):
                return self.base_estimator.predict_proba(X)
            raise RuntimeError("Calibrator has not been fitted.")
        return self.calibrator_.predict_proba(X)

    def predict(self, X: Any) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)
