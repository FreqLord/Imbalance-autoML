"""Probability calibration: Platt Scaling (<5k positives) vs Isotonic Regression (>=5k positives)."""

from typing import Union
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV


def build_calibrator(
    base_estimator,
    y: Union[pd.Series, np.ndarray],
    pos_threshold: int = 5000,
    cv: Union[str, int] = "prefit",
) -> CalibratedClassifierCV:
    """Selects Platt scaling (sigmoid) or Isotonic regression based on minority count."""
    n_positives = int(np.sum(np.asarray(y) == 1))
    method = "isotonic" if n_positives >= pos_threshold else "sigmoid"
    return CalibratedClassifierCV(estimator=base_estimator, method=method, cv=cv)
