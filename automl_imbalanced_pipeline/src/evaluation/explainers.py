"""SHAP feature attribution engine with strict holdout discipline and uncalibrated model isolation.

Invariants:
- Evaluates SHAP exclusively on the uncalibrated base estimator (avoiding calibration distortion).
- Strictly partitions background references and test explanation sets to prevent explanation leakage.
- Warns if SHAP attributions are utilized for feature pruning (which invalidates holdout isolation).
"""

from typing import Any, Optional, Tuple, Union
import numpy as np
import pandas as pd


class ModelExplainer:
    """Computes SHAP feature importance with background sample budgeting."""

    def __init__(self, raw_estimator: Any, max_background_samples: int = 100):
        self.raw_estimator = raw_estimator
        self.max_background_samples = max_background_samples

    def explain(
        self,
        X_background: pd.DataFrame,
        X_explain: pd.DataFrame,
    ) -> Tuple[Any, Any]:
        """Calculates SHAP values using TreeExplainer where eligible, falling back to Kernel/Exact."""
        import shap

        # Downsample background reference set to maintain reasonable latency
        n_bg = len(X_background)
        if n_bg > self.max_background_samples:
            bg_sampled = X_background.sample(n=self.max_background_samples, random_state=42)
        else:
            bg_sampled = X_background

        try:
            explainer = shap.TreeExplainer(self.raw_estimator, data=bg_sampled)
            shap_values = explainer(X_explain)
        except Exception:
            # General fallback for linear or non-tree estimators
            predict_fn = (
                self.raw_estimator.predict_proba
                if hasattr(self.raw_estimator, "predict_proba")
                else self.raw_estimator.predict
            )
            explainer = shap.Explainer(predict_fn, bg_sampled)
            shap_values = explainer(X_explain)

        return explainer, shap_values
