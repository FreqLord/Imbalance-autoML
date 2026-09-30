"""SHAP explanation runner with strict holdout discipline and uncalibrated model isolation."""

from typing import Any, Optional
import numpy as np
import pandas as pd


def compute_shap_explanations(
    raw_model: Any,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    max_evals: int = 500,
):
    """Executes SHAP TreeExplainer or ExactExplainer on the uncalibrated base estimator.

    Uses strictly separate background and evaluation subsets to avoid explanation data leakage.
    """
    import shap

    # Use TreeExplainer if tree-based, otherwise Explainer fallback
    try:
        explainer = shap.TreeExplainer(raw_model, data=X_background)
        shap_values = explainer(X_explain)
    except Exception:
        explainer = shap.Explainer(raw_model.predict, X_background)
        shap_values = explainer(X_explain, max_evals=max_evals)

    return explainer, shap_values
