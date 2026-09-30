"""Imputation transformers with strict NaN eradication before SMOTENC.

Guarantees:
- Categorical columns: Nulls converted to explicit '__MISSING__' category token.
- Continuous columns: MICE (IterativeImputer) preserves multivariate covariance.
- Fail-safe: No NaNs leak to SMOTE/SMOTENC (which crash on NaNs).
"""

from typing import List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer


class CategoricalMissingImputer(BaseEstimator, TransformerMixin):
    """Replaces missing values in categorical columns with an explicit '__MISSING__' category token."""

    def __init__(self, categorical_columns: Optional[List[str]] = None, fill_value: str = "__MISSING__"):
        self.categorical_columns = categorical_columns
        self.fill_value = fill_value
        self.imputer = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        if isinstance(X, pd.DataFrame):
            X_out = X.copy()
            cols = self.categorical_columns if self.categorical_columns is not None else X_out.select_dtypes(include=["object", "category"]).columns
            for col in cols:
                if col in X_out.columns:
                    # Convert to string to avoid CategoricalDtype NaN insertion errors
                    s = X_out[col].astype(object).fillna(self.fill_value)
                    X_out[col] = s.astype(str)
            return X_out
        else:
            # Array fallback
            X_out = np.copy(X)
            # If object array, fill None/NaN
            mask = pd.isna(X_out)
            X_out[mask] = self.fill_value
            return X_out


class ContinuousIterativeImputer(BaseEstimator, TransformerMixin):
    """Multivariate Imputation by Chained Equations (MICE) for continuous columns.

    Falls back cleanly to median imputation if sample size is extremely small or singular.
    """

    def __init__(
        self,
        continuous_columns: Optional[List[str]] = None,
        max_iter: int = 10,
        random_state: int = 42,
    ):
        self.continuous_columns = continuous_columns
        self.max_iter = max_iter
        self.random_state = random_state
        self.imputer = None
        self.columns_: List[str] = []

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        if isinstance(X, pd.DataFrame):
            self.columns_ = self.continuous_columns if self.continuous_columns is not None else list(X.select_dtypes(include=[np.number]).columns)
            X_fit = X[self.columns_].values
        else:
            X_fit = X
            self.columns_ = []

        if X_fit.shape[1] == 0:
            return self

        try:
            self.imputer = IterativeImputer(
                max_iter=self.max_iter,
                random_state=self.random_state,
                initial_strategy="median",
                skip_complete=True,
            )
            self.imputer.fit(X_fit)
        except Exception:
            # Robust fallback for singular matrices or small sample regimes
            self.imputer = SimpleImputer(strategy="median")
            self.imputer.fit(X_fit)

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        if self.imputer is None:
            return X

        if isinstance(X, pd.DataFrame):
            X_out = X.copy()
            if self.columns_:
                imputed = self.imputer.transform(X_out[self.columns_].values)
                X_out[self.columns_] = imputed
            return X_out
        else:
            return self.imputer.transform(X)
