"""Conditional feature scaling: RobustScaler with strict SMOTE/Tree awareness.

Crucial Rule:
- Always scale continuous columns if SMOTE/SMOTENC is active, regardless of downstream model family,
  because SMOTENC's Euclidean distance calculation across continuous dimensions requires normalized variance.
- Skip scaling for tree models ONLY if the sampling strategy is not oversampling.
- Always scale for linear and distance-dependent models (Logistic Regression, etc.).
"""

from typing import List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler


class ConditionalRobustScaler(BaseEstimator, TransformerMixin):
    """RobustScaler adhering to strict SMOTE-dependency and tree-model bypass rules."""

    def __init__(
        self,
        continuous_columns: Optional[List[str]] = None,
        is_oversampling: bool = False,
        is_tree_model: bool = False,
    ):
        self.continuous_columns = continuous_columns
        self.is_oversampling = is_oversampling
        self.is_tree_model = is_tree_model
        self.scaler_: Optional[RobustScaler] = None
        self.columns_: List[str] = []
        self._should_scale = False

    def _determine_scaling_need(self) -> bool:
        # Mandatory if oversampling (SMOTE/SMOTENC distance metric depends on scale)
        if self.is_oversampling:
            return True
        # If no oversampling, tree models do not require monotonic continuous scaling
        if self.is_tree_model:
            return False
        # Default: scale for linear / distance-sensitive models
        return True

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        self._should_scale = self._determine_scaling_need()
        if not self._should_scale:
            return self

        if isinstance(X, pd.DataFrame):
            self.columns_ = (
                self.continuous_columns
                if self.continuous_columns is not None
                else list(X.select_dtypes(include=[np.number]).columns)
            )
            if len(self.columns_) == 0:
                return self
            self.scaler_ = RobustScaler()
            self.scaler_.fit(X[self.columns_])
        else:
            self.scaler_ = RobustScaler()
            self.scaler_.fit(X)

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        if not self._should_scale or self.scaler_ is None:
            return X

        if isinstance(X, pd.DataFrame):
            if len(self.columns_) == 0:
                return X
            X_out = X.copy()
            scaled_vals = self.scaler_.transform(X_out[self.columns_])
            for i, col in enumerate(self.columns_):
                X_out[col] = scaled_vals[:, i]
            return X_out
        else:
            return self.scaler_.transform(X)
