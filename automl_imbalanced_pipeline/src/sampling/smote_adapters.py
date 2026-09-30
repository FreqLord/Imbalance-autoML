"""SMOTE / SMOTENC adapters with dynamic column index recomputation and mathematical safety guards.

Invariants:
- Dynamically recomputes categorical_features integer indices prior to calling SMOTENC.
- Automatically falls back to standard SMOTE when no categorical features are present.
- Mathematical Guard: Disallows oversampling when CatBoost is coupled with un-encoded high-cardinality categoricals
  (prevents synthetic mode-voting degradation across hundreds of sparse levels).
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from imblearn.base import BaseSampler
from imblearn.over_sampling import SMOTE, SMOTENC


class DynamicIndexSMOTENC(BaseSampler):
    """SMOTENC adapter that dynamically locates categorical column positions in transformed DataFrames."""

    def __init__(
        self,
        categorical_column_names: Optional[List[str]] = None,
        k_neighbors: int = 5,
        random_state: Optional[int] = 42,
        sampling_strategy: Union[float, str] = "auto",
        is_catboost_high_cardinality: bool = False,
    ):
        super().__init__()
        self.categorical_column_names = categorical_column_names or []
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.sampling_strategy = sampling_strategy
        self.is_catboost_high_cardinality = is_catboost_high_cardinality
        self.sampler_ = None

    def _fit_resample(
        self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], np.ndarray]:
        # Guard: Mode-voting over high cardinality features is mathematically unsound
        if self.is_catboost_high_cardinality:
            raise ValueError(
                "Oversampling is disallowed for CatBoost when high-cardinality categorical features are unencoded. "
                "SMOTENC mode-voting over hundreds of categorical levels degrades feature fidelity."
            )

        # Inspect minority count to adjust k_neighbors if necessary
        n_pos = int(np.sum(y == 1))
        effective_k = min(self.k_neighbors, max(1, n_pos - 1))

        if isinstance(X, pd.DataFrame):
            col_list = list(X.columns)
            # Dynamically map categorical column names to integer indices
            cat_indices = [
                i for i, c in enumerate(col_list) if c in self.categorical_column_names
            ]

            if len(cat_indices) > 0:
                self.sampler_ = SMOTENC(
                    categorical_features=cat_indices,
                    k_neighbors=effective_k,
                    random_state=self.random_state,
                    sampling_strategy=self.sampling_strategy,
                )
                X_res, y_res = self.sampler_.fit_resample(X.values, y)
                return pd.DataFrame(X_res, columns=col_list), y_res
            else:
                # Fallback to plain SMOTE when no categoricals exist
                self.sampler_ = SMOTE(
                    k_neighbors=effective_k,
                    random_state=self.random_state,
                    sampling_strategy=self.sampling_strategy,
                )
                X_res, y_res = self.sampler_.fit_resample(X.values, y)
                return pd.DataFrame(X_res, columns=col_list), y_res

        else:
            # Array path: fallback to SMOTE if categorical names not resolvable
            self.sampler_ = SMOTE(
                k_neighbors=effective_k,
                random_state=self.random_state,
                sampling_strategy=self.sampling_strategy,
            )
            return self.sampler_.fit_resample(X, y)
