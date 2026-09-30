"""Encoders: Internally cross-fitted TargetEncoder, OrdinalEncoder, and conditional One-Hot."""

from typing import List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder


class CrossFittedTargetEncoder(BaseEstimator, TransformerMixin):
    """Target encoder with internal cross-fitting to eliminate target leakage on rare minority classes.

    Converts high-cardinality categoricals into smooth continuous posterior probabilities.
    Crucial: Must be skipped when training CatBoost, which calculates ordered target statistics internally.
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        cv: int = 5,
        smooth: Union[float, str] = "auto",
        random_state: int = 42,
    ):
        self.columns = columns
        self.cv = cv
        self.smooth = smooth
        self.random_state = random_state
        self.encoder_: Optional[TargetEncoder] = None
        self.columns_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CrossFittedTargetEncoder requires pandas.DataFrame input.")

        self.columns_ = self.columns if self.columns is not None else list(X.select_dtypes(include=["object", "category"]).columns)
        if len(self.columns_) == 0:
            return self

        self.encoder_ = TargetEncoder(
            categories="auto",
            target_type="binary",
            smooth=self.smooth,
            cv=self.cv,
            random_state=self.random_state,
        )
        self.encoder_.fit(X[self.columns_], np.asarray(y))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.encoder_ is None or len(self.columns_) == 0:
            return X

        X_out = X.copy()
        encoded = self.encoder_.transform(X_out[self.columns_])
        # Replace original categorical columns with their continuous target-encoded values
        for i, col in enumerate(self.columns_):
            X_out[col] = encoded[:, i]
        return X_out


class LowCardinalityOrdinalEncoder(BaseEstimator, TransformerMixin):
    """Encodes low-cardinality categoricals as non-negative integers for SMOTENC compatibility."""

    def __init__(self, columns: Optional[List[str]] = None):
        self.columns = columns
        self.encoder_: Optional[OrdinalEncoder] = None
        self.columns_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("LowCardinalityOrdinalEncoder requires pandas.DataFrame input.")

        self.columns_ = self.columns if self.columns is not None else list(X.select_dtypes(include=["object", "category"]).columns)
        if len(self.columns_) == 0:
            return self

        self.encoder_ = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )
        self.encoder_.fit(X[self.columns_].astype(str))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.encoder_ is None or len(self.columns_) == 0:
            return X

        X_out = X.copy()
        encoded = self.encoder_.transform(X_out[self.columns_].astype(str))
        for i, col in enumerate(self.columns_):
            X_out[col] = encoded[:, i]
        return X_out


class PostResamplingOneHotEncoder(BaseEstimator, TransformerMixin):
    """Conditional One-Hot Encoder applied AFTER resampling for linear & distance-based models.

    Skipped entirely for LightGBM and CatBoost, which consume native categorical indices directly.
    """

    def __init__(self, columns: Optional[List[str]] = None, enabled: bool = True):
        self.columns = columns
        self.enabled = enabled
        self.encoder_: Optional[OneHotEncoder] = None
        self.columns_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        if not self.enabled or not isinstance(X, pd.DataFrame):
            return self

        self.columns_ = self.columns if self.columns is not None else list(X.select_dtypes(include=["object", "category"]).columns)
        if len(self.columns_) == 0:
            return self

        self.encoder_ = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            drop="first",
        )
        self.encoder_.fit(X[self.columns_])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled or self.encoder_ is None or len(self.columns_) == 0:
            return X

        X_out = X.copy()
        ohe_arr = self.encoder_.transform(X_out[self.columns_])
        feature_names = self.encoder_.get_feature_names_out(self.columns_)
        df_ohe = pd.DataFrame(ohe_arr, columns=feature_names, index=X_out.index)

        X_out = X_out.drop(columns=self.columns_)
        return pd.concat([X_out, df_ohe], axis=1)
