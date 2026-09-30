"""Early stopping engine carving un-resampled, embargoed validation slices before pipeline transformations.

Monitors Average Precision (PR-AUC), strictly avoiding logloss which is distorted by SMOTE and class-weights.
"""

from typing import Any, Callable, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from src.data.embargo import LabelDelayEmbargo


def carve_unresampled_val_split(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    val_fraction: float = 0.15,
    timestamps: Optional[Union[pd.Series, np.ndarray]] = None,
    embargo_horizon: Optional[Union[pd.Timedelta, int, float, str]] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Partitions training fold into inner-train and honest, un-resampled inner-val indices."""
    n_samples = len(X)
    y_arr = np.asarray(y)

    if timestamps is not None:
        # Chronological forward split: earlier part is inner-train, later part is inner-val
        ts_order = np.argsort(pd.to_datetime(timestamps).values)
        val_start = int(np.floor(n_samples * (1.0 - val_fraction)))
        train_idx = ts_order[:val_start]
        val_idx = ts_order[val_start:]

        # Apply embargo gap between inner-train and inner-val
        if embargo_horizon is not None:
            embargo = LabelDelayEmbargo(embargo_horizon)
            train_idx = embargo.prune_train_indices(train_idx, val_idx, timestamps)

        return train_idx, val_idx

    else:
        # Stratified holdout slice
        skf = StratifiedKFold(
            n_splits=max(2, int(1.0 / val_fraction)),
            shuffle=True,
            random_state=random_state,
        )
        splits = list(skf.split(X, y_arr))
        train_idx, val_idx = splits[0]
        return train_idx, val_idx


class EarlyStoppingGBM(BaseEstimator, ClassifierMixin):
    """Wrapper that evaluates iterations using Average Precision on un-resampled validation features."""

    def __init__(
        self,
        base_estimator: Any,
        model_name: str,
        early_stopping_rounds: int = 25,
        max_iterations: int = 500,
    ):
        self.base_estimator = base_estimator
        self.model_name = model_name.lower()
        self.early_stopping_rounds = early_stopping_rounds
        self.max_iterations = max_iterations
        self.best_iteration_: Optional[int] = None
        self.best_score_: float = -1.0
        self.fitted_estimator_: Optional[Any] = None

    def fit_with_eval_set(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
    ) -> "EarlyStoppingGBM":
        """Fits the underlying tree booster using PR-AUC (Average Precision) early stopping."""
        y_train_arr = np.asarray(y_train)
        y_val_arr = np.asarray(y_val)

        if "lightgbm" in self.model_name:
            import lightgbm as lgb

            def lgb_average_precision(preds, train_data):
                labels = train_data.get_label()
                ap = average_precision_score(labels, preds)
                return "average_precision", ap, True

            self.base_estimator.set_params(n_estimators=self.max_iterations)
            self.base_estimator.fit(
                X_train,
                y_train_arr,
                eval_set=[(X_val, y_val_arr)],
                eval_metric=lgb_average_precision,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            self.best_iteration_ = getattr(self.base_estimator, "best_iteration_", None)
            self.fitted_estimator_ = self.base_estimator

        elif "catboost" in self.model_name:
            self.base_estimator.set_params(
                iterations=self.max_iterations,
                early_stopping_rounds=self.early_stopping_rounds,
                eval_metric="PRAUC",
            )
            self.base_estimator.fit(
                X_train,
                y_train_arr,
                eval_set=(X_val, y_val_arr),
                verbose=False,
            )
            self.best_iteration_ = getattr(self.base_estimator, "get_best_iteration", lambda: None)()
            self.fitted_estimator_ = self.base_estimator

        elif "xgboost" in self.model_name:
            self.base_estimator.set_params(
                n_estimators=self.max_iterations,
                early_stopping_rounds=self.early_stopping_rounds,
                eval_metric="aucpr",
            )
            self.base_estimator.fit(
                X_train,
                y_train_arr,
                eval_set=[(X_val, y_val_arr)],
                verbose=False,
            )
            self.best_iteration_ = getattr(self.base_estimator, "best_iteration", None)
            self.fitted_estimator_ = self.base_estimator

        else:
            # Fallback for non-GBM models (e.g. Random Forest, Logistic)
            self.base_estimator.fit(X_train, y_train_arr)
            self.fitted_estimator_ = self.base_estimator

        # Calculate final honest validation AP
        if hasattr(self.fitted_estimator_, "predict_proba"):
            val_probs = self.fitted_estimator_.predict_proba(X_val)[:, 1]
            self.best_score_ = float(average_precision_score(y_val_arr, val_probs))

        return self

    def predict(self, X: Any) -> np.ndarray:
        return self.fitted_estimator_.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.fitted_estimator_.predict_proba(X)
