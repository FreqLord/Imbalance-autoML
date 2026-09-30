"""Label-delay embargo gap injection for temporal imbalanced validation.

Prevents lookahead leakage arising from delayed ground-truth arrival (e.g. chargeback windows,
claim maturity, or delayed churn events) by inserting an embargo buffer before every evaluation boundary:
- Inside cross-validation splits.
- Preceding inner early-stopping validation splits.
- Between static Train, Calibration, and Holdout partitions.
"""

from typing import Generator, Iterable, Optional, Tuple, Union
import numpy as np
import pandas as pd


class LabelDelayEmbargo:
    """Manages label delay embargo horizons across splits and static partitions."""

    def __init__(self, embargo_horizon: Union[pd.Timedelta, int, float, str]):
        if isinstance(embargo_horizon, (int, float)):
            # If numeric, assume number of days
            self.embargo_delta = pd.Timedelta(days=embargo_horizon)
        elif isinstance(embargo_horizon, str):
            self.embargo_delta = pd.Timedelta(embargo_horizon)
        else:
            self.embargo_delta = embargo_horizon

    def prune_train_indices(
        self,
        train_idx: np.ndarray,
        eval_idx: np.ndarray,
        timestamps: Union[pd.Series, np.ndarray],
    ) -> np.ndarray:
        """Prunes training records falling within the embargo gap immediately prior to evaluation onset."""
        if len(train_idx) == 0 or len(eval_idx) == 0:
            return train_idx

        ts_series = pd.to_datetime(timestamps)
        eval_timestamps = ts_series.iloc[eval_idx].values
        train_timestamps = ts_series.iloc[train_idx].values

        eval_start = eval_timestamps.min()
        embargo_cutoff = eval_start - self.embargo_delta

        # Keep only training observations strictly earlier than embargo cutoff
        valid_mask = train_timestamps < embargo_cutoff
        return train_idx[valid_mask]

    def apply_to_splits(
        self,
        splits: Iterable[Tuple[np.ndarray, np.ndarray]],
        timestamps: Union[pd.Series, np.ndarray],
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Injects embargo gap into an iterable of (train_idx, test_idx) folds."""
        for train_idx, test_idx in splits:
            pruned_train = self.prune_train_indices(train_idx, test_idx, timestamps)
            yield pruned_train, test_idx

    def apply_to_static_partitions(
        self,
        train_idx: np.ndarray,
        cal_idx: np.ndarray,
        holdout_idx: np.ndarray,
        timestamps: Union[pd.Series, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Inserts embargo gaps between Train -> Cal and Cal -> Holdout partitions."""
        # Embargo gap between Train and Cal
        pruned_train = self.prune_train_indices(train_idx, cal_idx, timestamps)
        # Embargo gap between Cal and Holdout
        pruned_cal = self.prune_train_indices(cal_idx, holdout_idx, timestamps)

        return pruned_train, pruned_cal, holdout_idx
