"""Data partitioners and cross-validation splitters.

Supports:
- Group-aware StratifiedGroupKFold
- Rolling-Origin Forward Splitter for temporal panels (with group awareness and covered subset tracking)
- Static Train / Calibration / Holdout partitioner for Tier 3
"""

from typing import Generator, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


class RollingOriginSplitter:
    """Group-aware rolling-origin forward splitter for temporal imbalanced datasets.

    Properties:
    - Expanding training window.
    - Forward-facing out-of-sample test window.
    - Group-integrity: Group instances occurring in the test window are cleanly separated.
    - Tracks 'covered_indices': union of test fold indices to ensure fair multi-model comparison.
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_fraction: float = 0.15,
        min_train_fraction: float = 0.25,
    ):
        self.n_splits = n_splits
        self.test_fraction = test_fraction
        self.min_train_fraction = min_train_fraction
        self.covered_indices_: np.ndarray = np.array([], dtype=int)

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None,
        timestamps: Optional[pd.Series] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generates expanding-window forward train/test index pairs."""
        n_samples = len(X)
        if n_samples == 0:
            raise ValueError("Input dataset X cannot be empty.")

        # Order chronologically if timestamps provided
        if timestamps is not None:
            sort_order = np.argsort(pd.to_datetime(timestamps).values)
        else:
            sort_order = np.arange(n_samples)

        all_test_indices: List[np.ndarray] = []
        min_train_size = int(np.floor(n_samples * self.min_train_fraction))
        eval_budget = n_samples - min_train_size
        step_size = eval_budget // self.n_splits

        for fold in range(self.n_splits):
            train_end = min_train_size + fold * step_size
            test_end = min_train_size + (fold + 1) * step_size if fold < self.n_splits - 1 else n_samples

            train_pos_idx = sort_order[:train_end]
            test_pos_idx = sort_order[train_end:test_end]

            # If groups are specified, ensure test groups do not leak into train
            if groups is not None:
                groups_arr = np.asarray(groups)
                test_groups = set(groups_arr[test_pos_idx])
                # Remove overlapping groups from train to prevent cluster leakage
                train_mask = ~np.isin(groups_arr[train_pos_idx], list(test_groups))
                train_pos_idx = train_pos_idx[train_mask]

            all_test_indices.append(test_pos_idx)
            yield train_pos_idx, test_pos_idx

        # Union of all evaluated out-of-sample observations
        if all_test_indices:
            self.covered_indices_ = np.sort(np.concatenate(all_test_indices))
        else:
            self.covered_indices_ = np.array([], dtype=int)


class StaticPartitionSplitter:
    """Static 3-way partitioner (Train / Calibration / Holdout) for Tier 3 static path.

    Adheres strictly to chronological ordering when temporal columns are present,
    and preserves group isolation.
    """

    def __init__(
        self,
        cal_fraction: float = 0.15,
        holdout_fraction: float = 0.15,
        random_state: int = 42,
    ):
        self.cal_fraction = cal_fraction
        self.holdout_fraction = holdout_fraction
        self.random_state = random_state

    def partition(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: Optional[pd.Series] = None,
        timestamps: Optional[pd.Series] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (train_indices, cal_indices, holdout_indices)."""
        n_samples = len(X)

        if timestamps is not None:
            # Temporal path: Chronological slicing: Train -> Cal -> Holdout
            chronological_order = np.argsort(pd.to_datetime(timestamps).values)
            holdout_start = int(np.floor(n_samples * (1.0 - self.holdout_fraction)))
            cal_start = int(np.floor(n_samples * (1.0 - (self.cal_fraction + self.holdout_fraction))))

            train_idx = chronological_order[:cal_start]
            cal_idx = chronological_order[cal_start:holdout_start]
            holdout_idx = chronological_order[holdout_start:]

            if groups is not None:
                # Disallow group leakage from cal or holdout backward into train
                groups_arr = np.asarray(groups)
                holdout_groups = set(groups_arr[holdout_idx])
                cal_groups = set(groups_arr[cal_idx])
                excluded_train = holdout_groups | cal_groups
                train_idx = train_idx[~np.isin(groups_arr[train_idx], list(excluded_train))]

            return train_idx, cal_idx, holdout_idx

        elif groups is not None:
            # Group-aware stratified splitting
            sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            splits = list(sgkf.split(X, y, groups))
            holdout_idx = splits[0][1]
            rem_idx = splits[0][0]

            X_rem = X.iloc[rem_idx]
            y_rem = y.iloc[rem_idx]
            groups_rem = groups.iloc[rem_idx]

            sgkf_cal = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=self.random_state)
            cal_splits = list(sgkf_cal.split(X_rem, y_rem, groups_rem))
            cal_idx = rem_idx[cal_splits[0][1]]
            train_idx = rem_idx[cal_splits[0][0]]

            return train_idx, cal_idx, holdout_idx

        else:
            # Standard stratified splitting
            skf = StratifiedKFold(n_splits=int(1.0 / self.holdout_fraction), shuffle=True, random_state=self.random_state)
            splits = list(skf.split(X, y))
            holdout_idx = splits[0][1]
            rem_idx = splits[0][0]

            cal_n_splits = int(len(rem_idx) / (n_samples * self.cal_fraction))
            skf_cal = StratifiedKFold(n_splits=max(2, cal_n_splits), shuffle=True, random_state=self.random_state)
            cal_splits = list(skf_cal.split(X.iloc[rem_idx], y.iloc[rem_idx]))
            cal_idx = rem_idx[cal_splits[0][1]]
            train_idx = rem_idx[cal_splits[0][0]]

            return train_idx, cal_idx, holdout_idx
