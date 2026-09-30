"""Unit tests for Phase 1: Tier Router, Seed Manager, Splitters, and Embargo Logic."""

import numpy as np
import pandas as pd
import pytest

from src.core.tier_router import PipelineTier, PositivesGatewayRouter, determine_tier
from src.core.seed_manager import SeedManager
from src.data.splitters import RollingOriginSplitter, StaticPartitionSplitter
from src.data.embargo import LabelDelayEmbargo


def test_positives_gateway_scarcity():
    """Validates Tier 1 hard floor on extreme scarcity (< 30 positives)."""
    router = PositivesGatewayRouter(tier_1_max_positives=30)
    y = np.array([1] * 25 + [0] * 5000)
    budget = router.route(y)

    assert budget.tier == PipelineTier.TIER_1_SCARCITY
    assert budget.requires_alert_budget is True
    assert budget.supports_probability_thresholds is False
    assert "dummy" in budget.allowed_models
    assert "lightgbm" not in budget.allowed_models


def test_positives_gateway_nested_cv_guard():
    """Validates Tier 2 routing when positives are between 30 and 500 or partition starves."""
    router = PositivesGatewayRouter(
        tier_1_max_positives=30,
        tier_2_max_positives=500,
        min_calibration_positives=50,
        min_holdout_positives=50,
        cal_fraction=0.15,
        holdout_fraction=0.15,
    )
    # 200 positives: enough for overall CV, but 15% holdout = 30 (< 50 min), starves static split!
    y = np.array([1] * 200 + [0] * 10000)
    budget = router.route(y)

    assert budget.tier == PipelineTier.TIER_2_NESTED_CV
    assert budget.requires_nested_cv is True


def test_positives_gateway_static_path():
    """Validates Tier 3 routing when positives >= 500 and partitions are well-stocked."""
    router = PositivesGatewayRouter(
        tier_1_max_positives=30,
        tier_2_max_positives=500,
        min_calibration_positives=50,
        min_holdout_positives=50,
    )
    y = np.array([1] * 800 + [0] * 20000)
    budget = router.route(y)

    assert budget.tier == PipelineTier.TIER_3_STATIC_PATH
    assert budget.requires_nested_cv is False
    assert "catboost" in budget.allowed_models
    assert budget.holdout_positives_est >= 50


def test_seed_manager_deterministic_streams():
    """Validates that SeedSequence derives non-overlapping deterministic child streams."""
    sm1 = SeedManager(master_seed=12345)
    sm2 = SeedManager(master_seed=12345)

    seeds1 = [sm1.derive_int_seed("splitters", i) for i in range(5)]
    seeds2 = [sm2.derive_int_seed("splitters", i) for i in range(5)]
    assert seeds1 == seeds2

    # Different domain yields different sequence
    optuna_seed = sm1.derive_int_seed("optuna", 0)
    assert optuna_seed != seeds1[0]


def test_rolling_origin_splitter_with_groups():
    """Validates chronological expanding forward splits with group integrity and covered index tracking."""
    n = 100
    df = pd.DataFrame({
        "feature": np.arange(n),
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="D"),
        "customer_id": np.repeat(np.arange(20), 5),  # 20 customers, 5 records each
        "target": np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
    })

    splitter = RollingOriginSplitter(n_splits=3, min_train_fraction=0.4)
    splits = list(splitter.split(df, groups=df["customer_id"], timestamps=df["timestamp"]))

    assert len(splits) == 3
    assert len(splitter.covered_indices_) > 0

    for train_idx, test_idx in splits:
        # Group isolation check
        train_groups = set(df["customer_id"].iloc[train_idx])
        test_groups = set(df["customer_id"].iloc[test_idx])
        assert len(train_groups.intersection(test_groups)) == 0


def test_label_delay_embargo():
    """Validates pruning of training records within the embargo gap before evaluation windows."""
    timestamps = pd.date_range("2025-01-01", periods=30, freq="D")
    embargo = LabelDelayEmbargo(embargo_horizon="5D")

    train_idx = np.arange(0, 20)  # Days 1 to 20
    test_idx = np.arange(20, 30)  # Days 21 to 30

    pruned_train = embargo.prune_train_indices(train_idx, test_idx, timestamps)
    # Test starts on Day 21 (index 20). Embargo is 5 days -> cutoff is Day 16 (index 15).
    # Valid train indices should strictly be < index 15 (i.e. up to index 14)
    assert pruned_train.max() < 15
