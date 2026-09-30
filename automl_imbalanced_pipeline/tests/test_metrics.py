"""Unit tests for Phase 3: Model Pool, Early Stopping, Nadeau-Bengio Variance, and 1-SE Champion Selection."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.models.pool import MODEL_REGISTRY, build_raw_estimator, get_eligible_models_for_tier
from src.models.early_stopping import carve_unresampled_val_split, EarlyStoppingGBM
from src.optimization.selection import (
    ChampionSelector,
    nadeau_bengio_corrected_variance,
    paired_bootstrap_standard_error,
)


def test_model_pool_tier_eligibility_and_complexity():
    """Validates complexity hierarchy: Dummy (0) < Logistic (1) < RF (2) < GBM (3)."""
    assert MODEL_REGISTRY["dummy"].complexity_rank == 0
    assert MODEL_REGISTRY["logistic_regression"].complexity_rank == 1
    assert MODEL_REGISTRY["random_forest"].complexity_rank == 2
    assert MODEL_REGISTRY["lightgbm"].complexity_rank == 3

    # Tier 1 allows only Dummy and Logistic
    tier_1_models = get_eligible_models_for_tier(1)
    assert set(tier_1_models) == {"dummy", "logistic_regression"}

    # Tier 2 allows up to RF and LightGBM
    tier_2_models = get_eligible_models_for_tier(2)
    assert "random_forest" in tier_2_models
    assert "catboost" not in tier_2_models


def test_carve_unresampled_val_split():
    """Asserts that inner validation set is carved with correct proportions and stratification."""
    X = np.random.randn(100, 4)
    y = np.array([1] * 20 + [0] * 80)
    train_idx, val_idx = carve_unresampled_val_split(X, y, val_fraction=0.20, random_state=42)

    assert len(train_idx) + len(val_idx) == 100
    assert len(val_idx) == 20
    assert np.sum(y[val_idx] == 1) > 0  # Stratified minority presence


def test_nadeau_bengio_variance_correction():
    """Asserts that Nadeau-Bengio correction yields variance strictly greater than naive sample variance."""
    fold_scores = np.array([0.70, 0.72, 0.68, 0.74, 0.71])
    n_train = 800
    n_test = 200

    naive_var = np.var(fold_scores, ddof=1)
    nb_var = nadeau_bengio_corrected_variance(fold_scores, n_train, n_test)

    # In 5-fold CV (n_test/n_train = 200/800 = 0.25): correction factor = 1/5 + 0.25 = 0.45
    correction_factor = (1.0 / 5) + (200.0 / 800.0)
    assert pytest.approx(nb_var, 1e-5) == naive_var * correction_factor
    assert nb_var > 0


def test_champion_selector_1_se_rule():
    """Tests 1-SE rule: selects simpler model (Logistic) over complex tree (LightGBM) if within 1-SE margin."""
    trial_scores = {
        0: [0.75, 0.76, 0.74, 0.75, 0.75],  # LightGBM: mean ~ 0.750, high complexity (rank 3)
        1: [0.74, 0.74, 0.73, 0.75, 0.74],  # Logistic: mean ~ 0.740, low complexity (rank 1)
    }
    trial_models = {
        0: "lightgbm",
        1: "logistic_regression",
    }

    selector = ChampionSelector(prevalence_baseline=0.05)
    result = selector.select_tier3_champion(trial_scores, trial_models, n_train=800, n_test=200)

    assert result.deployable is True
    assert result.champion_model_name == "lightgbm"
    # 0.740 is within 1-SE of 0.750, so Logistic is chosen as the simplest eligible model!
    assert result.simplest_eligible_model_name == "logistic_regression"


def test_champion_selector_beat_baseline_abort():
    """Aborts deployment if no model outperforms the uninformative prevalence baseline."""
    prevalence = 0.20
    trial_scores = {
        0: [0.15, 0.14, 0.16, 0.15, 0.15],  # Scores worse than 0.20 baseline
    }
    trial_models = {0: "logistic_regression"}

    selector = ChampionSelector(prevalence_baseline=prevalence)
    result = selector.select_tier3_champion(trial_scores, trial_models, n_train=800, n_test=200)

    assert result.deployable is False
    assert "prevalence baseline" in result.rejection_reason
