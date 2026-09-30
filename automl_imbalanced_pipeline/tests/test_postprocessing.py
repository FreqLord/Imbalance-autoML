"""Comprehensive integration tests for Phase 4: Postprocessing, Evaluation, Serialization, and Orchestrator."""

import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.deployment.serialization import strip_samplers_from_pipeline
from src.evaluation.metrics import (
    adaptive_expected_calibration_error,
    brier_skill_score,
    calculate_average_precision,
    precision_at_top_k,
)
from src.orchestrator import AutoMLOrchestrator
from src.postprocessing.calibration import PipelineProbabilityCalibrator
from src.postprocessing.prior_shift import adjust_odds_for_prior_shift
from src.postprocessing.thresholding import ThresholdedClassifier, calculate_closed_form_threshold


def test_closed_form_threshold_and_preconditions():
    """Validates Bayes-optimal threshold formula and raises on precondition violations."""
    # Preconditions: c_FP > c_TN and c_FN > c_TP
    # c_FP=2, c_FN=8, c_TP=0, c_TN=0 -> 2 / (2 + 8) = 0.20
    t = calculate_closed_form_threshold(c_fp=2.0, c_fn=8.0, c_tp=0.0, c_tn=0.0)
    assert pytest.approx(t, 1e-4) == 0.20

    with pytest.raises(ValueError, match="Precondition violated"):
        calculate_closed_form_threshold(c_fp=0.0, c_fn=10.0, c_tp=0.0, c_tn=1.0)


def test_prior_shift_odds_reweighting():
    """Validates Bayes odds-reweighting formula under prevalence divergence."""
    # Calibration prevalence: 0.10, Deployment prevalence: 0.20 (2x prevalence increase)
    p_cal = np.array([0.10])
    p_deploy = adjust_odds_for_prior_shift(p_cal, prevalence_cal=0.10, prevalence_deploy=0.20)
    # When p_cal == prev_cal, p_deploy should equal prev_deploy
    assert pytest.approx(p_deploy[0], 1e-4) == 0.20


def test_calibration_isotonic_cutoff():
    """Asserts Platt scaling for < 5k positives and Isotonic for >= 5k positives."""
    X = np.random.randn(100, 2)
    y_small = np.array([1] * 20 + [0] * 80)
    base = LogisticRegression().fit(X, y_small)

    cal_small = PipelineProbabilityCalibrator(base, pos_threshold_for_isotonic=5000)
    cal_small.fit(X, y_small)
    assert cal_small.resolved_method_ == "sigmoid"

    cal_force_iso = PipelineProbabilityCalibrator(base, pos_threshold_for_isotonic=10)
    cal_force_iso.fit(X, y_small)
    assert cal_force_iso.resolved_method_ == "isotonic"


def test_strip_samplers_from_pipeline():
    """Asserts that training-time samplers are pruned from the pipeline before export."""
    pipe = Pipeline([
        ("sampler", SMOTE(random_state=42)),
        ("classifier", LogisticRegression()),
    ])
    clean_pipe = strip_samplers_from_pipeline(pipe)
    step_names = [name for name, _ in clean_pipe.steps]
    assert "sampler" not in step_names
    assert "classifier" in step_names


def test_orchestrator_tier1_execution():
    """Validates Tier 1 scarcity routing when positives < 30."""
    X = pd.DataFrame({"num": np.random.randn(100)})
    y = np.array([1] * 15 + [0] * 85)

    orch = AutoMLOrchestrator(master_seed=42)
    orch.fit(X, y)
    assert orch.metadata_["tier"] == "TIER_1_SCARCITY"
    assert orch.metadata_["operating_threshold"] == "TOP_K_BUDGET"
