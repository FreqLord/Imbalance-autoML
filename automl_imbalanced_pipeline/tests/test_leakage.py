"""Unit tests for Phase 2: Pipeline Assembler, Encoders, Imputers, Scalers, and Resampling Guards."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing.imputers import CategoricalMissingImputer, ContinuousIterativeImputer
from src.preprocessing.encoders import CrossFittedTargetEncoder, LowCardinalityOrdinalEncoder
from src.preprocessing.scalers import ConditionalRobustScaler
from src.sampling.smote_adapters import DynamicIndexSMOTENC
from src.sampling.strategies import ImbalanceStrategy
from src.preprocessing.pipeline_assembler import build_pipeline_assembler


@pytest.fixture
def sample_imbalanced_df():
    """Generates synthetic dataframe with mixed continuous, low, and high cardinality categoricals."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "num_1": np.random.randn(n) * 100,
        "num_2": np.random.exponential(scale=10, size=n),
        "cat_low": np.random.choice(["red", "blue", "green", None], size=n),
        "cat_high": np.random.choice([f"id_{i}" for i in range(30)], size=n),
    })
    # Inject missing continuous values
    df.loc[0:10, "num_1"] = np.nan
    y = np.array([1] * 20 + [0] * 180)
    return df, y


def test_categorical_and_continuous_imputation(sample_imbalanced_df):
    """Asserts that categorical and continuous imputers eradicate all NaNs before downstream transforms."""
    df, _ = sample_imbalanced_df
    cat_imp = CategoricalMissingImputer(categorical_columns=["cat_low", "cat_high"])
    df_cat = cat_imp.fit_transform(df)
    assert not df_cat["cat_low"].isna().any()
    assert (df_cat["cat_low"] == "__MISSING__").sum() > 0

    cont_imp = ContinuousIterativeImputer(continuous_columns=["num_1", "num_2"])
    df_clean = cont_imp.fit_transform(df_cat)
    assert not df_clean["num_1"].isna().any()


def test_conditional_robust_scaler_rules():
    """Tests scaling invariant: Mandatory for oversampling, bypassed for trees without oversampling."""
    # Tree without oversampling -> Should NOT scale
    scaler_tree_no_os = ConditionalRobustScaler(continuous_columns=["num_1"], is_oversampling=False, is_tree_model=True)
    assert scaler_tree_no_os._determine_scaling_need() is False

    # Tree WITH oversampling -> MUST scale
    scaler_tree_os = ConditionalRobustScaler(continuous_columns=["num_1"], is_oversampling=True, is_tree_model=True)
    assert scaler_tree_os._determine_scaling_need() is True

    # Logistic regression -> ALWAYS scale
    scaler_linear = ConditionalRobustScaler(continuous_columns=["num_1"], is_oversampling=False, is_tree_model=False)
    assert scaler_linear._determine_scaling_need() is True


def test_catboost_high_cardinality_oversampling_guard():
    """Validates that oversampling is disallowed when CatBoost handles raw high-cardinality features."""
    sampler = DynamicIndexSMOTENC(
        categorical_column_names=["cat_high"],
        is_catboost_high_cardinality=True,
    )
    X = pd.DataFrame({"cat_high": ["a", "b", "c"], "num": [1.0, 2.0, 3.0]})
    y = np.array([0, 1, 0])

    with pytest.raises(ValueError, match="disallowed for CatBoost"):
        sampler._fit_resample(X, y)


def test_end_to_end_pipeline_assembler(sample_imbalanced_df):
    """Validates that full assembly fits and predicts cleanly with no leakage and correct step ordering."""
    df, y = sample_imbalanced_df
    pipe = build_pipeline_assembler(
        estimator=LogisticRegression(solver="saga", max_iter=200),
        model_name="logistic_regression",
        categorical_low_cardinality=["cat_low"],
        categorical_high_cardinality=["cat_high"],
        continuous_columns=["num_1", "num_2"],
        imbalance_strategy=ImbalanceStrategy.OVERSAMPLING,
        random_state=42,
    )

    pipe.fit(df, y)
    preds = pipe.predict(df)
    probas = pipe.predict_proba(df)

    assert len(preds) == len(y)
    assert probas.shape == (len(y), 2)
    # Check that steps are strictly ordered
    step_names = [name for name, _ in pipe.steps]
    assert step_names == ["imputer_cat", "imputer_cont", "target_encoder", "ordinal_encoder", "scaler", "resampler", "post_ohe", "classifier"]
