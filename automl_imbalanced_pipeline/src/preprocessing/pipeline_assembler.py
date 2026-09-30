"""Strict sequence pipeline assembler enforcing mathematical component order.

Order:
1. Imputation (Categorical Missing token '__MISSING__' & Continuous MICE).
2. Target Encoding (High-cardinality categoricals; skipped for CatBoost).
3. Ordinal Encoding (Low-cardinality categoricals for SMOTENC compatibility).
4. Conditional RobustScaler (Continuous features; mandatory for SMOTENC, bypassed for trees without oversampling).
5. Resampling (DynamicIndexSMOTENC / RandomUnderSampler / None).
6. Post-Resampling One-Hot Encoding (Applied for linear/logistic; skipped for LightGBM/CatBoost).
7. Estimator.
"""

from typing import Any, List, Optional, Tuple
from imblearn.pipeline import Pipeline

from src.preprocessing.encoders import (
    CrossFittedTargetEncoder,
    LowCardinalityOrdinalEncoder,
    PostResamplingOneHotEncoder,
)
from src.preprocessing.imputers import (
    CategoricalMissingImputer,
    ContinuousIterativeImputer,
)
from src.preprocessing.scalers import ConditionalRobustScaler
from src.sampling.strategies import ImbalanceStrategy, resolve_sampler


def build_pipeline_assembler(
    estimator: Any,
    model_name: str,
    categorical_low_cardinality: Optional[List[str]] = None,
    categorical_high_cardinality: Optional[List[str]] = None,
    continuous_columns: Optional[List[str]] = None,
    imbalance_strategy: ImbalanceStrategy = ImbalanceStrategy.NONE,
    random_state: int = 42,
    cv_target_encode: int = 5,
) -> Pipeline:
    """Constructs an imblearn.pipeline.Pipeline preserving strict ordering invariants."""
    low_card = categorical_low_cardinality or []
    high_card = categorical_high_cardinality or []
    cont_cols = continuous_columns or []

    model_lower = model_name.lower()
    is_tree_model = any(m in model_lower for m in ["lightgbm", "catboost", "xgboost", "random_forest"])
    is_catboost = "catboost" in model_lower
    is_lightgbm = "lightgbm" in model_lower
    is_oversampling = imbalance_strategy == ImbalanceStrategy.OVERSAMPLING

    steps: List[Tuple[str, Any]] = []

    # 1. Imputation
    steps.append(("imputer_cat", CategoricalMissingImputer(categorical_columns=low_card + high_card)))
    steps.append(("imputer_cont", ContinuousIterativeImputer(continuous_columns=cont_cols, random_state=random_state)))

    # 2. Target Encoding (Skip for CatBoost)
    if high_card and not is_catboost:
        steps.append((
            "target_encoder",
            CrossFittedTargetEncoder(
                columns=high_card,
                cv=cv_target_encode,
                random_state=random_state,
            ),
        ))

    # 3. Ordinal Encoding for low cardinality
    if low_card:
        steps.append(("ordinal_encoder", LowCardinalityOrdinalEncoder(columns=low_card)))

    # 4. Conditional RobustScaler
    steps.append((
        "scaler",
        ConditionalRobustScaler(
            continuous_columns=cont_cols,
            is_oversampling=is_oversampling,
            is_tree_model=is_tree_model,
        ),
    ))

    # 5. Resampling Step
    sampler = resolve_sampler(
        strategy=imbalance_strategy,
        categorical_columns=low_card,
        random_state=random_state,
        is_catboost_high_cardinality=(is_catboost and bool(high_card)),
    )
    if sampler is not None:
        steps.append(("resampler", sampler))

    # 6. Post-Resampling One-Hot Encoding (Linear & Distance models only; skipped for LightGBM/CatBoost)
    enable_ohe = low_card and not (is_lightgbm or is_catboost)
    if enable_ohe:
        steps.append(("post_ohe", PostResamplingOneHotEncoder(columns=low_card, enabled=True)))

    # 7. Final Estimator
    steps.append(("classifier", estimator))

    return Pipeline(steps=steps)
