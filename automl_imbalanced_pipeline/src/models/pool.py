"""Candidate model pool initializers, complexity ranking, and tier eligibility.

Complexity Hierarchy:
- Rank 0: Dummy Classifier
- Rank 1: Logistic Regression
- Rank 2: Random Forest
- Rank 3: Gradient Boosted Trees (LightGBM, CatBoost, XGBoost)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class ModelCandidateMeta:
    name: str
    complexity_rank: int
    tier_eligibility: List[int]
    is_tree_model: bool
    supports_class_weights: bool


MODEL_REGISTRY: Dict[str, ModelCandidateMeta] = {
    "dummy": ModelCandidateMeta(
        name="dummy",
        complexity_rank=0,
        tier_eligibility=[1, 2, 3],
        is_tree_model=False,
        supports_class_weights=False,
    ),
    "logistic_regression": ModelCandidateMeta(
        name="logistic_regression",
        complexity_rank=1,
        tier_eligibility=[1, 2, 3],
        is_tree_model=False,
        supports_class_weights=True,
    ),
    "random_forest": ModelCandidateMeta(
        name="random_forest",
        complexity_rank=2,
        tier_eligibility=[2, 3],
        is_tree_model=True,
        supports_class_weights=True,
    ),
    "lightgbm": ModelCandidateMeta(
        name="lightgbm",
        complexity_rank=3,
        tier_eligibility=[2, 3],
        is_tree_model=True,
        supports_class_weights=True,
    ),
    "catboost": ModelCandidateMeta(
        name="catboost",
        complexity_rank=3,
        tier_eligibility=[3],
        is_tree_model=True,
        supports_class_weights=True,
    ),
    "xgboost": ModelCandidateMeta(
        name="xgboost",
        complexity_rank=3,
        tier_eligibility=[3],
        is_tree_model=True,
        supports_class_weights=True,
    ),
}


def build_raw_estimator(model_name: str, params: Optional[Dict[str, Any]] = None, random_state: int = 42):
    """Instantiates the raw estimator with assigned hyperparameters and deterministic random state."""
    name = model_name.lower()
    p = (params or {}).copy()

    if name == "dummy":
        strategy = p.get("strategy", "prior")
        return DummyClassifier(strategy=strategy, random_state=random_state)

    elif name == "logistic_regression":
        solver = p.get("solver", "saga")
        penalty = p.get("penalty", "l2")
        C = p.get("C", 1.0)
        class_weight = p.get("class_weight", None)
        return LogisticRegression(
            solver=solver,
            penalty=penalty,
            C=C,
            class_weight=class_weight,
            random_state=random_state,
            max_iter=500,
        )

    elif name == "random_forest":
        n_estimators = p.get("n_estimators", 100)
        max_depth = p.get("max_depth", 6)
        min_samples_split = p.get("min_samples_split", 5)
        class_weight = p.get("class_weight", None)
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=1,  # Single-threaded for strict determinism
        )

    elif name == "lightgbm":
        from lightgbm import LGBMClassifier
        p.setdefault("n_estimators", 150)
        p.setdefault("learning_rate", 0.05)
        p.setdefault("random_state", random_state)
        p.setdefault("n_jobs", 1)
        p.setdefault("verbose", -1)
        return LGBMClassifier(**p)

    elif name == "catboost":
        from catboost import CatBoostClassifier
        p.setdefault("iterations", 200)
        p.setdefault("learning_rate", 0.05)
        p.setdefault("random_seed", random_state)
        p.setdefault("thread_count", 1)
        p.setdefault("verbose", 0)
        return CatBoostClassifier(**p)

    elif name == "xgboost":
        from xgboost import XGBClassifier
        p.setdefault("n_estimators", 150)
        p.setdefault("learning_rate", 0.05)
        p.setdefault("random_state", random_state)
        p.setdefault("n_jobs", 1)
        p.setdefault("eval_metric", "logloss")
        return XGBClassifier(**p)

    else:
        raise ValueError(f"Unknown model candidate '{model_name}'. Allowed: {list(MODEL_REGISTRY.keys())}")


def get_eligible_models_for_tier(tier: int) -> List[str]:
    """Filters model candidates by tier restriction."""
    return [name for name, meta in MODEL_REGISTRY.items() if tier in meta.tier_eligibility]
