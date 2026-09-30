"""Positives Gateway: Dynamic partition-aware tier routing logic for imbalanced datasets.

Evaluates minority class representation against partition fractions and absolute counts:
- Tier 1: Extreme Scarcity (Total positives < 30) -> Hard floor, top-k alert budget.
- Tier 2: Low-Count Guard (30 <= Total positives < 500 or partition starvation) -> Nested CV.
- Tier 3: The Static Path (Total positives >= 500 with viable partitions) -> Static Train/Cal/Holdout.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class PipelineTier(IntEnum):
    """Execution tiers for imbalanced model selection."""
    TIER_1_SCARCITY = 1
    TIER_2_NESTED_CV = 2
    TIER_3_STATIC_PATH = 3


@dataclass(frozen=True)
class PartitionBudget:
    """Calculated partition counts and eligibility constraints."""
    total_samples: int
    total_positives: int
    prevalence: float
    tier: PipelineTier
    train_positives_est: int
    cal_positives_est: int
    holdout_positives_est: int
    requires_nested_cv: bool
    allowed_models: List[str]
    supports_probability_thresholds: bool
    requires_alert_budget: bool
    reason: str


class PositivesGatewayRouter:
    """Gateway router determining execution topology and split safety.

    Guarantees:
    - Never starves static calibration or holdout partitions.
    - Accurately checks Partition_Positives = Total_Positives * Partition_Fraction >= Minimum.
    """

    def __init__(
        self,
        tier_1_max_positives: int = 30,
        tier_2_max_positives: int = 500,
        min_calibration_positives: int = 50,
        min_holdout_positives: int = 50,
        cal_fraction: float = 0.15,
        holdout_fraction: float = 0.15,
    ):
        self.tier_1_max_positives = tier_1_max_positives
        self.tier_2_max_positives = tier_2_max_positives
        self.min_calibration_positives = min_calibration_positives
        self.min_holdout_positives = min_holdout_positives
        self.cal_fraction = cal_fraction
        self.holdout_fraction = holdout_fraction
        self.train_fraction = 1.0 - (cal_fraction + holdout_fraction)

    def route(self, y: Union[pd.Series, np.ndarray, List[int]]) -> PartitionBudget:
        """Evaluates target distribution and routes to the appropriate tier."""
        y_arr = np.asarray(y)
        total_samples = len(y_arr)
        if total_samples == 0:
            raise ValueError("Input target sequence cannot be empty.")

        total_positives = int(np.sum(y_arr == 1))
        prevalence = float(total_positives / total_samples)

        # Tier 1: Extreme Scarcity
        if total_positives < self.tier_1_max_positives:
            return PartitionBudget(
                total_samples=total_samples,
                total_positives=total_positives,
                prevalence=prevalence,
                tier=PipelineTier.TIER_1_SCARCITY,
                train_positives_est=total_positives,
                cal_positives_est=0,
                holdout_positives_est=0,
                requires_nested_cv=False,
                allowed_models=["dummy", "logistic_regression"],
                supports_probability_thresholds=False,
                requires_alert_budget=True,
                reason=(
                    f"Extreme scarcity detected: Total positives ({total_positives}) < {self.tier_1_max_positives}. "
                    "Refusing automatic model search. Limited to baseline models and top-k capacity budget."
                ),
            )

        # Estimate positives per static partition
        holdout_pos = int(np.floor(total_positives * self.holdout_fraction))
        cal_pos = int(np.floor(total_positives * self.cal_fraction))
        train_pos = total_positives - (holdout_pos + cal_pos)

        # Tier 2: Low-Count Guard or Partition Starvation
        is_low_total = total_positives < self.tier_2_max_positives
        is_partition_starved = (
            holdout_pos < self.min_holdout_positives
            or cal_pos < self.min_calibration_positives
        )

        if is_low_total or is_partition_starved:
            reason = (
                f"Low-count guard activated ({total_positives} positives). "
                f"Static split would starve holdout ({holdout_pos} pos) or calibration ({cal_pos} pos). "
                "Routing to Nested Cross-Validation (Tier 2)."
            )
            return PartitionBudget(
                total_samples=total_samples,
                total_positives=total_positives,
                prevalence=prevalence,
                tier=PipelineTier.TIER_2_NESTED_CV,
                train_positives_est=train_pos,
                cal_positives_est=cal_pos,
                holdout_positives_est=holdout_pos,
                requires_nested_cv=True,
                allowed_models=["dummy", "logistic_regression", "random_forest", "lightgbm"],
                supports_probability_thresholds=True,
                requires_alert_budget=False,
                reason=reason,
            )

        # Tier 3: The Static Path
        return PartitionBudget(
            total_samples=total_samples,
            total_positives=total_positives,
            prevalence=prevalence,
            tier=PipelineTier.TIER_3_STATIC_PATH,
            train_positives_est=train_pos,
            cal_positives_est=cal_pos,
            holdout_positives_est=holdout_pos,
            requires_nested_cv=False,
            allowed_models=["dummy", "logistic_regression", "random_forest", "lightgbm", "catboost", "xgboost"],
            supports_probability_thresholds=True,
            requires_alert_budget=False,
            reason=(
                f"Sufficient positives detected ({total_positives} total). "
                f"Partitions adequately provisioned: Train (~{train_pos}), Cal (~{cal_pos}), Holdout (~{holdout_pos}). "
                "Routing to Static Train/Calibration/Holdout Path (Tier 3)."
            ),
        )


def determine_tier(y: Union[pd.Series, np.ndarray, List[int]], **kwargs) -> PipelineTier:
    """Convenience functional interface for tier routing."""
    router = PositivesGatewayRouter(**kwargs)
    return router.route(y).tier
