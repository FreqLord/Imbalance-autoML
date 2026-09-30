"""Mutually exclusive resampling strategy dispatcher.

Imbalance strategy is treated as a single categorical choice:
- 'none': Default raw distribution.
- 'class_weights': Algorithmic cost sensitivity via estimator loss function (class_weight='balanced').
- 'oversampling': Synthetic interpolation via DynamicIndexSMOTENC / SMOTE.
- 'undersampling': Majority thinning via RandomUnderSampler.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from imblearn.under_sampling import RandomUnderSampler
from src.sampling.smote_adapters import DynamicIndexSMOTENC


class ImbalanceStrategy(str, Enum):
    NONE = "none"
    CLASS_WEIGHTS = "class_weights"
    OVERSAMPLING = "oversampling"
    UNDERSAMPLING = "undersampling"


def resolve_sampler(
    strategy: Union[ImbalanceStrategy, str],
    categorical_columns: Optional[List[str]] = None,
    random_state: int = 42,
    is_catboost_high_cardinality: bool = False,
    sampling_ratio: float = 0.5,
) -> Optional[Any]:
    """Returns the configured imblearn sampler or None if algorithm-weighting or unadjusted."""
    strat = ImbalanceStrategy(strategy)

    if strat in (ImbalanceStrategy.NONE, ImbalanceStrategy.CLASS_WEIGHTS):
        return None

    elif strat == ImbalanceStrategy.OVERSAMPLING:
        return DynamicIndexSMOTENC(
            categorical_column_names=categorical_columns,
            random_state=random_state,
            sampling_strategy=sampling_ratio,
            is_catboost_high_cardinality=is_catboost_high_cardinality,
        )

    elif strat == ImbalanceStrategy.UNDERSAMPLING:
        return RandomUnderSampler(
            sampling_strategy=sampling_ratio,
            random_state=random_state,
        )

    return None
