"""Prior shift adjustment: Odds-reweighting for difference between training and deployment prevalence."""

import numpy as np


def adjust_probabilities_for_prior_shift(
    p_train: np.ndarray,
    prevalence_train: float,
    prevalence_deploy: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """Adjusts predicted probabilities using Bayes odds-reweighting:

    odds_deploy = odds_train * (p_deploy / (1 - p_deploy)) / (p_train / (1 - p_train))
    p_deploy_adj = odds_deploy / (1 + odds_deploy)
    """
    p_clipped = np.clip(p_train, eps, 1 - eps)
    odds_train = p_clipped / (1.0 - p_clipped)

    prior_ratio = (prevalence_deploy / (1.0 - prevalence_deploy)) / (
        prevalence_train / (1.0 - prevalence_train)
    )
    odds_deploy = odds_train * prior_ratio
    return odds_deploy / (1.0 + odds_deploy)
