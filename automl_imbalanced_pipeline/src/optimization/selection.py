"""Champion selection gates: 1-SE Rule, Nadeau-Bengio variance correction, and Beat-Baseline gate.

Mathematical Invariants:
- Tier 3: Corrects for fold overlap via Nadeau-Bengio:
  Var_corr = S^2 * (1/K + n_test / n_train), SE = sqrt(Var_corr)
- Tier 2: Paired bootstrap of pooled OOF predictions.
- 1-SE Rule: Selects the simplest model whose performance falls within 1-SE of the empirical champion.
- Beat-Baseline Gate:
  1. If Dummy / Logistic baseline falls within 1-SE of complex tree ensembles, the pipeline chooses the simpler model.
  2. If no candidate outperforms the uninformative prevalence baseline, deployment is aborted.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import average_precision_score

from src.models.pool import MODEL_REGISTRY


@dataclass(frozen=True)
class SelectionResult:
    champion_trial_id: int
    champion_model_name: str
    champion_mean_ap: float
    champion_se: float
    simplest_eligible_trial_id: int
    simplest_eligible_model_name: str
    prevalence_baseline: float
    deployable: bool
    rejection_reason: Optional[str]


def nadeau_bengio_corrected_variance(
    fold_scores: np.ndarray,
    n_train: int,
    n_test: int,
) -> float:
    """Computes Nadeau-Bengio corrected variance for cross-validation estimators:

    Var_corr = Var_sample * (1/K + n_test / n_train)
    """
    k = len(fold_scores)
    if k <= 1:
        return 0.0
    sample_var = float(np.var(fold_scores, ddof=1))
    correction = (1.0 / k) + (float(n_test) / float(n_train))
    return float(sample_var * correction)


def paired_bootstrap_standard_error(
    y_true: np.ndarray,
    y_prob_best: np.ndarray,
    y_prob_cand: np.ndarray,
    n_bootstraps: int = 1000,
    random_state: int = 42,
) -> float:
    """Computes standard error of the performance difference (Delta AP) via paired bootstrapping."""
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    diffs = []

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        if np.sum(y_true[idx]) == 0:
            continue
        ap_best = average_precision_score(y_true[idx], y_prob_best[idx])
        ap_cand = average_precision_score(y_true[idx], y_prob_cand[idx])
        diffs.append(ap_best - ap_cand)

    return float(np.std(diffs, ddof=1)) if diffs else 0.0


class ChampionSelector:
    """Executes champion selection gates adhering to the 1-SE rule and complexity penalization."""

    def __init__(self, prevalence_baseline: float):
        self.prevalence_baseline = prevalence_baseline

    def select_tier3_champion(
        self,
        trial_scores: Dict[int, List[float]],
        trial_models: Dict[int, str],
        n_train: int,
        n_test: int,
    ) -> SelectionResult:
        """Applies Nadeau-Bengio 1-SE rule and Beat-Baseline gate for Tier 3."""
        if not trial_scores:
            return SelectionResult(
                champion_trial_id=-1,
                champion_model_name="none",
                champion_mean_ap=0.0,
                champion_se=0.0,
                simplest_eligible_trial_id=-1,
                simplest_eligible_model_name="none",
                prevalence_baseline=self.prevalence_baseline,
                deployable=False,
                rejection_reason="No completed trials available.",
            )

        mean_scores = {t_id: float(np.mean(scores)) for t_id, scores in trial_scores.items()}
        best_trial_id = max(mean_scores, key=mean_scores.get)
        best_mean = mean_scores[best_trial_id]
        best_scores_arr = np.asarray(trial_scores[best_trial_id])

        # Nadeau-Bengio SE for best candidate
        nb_var = nadeau_bengio_corrected_variance(best_scores_arr, n_train, n_test)
        se_best = float(np.sqrt(max(0.0, nb_var)))

        # 1-SE Threshold
        se_threshold = best_mean - se_best

        # Gate 1: Check if best candidate demonstrates skill above uninformative prior
        if best_mean <= self.prevalence_baseline:
            return SelectionResult(
                champion_trial_id=best_trial_id,
                champion_model_name=trial_models[best_trial_id],
                champion_mean_ap=best_mean,
                champion_se=se_best,
                simplest_eligible_trial_id=best_trial_id,
                simplest_eligible_model_name=trial_models[best_trial_id],
                prevalence_baseline=self.prevalence_baseline,
                deployable=False,
                rejection_reason=(
                    f"No model demonstrates skill above the uninformative prevalence baseline "
                    f"({best_mean:.4f} <= {self.prevalence_baseline:.4f}). Deployment aborted."
                ),
            )

        # Eligible candidates within 1-SE bound
        eligible_trials = [t_id for t_id, score in mean_scores.items() if score >= se_threshold]

        # Rank eligible trials by complexity
        def get_rank(t_id: int) -> Tuple[int, float]:
            model_name = trial_models[t_id]
            rank = MODEL_REGISTRY.get(model_name).complexity_rank if model_name in MODEL_REGISTRY else 99
            # Secondary tiebreaker: negative score (higher score preferred among equal complexity)
            return rank, -mean_scores[t_id]

        simplest_trial_id = min(eligible_trials, key=get_rank)
        simplest_model = trial_models[simplest_trial_id]

        return SelectionResult(
            champion_trial_id=best_trial_id,
            champion_model_name=trial_models[best_trial_id],
            champion_mean_ap=best_mean,
            champion_se=se_best,
            simplest_eligible_trial_id=simplest_trial_id,
            simplest_eligible_model_name=simplest_model,
            prevalence_baseline=self.prevalence_baseline,
            deployable=True,
            rejection_reason=None,
        )
