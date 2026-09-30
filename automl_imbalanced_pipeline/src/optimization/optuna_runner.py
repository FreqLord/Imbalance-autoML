"""Optuna Bayesian hyperparameter search execution with early pruning.

Runs 5x1 search optimizing Average Precision (AP), capturing per-fold scores for
subsequent 1-SE rule and Nadeau-Bengio variance calculations.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score

from src.core.seed_manager import SeedManager
from src.models.pool import build_raw_estimator
from src.preprocessing.pipeline_assembler import build_pipeline_assembler
from src.sampling.strategies import ImbalanceStrategy


class OptunaSearchRunner:
    """Manages Bayesian hyperparameter optimization across model candidates and resampling strategies."""

    def __init__(
        self,
        seed_manager: SeedManager,
        n_trials: int = 40,
        timeout: Optional[int] = 1800,
    ):
        self.seed_manager = seed_manager
        self.n_trials = n_trials
        self.timeout = timeout
        self.study_: Optional[optuna.Study] = None
        self.trial_fold_scores_: Dict[int, List[float]] = {}
        self.trial_model_names_: Dict[int, str] = {}

    def _sample_params_for_model(self, trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
        """Samples hyperparameters based on model family."""
        params: Dict[str, Any] = {}

        if model_name == "dummy":
            params["strategy"] = trial.suggest_categorical("dummy_strategy", ["stratified", "most_frequent", "prior"])

        elif model_name == "logistic_regression":
            params["C"] = trial.suggest_float("lr_C", 1e-4, 1e2, log=True)
            params["penalty"] = trial.suggest_categorical("lr_penalty", ["l1", "l2"])
            params["solver"] = "saga"

        elif model_name == "random_forest":
            params["n_estimators"] = trial.suggest_int("rf_n_estimators", 50, 300, step=50)
            params["max_depth"] = trial.suggest_int("rf_max_depth", 3, 10)
            params["min_samples_split"] = trial.suggest_int("rf_min_samples_split", 2, 10)

        elif model_name == "lightgbm":
            params["n_estimators"] = trial.suggest_int("lgb_n_estimators", 50, 400, step=50)
            params["learning_rate"] = trial.suggest_float("lgb_lr", 0.01, 0.2, log=True)
            params["num_leaves"] = trial.suggest_int("lgb_num_leaves", 15, 63)
            params["subsample"] = trial.suggest_float("lgb_subsample", 0.6, 1.0)
            params["colsample_bytree"] = trial.suggest_float("lgb_colsample", 0.6, 1.0)

        elif model_name == "catboost":
            params["iterations"] = trial.suggest_int("cb_iterations", 100, 400, step=50)
            params["learning_rate"] = trial.suggest_float("cb_lr", 0.01, 0.2, log=True)
            params["depth"] = trial.suggest_int("cb_depth", 3, 8)

        elif model_name == "xgboost":
            params["n_estimators"] = trial.suggest_int("xgb_n_estimators", 50, 400, step=50)
            params["learning_rate"] = trial.suggest_float("xgb_lr", 0.01, 0.2, log=True)
            params["max_depth"] = trial.suggest_int("xgb_max_depth", 3, 8)
            params["subsample"] = trial.suggest_float("xgb_subsample", 0.6, 1.0)
            params["colsample_bytree"] = trial.suggest_float("xgb_colsample", 0.6, 1.0)

        return params

    def run_cv_search(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        splits: List[Tuple[np.ndarray, np.ndarray]],
        allowed_models: List[str],
        categorical_low: Optional[List[str]] = None,
        categorical_high: Optional[List[str]] = None,
        continuous_cols: Optional[List[str]] = None,
    ) -> optuna.Study:
        """Executes multi-model Bayesian optimization across provided CV folds."""
        optuna_seed = self.seed_manager.derive_int_seed("optuna", 0)
        sampler = optuna.samplers.TPESampler(seed=optuna_seed)
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=2)

        self.study_ = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )

        def objective(trial: optuna.Trial) -> float:
            model_name = trial.suggest_categorical("model_name", allowed_models)
            imbalance_strat_str = trial.suggest_categorical(
                "imbalance_strategy",
                ["none", "class_weights", "oversampling", "undersampling"],
            )
            imbalance_strategy = ImbalanceStrategy(imbalance_strat_str)

            # High-cardinality CatBoost guard
            if model_name == "catboost" and categorical_high and imbalance_strategy == ImbalanceStrategy.OVERSAMPLING:
                raise optuna.TrialPruned("Disallowed: CatBoost + unencoded high cardinality + oversampling.")

            params = self._sample_params_for_model(trial, model_name)
            if imbalance_strategy == ImbalanceStrategy.CLASS_WEIGHTS and model_name in ["logistic_regression", "random_forest"]:
                params["class_weight"] = "balanced"

            fold_scores: List[float] = []

            for step, (train_idx, val_idx) in enumerate(splits):
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

                # Ensure fold has at least 1 positive
                if y_val.sum() == 0:
                    continue

                trial_seed = self.seed_manager.derive_int_seed("trials", trial.number * 100 + step)
                raw_est = build_raw_estimator(model_name, params, random_state=trial_seed)

                pipe = build_pipeline_assembler(
                    estimator=raw_est,
                    model_name=model_name,
                    categorical_low_cardinality=categorical_low,
                    categorical_high_cardinality=categorical_high,
                    continuous_columns=continuous_cols,
                    imbalance_strategy=imbalance_strategy,
                    random_state=trial_seed,
                )

                pipe.fit(X_train, y_train)

                if hasattr(pipe, "predict_proba"):
                    y_prob = pipe.predict_proba(X_val)[:, 1]
                else:
                    y_prob = pipe.predict(X_val)

                fold_ap = float(average_precision_score(y_val, y_prob))
                fold_scores.append(fold_ap)

                # Report intermediate score for pruning
                trial.report(float(np.mean(fold_scores)), step=step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            mean_ap = float(np.mean(fold_scores)) if fold_scores else 0.0
            self.trial_fold_scores_[trial.number] = fold_scores
            self.trial_model_names_[trial.number] = model_name
            return mean_ap

        self.study_.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        return self.study_
