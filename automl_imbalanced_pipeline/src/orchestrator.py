"""End-to-end AutoML Orchestrator executing the conditional architecture DAG."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from src.core.seed_manager import SeedManager
from src.core.tier_router import PipelineTier, PositivesGatewayRouter
from src.data.embargo import LabelDelayEmbargo
from src.data.splitters import RollingOriginSplitter, StaticPartitionSplitter
from src.deployment.model_card import generate_enterprise_model_card
from src.deployment.serialization import export_pipeline_artifact, strip_samplers_from_pipeline
from src.evaluation.bootstrap import compute_bootstrap_ci
from src.evaluation.metrics import (
    adaptive_expected_calibration_error,
    brier_skill_score,
    calculate_average_precision,
    precision_at_top_k,
)
from src.models.pool import build_raw_estimator
from src.optimization.optuna_runner import OptunaSearchRunner
from src.optimization.selection import ChampionSelector
from src.postprocessing.calibration import PipelineProbabilityCalibrator
from src.postprocessing.prior_shift import PriorShiftAdjustedClassifier
from src.postprocessing.thresholding import (
    ThresholdedClassifier,
    calculate_closed_form_threshold,
    tune_metric_threshold,
)
from src.preprocessing.pipeline_assembler import build_pipeline_assembler
from src.sampling.strategies import ImbalanceStrategy


class AutoMLOrchestrator:
    """Master orchestrator implementing the Positives Gateway and conditional DAG routing."""

    def __init__(
        self,
        master_seed: int = 42,
        cost_matrix: Optional[Dict[str, float]] = None,
        deployment_prevalence: Optional[float] = None,
        group_column: Optional[str] = None,
        time_column: Optional[str] = None,
        embargo_horizon: Optional[Union[str, int, float]] = None,
        n_trials: int = 30,
    ):
        self.master_seed = master_seed
        self.seed_manager = SeedManager(master_seed)
        self.cost_matrix = cost_matrix or {"c_fp": 1.0, "c_fn": 10.0, "c_tp": 0.0, "c_tn": 0.0}
        self.deployment_prevalence = deployment_prevalence
        self.group_column = group_column
        self.time_column = time_column
        self.embargo_horizon = embargo_horizon
        self.n_trials = n_trials

        self.router = PositivesGatewayRouter()
        self.fitted_artifact_: Optional[Any] = None
        self.metadata_: Dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> "AutoMLOrchestrator":
        """Executes the pipeline training DAG according to the determined tier."""
        y_series = pd.Series(y).reset_index(drop=True)
        X_df = X.reset_index(drop=True)

        # 1. Positives Gateway Routing
        budget = self.router.route(y_series)
        prevalence = float(np.mean(y_series))

        self.metadata_ = {
            "tier": budget.tier.name,
            "training_prevalence": prevalence,
            "deployment_prevalence": self.deployment_prevalence or prevalence,
            "total_samples": budget.total_samples,
            "total_positives": budget.total_positives,
            "cost_matrix": self.cost_matrix,
            "master_seed": self.master_seed,
        }

        # Identify feature categories
        cat_cols = list(X_df.select_dtypes(include=["object", "category"]).columns)
        low_card = [c for c in cat_cols if X_df[c].nunique() <= 10]
        high_card = [c for c in cat_cols if X_df[c].nunique() > 10]
        num_cols = list(X_df.select_dtypes(include=[np.number]).columns)

        # Filter out meta columns
        if self.group_column and self.group_column in num_cols:
            num_cols.remove(self.group_column)
        if self.time_column and self.time_column in num_cols:
            num_cols.remove(self.time_column)

        # -------------------------------------------------------------
        # TIER 1: Extreme Scarcity (< 30 Positives)
        # -------------------------------------------------------------
        if budget.tier == PipelineTier.TIER_1_SCARCITY:
            # Baseline-only fitting with top-k capacity allocation
            pipe = build_pipeline_assembler(
                estimator=build_raw_estimator("logistic_regression", random_state=self.master_seed),
                model_name="logistic_regression",
                categorical_low_cardinality=low_card,
                categorical_high_cardinality=high_card,
                continuous_columns=num_cols,
                imbalance_strategy=ImbalanceStrategy.NONE,
                random_state=self.master_seed,
            )
            pipe.fit(X_df, y_series)
            self.fitted_artifact_ = pipe
            self.metadata_["operating_threshold"] = "TOP_K_BUDGET"
            self.metadata_["model_name"] = "logistic_regression_baseline"
            return self

        # -------------------------------------------------------------
        # TIER 2: Low-Count Guard (30 <= Positives < 500) -> Nested CV
        # -------------------------------------------------------------
        elif budget.tier == PipelineTier.TIER_2_NESTED_CV:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.master_seed)
            splits = list(skf.split(X_df, y_series))

            runner = OptunaSearchRunner(seed_manager=self.seed_manager, n_trials=self.n_trials)
            runner.run_cv_search(
                X=X_df,
                y=y_series,
                splits=splits,
                allowed_models=budget.allowed_models,
                categorical_low=low_card,
                categorical_high=high_card,
                continuous_cols=num_cols,
            )

            # Fit calibrator on OOF predictions
            best_trial = runner.study_.best_trial
            best_model_name = best_trial.params.get("model_name", "logistic_regression")
            best_strategy = ImbalanceStrategy(best_trial.params.get("imbalance_strategy", "none"))

            champion_pipe = build_pipeline_assembler(
                estimator=build_raw_estimator(best_model_name, random_state=self.master_seed),
                model_name=best_model_name,
                categorical_low_cardinality=low_card,
                categorical_high_cardinality=high_card,
                continuous_columns=num_cols,
                imbalance_strategy=best_strategy,
                random_state=self.master_seed,
            )
            champion_pipe.fit(X_df, y_series)

            calibrator = PipelineProbabilityCalibrator(base_estimator=champion_pipe, cv=3)
            calibrator.fit(X_df, y_series)

            threshold = calculate_closed_form_threshold(
                c_fp=self.cost_matrix["c_fp"],
                c_fn=self.cost_matrix["c_fn"],
                c_tp=self.cost_matrix.get("c_tp", 0.0),
                c_tn=self.cost_matrix.get("c_tn", 0.0),
            )
            self.fitted_artifact_ = ThresholdedClassifier(estimator=calibrator, threshold=threshold)
            self.metadata_["model_name"] = best_model_name
            self.metadata_["operating_threshold"] = threshold
            return self

        # -------------------------------------------------------------
        # TIER 3: The Static Path (Positives >= 500)
        # -------------------------------------------------------------
        else:
            partitioner = StaticPartitionSplitter(cal_fraction=0.15, holdout_fraction=0.15, random_state=self.master_seed)
            train_idx, cal_idx, holdout_idx = partitioner.partition(
                X_df,
                y_series,
                groups=X_df[self.group_column] if self.group_column else None,
                timestamps=X_df[self.time_column] if self.time_column else None,
            )

            # Apply Embargo
            if self.time_column and self.embargo_horizon:
                embargo = LabelDelayEmbargo(self.embargo_horizon)
                train_idx, cal_idx, holdout_idx = embargo.apply_to_static_partitions(
                    train_idx, cal_idx, holdout_idx, X_df[self.time_column]
                )

            X_train, y_train = X_df.iloc[train_idx], y_series.iloc[train_idx]
            X_cal, y_cal = X_df.iloc[cal_idx], y_series.iloc[cal_idx]
            X_holdout, y_holdout = X_df.iloc[holdout_idx], y_series.iloc[holdout_idx]

            # 5x1 CV Search on Train Set
            cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.master_seed)
            cv_splits = list(cv_splitter.split(X_train, y_train))

            runner = OptunaSearchRunner(seed_manager=self.seed_manager, n_trials=self.n_trials)
            runner.run_cv_search(
                X=X_train,
                y=y_train,
                splits=cv_splits,
                allowed_models=budget.allowed_models,
                categorical_low=low_card,
                categorical_high=high_card,
                continuous_cols=num_cols,
            )

            # Champion Selection Gate (1-SE Rule & Beat-Baseline)
            selector = ChampionSelector(prevalence_baseline=float(np.mean(y_train)))
            selection = selector.select_tier3_champion(
                trial_scores=runner.trial_fold_scores_,
                trial_models=runner.trial_model_names_,
                n_train=len(train_idx) * 4 // 5,
                n_test=len(train_idx) // 5,
            )

            if not selection.deployable:
                raise RuntimeError(f"Model selection rejected deployment: {selection.rejection_reason}")

            chosen_trial = runner.study_.trials[selection.simplest_eligible_trial_id]
            chosen_model = selection.simplest_eligible_model_name
            chosen_strategy = ImbalanceStrategy(chosen_trial.params.get("imbalance_strategy", "none"))

            # Fit final pipeline on Train Set
            champion_pipe = build_pipeline_assembler(
                estimator=build_raw_estimator(chosen_model, random_state=self.master_seed),
                model_name=chosen_model,
                categorical_low_cardinality=low_card,
                categorical_high_cardinality=high_card,
                continuous_columns=num_cols,
                imbalance_strategy=chosen_strategy,
                random_state=self.master_seed,
            )
            champion_pipe.fit(X_train, y_train)

            # Calibrate on Calibration Set
            cal_positives = int(np.sum(y_cal == 1))
            calibrator = PipelineProbabilityCalibrator(
                base_estimator=champion_pipe,
                pos_threshold_for_isotonic=5000,
                cv="prefit",
            )
            calibrator.fit(X_cal, y_cal)

            # Calculate Optimal Decision Threshold (Closed-form)
            threshold = calculate_closed_form_threshold(
                c_fp=self.cost_matrix["c_fp"],
                c_fn=self.cost_matrix["c_fn"],
                c_tp=self.cost_matrix.get("c_tp", 0.0),
                c_tn=self.cost_matrix.get("c_tn", 0.0),
            )

            # Assemble Final Thresholded Classifier
            final_clf = ThresholdedClassifier(estimator=calibrator, threshold=threshold)

            # Optional Prior Shift Adjustment
            if self.deployment_prevalence and abs(self.deployment_prevalence - prevalence) > 1e-4:
                final_clf = PriorShiftAdjustedClassifier(
                    base_estimator=final_clf,
                    prevalence_cal=prevalence,
                    prevalence_deploy=self.deployment_prevalence,
                    threshold=threshold,
                )

            # Unbiased Holdout Evaluation
            holdout_probs = calibrator.predict_proba(X_holdout)[:, 1]
            holdout_metrics = {
                "average_precision": calculate_average_precision(y_holdout, holdout_probs),
                "brier_skill_score": brier_skill_score(y_holdout, holdout_probs),
                "adaptive_ece": adaptive_expected_calibration_error(y_holdout, holdout_probs),
                "precision_at_top_k": precision_at_top_k(y_holdout, holdout_probs, k=min(50, len(y_holdout))),
            }

            self.metadata_.update({
                "model_name": chosen_model,
                "imbalance_strategy": chosen_strategy.value,
                "calibration_method": calibrator.resolved_method_,
                "operating_threshold": threshold,
                "metrics": holdout_metrics,
            })

            self.fitted_artifact_ = final_clf
            return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Inference scoring with fixed decision threshold."""
        if self.fitted_artifact_ is None:
            raise RuntimeError("Pipeline must be fitted before predict.")
        return self.fitted_artifact_.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated probability estimation."""
        if self.fitted_artifact_ is None:
            raise RuntimeError("Pipeline must be fitted before predict_proba.")
        return self.fitted_artifact_.predict_proba(X)

    def export(self, output_path: str = "artifacts/model.skops") -> Path:
        """Strips samplers and serializes the frozen pipeline artifact and model card."""
        artifact_file = export_pipeline_artifact(self.fitted_artifact_, output_path)
        self.metadata_["artifact_path"] = str(artifact_file)
        generate_enterprise_model_card(self.metadata_, output_path="MODEL_CARD.md")
        return artifact_file
