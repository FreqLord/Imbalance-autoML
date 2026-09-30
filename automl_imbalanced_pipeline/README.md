# AutoML Imbalanced Pipeline

A production-grade AutoML framework designed specifically for extreme class imbalance, temporal dependency, label-arrival delays, and asymmetric business misclassification costs.

## Architecture

```
automl_imbalanced_pipeline/
├── configs/
│   ├── search_spaces.yaml          # Optuna hyperparameter bounds per model family
│   ├── model_hierarchy.yaml        # Complexity proxy (Dummy < Logistic < RF < GBMs)
│   ├── cost_matrices.yaml          # Business costs (c_FP, c_FN, c_TP, c_TN)
│   └── feature_schema.yaml         # Group keys, temporal columns, categorical/continuous flags
├── src/
│   ├── __init__.py
│   ├── orchestrator.py             # Main entrypoint executing the end-to-end DAG
│   ├── core/
│   │   ├── tier_router.py          # Gateway logic: Total Positives vs Tier 1/2/3 routing
│   │   └── seed_manager.py         # numpy.random.SeedSequence child seed derivation
│   ├── data/
│   │   ├── splitters.py            # StratifiedGroupKFold and Rolling-Origin Forward Split
│   │   └── embargo.py              # Label-delay embargo gap injection for temporal data
│   ├── preprocessing/
│   │   ├── encoders.py             # Internal cross-fitted target encoding, ordinal/OHE
│   │   ├── imputers.py             # Explicit 'Missing' categorical imputer, continuous iterative imputer
│   │   ├── scalers.py              # Conditional RobustScaler (skips for trees unless oversampling)
│   │   └── pipeline_assembler.py   # imblearn.pipeline.Pipeline strict sequence builder
│   ├── sampling/
│   │   ├── strategies.py           # Mutually exclusive selector (None / Weights / Oversample / Undersample)
│   │   └── smote_adapters.py       # SMOTENC/SMOTEN wrappers, dynamic categorical_features index recomputation
│   ├── models/
│   │   ├── pool.py                 # Candidate initializers (Dummy, Logistic, RF, LightGBM, CatBoost, XGBoost)
│   │   └── early_stopping.py       # Custom GBM wrapper: carves un-resampled/embargoed inner split before transforms
│   ├── optimization/
│   │   ├── optuna_runner.py        # 5x1 Bayesian search execution and pruning
│   │   └── selection.py            # Champion gates: Pooled AP, Nadeau-Bengio variance, Bootstrap SE, 1-SE rule
│   ├── postprocessing/
│   │   ├── calibration.py          # Platt (<5k pos) vs Isotonic (>=5k pos), CalibratedClassifierCV logic
│   │   ├── thresholding.py         # Closed-form cost matrix math and TunedThresholdClassifierCV (F1/MCC) cross-fitting
│   │   └── prior_shift.py          # Odds-reweighting wrapper based on calibration prevalence vs deployment prevalence
│   ├── evaluation/
│   │   ├── metrics.py              # Brier Skill Score, ECE (adaptive binning), Precision @ top-k budget
│   │   ├── bootstrap.py            # Statistical confidence intervals for holdout/OOF reporting
│   │   └── explainers.py           # SHAP execution (uncalibrated model, strict holdout discipline)
│   └── deployment/
│       ├── serialization.py        # skops/MLflow exporter, dynamically drops samplers from the pipeline before save
│       └── model_card.py           # Generates markdown artifact logging prevalence, thresholds, and CIs
├── tests/
│   ├── test_leakage.py             # Asserts samplers/scalers never touch validation/holdout folds
│   ├── test_tier_logic.py          # Validates correct Nested CV vs Static split branching on low positives
│   └── test_metrics.py             # Verifies custom pooled AP and closed-form threshold math
├── scripts/
│   ├── train.py                    # CLI to trigger the pipeline (loads configs, runs orchestrator)
│   └── predict.py                  # CLI to load the frozen artifact and score new data
├── pyproject.toml                  # Explicit library pinning (scikit-learn, imbalanced-learn, optuna, skops, catboost)
└── README.md
```

## Quickstart

### Installation
```bash
pip install -e .
```

### Run Pipeline Training
```bash
python scripts/train.py --data path/to/dataset.csv --target target
```

### Run Batch Prediction
```bash
python scripts/predict.py --model artifacts/model.skops --data path/to/test.csv --output predictions.csv
```
