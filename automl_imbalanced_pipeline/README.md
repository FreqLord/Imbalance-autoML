# AutoML Imbalanced Pipeline

An enterprise-grade, leakage-free automated model selection and deployment framework designed specifically for datasets with **extreme class imbalance**, **temporal dependencies**, **label-arrival delays**, and **asymmetric business misclassification costs**.

---

## Table of Contents
- [1. Executive Overview](#1-executive-overview)
- [2. The Positives Gateway Architecture](#2-the-positives-gateway-architecture)
- [3. Core Mathematical Formulations & Invariants](#3-core-mathematical-formulations--invariants)
  - [3.1 Primary Metric: Average Precision & Baseline Grounding](#31-primary-metric-average-precision--baseline-grounding)
  - [3.2 Preprocessing Order & SMOTENC Invariants](#32-preprocessing-order--smotenc-invariants)
  - [3.3 Honest PR-AUC Early Stopping](#33-honest-pr-auc-early-stopping)
  - [3.4 Nadeau-Bengio 1-SE Rule & Beat-Baseline Gate](#34-nadeau-bengio-1-se-rule--beat-baseline-gate)
  - [3.5 Calibration: Platt Scaling vs. Isotonic Cutoff](#35-calibration-platt-scaling-vs-isotonic-cutoff)
  - [3.6 Closed-Form Bayes Optimal Thresholding](#36-closed-form-bayes-optimal-thresholding)
  - [3.7 Bayes Odds-Reweighting for Prior Shift](#37-bayes-odds-reweighting-for-prior-shift)
  - [3.8 Unbiased Holdout Metrics & Adaptive ECE](#38-unbiased-holdout-metrics--adaptive-ece)
- [4. Repository Structure](#4-repository-structure)
- [5. Configuration Guide](#5-configuration-guide)
- [6. Installation & Environment Setup](#6-installation--environment-setup)
- [7. CLI Usage](#7-cli-usage)
  - [7.1 Training Pipeline](#71-training-pipeline)
  - [7.2 Batch Inference](#72-batch-inference)
- [8. Python API Usage](#8-python-api-usage)
- [9. Testing Suite](#9-testing-suite)
- [10. Governance & Model Cards](#10-governance--model-cards)

---

## 1. Executive Overview

Standard machine learning pipelines fail catastrophically under extreme class imbalance (e.g., fraud detection, rare disease screening, ad conversion, churn prediction) due to five systemic failure modes:
1. **Data Leakage in Resampling:** Applying SMOTE, scaling, or target encoding across an entire dataset prior to splitting artificially inflates cross-validation performance.
2. **Metric Conflation:** Optimizing for Accuracy or ROC-AUC creates a false sense of security. ROC-AUC is visually flattered by a large true-negative population, whereas logloss is severely distorted by synthetic oversampling and class weights.
3. **Partition Starvation:** Arbitrarily applying a static 80/20 train/test split on a dataset with only 80 positive instances leaves the test partition with fewer than 16 positives, making variance explode and rendering threshold selection arbitrary.
4. **Uncalibrated Probability Shift:** Thresholding raw model outputs or applying non-parametric Isotonic regression on small sample sizes causes severe probability collapse.
5. **Production Incompatibility:** Serializing pipelines containing active resamplers corrupts real-time inference latency and predictions.

**AutoML Imbalanced Pipeline** resolves these challenges by introducing mathematically verified component ordering, dynamic sample-size routing via the **Positives Gateway**, and complete inference encapsulation.

---

## 2. The Positives Gateway Architecture

The pipeline calculates the expected number of rare instances per partition:
$$\text{Positives}_{\text{partition}} = N_{\text{pos}} \times f_{\text{partition}}$$

Based on absolute positive counts and partition feasibility, execution routes dynamically into one of three tiers:

```
                               Raw Data & Feature Setup
                       (Group keys & time columns declared upfront)
                                          │
                                          ▼
                             Positives Gateway Partition
                  (Total_Positives * Partition_Fraction >= Threshold)
                                          │
          ┌───────────────────────────────┼───────────────────────────────┐
          │ < 30 Positives                │ 30 - 499 Positives            │ >= 500 Positives
          ▼                               ▼                               ▼
 ┌──────────────────┐           ┌───────────────────┐           ┌───────────────────┐
 │      TIER 1      │           │      TIER 2       │           │      TIER 3       │
 │ Extreme Scarcity │           │  Low-Count Guard  │           │    Static Path    │
 └────────┬─────────┘           └─────────┬─────────┘           └─────────┬─────────┘
          │                               │                               │
 • Refuse AutoML search         • Nested Cross-Validation       • Static Split:
 • Allowed pool:                • Single global Optuna search     Train / Cal / Holdout
   Dummy & Logistic only          over candidate pool           • 5x1 Search -> 5x3 Re-eval
 • Top-k capacity alert budget  • CalibratedClassifierCV          (or Rolling-Origin CV)
   (No prob thresholds)           (cv=k, ensemble=False) on OOF • Nadeau-Bengio 1-SE Gate
                                • Final refit on all data +     • Beat-Baseline Gate
                                  OOF calibrator (mismatch doc) • Platt (<5k pos) vs
                                • Paired Bootstrap 1-SE Rule      Isotonic (>=5k pos)
                                • Abandon F1 threshold tuning   • Closed-Form Cost / TunedThresh
                                                                • Pure Untouched Holdout Eval
```

### Tier Comparison Table

| Feature | Tier 1: Extreme Scarcity | Tier 2: Low-Count Guard | Tier 3: Static Path |
| :--- | :--- | :--- | :--- |
| **Positive Count ($N_{\text{pos}}$)** | $N_{\text{pos}} < 30$ | $30 \le N_{\text{pos}} < 500$ (or starvation) | $N_{\text{pos}} \ge 500$ (well-stocked) |
| **Partitioning Strategy** | Static Baseline Check | Nested Cross-Validation ($5 \times 5$) | Static Train ($70\%$) / Cal ($15\%$) / Holdout ($15\%$) |
| **Temporal Data** | Chronological slice | Rolling-Origin with Embargo Gaps | 3-way Chronological Split with Embargo Gaps |
| **Model Candidate Pool** | Dummy, Logistic Regression | Dummy, Logistic, Random Forest, LightGBM | Full Suite (Dummy, Logistic, RF, LightGBM, CatBoost, XGBoost) |
| **Probability Calibration** | None (Uncalibrated) | `CalibratedClassifierCV(cv=k, ensemble=False)` on OOF | Platt (<5k cal pos) or Isotonic ($\ge 5\text{k}$ cal pos) on Cal |
| **Threshold Decision** | Top-$k$ Alert Capacity Budget | Closed-Form Cost Matrix or Default (0.5) | Closed-Form Cost Matrix ($t^*$) or `TunedThresholdClassifierCV` |
| **Holdout Integrity** | N/A | Evaluated on Out-of-Fold (OOF) | Evaluated on Strictly Untouched Holdout |

---

## 3. Core Mathematical Formulations & Invariants

### 3.1 Primary Metric: Average Precision & Baseline Grounding
The pipeline optimizes for **Average Precision (AP / PR-AUC)**:
$$\text{AP} = \sum_{n} (R_n - R_{n-1}) P_n$$

**Prevalence-Baseline Grounding:** For any scored subset (fold OOF, calibration partition, or final holdout), the random-guess baseline performance equals the minority prevalence:
$$\text{AP}_{\text{baseline}} = \pi = \frac{N_{\text{pos}}}{N_{\text{total}}}$$
All candidate configurations must demonstrate statistically significant skill above $\pi$ to be considered for production deployment.

### 3.2 Preprocessing Order & SMOTENC Invariants
Transformations are encapsulated inside an `imblearn.pipeline.Pipeline` with strict component ordering:
```
Raw Features 
   │
   ├─► 1. Categorical Missing Imputation ('__MISSING__' token)
   ├─► 2. Continuous Iterative Imputation (MICE / IterativeImputer)
   │
   ├─► 3. Target Encoding (High-cardinality categoricals; internal cross-fitting; SKIPPED for CatBoost)
   │
   ├─► 4. Ordinal Encoding (Low-cardinality categoricals prepared for SMOTENC)
   │
   ├─► 5. Conditional RobustScaler (Continuous features; MANDATORY for SMOTE/SMOTENC; bypassed for trees if no oversampling)
   │
   ├─► 6. Dynamic Column Index Recomputation (Maps post-transform column indices to categorical indices)
   │
   ├─► 7. Resampling Gate (SMOTENC / SMOTE / RandomUnderSampler; DISALLOWED if CatBoost + high-cardinality)
   │
   ├─► 8. Conditional One-Hot Encoding (Post-resampling for Logistic/RF; BYPASSED for LightGBM/CatBoost)
   │
   └─► 9. Estimator
```

#### Mathematical Guards:
1. **SMOTENC NaN Crash Guard:** SMOTENC computes Euclidean distances along continuous dimensions and overlap metrics along categoricals. Any NaN in either partition triggers fatal runtime failure. Imputation must precede resampling.
2. **Scale Distortion in SMOTENC:** The distance between samples $x_i$ and $x_j$ in SMOTENC is:
   $$d(x_i, x_j) = \sqrt{\sum_{c \in \text{continuous}} \left(\frac{x_{i,c} - x_{j,c}}{\sigma_c}\right)^2 + m \cdot \text{mismatches}}$$
   Without robust scaling, unscaled continuous features artificially dominate nearest-neighbor graphs. Hence, continuous scaling is **mandatory** whenever oversampling is enabled, regardless of whether downstream models are tree-based.
3. **CatBoost High-Cardinality Trap:** CatBoost computes native ordered target statistics directly from raw categoricals. Performing SMOTENC on hundreds of levels requires synthetic mode-voting, which destroys feature correlation. Oversampling is disallowed for CatBoost when high-cardinality categoricals remain unencoded.

### 3.3 Honest PR-AUC Early Stopping
Standard early stopping evaluates logloss on resampled data, leading to severe overfitting on synthetic artifacts. The custom wrapper `EarlyStoppingGBM`:
1. Carves an un-resampled, embargoed inner validation split ($15\%$) *before* transformations and resampling are fitted.
2. Monitors **Average Precision (PR-AUC)** at every boosting iteration.

### 3.4 Nadeau-Bengio 1-SE Rule & Beat-Baseline Gate
To select the simplest model that does not sacrifice significant predictive power:
$$\text{Threshold} = \text{Score}_{\text{best}} - \text{SE}_{\text{best}}$$

In standard cross-validation, training folds overlap, violating the independence assumption and causing naive sample variance to underestimate true variance. For Tier 3, we compute the **Nadeau-Bengio corrected variance**:
$$\sigma^2_{\text{corrected}} = S^2 \left(\frac{1}{K} + \frac{n_{\text{test}}}{n_{\text{train}}}\right), \quad \text{SE} = \sqrt{\sigma^2_{\text{corrected}}}$$
- **Beat-Baseline Gate:** If the Dummy Classifier or simple Logistic Regression falls within the 1-SE window of a complex gradient booster, the simpler model is selected. If no model outperforms the uninformative prevalence baseline $\pi$, deployment is rejected.

### 3.5 Calibration: Platt Scaling vs. Isotonic Cutoff
- **Platt Scaling (Sigmoid):** Parametric logistic calibration ($P(y=1|f) = \frac{1}{1 + \exp(Af + B)}$). Default method for small to moderate minority counts ($< 5,000$ calibration positives).
- **Isotonic Regression:** Monotonic non-parametric step-function. Restricted to large partitions ($\ge 5,000$ calibration positives) where bin sample densities prevent probability collapse.

### 3.6 Closed-Form Bayes Optimal Thresholding
Given business costs $c_{\text{FP}}, c_{\text{FN}}, c_{\text{TP}}, c_{\text{TN}}$ with preconditions $c_{\text{FP}} > c_{\text{TN}}$ and $c_{\text{FN}} > c_{\text{TP}}$, the expected risk minimization yields the closed-form threshold:
$$t^* = \frac{c_{\text{FP}} - c_{\text{TN}}}{(c_{\text{FP}} - c_{\text{TN}}) + (c_{\text{FN}} - c_{\text{TP}})}$$

### 3.7 Bayes Odds-Reweighting for Prior Shift
When target deployment prevalence $\pi_{\text{deploy}}$ diverges from calibration prevalence $\pi_{\text{cal}}$:
$$\text{Odds}_{\text{cal}} = \frac{p_{\text{cal}}}{1 - p_{\text{cal}}}$$
$$\text{Odds}_{\text{deploy}} = \text{Odds}_{\text{cal}} \times \left(\frac{\pi_{\text{deploy}}}{1 - \pi_{\text{deploy}}}\right) \left(\frac{1 - \pi_{\text{cal}}}{\pi_{\text{cal}}}\right)$$
$$p_{\text{deploy}} = \frac{\text{Odds}_{\text{deploy}}}{1 + \text{Odds}_{\text{deploy}}}$$

### 3.8 Unbiased Holdout Metrics & Adaptive ECE
Evaluated strictly on the untouched Holdout set (Tier 3) or pooled outer OOF predictions (Tier 2):
- **Brier Skill Score (BSS):**
  $$\text{BSS} = 1 - \frac{\text{BS}_{\text{model}}}{\pi(1 - \pi)}$$
- **Adaptive ECE (Equal-Frequency Binning):** Quantile-based bin edges prevent empty bin anomalies in sparse extreme-probability regimes:
  $$\text{ECE}_{\text{adaptive}} = \sum_{b=1}^{B} \frac{|B_b|}{N} \left| \text{acc}(B_b) - \text{conf}(B_b) \right|$$
- **Precision @ Top-$k$ Alert Budget:** Evaluates precision under operational review constraints.
- **Empirical 95% Bootstrap Confidence Intervals:** 1,000 stratified resamples.

---

## 4. Repository Structure

```
automl_imbalanced_pipeline/
├── configs/
│   ├── search_spaces.yaml          # Optuna hyperparameter bounds per model family
│   ├── model_hierarchy.yaml        # Model complexity rankings and tier eligibility
│   ├── cost_matrices.yaml          # Business loss costs (c_FP, c_FN, c_TP, c_TN)
│   └── feature_schema.yaml         # Group keys, temporal columns, categorical & continuous flags
├── src/
│   ├── __init__.py
│   ├── orchestrator.py             # End-to-end master DAG orchestrator
│   ├── core/
│   │   ├── tier_router.py          # Positives Gateway routing logic (Tiers 1, 2, 3)
│   │   └── seed_manager.py         # SeedSequence hierarchical child derivation
│   ├── data/
│   │   ├── splitters.py            # StratifiedGroupKFold, RollingOriginSplitter, StaticPartitioner
│   │   └── embargo.py              # Label-delay embargo gap injection
│   ├── preprocessing/
│   │   ├── encoders.py             # Cross-fitted TargetEncoder, OrdinalEncoder, Post-OHE
│   │   ├── imputers.py             # '__MISSING__' categorical & MICE continuous imputers
│   │   ├── scalers.py              # Conditional RobustScaler (SMOTE/tree-aware)
│   │   └── pipeline_assembler.py   # imblearn.pipeline.Pipeline strict sequence builder
│   ├── sampling/
│   │   ├── strategies.py           # Mutually exclusive resampling dispatcher
│   │   └── smote_adapters.py       # DynamicIndexSMOTENC & CatBoost high-cardinality guards
│   ├── models/
│   │   ├── pool.py                 # Candidate initializers (Dummy, Logistic, RF, GBMs)
│   │   └── early_stopping.py       # Custom GBM wrapper: PR-AUC early stopping on unresampled val
│   ├── optimization/
│   │   ├── optuna_runner.py        # 5x1 Bayesian search with MedianPruner
│   │   └── selection.py            # Champion selection: Nadeau-Bengio 1-SE & Beat-Baseline
│   ├── postprocessing/
│   │   ├── calibration.py          # Platt (<5k pos) vs Isotonic (>=5k pos) calibrators
│   │   ├── thresholding.py         # Closed-form cost matrix threshold & TunedThresholdClassifierCV
│   │   └── prior_shift.py          # Odds-reweighting for deployment prevalence divergence
│   ├── evaluation/
│   │   ├── metrics.py              # AP, Brier Skill Score, Adaptive ECE, Precision @ top-k
│   │   ├── bootstrap.py            # 95% Bootstrap percentile confidence intervals
│   │   └── explainers.py           # SHAP execution on uncalibrated models with holdout isolation
│   └── deployment/
│       ├── serialization.py        # Strips BaseSamplers from pipeline, skops/MLflow exporter
│       └── model_card.py           # Automated enterprise markdown model card generator
├── tests/
│   ├── test_tier_logic.py          # Gateway, SeedSequence, splitters, & embargo tests
│   ├── test_leakage.py             # Imputation, scaling invariants, & SMOTE leakage tests
│   ├── test_metrics.py             # Pool, early stopping, Nadeau-Bengio, & 1-SE tests
│   └── test_postprocessing.py      # Calibration, thresholding math, prior shift, & serialization tests
├── scripts/
│   ├── train.py                    # CLI command to execute the pipeline
│   └── predict.py                  # CLI command to score data with the frozen artifact
├── pyproject.toml                  # Explicit library dependencies and build configuration
└── README.md
```

---

## 5. Configuration Guide

Configurations are stored in `configs/` and loaded dynamically during training.

### `configs/feature_schema.yaml`
```yaml
group_column: "customer_id"             # Key for GroupKFold or group-aware rolling origin
time_column: "transaction_timestamp"    # Timestamp for chronological ordering
label_column: "is_fraud"                # Binary target (0/1)
embargo_horizon_days: 7                 # Label-arrival delay gap (in days)
categorical_features: ["channel", "device_type"]
continuous_features: ["amount", "velocity_24h"]
high_cardinality_features: ["merchant_id", "zip_code"]
```

### `configs/cost_matrices.yaml`
```yaml
default:
  c_fp: 1.0     # Cost of False Alarm
  c_fn: 10.0    # Cost of Missed Fraud
  c_tp: 0.0     # Value of Catching Fraud
  c_tn: 0.0     # Value of Correct Non-Fraud

fraud_detection:
  c_fp: 5.0
  c_fn: 100.0
  c_tp: -10.0
  c_tn: 0.0
```

---

## 6. Installation & Environment Setup

```bash
# Clone repository
git clone https://github.com/FreqLord/Imbalance-autoML.git
cd Imbalance-autoML/automl_imbalanced_pipeline

# Create dedicated virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

---

## 7. CLI Usage

### 7.1 Training Pipeline
Execute the end-to-end DAG using `scripts/train.py`:
```bash
python scripts/train.py \
  --data data/credit_card_fraud.csv \
  --target is_fraud \
  --schema configs/feature_schema.yaml \
  --costs configs/cost_matrices.yaml \
  --output artifacts/champion_model.skops \
  --trials 40
```

**Artifacts Generated:**
- `artifacts/champion_model.skops`: The frozen inference pipeline (samplers stripped).
- `MODEL_CARD.md`: Comprehensive governance card with prevalence, threshold, and 95% CIs.

### 7.2 Batch Inference
Generate calibrated probability estimates and thresholded predictions:
```bash
python scripts/predict.py \
  --model artifacts/champion_model.skops \
  --data data/new_transactions.csv \
  --output predictions.csv
```

---

## 8. Python API Usage

```python
import pandas as pd
from src.orchestrator import AutoMLOrchestrator

# 1. Load dataset
df = pd.read_csv("data/transactions.csv")
X = df.drop(columns=["target"])
y = df["target"]

# 2. Initialize orchestrator
orchestrator = AutoMLOrchestrator(
    master_seed=42,
    cost_matrix={"c_fp": 2.0, "c_fn": 18.0, "c_tp": 0.0, "c_tn": 0.0},
    deployment_prevalence=0.015,  # Optional target prevalence for odds-reweighting
    group_column="customer_id",
    time_column="timestamp",
    embargo_horizon="7D",
    n_trials=30,
)

# 3. Fit pipeline DAG (auto-routes through Tier 1, Tier 2, or Tier 3)
orchestrator.fit(X, y)

# 4. Generate predictions
probs = orchestrator.predict_proba(X)
preds = orchestrator.predict(X)

# 5. Export frozen production artifact and model card
artifact_path = orchestrator.export("artifacts/frozen_model.skops")
print(f"Artifact exported to: {artifact_path}")
```

---

## 9. Testing Suite

The repository includes a comprehensive unit and integration test suite verifying every component invariant:

```bash
# Run entire test suite
pytest -v

# Run with test coverage
pytest --cov=src tests/
```

### Covered Test Dimensions:
- **`tests/test_tier_logic.py`**: Validates Positives Gateway boundaries, partition starvation detection, `SeedSequence` child stream determinism, rolling-origin group isolation, and embargo pruning.
- **`tests/test_leakage.py`**: Verifies categorical and continuous imputation, conditional scaling bypass/mandate rules, and CatBoost high-cardinality oversampling guards.
- **`tests/test_metrics.py`**: Tests model pool complexity hierarchy, un-resampled inner validation carving, Nadeau-Bengio variance correction formulas, 1-SE model selection, and baseline rejection.
- **`tests/test_postprocessing.py`**: Validates Platt vs. Isotonic 5k cutoff switching, closed-form Bayes optimal threshold math, prior shift odds-reweighting, and dynamic sampler stripping.

---

## 10. Governance & Model Cards

Every pipeline run automatically produces an enterprise **Model Card** (`MODEL_CARD.md`) logging:
1. **Operating Regime**: Training vs. deployment prevalence, pipeline tier, and prior-shift configuration.
2. **Business Decision Thresholds**: Calculated Bayes-optimal threshold $t^*$ and the input cost matrix.
3. **Validation & Confidence Intervals**: 95% bootstrap confidence intervals for Average Precision, Brier Skill Score, and Adaptive ECE.
4. **Lineage**: Frozen random seeds, stripped sampler verification, and serialization metadata.
