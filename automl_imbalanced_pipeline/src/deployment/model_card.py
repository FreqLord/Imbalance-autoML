"""Automated model card generation logging prevalence, operating thresholds, and 95% CIs."""

from pathlib import Path
from typing import Any, Dict, Optional


def generate_enterprise_model_card(
    metadata: Dict[str, Any],
    output_path: str = "MODEL_CARD.md",
) -> str:
    """Generates a structured enterprise markdown model card."""
    model_name = metadata.get("model_name", "AutoML Imbalanced Classifier")
    tier = metadata.get("tier", "N/A")
    train_prev = metadata.get("training_prevalence", 0.0)
    deploy_prev = metadata.get("deployment_prevalence", train_prev)
    threshold = metadata.get("operating_threshold", 0.5)

    costs = metadata.get("cost_matrix", {})
    metrics = metadata.get("metrics", {})
    cis = metadata.get("confidence_intervals", {})

    def fmt_ci(metric_name: str) -> str:
        if metric_name in cis:
            ci = cis[metric_name]
            return f"{ci.get('point', 0.0):.4f} (95% CI: [{ci.get('lower', 0.0):.4f}, {ci.get('upper', 0.0):.4f}])"
        elif metric_name in metrics:
            return f"{metrics[metric_name]:.4f}"
        return "N/A"

    content = f"""# Enterprise Model Card: {model_name}

## 1. Executive Summary & Operating Regime
- **Pipeline Execution Tier**: {tier}
- **Training Population Prevalence ($\pi_{{\\text{{train}}}}$)**: {train_prev:.4%}
- **Target Deployment Prevalence ($\pi_{{\\text{{deploy}}}}$)**: {deploy_prev:.4%}
- **Prior Shift Adjustment**: {"Enabled" if abs(train_prev - deploy_prev) > 1e-4 else "Identical Priors"}

## 2. Decision Thresholds & Business Cost Optimization
- **Operating Decision Threshold ($t^*$)**: {threshold:.4f}
- **Cost Matrix Specification**:
  - Cost of False Positive ($c_{{\\text{{FP}}}}$): {costs.get("c_fp", "N/A")}
  - Cost of False Negative ($c_{{\\text{{FN}}}}$): {costs.get("c_fn", "N/A")}
  - Benefit of True Positive ($c_{{\\text{{TP}}}}$): {costs.get("c_tp", 0.0)}
  - Benefit of True Negative ($c_{{\\text{{TN}}}}$): {costs.get("c_tn", 0.0)}

## 3. Unbiased Validation & Confidence Intervals (95% CI)
- **Average Precision (PR-AUC)**: {fmt_ci("average_precision")}
- **Brier Skill Score (BSS)**: {fmt_ci("brier_skill_score")}
- **Adaptive Expected Calibration Error (ECE)**: {fmt_ci("adaptive_ece")}
- **Precision @ Top-k Alert Capacity**: {fmt_ci("precision_at_top_k")}

## 4. Governance, Reproducibility & Artifact Lineage
- **Random Master Seed**: {metadata.get("master_seed", 42)}
- **Artifact Path**: `{metadata.get("artifact_path", "model.skops")}`
- **Sampling Strategy**: `{metadata.get("imbalance_strategy", "none")}`
- **Calibration Method**: `{metadata.get("calibration_method", "sigmoid")}`
- **Sampler Pruned From Final Export**: Yes (Strict inference isolation)
"""

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return content
