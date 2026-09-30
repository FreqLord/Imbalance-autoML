"""Generates markdown model card logging prevalence, optimal thresholds, and confidence intervals."""

from pathlib import Path
from typing import Any, Dict


def generate_model_card(metadata: Dict[str, Any], output_path: str = "MODEL_CARD.md") -> str:
    """Generates markdown model documentation summarizing validation results and operating constraints."""
    content = f"""# Model Card: {metadata.get('model_name', 'AutoML Imbalanced Classifier')}

## 1. Operating Regime & Prevalence
- **Training Prevalence**: {metadata.get('training_prevalence', 'N/A')}
- **Target Deployment Prevalence**: {metadata.get('deployment_prevalence', 'N/A')}
- **Pipeline Tier**: {metadata.get('tier', 'N/A')}

## 2. Decision Thresholds & Cost Matrix
- **Operating Threshold (t*)**: {metadata.get('operating_threshold', 'N/A')}
- **Cost Parameters**:
  - Cost of FP: {metadata.get('c_fp', 'N/A')}
  - Cost of FN: {metadata.get('c_fn', 'N/A')}

## 3. Validation Performance & Confidence Intervals (95% CI)
- **Average Precision (PR-AUC)**: {metadata.get('ap_score', 'N/A')} (CI: {metadata.get('ap_ci', 'N/A')})
- **Brier Skill Score (BSS)**: {metadata.get('bss_score', 'N/A')} (CI: {metadata.get('bss_ci', 'N/A')})
- **Adaptive ECE**: {metadata.get('ece_score', 'N/A')}

## 4. Model Lineage & Serialization
- **Artifact Type**: {metadata.get('format', 'skops')}
- **Random Seed**: {metadata.get('seed', 42)}
"""
    Path(output_path).write_text(content, encoding="utf-8")
    return content
