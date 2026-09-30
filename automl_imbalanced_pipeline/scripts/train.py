"""CLI command to trigger end-to-end training pipeline."""

import argparse
from pathlib import Path
import pandas as pd
import yaml

from src.orchestrator import AutoMLOrchestrator


def main():
    parser = argparse.ArgumentParser(description="Run AutoML Imbalanced Pipeline Training")
    parser.add_argument("--data", required=True, help="Path to input dataset (CSV/Parquet)")
    parser.add_argument("--target", default="target", help="Name of target label column")
    parser.add_argument("--schema", default="configs/feature_schema.yaml", help="Path to feature schema configuration")
    parser.add_argument("--costs", default="configs/cost_matrices.yaml", help="Path to business cost matrix configuration")
    parser.add_argument("--output", default="artifacts/model.skops", help="Output model artifact path")
    parser.add_argument("--trials", type=int, default=30, help="Number of Optuna search trials")
    args = parser.parse_args()

    # Load configuration files if present
    schema_cfg = {}
    if Path(args.schema).exists():
        with open(args.schema, "r", encoding="utf-8") as f:
            schema_cfg = yaml.safe_load(f) or {}

    cost_cfg = {"c_fp": 1.0, "c_fn": 10.0, "c_tp": 0.0, "c_tn": 0.0}
    if Path(args.costs).exists():
        with open(args.costs, "r", encoding="utf-8") as f:
            costs_all = yaml.safe_load(f) or {}
            cost_cfg = costs_all.get("default", cost_cfg)

    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data) if args.data.endswith(".csv") else pd.read_parquet(args.data)
    target_col = schema_cfg.get("label_column", args.target)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"Initializing AutoML Orchestrator...")
    orchestrator = AutoMLOrchestrator(
        cost_matrix=cost_cfg,
        group_column=schema_cfg.get("group_column"),
        time_column=schema_cfg.get("time_column"),
        embargo_horizon=schema_cfg.get("embargo_horizon_days"),
        n_trials=args.trials,
    )
    orchestrator.fit(X, y)
    saved_path = orchestrator.export(args.output)
    print(f"Training completed successfully. Frozen artifact exported to: {saved_path}")
    print("Generated Model Card: MODEL_CARD.md")


if __name__ == "__main__":
    main()
