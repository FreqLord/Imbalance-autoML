"""CLI command to trigger end-to-end training pipeline."""

import argparse
import sys
from pathlib import Path
import pandas as pd
import yaml

from src.orchestrator import AutoMLOrchestrator


def main():
    parser = argparse.ArgumentParser(description="Run AutoML Imbalanced Pipeline Training")
    parser.add_argument("--data", required=True, help="Path to input dataset (CSV/Parquet)")
    parser.add_argument("--target", default="target", help="Name of target label column")
    parser.add_argument("--config", default="configs/search_spaces.yaml", help="Path to configuration file")
    parser.add_argument("--output", default="artifacts/model.skops", help="Output model artifact path")
    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data) if args.data.endswith(".csv") else pd.read_parquet(args.data)
    X = df.drop(columns=[args.target])
    y = df[args.target]

    print(f"Initializing AutoML Orchestrator...")
    orchestrator = AutoMLOrchestrator()
    orchestrator.fit(X, y)
    print("Training completed successfully.")


if __name__ == "__main__":
    main()
