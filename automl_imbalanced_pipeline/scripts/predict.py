"""CLI command to load frozen artifact and generate inferences on new data."""

import argparse
from pathlib import Path
import pandas as pd
import skops.io as sio


def main():
    parser = argparse.ArgumentParser(description="Run inference using frozen AutoML pipeline")
    parser.add_argument("--model", required=True, help="Path to frozen .skops artifact")
    parser.add_argument("--data", required=True, help="Path to scoring data (CSV/Parquet)")
    parser.add_argument("--output", default="predictions.csv", help="Output path for predictions")
    args = parser.parse_args()

    print(f"Loading frozen pipeline from {args.model}...")
    pipeline = sio.load(args.model, trusted=True)

    print(f"Scoring dataset from {args.data}...")
    df = pd.read_csv(args.data) if args.data.endswith(".csv") else pd.read_parquet(args.data)
    
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(df)[:, 1]
        df["predicted_probability"] = probabilities

    predictions = pipeline.predict(df)
    df["prediction"] = predictions

    df.to_csv(args.output, index=False)
    print(f"Predictions written to {args.output}.")


if __name__ == "__main__":
    main()
