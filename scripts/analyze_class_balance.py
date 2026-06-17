"""
Class balance analysis for the Telco Churn dataset.

Computes churn vs. non-churn counts and percentages, the imbalance ratio,
and basic dataset info.  Optionally saves a CSV of the class distribution.

Usage:
    python scripts/analyze_class_balance.py
    python scripts/analyze_class_balance.py --output results/class_balance.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import DATA_PATH, TARGET_COLUMN
from utils import load_data


def analyze_class_balance(output: str | None = None) -> pd.DataFrame:
    """Compute churn class balance and return the distribution as a DataFrame."""
    df = load_data(DATA_PATH)

    counts = df[TARGET_COLUMN].value_counts().to_dict()
    percentages = (df[TARGET_COLUMN].value_counts(normalize=True) * 100).to_dict()
    imbalance_ratio = counts["No"] / counts["Yes"]

    dist = pd.DataFrame({
        "class": ["No (non-churn)", "Yes (churn)"],
        "count": [int(counts["No"]), int(counts["Yes"])],
        "percentage": [round(percentages["No"], 2), round(percentages["Yes"], 2)],
    })

    print("\n=== Class Balance Analysis ===")
    print(dist.to_string(index=False))
    print(f"\nTotal Records:    {len(df):,}")
    print(f"Churn Rate:       {percentages['Yes']:.2f}%")
    print(f"Non-Churn Rate:   {percentages['No']:.2f}%")
    print(f"Imbalance Ratio:  {imbalance_ratio:.2f}:1 (No:Yes)")
    print(f"Missing in {TARGET_COLUMN}: {df[TARGET_COLUMN].isnull().sum()}")

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dist.to_csv(out_path, index=False)
        print(f"\nSaved to {out_path}")

    return dist


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Class balance analysis for the Telco Churn dataset")
    parser.add_argument("--output", type=str, default=None, help="Optional CSV output path")
    args = parser.parse_args()

    analyze_class_balance(output=args.output)
