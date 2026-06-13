"""
Threshold analysis for the saved best_model.pkl.

Computes precision, recall, and F1 across a range of classification thresholds
and prints a summary table.  Optionally saves a CSV of the full sweep.

Usage:
    python scripts/threshold_analysis.py
    python scripts/threshold_analysis.py --output results/threshold_sweep.csv
    python scripts/threshold_analysis.py --step 0.02 --output results/threshold_sweep.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import BEST_MODEL_FILE, TEST_DATA_FILE, TEST_LABELS_FILE, IDENTIFIER_COLUMN
from utils import load_parquet, load_pickle


def run_threshold_analysis(step: float = 0.05, output: str | None = None) -> pd.DataFrame:
    """Compute precision/recall/F1 across thresholds and return as a DataFrame."""
    model  = load_pickle(BEST_MODEL_FILE)
    X_test = load_parquet(TEST_DATA_FILE).drop(columns=[IDENTIFIER_COLUMN])
    y_test = load_parquet(TEST_LABELS_FILE).squeeze()
    y_bin  = pd.Series(y_test == "Yes", dtype=int)

    proba = model.predict_proba(X_test)[:, 1]

    precision_pts, recall_pts, thresholds = precision_recall_curve(y_bin, proba)
    full = pd.DataFrame({
        "threshold": np.append(thresholds, 1.0),
        "precision": precision_pts,
        "recall":    recall_pts,
    })
    full["f1"] = (
        2 * full["precision"] * full["recall"]
        / (full["precision"] + full["recall"] + 1e-9)
    )

    # Sample at the requested step size
    sweep_points = np.arange(0.10, 0.91, step)
    rows = []
    for t in sweep_points:
        candidates = full[full["threshold"] >= t]
        if candidates.empty:
            continue
        row = candidates.iloc[0]
        rows.append({
            "threshold": round(t, 4),
            "precision": round(row["precision"], 4),
            "recall":    round(row["recall"],    4),
            "f1":        round(row["f1"],         4),
        })
    sweep_df = pd.DataFrame(rows)

    # Key operating points
    default_row = full[full["threshold"] >= 0.50].iloc[0]
    best_f1_row = full.loc[full["f1"].idxmax()]
    best_recall_70 = pd.DataFrame(full[full["recall"] >= 0.70]).sort_values(by="precision", ascending=False).iloc[0]

    print("\n=== Threshold Analysis ===")
    print(f"\nDefault (0.50):  precision={default_row['precision']:.3f}  recall={default_row['recall']:.3f}  f1={default_row['f1']:.3f}")
    print(f"Best F1:         threshold={best_f1_row['threshold']:.3f}  precision={best_f1_row['precision']:.3f}  recall={best_f1_row['recall']:.3f}  f1={best_f1_row['f1']:.3f}")
    print(f"Recall >= 0.70:  threshold={best_recall_70['threshold']:.3f}  precision={best_recall_70['precision']:.3f}  recall={best_recall_70['recall']:.3f}  f1={best_recall_70['f1']:.3f}")

    print(f"\nFull sweep (step={step}):")
    print(sweep_df.to_string(index=False))

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sweep_df.to_csv(out_path, index=False)
        print(f"\nSaved to {out_path}")

    return sweep_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Threshold sweep for best_model.pkl")
    parser.add_argument("--step",   type=float, default=0.05,  help="Threshold step size (default: 0.05)")
    parser.add_argument("--output", type=str,   default=None,  help="Optional CSV output path")
    args = parser.parse_args()

    run_threshold_analysis(step=args.step, output=args.output)
