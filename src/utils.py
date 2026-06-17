"""
Utility functions for the Telco Churn Prediction project.
"""

import pandas as pd
import pickle
from pathlib import Path
from typing import Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_data(data_path: Path) -> pd.DataFrame:
    """Load dataset from CSV file."""
    logger.info(f"Loading data from {data_path}")
    return pd.read_csv(data_path)


def save_pickle(obj: Any, file_path: Path) -> None:
    """Save object to pickle file."""
    logger.info(f"Saving object to {file_path}")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(file_path: Path) -> Any:
    """Load object from pickle file."""
    logger.info(f"Loading object from {file_path}")
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_parquet(df: pd.DataFrame, file_path: Path) -> None:
    """Save DataFrame to parquet file."""
    logger.info(f"Saving DataFrame to {file_path}")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(file_path)


def load_parquet(file_path: Path) -> pd.DataFrame:
    """Load DataFrame from parquet file."""
    logger.info(f"Loading DataFrame from {file_path}")
    return pd.read_parquet(file_path)


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate model and return metrics."""
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
    )

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        # Get probability of positive class (Yes)
        classes = model.classes_
        if "Yes" in classes:
            yes_index = list(classes).index("Yes")
            y_proba = model.predict_proba(X_test)[:, yes_index]
        else:
            y_proba = (y_pred == "Yes").astype(int)
    else:
        y_proba = (y_pred == "Yes").astype(int)

    # Convert string labels to binary for metrics calculation
    y_test_binary = (y_test == "Yes").astype(int)
    y_pred_binary = (y_pred == "Yes").astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test_binary, y_pred_binary),
        "precision": precision_score(y_test_binary, y_pred_binary),
        "recall": recall_score(y_test_binary, y_pred_binary),
        "f1": f1_score(y_test_binary, y_pred_binary),
        "roc_auc": roc_auc_score(y_test_binary, y_proba),
    }

    return metrics
