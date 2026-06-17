"""
Configuration parameters for the Telco Churn Prediction project.
"""

from pathlib import Path
from typing import Any, List, Dict

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure directories exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Data configuration
DATA_PATH = RAW_DATA_DIR / "telco_customer_churn.csv"

# Feature configuration
NUMERICAL_COLUMNS: List[str] = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_COLUMNS: List[str] = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

IDENTIFIER_COLUMN: str = "customerID"
TARGET_COLUMN: str = "Churn"

# Model configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Model parameters
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "logistic_regression": {"random_state": RANDOM_STATE, "max_iter": 1000},
    "random_forest": {"random_state": RANDOM_STATE, "n_estimators": 100},
    "gradient_boosting": {"random_state": RANDOM_STATE, "n_estimators": 100},
}

# Balanced variants — class_weight='balanced' to address low recall from class imbalance
# GradientBoostingClassifier does not accept class_weight;
# handled via sample_weight in train.py
MODEL_CONFIGS_BALANCED: Dict[str, Dict[str, Any]] = {
    "logistic_regression_balanced": {
        "random_state": RANDOM_STATE,
        "max_iter": 1000,
        "class_weight": "balanced",
    },
    "random_forest_balanced": {
        "random_state": RANDOM_STATE,
        "n_estimators": 100,
        "class_weight": "balanced",
    },
    "gradient_boosting_balanced": {"random_state": RANDOM_STATE, "n_estimators": 100},
}

# GridSearch parameter grids for hyperparameter tuning (used in train.py)
PARAM_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "gradient_boosting": {
        "n_estimators": [100, 200],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
    },
    "logistic_regression": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "penalty": ["l2"],
        "class_weight": [None, "balanced"],
    },
    "random_forest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "class_weight": [None, "balanced"],
    },
}

# File names
TRAIN_DATA_FILE = PROCESSED_DATA_DIR / "train.parquet"
TEST_DATA_FILE = PROCESSED_DATA_DIR / "test.parquet"
TRAIN_LABELS_FILE = PROCESSED_DATA_DIR / "train_labels.parquet"
TEST_LABELS_FILE = PROCESSED_DATA_DIR / "test_labels.parquet"
SCALER_FILE = PROCESSED_DATA_DIR / "scaler.pkl"
ENCODER_FILE = PROCESSED_DATA_DIR / "encoder.pkl"
BEST_MODEL_FILE = MODELS_DIR / "best_model.pkl"
RESULTS_FILE = MODELS_DIR / "model_comparison_results.csv"

# Engineered feature files
TRAIN_ENGINEERED_FILE = PROCESSED_DATA_DIR / "train_engineered.parquet"
TEST_ENGINEERED_FILE = PROCESSED_DATA_DIR / "test_engineered.parquet"
FEATURE_ENGINEERING_OBJECTS_FILE = (
    PROCESSED_DATA_DIR / "feature_engineering_objects.pkl"
)
