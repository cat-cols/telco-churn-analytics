# Technical Implementation Guide

## Table of Contents
1. [Project Architecture](#project-architecture)
2. [Data Understanding & Preprocessing](#data-understanding--preprocessing)
3. [Feature Engineering Decisions](#feature-engineering-decisions)
4. [Model Selection Strategy](#model-selection-strategy)
5. [Implementation: Building Each Component](#implementation-building-each-component)
6. [Evaluation Framework](#evaluation-framework)
7. [Dataset Schema Validation](#dataset-schema-validation)
8. [Testing](#testing)
9. [Artifact Lifecycle Management](#artifact-lifecycle-management)
10. [Class Imbalance, Hyperparameter Tuning, and Threshold Analysis](#class-imbalance-hyperparameter-tuning-and-threshold-analysis)

---

## Project Architecture

### Why This Structure?

```
telco-churn-analytics/
├── data/
│   ├── raw/                    # Immutable source data
│   └── processed/              # Version-controlled transformations
├── src/                        # Production-ready scripts
│   ├── config.py              # Centralized configuration
│   ├── utils.py               # Reusable utilities
│   ├── preprocess.py          # Data pipeline
│   ├── train.py               # Model training
│   ├── predict.py             # Inference engine
│   └── run_pipeline.py        # Orchestration
├── notebooks/                  # Exploration & documentation
└── models/                     # Serialized artifacts
```

**Design Philosophy:**
- **Separation of concerns**: Notebooks for exploration, scripts for production
- **Reproducibility**: Every transformation is scripted, not manual
- **Configurability**: All parameters in `config.py`, not hardcoded

---

## Data Understanding & Preprocessing

### The Dataset

The IBM Telco Customer Churn dataset contains 7,043 customer records with 21 columns:

**Customer Demographics:**
- `customerID`: Unique identifier (preserved for predictions)
- `gender`: Male/Female
- `SeniorCitizen`: 0/1 indicator
- `Partner`: Yes/No (has partner)
- `Dependents`: Yes/No (has dependents)

**Service Details:**
- `tenure`: Months with company
- `PhoneService`, `MultipleLines`: Phone service flags
- `InternetService`: DSL, Fiber optic, or No
- `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`: Addon services

**Contract & Billing:**
- `Contract`: Month-to-month, One year, Two year
- `PaperlessBilling`: Yes/No
- `PaymentMethod`: Electronic check, Mailed check, Bank transfer, Credit card
- `MonthlyCharges`: Current monthly bill
- `TotalCharges`: Lifetime charges (STRING - critical data quality issue)

**Target:**
- `Churn`: Yes/No (whether customer left in the last month)

### The TotalCharges Data Quality Issue

```python
# Critical bug in the raw data
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
```

**Why this happens:**
- 11 customers have whitespace-only values in `TotalCharges`
- These are all brand new customers (tenure = 0)
- Pandas `to_numeric()` with `errors='coerce'` converts invalid parsing to NaN

### Decision Analysis: Handling the 11 Missing TotalCharges

**Root Cause Investigation:**
```python
# All 11 problematic rows have tenure = 0
whitespace_mask = df['TotalCharges'].str.strip() == ''
problematic = df[whitespace_mask]
print(problematic['tenure'].value_counts())
# Output: 0    11
```

This is an ETL bug: TotalCharges = tenure × MonthlyCharges, but for tenure=0, the system wrote whitespace instead of "0.00".

**Option Analysis:**

| Option | Approach | Pros | Cons | Recommendation |
|--------|----------|------|------|----------------|
| **A** | Drop 11 rows | Simple, clean dataset | Loses 0.16% of data; new customers may have different patterns | Acceptable |
| **B** | Impute with $0 | Retains all data; logically correct (tenure=0 → $0) | None significant | **Preferred** |
| **C** | Impute with tenure × MonthlyCharges | Data-driven estimation | For tenure=0, still gives $0 | Overly complex |
| **D** | Flag + impute | Preserves "new customer" signal | Adds complexity for marginal gain | Optional |

**Recommended Implementation (Option B):**
```python
# Convert TotalCharges from string to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Since all 11 missing have tenure=0, they SHOULD have $0 TotalCharges
# This is logically consistent: new customers haven't been charged yet
missing_count = df['TotalCharges'].isna().sum()
df['TotalCharges'] = df['TotalCharges'].fillna(0)
print(f"Imputed {missing_count} missing TotalCharges with $0 (all tenure=0 customers)")
```

**Rationale for Option B over Option A:**
1. **Business logic:** tenure=0 means new customer, so $0 total charges is correct
2. **No data loss:** Retains all 7,043 customers for analysis
3. **New customer patterns:** New customers may have unique churn patterns worth studying
4. **Simplicity:** More straightforward than dropping with explanation
5. **Production consistency:** New customers in production will also have TotalCharges=0 initially

**Note:** If implementing Option D (flagging), add:
```python
df['Is_New_Customer'] = (df['tenure'] == 0).astype(int)
# This captures the new customer signal explicitly
```

---

## Feature Engineering Decisions

### Numerical Features: Standardization

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[NUMERICAL_COLUMNS])
```

**Why StandardScaler?**
- **Algorithm requirement**: Logistic regression, gradient boosting, and neural networks perform better with normalized features
- **Interpretability**: All features on same scale (mean=0, std=1)
- **Convergence**: Faster training for gradient-based methods

**Why not MinMaxScaler?**
- StandardScaler handles outliers better (Telco data has some high-value customers)
- Preserves relative distances between points

### Categorical Features: One-Hot Encoding

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)
X_train_encoded = encoder.fit_transform(X_train[CATEGORICAL_COLUMNS])
```

**Why One-Hot Encoding?**
- **No ordinal assumption**: Contract types (Month-to-month vs Two-year) aren't ordered
- **Tree-based models**: Can split on specific categories
- **Linear models**: Creates binary features for coefficient interpretation

**Why sparse_output=False?**
- Pandas DataFrame output for easier debugging
- Memory not a concern (7K rows, ~40 features after encoding)

### Feature Selection: Domain-Driven

We keep all features except `customerID` (identifier only). Rationale:

1. **Business knowledge**: Telco industry knows these factors affect churn
2. **No curse of dimensionality**: ~40 features after encoding is manageable
3. **Correlation analysis**: No highly correlated feature pairs requiring removal

---

## Model Selection Strategy

### Why These Four Models?

| Model | Type | Strengths for Churn |
|-------|------|---------------------|
| **Logistic Regression** | Linear | Interpretable coefficients, fast, baseline |
| **Random Forest** | Ensemble | Handles non-linear, feature importance, robust |
| **Gradient Boosting** | Sequential Ensemble | Often best performance, handles imbalanced data |

### Algorithm Deep Dives

#### 1. Logistic Regression

**What It Is:**
Logistic Regression is a linear classifier that predicts the probability of a binary outcome using a logistic (sigmoid) function. Despite the name "regression," it's used for classification.

**How It Works:**
```
Input: Features (x1, x2, ..., xn)
Linear combination: z = w0 + w1*x1 + w2*x2 + ... + wn*xn
Sigmoid function: p = 1 / (1 + e^(-z))
Output: Probability between 0 and 1
```

The sigmoid function squashes any real-valued number into the range (0, 1), interpreting it as probability.

**Training Process:**
1. Initialize weights (coefficients) randomly or to zero
2. Calculate predicted probabilities for all training samples
3. Compute log-loss (how wrong the predictions are)
4. Update weights using gradient descent to minimize loss
5. Repeat until convergence

**Why We Use It for Churn:**
- **Interpretability:** Coefficients show direction and magnitude of feature impact
  ```python
  # Example output:
  # tenure: -0.15 (longer tenure = lower churn probability)
  # MonthlyCharges: +0.08 (higher charges = slightly higher churn)
  ```
- **Fast training:** Seconds even on large datasets
- **Probability output:** Natural churn probability scores
- **Baseline:** Simple model to benchmark against

**Limitations:**
- Assumes linear relationship between features and log-odds
- Can't capture complex feature interactions automatically
- Requires feature engineering for non-linear patterns

**When to Choose:**
- Need explainable coefficients for business stakeholders
- Quick prototyping and baseline
- Linear relationships dominate

---

#### 2. Gradient Boosting (Gradient Boosting Classifier)

**What It Is:**
Gradient Boosting builds an ensemble of weak prediction models (typically decision trees) sequentially, where each new tree corrects the errors of the previous ones.

**How It Works:**

**Conceptual Analogy:**
Imagine learning to play golf:
1. First attempt: You miss by 50 yards (high error)
2. Second attempt: You adjust based on first error, miss by 20 yards
3. Third attempt: Adjust again, miss by 5 yards
4. Continue until you're accurate

Each "attempt" is a weak learner (shallow tree), and the combination of many attempts produces a strong prediction.

**Technical Process:**
```
Step 1: Start with simple prediction (e.g., log-odds of churn rate)
Step 2: Calculate residuals (errors) of current model
Step 3: Train new tree to predict these residuals
Step 4: Update model: F_new(x) = F_old(x) + learning_rate * tree_prediction
Step 5: Repeat for n_estimators trees
```

**Key Hyperparameters:**
- `n_estimators`: Number of trees (100 in our config)
- `learning_rate`: How much each tree contributes (0.1 typical)
- `max_depth`: Depth of each tree (3-5 typical)

**Why We Use It for Churn:**
- **High accuracy:** Often best performance on tabular data
- **Handles non-linearity:** Captures complex feature interactions automatically
- **Feature importance:** Built-in ranking of predictive features
- **No scaling needed:** Works with raw feature values
- **Handles missing values:** Trees can split on "is_missing"

**Limitations:**
- Prone to overfitting if not tuned (too many trees, too deep)
- Sequential training = slower than random forests (parallel)
- Harder to interpret than single decision tree

**Why Not XGBoost/LightGBM:**
```python
# Optional: Advanced models
# xgboost>=1.5.0
# lightgbm>=3.2.0
```
- **Portability:** Extra dependencies complicate deployment
- **Diminishing returns:** On 7K rows, scikit-learn's gradient boosting performs comparably
- **Production simplicity:** Fewer installation issues with pure scikit-learn

**When to Choose:**
- Need best predictive accuracy on structured data
- Complex non-linear relationships expected
- Interpretability less important than performance

---

#### 3. Neural Networks (Not Used in This Project)

**What They Are:**
Neural networks are computational models inspired by biological neurons. They consist of layers of interconnected nodes (neurons) that transform inputs through weighted connections and non-linear activation functions.

**Architecture for Classification:**
```
Input Layer: [feature1, feature2, ..., featureN]
    ↓
Hidden Layer 1: [node1, node2, ..., node32] → Activation (ReLU)
    ↓
Hidden Layer 2: [node1, node2, ..., node16] → Activation (ReLU)
    ↓
Output Layer: [probability] → Activation (Sigmoid)
```

**How Training Works:**
1. **Forward pass:** Input flows through layers, produces prediction
2. **Calculate loss:** Compare prediction to actual (binary cross-entropy for classification)
3. **Backpropagation:** Calculate gradient of loss with respect to each weight
4. **Update weights:** Adjust weights using gradient descent
5. **Repeat:** For many epochs (passes through data)

**Why We Didn't Use for Telco Churn:**

| Factor | Neural Networks | Our Choice (Gradient Boosting) |
|--------|----------------|--------------------------------|
| **Data size** | Need 10k+ samples to shine | 7K samples sufficient for boosting |
| **Tabular data** | Not optimal; excel at images/text | Trees excel at tabular data |
| **Interpretability** | Black box (hard to explain) | Feature importance rankings |
| **Training time** | Minutes to hours | Seconds |
| **Hyperparameter tuning** | Many parameters (layers, units, dropout) | Fewer parameters |
| **Deployment** | Heavier dependencies | Lightweight scikit-learn |

**When Neural Networks Shine:**
- Computer vision (CNNs for image classification)
- Natural language processing (RNNs, Transformers)
- Audio/signal processing
- Massive datasets (millions of rows)

**Simple Neural Network Example (for reference):**
```python
from sklearn.neural_network import MLPClassifier

# MLP = Multi-Layer Perceptron (simple neural network)
# Not used in this project, but shown for comparison
nn_model = MLPClassifier(
    hidden_layer_sizes=(64, 32),  # Two hidden layers
    activation='relu',            # Rectified Linear Unit
    solver='adam',                # Optimizer
    max_iter=500,                 # Training iterations
    random_state=42
)

# Would achieve similar ~79% accuracy but:
# - Takes longer to train
# - Harder to interpret
# - More hyperparameters to tune
```

**Summary:**
For structured/tabular data like Telco customer records (7K rows, 20 features), tree-based methods (Gradient Boosting, Random Forest) typically outperform neural networks while being faster and more interpretable.

### Imbalanced Data Handling

**Class Distribution:** ~27% churn, 73% no churn

**Why NOT use SMOTE/oversampling?**
1. **Real-world simulation**: We want predictions on actual class distribution
2. **Calibration**: Probability scores more meaningful on original data
3. **Sufficient minority class**: 27% is not severely imbalanced

**Alternative approach:** Use ROC-AUC as primary metric (handles imbalance)

---

### Algorithm Selection Summary

| Algorithm | Why Chosen | Best For | Limitations |
|-----------|-----------|----------|-------------|
| **Logistic Regression** | Baseline, interpretable coefficients | Quick baseline, explaining feature direction | Linear relationships only |
| **Random Forest** | Robust ensemble, feature importance | Handling outliers, robust predictions | May overfit, slower than single tree |
| **Gradient Boosting** | Best accuracy on tabular data | Maximum predictive performance | Prone to overfitting, sequential training |

**Our Strategy:**
1. Start with Logistic Regression (baseline, interpretable)
2. Add Random Forest (non-linear patterns, robust)
3. Use Gradient Boosting (best performance, ~83% ROC-AUC)
4. Exclude Neural Networks (wrong tool for tabular data at this scale)

---

## Implementation: Building Each Component

### 1. Configuration Layer (`config.py`)

```python
"""
Centralize ALL tunable parameters here.
Benefits:
- Single source of truth
- Easy A/B testing (change config, rerun)
- No magic numbers in code
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

# Feature lists - defined once, used everywhere
NUMERICAL_COLUMNS = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges'
]

CATEGORICAL_COLUMNS = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

# ML parameters
RANDOM_STATE = 42  # Reproducibility
TEST_SIZE = 0.2    # 80/20 split - sufficient for 7K rows

# Model hyperparameters (grid search could improve these)
MODEL_CONFIGS = {
    'logistic_regression': {
        'random_state': RANDOM_STATE,
        'max_iter': 1000  # Ensure convergence
    },
    'random_forest': {
        'random_state': RANDOM_STATE,
        'n_estimators': 100  # Balance speed/performance
    },
    'gradient_boosting': {
        'random_state': RANDOM_STATE,
        'n_estimators': 100
    }
}
```

### 2. Data I/O Utilities (`utils.py`)

```python
"""
Abstract data operations for flexibility.
Tomorrow we might switch from parquet to HDF5 - only change here.
"""

import pandas as pd
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_data(filepath: Path) -> pd.DataFrame:
    """Load CSV with validation."""
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows from {filepath}")
    return df

def save_parquet(df: pd.DataFrame, filepath: Path):
    """Save to parquet with compression."""
    df.to_parquet(filepath, compression='snappy')
    logger.info(f"Saved {len(df)} rows to {filepath}")

def load_parquet(filepath: Path) -> pd.DataFrame:
    """Load parquet file."""
    df = pd.read_parquet(filepath)
    logger.info(f"Loaded {len(df)} rows from {filepath}")
    return df

def save_pickle(obj, filepath: Path):
    """Serialize Python objects (models, scalers, encoders)."""
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    logger.info(f"Saved object to {filepath}")

def load_pickle(filepath: Path):
    """Deserialize Python objects."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)
```

### 3. Preprocessing Pipeline (`preprocess.py`)

```python
"""
Complete preprocessing workflow.
Key principle: Fit on train, transform both train and test.
This prevents data leakage.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging

from config import (
    DATA_PATH, PROCESSED_DATA_DIR, NUMERICAL_COLUMNS,
    CATEGORICAL_COLUMNS, IDENTIFIER_COLUMN, TARGET_COLUMN,
    RANDOM_STATE, TEST_SIZE
)
from utils import load_data, save_parquet, save_pickle

logger = logging.getLogger(__name__)

def preprocess_data():
    """
    Execute the complete preprocessing pipeline.
    
    Pipeline Steps:
    1. Load raw data
    2. Clean TotalCharges (string -> numeric)
    3. Remove rows with missing values
    4. Train/test split (stratify on target for class balance)
    5. Fit StandardScaler on train numerical features
    6. Fit OneHotEncoder on train categorical features
    7. Transform both train and test
    8. Save processed data and preprocessing artifacts
    """
    logger.info("Starting data preprocessing pipeline")
    
    # 1. Load
    data = load_data(DATA_PATH)
    
    # 2. Clean TotalCharges
    # Critical: Handle whitespace strings in TotalCharges
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    
    # 3. Remove missing
    # Only 11 rows affected (0.16%), safe to drop
    data_clean = data.dropna()
    
    # 4. Split BEFORE any transformation (prevent leakage)
    # stratify ensures same churn rate in train/test
    X_train, X_test, y_train, y_test = train_test_split(
        data_clean.drop(columns=[TARGET_COLUMN]),
        data_clean[TARGET_COLUMN],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data_clean[TARGET_COLUMN]  # Maintain class balance
    )
    
    # 5. Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[NUMERICAL_COLUMNS])
    X_test_scaled = scaler.transform(X_test[NUMERICAL_COLUMNS])
    
    # 6. Encode categorical features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_encoded = encoder.fit_transform(X_train[CATEGORICAL_COLUMNS])
    X_test_encoded = encoder.transform(X_test[CATEGORICAL_COLUMNS])
    
    # 7. Reconstruct DataFrames with proper column names
    X_train_processed = pd.concat([
        X_train[IDENTIFIER_COLUMN].reset_index(drop=True),
        pd.DataFrame(X_train_scaled, columns=NUMERICAL_COLUMNS),
        pd.DataFrame(X_train_encoded, columns=encoder.get_feature_names_out())
    ], axis=1)
    
    X_test_processed = pd.concat([
        X_test[IDENTIFIER_COLUMN].reset_index(drop=True),
        pd.DataFrame(X_test_scaled, columns=NUMERICAL_COLUMNS),
        pd.DataFrame(X_test_encoded, columns=encoder.get_feature_names_out())
    ], axis=1)
    
    # 8. Save everything
    save_parquet(X_train_processed, PROCESSED_DATA_DIR / "train.parquet")
    save_parquet(X_test_processed, PROCESSED_DATA_DIR / "test.parquet")
    save_parquet(y_train.to_frame(), PROCESSED_DATA_DIR / "train_labels.parquet")
    save_parquet(y_test.to_frame(), PROCESSED_DATA_DIR / "test_labels.parquet")
    save_pickle(scaler, PROCESSED_DATA_DIR / "scaler.pkl")
    save_pickle(encoder, PROCESSED_DATA_DIR / "encoder.pkl")
    
    return X_train_processed, X_test_processed, y_train, y_test
```

### 4. Model Training (`train.py`)

```python
"""
Model training with comparison framework.
Design goal: Easy to add new models, consistent evaluation.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging

from config import (
    PROCESSED_DATA_DIR, MODELS_DIR, MODEL_CONFIGS, RANDOM_STATE
)
from utils import load_parquet, save_pickle

logger = logging.getLogger(__name__)

def train_models():
    """
    Train multiple models and select best based on ROC-AUC.
    
    Why ROC-AUC for imbalanced data?
    - Threshold-independent (unlike accuracy)
    - Measures discrimination across all thresholds
    - Robust to class imbalance
    """
    # Load processed data
    X_train = load_parquet(PROCESSED_DATA_DIR / "train.parquet")
    y_train = load_parquet(PROCESSED_DATA_DIR / "train_labels.parquet").squeeze()
    X_test = load_parquet(PROCESSED_DATA_DIR / "test.parquet")
    y_test = load_parquet(PROCESSED_DATA_DIR / "test_labels.parquet").squeeze()
    
    # Drop identifier for modeling
    X_train_features = X_train.drop(columns=['customerID'])
    X_test_features = X_test.drop(columns=['customerID'])
    
    # Convert target to binary (Yes/No -> 1/0)
    y_train_binary = (y_train == 'Yes').astype(int)
    y_test_binary = (y_test == 'Yes').astype(int)
    
    # Define models
    models = {
        'logistic_regression': LogisticRegression(**MODEL_CONFIGS['logistic_regression']),
        'random_forest': RandomForestClassifier(**MODEL_CONFIGS['random_forest']),
        'gradient_boosting': GradientBoostingClassifier(**MODEL_CONFIGS['gradient_boosting'])
    }
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        
        # Train
        model.fit(X_train_features, y_train_binary)
        trained_models[name] = model
        
        # Predict
        y_pred = model.predict(X_test_features)
        y_proba = model.predict_proba(X_test_features)[:, 1]
        
        # Evaluate
        metrics = {
            'model': name,
            'accuracy': accuracy_score(y_test_binary, y_pred),
            'precision': precision_score(y_test_binary, y_pred),
            'recall': recall_score(y_test_binary, y_pred),
            'f1': f1_score(y_test_binary, y_pred),
            'roc_auc': roc_auc_score(y_test_binary, y_proba)
        }
        results.append(metrics)
        logger.info(f"{name} - ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Select best model by ROC-AUC
    results_df = pd.DataFrame(results)
    best_model_name = results_df.loc[results_df['roc_auc'].idxmax(), 'model']
    best_model = trained_models[best_model_name]
    
    logger.info(f"Best model: {best_model_name} with ROC-AUC: {results_df['roc_auc'].max():.4f}")
    
    # Save artifacts
    save_pickle(best_model, MODELS_DIR / "best_model.pkl")
    results_df.to_csv(MODELS_DIR / "model_comparison_results.csv", index=False)
    
    return best_model, results_df
```

### 5. Prediction Engine (`predict.py`)

```python
"""
Inference script for new data.
Handles the full preprocessing pipeline for unseen data.
"""

import argparse
import pandas as pd
import logging

from config import PROCESSED_DATA_DIR, MODELS_DIR, NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS
from utils import load_data, save_pickle, load_pickle

logger = logging.getLogger(__name__)

def predict_churn(input_path: str, output_path: str, threshold: float = 0.5):
    """
    Generate churn predictions for new customer data.
    
    Process:
    1. Load new data
    2. Apply same preprocessing (TotalCharges cleaning)
    3. Load saved scaler and encoder (fit on training data)
    4. Transform features
    5. Load trained model
    6. Generate predictions and probabilities
    7. Save results
    """
    # Load artifacts
    scaler = load_pickle(PROCESSED_DATA_DIR / "scaler.pkl")
    encoder = load_pickle(PROCESSED_DATA_DIR / "encoder.pkl")
    model = load_pickle(MODELS_DIR / "best_model.pkl")
    
    # Load and preprocess new data
    new_data = load_data(input_path)
    new_data['TotalCharges'] = pd.to_numeric(new_data['TotalCharges'], errors='coerce')
    
    # Handle missing values (drop or impute based on business rules)
    new_data_clean = new_data.dropna()
    if len(new_data_clean) < len(new_data):
        logger.warning(f"Dropped {len(new_data) - len(new_data_clean)} rows with missing values")
    
    # Transform features using TRAIN-FITTED scaler/encoder
    # Critical: Don't fit on new data, only transform
    X_numerical = scaler.transform(new_data_clean[NUMERICAL_COLUMNS])
    X_categorical = encoder.transform(new_data_clean[CATEGORICAL_COLUMNS])
    
    X_processed = pd.concat([
        new_data_clean['customerID'].reset_index(drop=True),
        pd.DataFrame(X_numerical, columns=NUMERICAL_COLUMNS),
        pd.DataFrame(X_categorical, columns=encoder.get_feature_names_out())
    ], axis=1)
    
    # Generate predictions
    X_features = X_processed.drop(columns=['customerID'])
    probabilities = model.predict_proba(X_features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    
    # Create results
    results = pd.DataFrame({
        'CustomerID': new_data_clean['customerID'],
        'Churn_Probability': probabilities,
        'Predicted_Churn': predictions,
        'Risk_Segment': pd.cut(probabilities, 
                               bins=[0, 0.3, 0.7, 1.0],
                               labels=['Low', 'Medium', 'High'])
    })
    
    results.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument('--output', required=True, help='Output CSV file')
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help='Churn probability threshold')
    args = parser.parse_args()
    
    predict_churn(args.input, args.output, args.threshold)
```

### 6. Pipeline Orchestration (`run_pipeline.py`)

```python
"""
End-to-end orchestration with optional step skipping.
Useful for iterative development:
- --skip-preprocessing: Data already cleaned
- --skip-training: Model already trained, just generate predictions
"""

import argparse
import logging
from pathlib import Path

from preprocess import preprocess_data
from train import train_models
from utils import load_parquet, load_pickle
from config import MODELS_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, IDENTIFIER_COLUMN

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_full_pipeline(skip_preprocessing: bool = False, skip_training: bool = False):
    """Orchestrate the complete ML workflow."""
    
    # Step 1: Preprocessing
    if not skip_preprocessing:
        X_train, X_test, y_train, y_test = preprocess_data()
    
    # Step 2: Training
    if not skip_training:
        best_model, results_df = train_models()
    else:
        best_model = load_pickle(MODELS_DIR / "best_model.pkl")
    
    # Step 3: Generate test set predictions for evaluation
    X_test = load_parquet(PROCESSED_DATA_DIR / "test.parquet")
    y_test = load_parquet(PROCESSED_DATA_DIR / "test_labels.parquet").squeeze()
    
    X_test_features = X_test.drop(columns=[IDENTIFIER_COLUMN])
    y_pred = best_model.predict(X_test_features)
    y_proba = best_model.predict_proba(X_test_features)[:, 1]
    
    predictions_df = pd.DataFrame({
        'CustomerID': X_test[IDENTIFIER_COLUMN],
        'Actual_Churn': y_test,
        'Predicted_Churn': y_pred,
        'Churn_Probability': y_proba
    })
    
    predictions_df.to_csv(RESULTS_DIR / 'test_predictions.csv', index=False)
    logger.info("Pipeline completed successfully")
    
    return best_model, predictions_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-preprocessing', action='store_true')
    parser.add_argument('--skip-training', action='store_true')
    args = parser.parse_args()
    
    run_full_pipeline(args.skip_preprocessing, args.skip_training)
```

---

## Edge Cases & Production Resilience

Real-world ML systems fail on edge cases. This section documents every edge case in the Telco Churn project and provides robust handling solutions.

### 1. New Customers (tenure = 0)

**The Problem:**
- TotalCharges = 0 (no billing history)
- Missing value pattern (whitespace in raw data)
- May have different churn patterns than established customers

**Detection:**
```python
def detect_new_customers(df):
    """Identify customers with no tenure."""
    new_customer_mask = df['tenure'] == 0
    count = new_customer_mask.sum()
    if count > 0:
        print(f"Found {count} new customers (tenure=0)")
        print(f"Churn rate: {df[new_customer_mask]['Churn'].mean():.1%}")
    return new_customer_mask
```

**Solutions:**
```python
# Option A: Handle in preprocessing
df['TotalCharges'] = df['TotalCharges'].fillna(0)  # tenure=0 → $0 charges

# Option B: Feature engineering - flag new customers
df['Is_New_Customer'] = (df['tenure'] == 0).astype(int)

# Option C: Separate model for new customers (if pattern differs significantly)
new_customers = df[df['tenure'] == 0]
established = df[df['tenure'] > 0]
# Train separate models or add interaction term
```

**Production Handling:**
```python
def handle_new_customer_edge_case(customer_data):
    """Ensure new customers get appropriate predictions."""
    if customer_data['tenure'] == 0:
        # Use only features available for new customers
        # Exclude TotalCharges-dependent features
        features = ['MonthlyCharges', 'Contract', 'PaymentMethod', ...]
        prediction = model.predict(customer_data[features])
        
        # Flag for manual review (new customers are higher uncertainty)
        return {
            'prediction': prediction,
            'confidence': 'LOW',
            'reason': 'New customer - limited history'
        }
```

---

### 2. Unseen Categories in Production

**The Problem:**
- New PaymentMethod added (e.g., "Crypto")
- New Contract type introduced
- Geographic expansion adds new regions

**Detection:**
```python
def detect_unseen_categories(train_data, prod_data, categorical_cols):
    """Find categories in production not seen during training."""
    unseen = {}
    for col in categorical_cols:
        train_cats = set(train_data[col].unique())
        prod_cats = set(prod_data[col].unique())
        new_cats = prod_cats - train_cats
        if new_cats:
            unseen[col] = new_cats
            print(f"⚠️  Unseen in {col}: {new_cats}")
    return unseen
```

**Solutions:**

**Option A: Robust Encoder Configuration**
```python
from sklearn.preprocessing import OneHotEncoder

# Configure encoder to handle unknown categories
encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'  # Creates all-zero vector for unknowns
)
encoder.fit(train_data[CATEGORICAL_COLUMNS])

# In production
encoded = encoder.transform(prod_data[CATEGORICAL_COLUMNS])
# Unknown categories become [0, 0, 0, ...]
```

**Option B: Category Mapping to 'Unknown'**
```python
class RobustCategoricalEncoder:
    """Map unknown categories to 'Unknown' bucket."""
    
    def __init__(self, known_categories):
        self.known_categories = known_categories
    
    def transform(self, series):
        return series.apply(
            lambda x: x if x in self.known_categories else 'Unknown'
        )

# Usage
payment_encoder = RobustCategoricalEncoder(
    known_categories={'Electronic check', 'Mailed check', ...}
)
prod_data['PaymentMethod'] = payment_encoder.transform(prod_data['PaymentMethod'])
```

**Option C: Monitoring & Alerting**
```python
def monitor_category_drift(train_dist, prod_dist, threshold=0.05):
    """Alert if new categories exceed threshold % of traffic."""
    total_new = sum(prod_dist.get(cat, 0) for cat in prod_dist 
                    if cat not in train_dist)
    if total_new > threshold:
        alert(f"New categories represent {total_new:.1%} of traffic. Retrain needed.")
```

---

### 3. Missing Values in Production

**The Problem:**
- Customer data incomplete (e.g., missing TotalCharges)
- System outages cause null values
- Data integration issues

**Detection:**
```python
def check_missing_in_production(df, required_cols, warning_threshold=0.01):
    """Check for unexpected missing values."""
    missing_report = {}
    for col in required_cols:
        missing_pct = df[col].isnull().mean()
        missing_report[col] = missing_pct
        
        if missing_pct > warning_threshold:
            logger.warning(f"{col}: {missing_pct:.1%} missing (threshold: {warning_threshold:.1%})")
        elif missing_pct > 0:
            logger.info(f"{col}: {missing_pct:.1%} missing (within tolerance)")
    
    return missing_report
```

**Solutions:**

**Option A: Imputation Strategy**
```python
# Store imputation values from training
IMPUTATION_VALUES = {
    'TotalCharges': 0,  # New customers
    'MonthlyCharges': train_data['MonthlyCharges'].median(),
    'tenure': train_data['tenure'].median()
}

def impute_production_data(df):
    """Apply training-derived imputation."""
    df_imputed = df.copy()
    for col, value in IMPUTATION_VALUES.items():
        df_imputed[col] = df_imputed[col].fillna(value)
    return df_imputed
```

**Option B: Prediction with Missing Features**
```python
# Tree-based models can handle missing values natively
# For linear models, use imputation or skip prediction

def safe_predict(model, data, required_fill_rate=0.8):
    """Only predict if sufficient data available."""
    fill_rate = 1 - data.isnull().mean()
    
    if fill_rate < required_fill_rate:
        return {
            'prediction': None,
            'error': f'Insufficient data ({fill_rate:.1%} filled, need {required_fill_rate:.1%})'
        }
    
    # Impute and predict
    data_imputed = impute_production_data(data)
    return model.predict(data_imputed)
```

**Option C: Circuit Breaker**
```python
def production_prediction_with_circuit_breaker(model, data):
    """Stop predictions if data quality degrades."""
    missing_report = check_missing_in_production(data, REQUIRED_COLUMNS)
    
    # If any critical feature >50% missing, stop
    if any(pct > 0.5 for pct in missing_report.values()):
        raise DataQualityError("Critical features missing. Halting predictions.")
    
    return model.predict(data)
```

---

### 4. Extreme Outliers

**The Problem:**
- Customer with $10,000 MonthlyCharges (data entry error?)
- tenure = 999 (impossible)
- Negative charges (refunds? fraud?)

**Detection:**
```python
def detect_outliers(df, column, method='iqr'):
    """Identify outlier values."""
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
    elif method == 'business_rules':
        # Domain-specific limits
        limits = {
            'tenure': (0, 72),  # Max 6 years reasonable
            'MonthlyCharges': (0, 200),
            'TotalCharges': (0, 15000)
        }
        lower, upper = limits.get(column, (None, None))
    
    outliers = df[(df[column] < lower) | (df[column] > upper)]
    return outliers
```

**Solutions:**

**Option A: Validation & Rejection**
```python
VALIDATION_RULES = {
    'tenure': {'min': 0, 'max': 72},
    'MonthlyCharges': {'min': 0, 'max': 200},
    'TotalCharges': {'min': 0, 'max': 15000}
}

def validate_customer_data(customer):
    """Check if values are within acceptable ranges."""
    errors = []
    for col, rules in VALIDATION_RULES.items():
        value = customer.get(col)
        if value is not None:
            if value < rules['min'] or value > rules['max']:
                errors.append(f"{col}={value} outside range [{rules['min']}, {rules['max']}]")
    
    if errors:
        return {'valid': False, 'errors': errors}
    return {'valid': True}
```

**Option B: Winsorization (Capping)**
```python
def winsorize_series(series, limits):
    """Cap values at business-defined limits."""
    return series.clip(lower=limits['min'], upper=limits['max'])

# Usage
df['tenure'] = winsorize_series(df['tenure'], {'min': 0, 'max': 72})
```

**Option C: Flag for Review**
```python
def flag_extreme_customers(df):
    """Mark customers with unusual values for manual review."""
    flags = pd.DataFrame(index=df.index)
    
    flags['high_tenure'] = df['tenure'] > 72
    flags['high_monthly'] = df['MonthlyCharges'] > 150
    flags['high_total'] = df['TotalCharges'] > 10000
    
    flags['any_flag'] = flags.any(axis=1)
    
    return flags
```

---

### 5. Duplicate Customers

**The Problem:**
- Same customerID appears multiple times (system bug)
- Near-duplicates (same person, different IDs)
- Retrain vs re-score confusion

**Detection:**
```python
def detect_duplicates(df):
    """Find exact and near-duplicates."""
    reports = {}
    
    # Exact duplicates (all columns match)
    exact_dups = df.duplicated().sum()
    reports['exact'] = exact_dups
    
    # ID duplicates (same customerID, different data)
    id_dups = df.duplicated(subset=['customerID'], keep=False).sum()
    reports['id_duplicates'] = id_dups
    
    if id_dups > 0:
        print("Duplicate customerIDs found:")
        print(df[df.duplicated(subset=['customerID'], keep=False)]
              .sort_values('customerID')[['customerID', 'tenure', 'MonthlyCharges']])
    
    return reports
```

**Solutions:**

**Option A: Deduplication**
```python
def deduplicate_customers(df, strategy='keep_last'):
    """Remove duplicate customer records."""
    if strategy == 'keep_last':
        # Keep most recent record (assuming sort by date)
        return df.drop_duplicates(subset=['customerID'], keep='last')
    elif strategy == 'keep_first':
        return df.drop_duplicates(subset=['customerID'], keep='first')
    elif strategy == 'merge':
        # Aggregate: take mean of numerical, mode of categorical
        # Complex - use only if business logic requires
        pass
```

**Option B: Prediction Consistency Check**
```python
def check_prediction_consistency(model, duplicate_records):
    """Ensure same customer gets same prediction."""
    predictions = model.predict(duplicate_records)
    
    if len(set(predictions)) > 1:
        logger.warning("Same customer getting different predictions!")
        # Return most common or average probability
        return max(set(predictions), key=predictions.count)
    
    return predictions[0]
```

---

### 6. Model Version Mismatch

**The Problem:**
- Model saved with scikit-learn 1.0, loaded with 1.3
- Pickle deserialization errors
- Feature order changed between training and production

**Detection:**
```python
def verify_model_compatibility(model, expected_version):
    """Check model was saved with compatible library version."""
    import sklearn
    current_version = sklearn.__version__
    
    # Major version must match
    if current_version.split('.')[0] != expected_version.split('.')[0]:
        raise VersionError(
            f"Model saved with sklearn {expected_version}, "
            f"but running with {current_version}"
        )
```

**Solutions:**

**Option A: Version Pinning**
```python
# requirements.txt
scikit-learn==1.2.0  # Pin exact version used for training

# Save version info with model
model_metadata = {
    'sklearn_version': sklearn.__version__,
    'python_version': sys.version,
    'training_date': datetime.now().isoformat(),
    'feature_columns': list(X.columns)  # Critical: save feature order
}

import joblib
joblib.dump({'model': model, 'metadata': model_metadata}, 'model.pkl')
```

**Option B: Feature Order Validation**
```python
def load_model_with_validation(path, expected_features):
    """Load model and verify feature compatibility."""
    package = joblib.load(path)
    model = package['model']
    metadata = package['metadata']
    
    # Verify feature order matches
    saved_features = metadata['feature_columns']
    if saved_features != expected_features:
        raise FeatureMismatchError(
            f"Feature mismatch! Expected {len(expected_features)} features, "
            f"model trained with {len(saved_features)}"
        )
    
    return model
```

**Option C: Model Serialization Alternatives**
```python
# Option 1: joblib (scikit-learn standard)
import joblib
joblib.dump(model, 'model.joblib')

# Option 2: ONNX (framework-agnostic, no version issues)
# !pip install skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# Option 3: PMML (industry standard for model exchange)
# !pip install sklearn2pmml
from sklearn2pmml import sklearn2pmml
sklearn2pmml(model, 'model.pmml')
```

---

### 7. Batch Prediction Failures

**The Problem:**
- Single bad row crashes entire batch
- Memory limits with large files
- Timeout on long-running predictions

**Solutions:**

**Option A: Row-Level Error Handling**
```python
def safe_batch_predict(model, data, batch_size=1000):
    """Predict on batches with per-row error handling."""
    results = []
    errors = []
    
    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i+batch_size]
        
        try:
            predictions = model.predict(batch)
            results.extend(predictions)
        except Exception as e:
            # Log error, continue with next batch
            logger.error(f"Batch {i//batch_size} failed: {e}")
            errors.append({'batch': i, 'error': str(e)})
            # Fill with default/None
            results.extend([None] * len(batch))
    
    return results, errors
```

**Option B: Individual Row Fallback**
```python
def robust_predict_row(model, row):
    """Try to predict on single row, return default on failure."""
    try:
        return model.predict(row.values.reshape(1, -1))[0]
    except Exception as e:
        logger.warning(f"Prediction failed for customer {row.get('customerID')}: {e}")
        return {
            'prediction': None,
            'error': str(e),
            'fallback': 'USE_BUSINESS_RULE'  # Flag for manual handling
        }

# Apply to each row
def batch_predict_safe(model, data):
    return data.apply(lambda row: robust_predict_row(model, row), axis=1)
```

**Option C: Memory-Efficient Processing**
```python
def predict_large_file_streaming(model, input_path, output_path, chunksize=10000):
    """Process large files in chunks to avoid memory issues."""
    import pandas as pd
    
    first_chunk = True
    for chunk in pd.read_csv(input_path, chunksize=chunksize):
        predictions = model.predict(chunk)
        chunk['prediction'] = predictions
        
        # Write incrementally
        mode = 'w' if first_chunk else 'a'
        header = first_chunk
        chunk.to_csv(output_path, mode=mode, header=header, index=False)
        first_chunk = False
```

---

### 8. Schema Changes in Production

**The Problem:**
- Column renamed: `TotalCharges` → `Total_Charges`
- New column added: `LoyaltyScore`
- Column removed: `PaperlessBilling` discontinued

**Detection:**
```python
def validate_schema(data, expected_columns):
    """Check if production data matches expected schema."""
    actual_columns = set(data.columns)
    expected = set(expected_columns)
    
    missing = expected - actual_columns
    extra = actual_columns - expected
    
    report = {
        'valid': len(missing) == 0,
        'missing_columns': list(missing),
        'extra_columns': list(extra)
    }
    
    if missing:
        raise SchemaError(f"Missing required columns: {missing}")
    
    return report
```

**Solutions:**

**Option A: Schema Versioning**
```python
SCHEMA_VERSIONS = {
    'v1': {
        'columns': ['customerID', 'gender', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn'],
        'date': '2023-01-01'
    },
    'v2': {
        'columns': ['customerID', 'gender', 'tenure', 'MonthlyCharges', 'TotalCharges', 'PaymentMethod', 'Churn'],
        'date': '2024-01-01'
    }
}

def validate_data_version(data, version='v2'):
    """Ensure data matches expected schema version."""
    expected_cols = SCHEMA_VERSIONS[version]['columns']
    return validate_schema(data, expected_cols)
```

**Option B: Flexible Column Mapping**
```python
COLUMN_ALIASES = {
    'TotalCharges': ['TotalCharges', 'Total_Charges', 'total_charges'],
    'customerID': ['customerID', 'CustomerID', 'customer_id']
}

def normalize_columns(df):
    """Map various column naming conventions to standard names."""
    for standard, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                df = df.rename(columns={alias: standard})
                break
    return df
```

**Option C: Graceful Degradation**
```python
def predict_with_missing_features(model, data, required_features):
    """Predict even if some features are missing."""
    available = [f for f in required_features if f in data.columns]
    missing = [f for f in required_features if f not in data.columns]
    
    if missing:
        logger.warning(f"Missing features {missing}, using available only")
        # Impute missing or use model that handles missing features
        for col in missing:
            data[col] = 0  # or median value
    
    return model.predict(data[required_features])
```

---

### Edge Case Summary Table

| Edge Case | Detection | Primary Solution | Fallback |
|-----------|-----------|------------------|----------|
| New customers (tenure=0) | `tenure == 0` | Impute TotalCharges=0 | Flag for review |
| Unseen categories | Set difference | `handle_unknown='ignore'` | Map to 'Unknown' |
| Missing values | `isnull().sum()` | Training-derived imputation | Skip prediction |
| Extreme outliers | IQR / business rules | Winsorization | Validation error |
| Duplicates | `duplicated()` | Drop duplicates | Manual merge |
| Model version mismatch | Version string check | Pin dependencies | ONNX format |
| Batch failures | Try/catch per row | Row-level handling | Default prediction |
| Schema changes | Column name check | Column mapping | Graceful degradation |

---

## Evaluation Framework

### Why These Metrics?

| Metric | Formula | Business Interpretation |
|--------|---------|---------------------------|
| **Accuracy** | (TP + TN) / Total | Overall correctness, but misleading with imbalance |
| **Precision** | TP / (TP + FP) | Of predicted churners, how many actually churned? |
| **Recall** | TP / (TP + FN) | Of actual churners, how many did we catch? |
| **F1** | 2 * (Precision * Recall) / (Precision + Recall) | Balance between precision and recall |
| **ROC-AUC** | Area under ROC curve | Discrimination ability across all thresholds |

### Business Context for Churn

**Precision vs Recall Trade-off:**
- **High Precision**: Fewer false alarms (don't waste retention offers)
- **High Recall**: Catch more potential churners (don't miss at-risk customers)

**Threshold Selection:**
- Default 0.5: Balanced
- Lower threshold (0.3): Catch more at-risk customers (higher recall)
- Higher threshold (0.7): Only flag high-confidence churners (higher precision)

### Feature Importance Analysis

```python
# Add to train.py for interpretability
if hasattr(model, 'feature_importances_'):
    importances = pd.DataFrame({
        'feature': X_train_features.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importances.head(10))
```

**Why it matters:**
- Validate model learned meaningful patterns
- Guide business strategy (focus on high-importance factors)
- Debug unexpected predictions

---

## Deployment Considerations

### Model Serialization

We use `pickle` for simplicity. Alternatives:
- **joblib**: Better for large numpy arrays
- **ONNX**: Framework-agnostic, language-independent
- **PMML**: Industry standard for model exchange

### Preprocessing Consistency

**Critical rule:** Never fit on test/production data.
```python
# WRONG (data leakage)
scaler.fit_transform(new_data)

# RIGHT (use training-fitted scaler)
scaler.transform(new_data)
```

### Monitoring

In production, track:
1. **Prediction distribution drift**: Are churn probabilities changing?
2. **Feature drift**: Are input distributions shifting?
3. **Label collection**: Capture actual churn outcomes for retraining

---

## Next Steps for Improvement

1. **Hyperparameter tuning**: Grid search or Bayesian optimization
2. **Cross-validation**: K-fold for more robust evaluation
3. **Ensemble methods**: Stack multiple models
4. **Feature engineering**: Create interaction terms, tenure bins
5. **Threshold optimization**: Use precision-recall curve to select optimal threshold

---

## Running the Pipeline

```bash
# Full pipeline (preprocess → train → predict)
python src/run_pipeline.py

# Skip preprocessing (use existing cleaned data)
python src/run_pipeline.py --skip-preprocessing

# Skip training (use existing model)
python src/run_pipeline.py --skip-training

# Predict on new data
python src/predict.py --input data/new_customers.csv --output predictions.csv --threshold 0.5
```

---

## Dataset Schema Validation

### Why Validate the Schema?

The preprocessing pipeline silently produces wrong results if the input data changes shape. For example:

- A column gets renamed upstream (`MonthlyCharge` instead of `MonthlyCharges`) → scaler throws a `KeyError` mid-pipeline
- An ETL job writes `SeniorCitizen` as `"0"`/`"1"` strings instead of integers → downstream one-hot encoding silently treats it as categorical
- A daily export contains only today's records (50 rows) instead of the full dataset → model trains on a non-representative sample with no warning

Catching these **at load time** — before any transformation — means the pipeline fails loudly with a clear message instead of silently producing bad artifacts.

### The Three Checks

#### 1. Required Columns

```python
required_columns = set(NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS + [IDENTIFIER_COLUMN, TARGET_COLUMN])
missing = required_columns - set(data.columns)
if missing:
    errors.append(f"Missing required columns: {sorted(missing)}")
```

Compares the full set of columns the pipeline expects (sourced from `config.py`) against what the CSV actually contains. Using `set` difference means extra columns in the data are ignored — only missing ones are flagged. Sorting the output makes the error message deterministic.

**When this fires:** Column was renamed, dropped, or the wrong file was passed as input.

#### 2. Minimum Row Count

```python
MIN_EXPECTED_ROWS = 100

if len(data) < MIN_EXPECTED_ROWS:
    errors.append(f"Dataset has only {len(data)} rows (expected >= {MIN_EXPECTED_ROWS})")
```

Guards against accidentally passing a sample file, an empty export, or a truncated download. The threshold (`100`) is intentionally conservative — the real dataset has 7,043 rows. Adjust `MIN_EXPECTED_ROWS` in `preprocess.py` if the dataset size is expected to change.

**When this fires:** Wrong file path, partial export, or test fixture accidentally used in production.

#### 3. Key Column Dtypes

```python
EXPECTED_DTYPES: dict[str, str] = {
    'SeniorCitizen': 'int64',
    'tenure': 'int64',
    'MonthlyCharges': 'float64',
}

for col, expected_dtype in EXPECTED_DTYPES.items():
    if col in data.columns and str(data[col].dtype) != expected_dtype:
        errors.append(
            f"Column '{col}' has dtype '{data[col].dtype}', expected '{expected_dtype}'"
        )
```

Only the columns most likely to be silently mistyped are checked here — not all 21. The rationale:

| Column | Expected dtype | Risk if wrong |
|---|---|---|
| `SeniorCitizen` | `int64` | Treated as categorical, inflates feature space |
| `tenure` | `int64` | Scaler produces wrong mean/std if read as string |
| `MonthlyCharges` | `float64` | Comparison errors if read as object |

`TotalCharges` is intentionally excluded — it is **known** to arrive as `object` (string) from the CSV and is converted to numeric explicitly as the first preprocessing step.

**When this fires:** Source system changes column format (e.g. adds currency symbols, changes integer to string encoding).

### Design Decisions

**Collect all errors before raising** — the function accumulates all failures into a list and raises once at the end. This means you see every problem in a single run rather than fixing one issue, re-running, and finding the next.

```python
if errors:
    raise ValueError("Schema validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
```

**Raise `ValueError`, not `AssertionError`** — `assert` statements are disabled when Python runs with the `-O` (optimise) flag. `ValueError` always fires and is the conventional exception for bad input data.

**Don't validate after transformation** — `validate_schema` is called on the raw loaded DataFrame, before `TotalCharges` is coerced, before `dropna`, before any splits. This ensures the check reflects what was actually in the file.

### Where It Lives in the Pipeline

```
load_data(DATA_PATH)
    ↓
validate_schema(data)       ← fails fast here if data is malformed
    ↓
pd.to_numeric(TotalCharges)
    ↓
dropna()
    ↓
train_test_split()
    ↓
fit / transform / save
```

### Adding New Checks

To extend validation, add to the `errors` list inside `validate_schema` before the final `if errors` raise:

```python
# Example: warn if churn rate is implausibly low or high
if TARGET_COLUMN in data.columns:
    churn_rate = (data[TARGET_COLUMN] == 'Yes').mean()
    if not (0.05 <= churn_rate <= 0.60):
        errors.append(f"Churn rate {churn_rate:.1%} is outside expected range [5%, 60%]")
```

---

## Testing

### Why Write Tests at This Stage?

Tests are written after the pipeline is stable because at that point you know:
- Which functions exist and what their contracts are
- Which failure modes have already been encountered (the debugging log is your test specification)
- What "correct behaviour" looks like (you've run the pipeline and seen real output)

Writing tests too early — before the code stabilises — means constantly rewriting them as the API changes. Writing them too late means you've already shipped bugs you can't detect.

### The Three Test Files

The test suite is split into three files by scope:

| File | Scope | Runs without | Speed |
|---|---|---|---|
| `tests/test_predict.py` | Unit — individual functions in `predict.py` | Model artifacts | Fast |
| `tests/test_preprocess.py` | Unit — individual functions in `preprocess.py` | Model artifacts, raw data | Fast |
| `tests/test_integration.py` | Integration — full `predict_from_file` call | Nothing (needs artifacts) | Slow |

**Why separate unit from integration?** Unit tests should run in milliseconds with no filesystem dependencies. Integration tests can take seconds and require the full artifact chain to be in place. Keeping them separate means you can run `pytest tests/test_predict.py tests/test_preprocess.py` quickly during development without needing the model loaded.

---

### Thinking Process: How to Identify What to Test

The starting point is the debugging log — every runtime bug that was fixed is a test waiting to be written. If a bug happened once, it can happen again after a refactor.

**Step 1: List every function that has had a bug or handles an edge case.**

From the debugging log and code review:
- `validate_input_data` — had TypeError on string TotalCharges (fix #1)
- `handle_unseen_categories` — had ValueError from set() vs list() (fix #3)
- `handle_new_customers` — core business logic, no test coverage
- `validate_schema` — brand new function, untested
- `preprocess_new_data` — feature order must match training exactly

**Step 2: For each function, ask: what inputs would break it, and what outputs must always be true?**

This produces two kinds of tests:
- **Regression tests** — prove a previously broken input now works
- **Contract tests** — prove the function's output always has certain properties

**Step 3: Write the simplest possible input that exercises the behaviour.**

Don't use the full 7,043-row dataset in unit tests. Build a minimal DataFrame with exactly the rows needed — 2–5 rows is usually enough. This makes tests fast, readable, and independent of the data file.

---

### Test Fixture Strategy

The fixture at `tests/fixtures/sample_customers.csv` serves the integration test and any unit test that needs a realistic full-row input. It is hand-crafted (not sampled from the real data) so it:

1. **Covers every edge case in one file** — one normal row, one new customer (tenure=0), one row with a missing value, one with an unseen category, one with an out-of-range value
2. **Is committed to git** — unlike the real data which is gitignored, this small fixture is tracked so tests always run in CI
3. **Is deterministic** — same input always produces same output, making assertions reliable

The fixture has exactly 10 rows: a mix of normal cases and edge cases, all with the full column set required by `config.py`.

---

### Anatomy of a Unit Test

```python
import pandas as pd
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / 'src'))

from predict import handle_new_customers

def test_handle_new_customers_sets_total_charges_to_zero():
    # Arrange — minimal DataFrame with exactly what the function needs
    data = pd.DataFrame({
        'customerID': ['A001', 'A002'],
        'tenure':     [0,      12   ],
        'TotalCharges': [' ',  '450.0'],
        'MonthlyCharges': [29.85, 37.50],
    })

    # Act
    result = handle_new_customers(data)

    # Assert — one specific, falsifiable claim
    assert result.loc[result['tenure'] == 0, 'TotalCharges'].iloc[0] == 0
    assert result.loc[result['tenure'] == 12, 'TotalCharges'].iloc[0] == '450.0'
```

**Arrange / Act / Assert** is the standard pattern:
- **Arrange** — build the minimum input to exercise the behaviour
- **Act** — call exactly one function
- **Assert** — make one or two specific claims about the output

Avoid asserting everything. One focused assertion per test makes it immediately clear what broke when a test fails.

---

### Anatomy of a Regression Test

A regression test proves a previously broken input no longer breaks the code.

```python
def test_validate_input_data_handles_string_total_charges():
    """Regression for fix #1 — TypeError when TotalCharges is string dtype."""
    data = pd.DataFrame({
        'customerID':    ['A001'],
        'tenure':        [12],
        'MonthlyCharges': [50.0],
        'TotalCharges':  ['600.0'],   # <-- string, not float
        # ... all required columns ...
    })

    # Must not raise TypeError
    is_valid, errors = validate_input_data(data)

    # TotalCharges='600.0' is within range [0, 15000] — should pass
    assert is_valid is True
```

The docstring references the fix number. When this test fails in the future, the reader immediately knows where to look in the debugging log for context.

---

### Anatomy of an Integration Test

```python
def test_predict_from_file_produces_valid_output(tmp_path):
    """Full pipeline: raw CSV in → predictions CSV out."""
    input_file = pathlib.Path('tests/fixtures/sample_customers.csv')
    output_file = tmp_path / 'predictions.csv'

    predict_from_file(str(input_file), str(output_file))

    result = pd.read_csv(output_file)

    # Schema assertions
    assert 'CustomerID' in result.columns
    assert 'Churn_Probability' in result.columns
    assert 'Predicted_Churn' in result.columns
    assert 'Risk_Level' in result.columns

    # Value range assertions
    assert result['Churn_Probability'].between(0, 1).all()
    assert result['Predicted_Churn'].isin([0, 1]).all()
    assert result['Risk_Level'].isin(['Low', 'Medium', 'High']).all()
```

`tmp_path` is a pytest built-in fixture that creates a temporary directory for the test — output files written here are automatically cleaned up after the test run.

---

### pytest Configuration

pytest is configured in `pyproject.toml` rather than a separate `pytest.ini` to keep configuration in one file:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- `testpaths` — tells pytest where to look for tests (avoids it scanning the whole project)
- `pythonpath` — adds `src/` to `sys.path` so test files can `import predict` without path manipulation

With this in place, running the full suite is simply:

```bash
pytest
```

And running only unit tests (fast, no artifacts needed):

```bash
pytest tests/test_predict.py tests/test_preprocess.py -v
```

---

### What Not to Test

- **`load_data`, `save_pickle`, `load_pickle`** — these are thin wrappers around stdlib/pandas I/O. Testing them means testing pandas, not your code.
- **Model accuracy** — whether ROC-AUC is above a threshold is a monitoring concern, not a unit test concern. It changes with every retrain.
- **Exact prediction values** — the model output will change when artifacts are regenerated. Assert output *shape and schema*, not specific probabilities.

---

## Artifact Lifecycle Management

### The Problem: Code Changes Don't Automatically Update Artifacts

The preprocessing and training scripts produce serialised artifacts — `encoder.pkl`, `scaler.pkl`, `best_model.pkl` — that are saved to disk and loaded at inference time. These artifacts are snapshots of fitted objects taken at a specific point in time.

This creates a class of silent failure: **code changes to `preprocess.py` or `train.py` do not automatically propagate into the saved artifacts.** The pipeline will continue running using the old fitted objects until someone explicitly regenerates them.

This project hit this exact issue. Fix #2 added `handle_unknown='ignore'` to the `OneHotEncoder` initialisation in `preprocess.py`:

```python
# Fix #2: handle_unknown='ignore' — encoder raised ValueError on unseen categories
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
```

The source code was correct from that point forward. But the artifact on disk was unchanged:

```python
# Checking the stored artifact revealed the old setting was still in place
import pickle
enc = pickle.load(open('data/processed/encoder.pkl', 'rb'))
print(enc.handle_unknown)  # → 'error'  ← stale
```

This meant the production prediction pipeline would still raise `ValueError` on any unseen category, despite the code appearing to be fixed.

---

### How to Detect Stale Artifacts

There is no automatic detection — you have to check deliberately. The three approaches from cheapest to most thorough:

**1. Inspect the artifact directly**

Load the artifact and check its parameters against what the current source code configures:

```python
import pickle
enc = pickle.load(open('data/processed/encoder.pkl', 'rb'))
print(enc.handle_unknown)   # should be 'ignore'
print(enc.sparse_output)    # should be False

scaler = pickle.load(open('data/processed/scaler.pkl', 'rb'))
print(scaler.feature_names_in_)  # should match current NUMERICAL_COLUMNS
```

**2. Check artifact modification timestamps against source file timestamps**

If `encoder.pkl` is older than `preprocess.py`, the artifact predates the latest code change:

```bash
ls -lt data/processed/encoder.pkl src/preprocess.py
```

**3. Run the integration tests**

`tests/test_integration.py` exercises the full `predict_from_file` pipeline end-to-end on the fixture. If artifacts are stale in a way that causes runtime failures, integration tests will catch it:

```bash
.venv/bin/python -m pytest tests/test_integration.py -v
```

---

### When to Regenerate Artifacts

Regeneration is required whenever any of the following change:

| Change | Affected artifacts |
|---|---|
| Column list in `config.py` (`NUMERICAL_COLUMNS`, `CATEGORICAL_COLUMNS`) | `scaler.pkl`, `encoder.pkl`, all `.parquet` splits |
| `StandardScaler` or `OneHotEncoder` constructor arguments in `preprocess.py` | `scaler.pkl`, `encoder.pkl` |
| Imputation logic or `dropna` behaviour in `preprocess.py` | All `.parquet` splits, `scaler.pkl`, `encoder.pkl` |
| `train_test_split` parameters (`test_size`, `random_state`) | All `.parquet` splits, `best_model.pkl` |
| Model hyperparameters in `MODEL_CONFIGS` in `config.py` | `best_model.pkl`, `model_comparison_results.csv` |
| Raw dataset replaced or updated | Everything |

Changes that do **not** require regeneration: docstrings, logging messages, `predict.py` edge case handling, test files, documentation.

---

### How to Regenerate

The full pipeline is orchestrated by a single command:

```bash
python3 src/run_pipeline.py
```

This runs all three stages in order:

1. **Preprocessing** (`src/preprocess.py`) — validates schema, scales, encodes, saves `.parquet` splits and `.pkl` artifacts
2. **Training** (`src/train.py`) — fits LR, RF, GB; selects best by ROC-AUC; saves `best_model.pkl`
3. **Prediction** (`src/run_pipeline.py`) — generates `results/test_predictions.csv` on the test split

To skip a stage (e.g. retraining only after preprocessing is already fresh):

```bash
python3 src/run_pipeline.py --skip-preprocessing
python3 src/run_pipeline.py --skip-preprocessing --skip-training  # predictions only
```

---

### Verifying the Regeneration Succeeded

After running the pipeline, confirm the key artifact parameters match the current source:

```bash
.venv/bin/python -c "
import pickle
enc = pickle.load(open('data/processed/encoder.pkl', 'rb'))
print('encoder handle_unknown:', enc.handle_unknown)   # expect: ignore

scaler = pickle.load(open('data/processed/scaler.pkl', 'rb'))
print('scaler n_features:', scaler.n_features_in_)     # expect: 3

model = pickle.load(open('models/best_model.pkl', 'rb'))
print('model type:', type(model).__name__)
"
```

Expected output after this project's regeneration run:

```
encoder handle_unknown: ignore
scaler n_features: 3
model type: GradientBoostingClassifier
```

Then run the unit and integration tests to confirm the full pipeline is coherent:

```bash
.venv/bin/python -m pytest -v
```

---

### Thought Process: Why This Is Easy to Miss

The root cause is that `.pkl` files are gitignored. They exist only on the local filesystem and are not versioned alongside the source code that produces them. This decoupling is intentional — model artifacts can be large and change frequently — but it means:

- A new contributor who clones the repo and runs `predict.py` without first running the pipeline will get an error or, worse, silently wrong results from a stale artifact
- A code review of `preprocess.py` will look correct even if the artifact on disk doesn't match

**Long-term mitigations** (documented here for future reference, not yet implemented):
- Add artifact parameter assertions to `load_model_and_preprocessors()` in `predict.py` — fail fast if loaded encoder has `handle_unknown='error'`
- Tag artifact filenames with a hash or timestamp: `encoder_20260613.pkl` so staleness is visible
- Add a CI step that regenerates artifacts from scratch and runs the full test suite on every push to `main`

---

## Class Imbalance, Hyperparameter Tuning, and Threshold Analysis

### The Problem: Low Recall at Default Threshold

The baseline models trained with default settings produced a Gradient Boosting model with ROC-AUC of **0.8340** but recall of only **0.500** — meaning half of all actual churners were missed. In a churn prediction context this is costly: a missed churner is a customer who leaves without any retention intervention.

The root cause is **class imbalance**. The Telco dataset has roughly 73% non-churn / 27% churn. All three baseline models were trained treating every sample equally, so the models learned a bias toward predicting the majority class (no churn). The default classification threshold of 0.5 compounds this — a model that assigns churners a probability of 0.35 will still classify them as non-churn.

---

### Stage 1+2: Baseline vs Balanced Variants

The fix was to train two parallel sets of models:

**Baseline** — default settings, no class weighting.

**Balanced** — `class_weight='balanced'` for `LogisticRegression` and `RandomForestClassifier`. This automatically reweights each training sample so the minority class contributes proportionally more to the loss.

`GradientBoostingClassifier` does not support `class_weight` as a constructor argument. Instead, `compute_sample_weight('balanced', y_train)` from `sklearn.utils.class_weight` generates per-sample weights that are passed to `model.fit(..., sample_weight=weights)`. The effect is equivalent.

**Results:**

| Model | Stage | ROC-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| logistic_regression | baseline | 0.8320 | 0.516 | 0.621 | 0.564 |
| random_forest | baseline | 0.8098 | 0.463 | 0.607 | 0.525 |
| gradient_boosting | baseline | 0.8340 | 0.500 | 0.630 | 0.557 |
| logistic_regression_balanced | balanced | 0.8317 | **0.791** | 0.498 | 0.611 |
| random_forest_balanced | balanced | 0.8096 | 0.610 | 0.536 | 0.571 |
| gradient_boosting_balanced | balanced | 0.8320 | **0.775** | 0.501 | 0.609 |

**Key insight:** Class weighting dramatically improved recall (0.516 → 0.791 for LR balanced) with only a small drop in ROC-AUC. The trade-off is lower precision — the model now flags more non-churners as churners. Whether this is acceptable depends on the cost of a false positive (unnecessary retention offer) vs. a false negative (missed churner).

---

### Stage 3: GridSearchCV on the Best Model

The best pre-tuning model by ROC-AUC was `gradient_boosting` (baseline, 0.8340). `GridSearchCV` with 5-fold cross-validation was run over:

```python
PARAM_GRIDS['gradient_boosting'] = {
    'n_estimators':  [100, 200],
    'max_depth':     [3, 4, 5],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample':     [0.8, 1.0]
}
```

Scoring metric: `roc_auc` (maximises discrimination across all thresholds).

**Best parameters found:**

```
learning_rate=0.05, max_depth=3, n_estimators=100, subsample=1.0
```

**Tuned model result:** ROC-AUC **0.8347** (+0.0007), Recall 0.476 at default threshold.

The tuning gain in ROC-AUC is marginal. This is expected — Gradient Boosting with 100 trees at depth 3 is already a well-specified model on a 5,000-row dataset. Deeper trees or more estimators tend to overfit without regularisation.

The `train.py` logic selects the tuned model as overall best by ROC-AUC:

```python
best_name = results_df.loc[results_df['roc_auc'].idxmax(), 'model']
# → 'gradient_boosting_tuned'
```

---

### Stage 4: Threshold Analysis

Lowering the classification threshold shifts the precision/recall tradeoff without retraining. At threshold `t`, a customer is predicted to churn if `predict_proba(X)[:,1] >= t`.

**Full precision/recall sweep on `gradient_boosting_tuned`:**

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.30 | 0.522 | 0.773 | 0.623 |
| 0.35 | 0.549 | 0.687 | 0.610 |
| 0.40 | 0.595 | 0.647 | 0.620 |
| 0.45 | 0.615 | 0.551 | 0.581 |
| **0.50** | **0.647** | **0.476** | **0.549** ← default |
| 0.55 | 0.671 | 0.409 | 0.508 |
| 0.60 | 0.712 | 0.350 | 0.470 |

**Best F1 threshold:** 0.30 → precision=0.522, recall=0.773, F1=0.623

**Recommended operating point:** threshold **0.35** provides a better precision/recall balance than 0.30 (precision 0.549 vs 0.522) while still substantially improving recall over the 0.50 default (0.687 vs 0.476). This is the threshold documented in `docs/usage.md`.

---

### Thought Process: Why Not Use the Balanced Model as the Saved Artifact?

The saved `best_model.pkl` is `gradient_boosting_tuned` (baseline, no class weighting) rather than `logistic_regression_balanced` (highest recall). The reasoning:

1. **ROC-AUC is threshold-independent** — it measures discrimination across the entire operating range. The tuned GB has the best ROC-AUC (0.8347), meaning it has the best raw signal regardless of threshold.
2. **Threshold tuning achieves the same recall improvement** — lowering the GB tuned threshold to 0.30 gives recall 0.773, close to the balanced LR's 0.791, while preserving higher ROC-AUC.
3. **One model, configurable threshold** — it's operationally simpler to deploy one model and document a threshold recommendation than to swap which artifact is "best" depending on the business objective.

If the business objective changes to maximise recall above all else (e.g. very high cost of missed churners), `logistic_regression_balanced` at threshold 0.30 is the right choice. The full comparison table in `models/model_comparison_results.csv` preserves all variants for this decision.

---

### `scripts/threshold_analysis.py` — Reusable Threshold Sweep Script

The threshold analysis results above were produced by `scripts/threshold_analysis.py`, a standalone CLI tool that can be re-run any time the model artifact is updated.

#### Why a separate script?

The threshold sweep doesn't belong in `train.py` — it's a post-hoc analysis tool, not part of the training pipeline. Keeping it in `scripts/` makes it:
- Runnable independently after any retrain without re-executing GridSearchCV
- Importable as a function (`run_threshold_analysis()`) from notebooks or other scripts
- Configurable via CLI flags without touching source code

#### How it works

```
scripts/threshold_analysis.py
    ├── load best_model.pkl + test split from data/processed/
    ├── call model.predict_proba(X_test)[:, 1]  → raw churn probabilities
    ├── sklearn.metrics.precision_recall_curve() → full (precision, recall, threshold) arrays
    ├── sample the curve at --step intervals (default 0.05)
    └── print key operating points + full sweep table
```

`precision_recall_curve` returns one point per unique predicted probability — typically thousands of rows. The script samples this at a coarser step for readability and to make the CSV output useful.

#### Key implementation decisions

**Why `precision_recall_curve` instead of a manual loop?**
Computing metrics by iterating over thresholds and calling `classification_report` each time is slow and redundant. `precision_recall_curve` computes the full curve in one pass over sorted probabilities — O(n log n) total instead of O(n × k) for k thresholds.

**Why `full[full["threshold"] >= t].iloc[0]`?**
The curve array has one entry per decision boundary. To find precision/recall at a given threshold `t`, we want the first row where `threshold >= t` — this gives the operating point immediately at or above `t`, which matches what `model.predict(X, threshold=t)` would produce.

**Why `+ 1e-9` in the F1 denominator?**
Prevents division-by-zero when both `precision` and `recall` are 0 (at very high thresholds near 1.0). Using `np.finfo(float).eps` would work too but `1e-9` is more readable.

#### Usage

```bash
# Default: print sweep at step=0.05
python scripts/threshold_analysis.py

# Finer sweep saved to CSV
python scripts/threshold_analysis.py --step 0.02 --output results/threshold_sweep.csv
```

#### Output interpretation

The three key lines printed at the top:

```
Default (0.50):  precision=0.647  recall=0.476  f1=0.549
Best F1:         threshold=0.301  precision=0.524  recall=0.770  f1=0.623
Recall >= 0.70:  threshold=0.333  precision=0.539  recall=0.706  f1=0.611
```

- **Default (0.50)** — what you get with `model.predict()` out of the box
- **Best F1** — the threshold that maximises the harmonic mean of precision and recall
- **Recall >= 0.70** — the highest-precision operating point that still catches ≥70% of churners; useful when there's a minimum recall SLA

---

### Implementation Notes

**`config.py` changes:**
- Added `MODEL_CONFIGS_BALANCED` — parallel dict to `MODEL_CONFIGS` with `class_weight='balanced'` for LR and RF
- Added `PARAM_GRIDS` — GridSearchCV parameter grids for all three model families
- Both typed as `dict[str, dict[str, Any]]` (same pattern as `MODEL_CONFIGS`)

**`train.py` changes:**
- `_build_models()` helper maps config name → sklearn class for clean instantiation
- Stages 1+2 loop over `{**MODEL_CONFIGS, **MODEL_CONFIGS_BALANCED}` in one pass
- `compute_sample_weight('balanced', y_train)` passed as `fit_kwargs` for `gradient_boosting_balanced` only
- Stage 3 uses `best_pre_tune.replace('_balanced', '')` to identify the base model family for the param grid
- Results DataFrame gains a `stage` column (`baseline` / `balanced` / `tuned`) for easy filtering

**`scripts/threshold_analysis.py`:**
- Standalone CLI wrapping `run_threshold_analysis(step, output)`
- Reuses `load_pickle` / `load_parquet` from `utils.py` — no duplicated I/O logic
- `sys.path.insert(0, "src/")` allows clean imports from `config` and `utils` without installing the package

---

*This guide provides the foundation for understanding, modifying, and extending the Telco Churn Analytics project.*
