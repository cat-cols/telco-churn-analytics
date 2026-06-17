# Usage Guide

## Overview
This guide explains how to use the Telco Churn Analytics project, including running notebooks, processing data, and understanding the workflow.

## Quick Start

### 1. Run the Preprocessing Pipeline
```bash
python3 src/run_pipeline.py
```

This runs the complete data preprocessing pipeline and generates processed parquet files. For exploratory analysis, open `notebooks/01_eda.ipynb`.

### 2. Expected Outputs
After running the notebook, you'll find:
- `train.parquet` - Processed training data (80% of dataset)
- `test.parquet` - Processed test data (20% of dataset)

## Detailed Workflow

### Step 1: Data Loading
The notebook loads the customer churn dataset from the specified path. Update the path in the notebook if your dataset location differs:

```python
data = pd.read_csv("data/raw/telco_customer_churn.csv")
```

### Step 2: Data Inspection
- Check dataset shape: `data.shape`
- View first few rows: `data.head()`
- Review statistics: `data.describe()`
- Identify missing values: `data.isnull().sum()`

### Step 3: Data Cleaning
- Convert `TotalCharges` from string to numeric (`pd.to_numeric(errors='coerce')`)
- Remove rows with null values (11 rows with blank `TotalCharges`, all `tenure = 0`)
- Verify no missing values remain

### Step 4: Feature Engineering
The dataset is split into:
- **Features (X)**: All columns except the target
- **Labels (y)**: Churn column (target variable)

### Step 5: Train/Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)
```

### Step 6: Feature Preprocessing

#### Numerical Features (StandardScaler)
Applied to: SeniorCitizen, tenure, MonthlyCharges, TotalCharges

Process:
1. Fit scaler on training data
2. Transform training data
3. Transform test data using fitted scaler

#### Categorical Features (OneHotEncoder)
Applied to: gender, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod

Process:
1. Fit encoder on training data
2. Transform training data
3. Transform test data using fitted encoder
4. Convert sparse matrix to dense format

### Step 7: Data Integration
Combine processed numerical and categorical features with customerID for tracking.

### Step 8: Export Processed Data
Save processed datasets as parquet files for efficient storage and loading:
```python
X_train_transf.to_parquet("train.parquet")
X_test_trans.to_parquet("test.parquet")
```

## Using Processed Data

### Load Processed Data
```python
import pandas as pd

train_data = pd.read_parquet("train.parquet")
test_data = pd.read_parquet("test.parquet")
```

### Model Training Example
```python
from sklearn.ensemble import RandomForestClassifier

# Separate features and target
X_train = train_data.drop('customerID', axis=1)
y_train = y_train  # Use labels from original split

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
X_test = test_data.drop('customerID', axis=1)
predictions = model.predict(X_test)
```

## Making Predictions (`src/predict.py`)

Once the model and preprocessors are trained, score new customers from a raw CSV:

```bash
python3 src/predict.py --input data/new_customers.csv --output results/predictions.csv
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | (required) | Raw customer CSV (same columns as the training data). |
| `--output` | none | Where to write predictions; also writes a `_summary.txt`. Prints to stdout if omitted. |
| `--threshold` | `0.5` | Probability cutoff used to convert `Churn_Probability` into the `Predicted_Churn` 0/1 label. |
| `--verbose` | off | Enable INFO-level logging (edge-case handling, imputation report, etc.). |

The output always includes the raw `Churn_Probability` and a `Risk_Level`
(Low / Medium / High), so changing `--threshold` only affects the binary
`Predicted_Churn` column — you never lose the underlying probability.

### Choosing the `--threshold`

The default `0.5` is rarely the right business choice for churn. Because the data is
imbalanced (~27% churn), a 0.5 cutoff **maximizes accuracy but misses about half of
real churners** (low recall). Lowering the threshold flags more customers as at-risk:
**recall goes up, precision goes down**. The right value depends on the relative cost
of a missed churner vs. a wasted retention offer.

**Reference operating points** (saved best model, evaluated on the test set — regenerate
with the command below):

| Threshold | Precision | Recall | F1 | When to use |
|----------:|----------:|-------:|----:|-------------|
| 0.50 (default) | 0.65 | 0.48 | 0.55 | Accuracy-oriented; under-detects churn. |
| **0.30 (best F1)** | 0.52 | 0.77 | **0.62** | Best precision/recall balance — recommended default for retention. |
| 0.33 (recall ≥ 0.70) | 0.54 | 0.71 | 0.61 | Highest precision while still catching ≥70% of churners. |
| 0.20 | 0.46 | 0.85 | 0.60 | Aggressive: catch most churners, accept more false alarms. |
| 0.65 | 0.77 | 0.28 | 0.41 | Conservative: act only on high-confidence churners. |

Rules of thumb:

- **Missed churner costs more than a wasted offer** (the usual case) → choose a
  **lower** threshold (e.g. `0.30`) to maximize recall.
- **Retention budget is tight / offers are expensive** → choose a **higher** threshold
  to maximize precision and only target high-confidence cases.
- **No strong preference** → use the **F1-maximizing** threshold (~`0.30` here), which
  balances the two.

Example — run a recall-oriented batch at threshold 0.30:

```bash
python3 src/predict.py --input data/new_customers.csv \
    --output results/predictions.csv --threshold 0.30
```

### Re-deriving the thresholds for your model

Do not hard-code the numbers above if you retrain. Regenerate the full
precision/recall/F1 sweep (and the key operating points) against the current
`models/best_model.pkl` with:

```bash
python3 scripts/threshold_analysis.py                 # prints sweep + key points
python3 scripts/threshold_analysis.py --step 0.02 \
    --output results/threshold_sweep.csv              # finer sweep, saved to CSV
```

The script reports the default (0.50), the F1-maximizing threshold, and the
highest-precision threshold achieving recall ≥ 0.70 — pick the one matching your
business objective and pass it to `predict.py --threshold`. The same trade-off is
visualized in the "Threshold Tuning" section of `notebooks/03_modeling.ipynb`.

### Edge-case behaviour

`predict.py` is designed to be robust to messy real-world input. Rather than crashing,
it handles the following cases automatically and records what it did in the logs (run
with `--verbose` to see them):

| Situation | What the pipeline does |
|-----------|------------------------|
| **New customer (`tenure = 0`)** | Sets `TotalCharges` to `0` (no billing history yet) and adds an `Is_New_Customer` flag. |
| **Missing `TotalCharges`** (blank/whitespace) | Coerced to numeric; missing values imputed with `0`. |
| **Missing `MonthlyCharges`** | Imputed with the training median (`65.0`). |
| **Missing `tenure`** | Imputed with the training median (`29`). |
| **Row missing a critical field** (`customerID`, `tenure`, `MonthlyCharges`) | Dropped (cannot be scored reliably); count is logged. |
| **Unseen category value** (e.g. a new `PaymentMethod`) | Mapped to an `"Unknown"` bucket; the encoder uses `handle_unknown='ignore'`, so it will not crash. |
| **Out-of-range numeric value** (e.g. `MonthlyCharges` > 200) | **Flagged, not removed** — a `*_outlier_flag` column and a combined `Has_Outlier_Flag` are added for review. |
| **Missing required columns / duplicate `customerID`s** | Reported as validation warnings; processing continues where possible. |

Two validation layers bracket the prediction:

- **Input validation** (`validate_input_data`) runs first. It checks required columns,
  value ranges (`tenure` 0–72, `MonthlyCharges` 0–200, `TotalCharges` 0–15000), and
  duplicate IDs. Problems are **logged as warnings but do not stop the run** — the
  pipeline tries to handle them in preprocessing.
- **Output validation** (`validate_output_schema`) runs last, immediately **before
  saving**. It enforces that the result has the required columns and sensible values
  (`Churn_Probability` ∈ [0, 1], `Predicted_Churn` ∈ {0, 1}, `Risk_Level` ∈
  {Low, Medium, High}). If the output is malformed, it **raises an error instead of
  writing a bad file**.

The output always contains `CustomerID`, `Churn_Probability`, `Predicted_Churn`, and
`Risk_Level`, plus `Is_New_Customer` and `Has_Outlier_Flag` when those situations
occur. A `*_summary.txt` is written alongside the predictions CSV with totals, the
predicted churn rate, and the risk-level distribution.

## Notebook Structure

### Cell-by-Cell Overview
1. **Dependencies**: Install required packages
2. **Data Loading**: Load raw dataset
3. **Initial Inspection**: Check shape and sample data
4. **Statistical Summary**: View descriptive statistics
5. **Missing Values**: Identify null values
6. **Data Cleaning**: Remove null rows
7. **Categorical Analysis**: Examine value distributions
8. **Feature Separation**: Split features and labels
9. **Train/Test Split**: Create training and test sets
10. **Numerical Scaling**: Apply StandardScaler
11. **Categorical Encoding**: Apply OneHotEncoder
12. **Data Integration**: Combine processed features
13. **Export**: Save processed data

## Customization

### Change Train/Test Split Ratio
```python
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.3, random_state=42  # 70/30 split
)
```

### Add Additional Features
Modify the `NUMERICAL_COLUMNS` or `CATEGORICAL_COLUMNS` lists in `src/config.py`:
```python
NUMERICAL_COLUMNS = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'NewFeature']
```

### Use Different Scalers
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
```

## Best Practices

1. **Always use the same random_state** for reproducibility
2. **Fit transformers on training data only**, then transform test data
3. **Keep CustomerID** for tracking but exclude from modeling
4. **Save processed data** in efficient formats (parquet recommended)
5. **Document any changes** to the preprocessing pipeline

## Troubleshooting

### Issue: File not found
- **Solution**: Verify dataset path in the notebook matches your actual file location

### Issue: Memory errors
- **Solution**: Process data in chunks or use a machine with more RAM

### Issue: Encoder/scaler mismatch
- **Solution**: Ensure you fit on training data and use the same fitted object for test data

### Issue: Parquet import errors
- **Solution**: Install pyarrow: `pip install pyarrow`

## Next Steps
After preprocessing, refer to [methodology.md](methodology.md) for detailed analysis approach and [model_evaluation.md](model_evaluation.md) for model evaluation guidelines.
