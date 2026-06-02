# Usage Guide

## Overview
This guide explains how to use the Telco Churn Analytics project, including running notebooks, processing data, and understanding the workflow.

## Quick Start

### 1. Run the Main Analysis Notebook
```bash
jupyter notebook notebooks/customer_churn.ipynb
```

This notebook performs the complete data preprocessing pipeline and generates processed parquet files.

### 2. Expected Outputs
After running the notebook, you'll find:
- `train.parquet` - Processed training data (80% of dataset)
- `test.parquet` - Processed test data (20% of dataset)

## Detailed Workflow

### Step 1: Data Loading
The notebook loads the customer churn dataset from the specified path. Update the path in the notebook if your dataset location differs:

```python
data = pd.read_csv("/path/to/your/customer_churn_dataset-training-master.csv")
```

### Step 2: Data Inspection
- Check dataset shape: `data.shape`
- View first few rows: `data.head()`
- Review statistics: `data.describe()`
- Identify missing values: `data.isnull().sum()`

### Step 3: Data Cleaning
- Remove rows with null values (1 row in current dataset)
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
Applied to: Age, Tenure, Usage Frequency, Support Calls, Payment Delay, Total Spend, Last Interaction

Process:
1. Fit scaler on training data
2. Transform training data
3. Transform test data using fitted scaler

#### Categorical Features (OneHotEncoder)
Applied to: Subscription Type, Contract Length

Process:
1. Fit encoder on training data
2. Transform training data
3. Transform test data using fitted encoder
4. Convert sparse matrix to dense format

### Step 7: Data Integration
Combine processed numerical and categorical features with CustomerID for tracking.

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
X_train = train_data.drop('CustomerID', axis=1)
y_train = y_train  # Use labels from original split

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
X_test = test_data.drop('CustomerID', axis=1)
predictions = model.predict(X_test)
```

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
Modify the numerical_columns or categorical_columns lists:
```python
numerical_columns = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls',
                     'Payment Delay', 'Total Spend', 'Last Interaction', 'NewFeature']
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
