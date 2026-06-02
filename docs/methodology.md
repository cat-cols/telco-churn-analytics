# Methodology

## Overview
This document outlines the analytical methodology used for the Telco Customer Churn prediction project, including data preprocessing, feature engineering, and the modeling approach.

## Problem Statement
Predict customer churn for a telecommunications company using customer demographic, usage, and subscription data. This is a binary classification problem where the target variable is whether a customer churned (1) or not (0).

## Data Preprocessing Pipeline

### 1. Data Loading and Inspection
- Load dataset from CSV/Excel source
- Initial inspection of dataset shape and structure
- Review statistical summary of numerical features
- Identify data quality issues (missing values, outliers)

### 2. Data Cleaning
**Missing Value Handling:**
- Identified 1 row with null values across all columns (row index 199,295)
- Removed the incomplete row using `data.dropna()`
- Final dataset: 440,832 complete records

**Rationale:** Complete case analysis is appropriate here since only 1 row was affected (0.0002% of data).

### 3. Exploratory Data Analysis (EDA)
**Categorical Variable Distribution:**
- Subscription Type: Examined value counts and proportions
  - Basic: 32.4%
  - Standard: 33.8%
  - Premium: 33.7%
- Contract Length: Examined value counts and proportions
  - Annual: 40.2%
  - Quarterly: 40.0%
  - Monthly: 19.8%

**Numerical Variable Statistics:**
- Computed descriptive statistics (mean, std, min, max, quartiles)
- Identified ranges and distributions for all numerical features
- Noted churn rate: 56.7% (indicating class imbalance)

### 4. Feature Engineering

#### Feature Separation
- **Features (X):** All columns except the target variable
- **Target (y):** Churn column (binary classification)

#### Feature Categorization
**Numerical Features (7 variables):**
- Age
- Tenure
- Usage Frequency
- Support Calls
- Payment Delay
- Total Spend
- Last Interaction

**Categorical Features (2 variables):**
- Subscription Type
- Contract Length

**Identifier:**
- CustomerID (retained for tracking, excluded from modeling)

### 5. Train/Test Split
**Parameters:**
- Test size: 20% (80% training, 20% testing)
- Random state: 42 (for reproducibility)
- Stratification: Not applied (could be considered for imbalanced target)

**Rationale:** Standard 80/20 split provides sufficient training data while maintaining a robust test set for evaluation.

### 6. Feature Transformation

#### Numerical Feature Scaling
**Method:** StandardScaler (Z-score normalization)

**Process:**
1. Fit scaler on training data only (computes mean and std)
2. Transform training data using fitted scaler
3. Transform test data using the same fitted scaler

**Formula:** z = (x - μ) / σ
- μ = mean of training data
- σ = standard deviation of training data

**Rationale:** StandardScaler is appropriate when features have different scales and we assume roughly Gaussian distributions. It centers features around 0 with unit variance, preventing features with larger scales from dominating model learning.

#### Categorical Feature Encoding
**Method:** OneHotEncoder

**Process:**
1. Fit encoder on training data (identifies unique categories)
2. Transform training data to binary columns
3. Transform test data using fitted encoder
4. Convert sparse matrix to dense format for DataFrame creation

**Output:** Binary columns for each category
- Subscription Type → Subscription Type_Basic, Subscription Type_Standard, Subscription Type_Premium
- Contract Length → Contract Length_Annual, Contract Length_Quarterly, Contract Length_Monthly

**Rationale:** OneHotEncoder is appropriate for nominal categorical variables with no inherent ordering. It creates binary features that can be effectively used by most machine learning algorithms.

### 7. Data Integration
Combine processed components:
- CustomerID (for tracking)
- Scaled numerical features
- Encoded categorical features

Concatenation along axis=1 (column-wise) to create final feature matrix.

### 8. Data Export
Save processed datasets in Parquet format:
- `train.parquet`: Processed training features
- `test.parquet`: Processed test features

**Rationale:** Parquet format provides efficient storage and fast loading for large datasets, with built-in compression and schema preservation.

## Modeling Approach (Future Work)

### Recommended Algorithms
Based on the problem characteristics:
- **Logistic Regression**: Baseline model, interpretable
- **Random Forest**: Handles non-linear relationships, feature importance
- **Gradient Boosting (XGBoost/LightGBM)**: High performance on tabular data
- **Neural Networks**: For complex pattern recognition

### Evaluation Metrics
Given the class imbalance (56.7% churn rate):
- **Primary:** AUC-ROC (robust to imbalance)
- **Secondary:** Precision, Recall, F1-Score
- **Additional:** Confusion Matrix, Accuracy

### Cross-Validation Strategy
- K-fold cross-validation (k=5 or k=10)
- Stratified sampling to maintain churn rate distribution
- Repeated runs for stability assessment

## Assumptions and Limitations

### Assumptions
1. Data is representative of the customer population
2. Missing values are missing completely at random (MCAR)
3. Feature relationships are consistent across train and test sets
4. CustomerID has no predictive power (used only for tracking)

### Limitations
1. Class imbalance may require specialized techniques (SMOTE, class weights)
2. No temporal information (time-series analysis not possible)
3. CustomerID and Gender were not used in feature engineering
4. No feature selection performed (all features used)

## Reproducibility
- Random state fixed at 42 for train/test split
- Scaler and encoder fit on training data only
- All transformations applied consistently to train and test sets
- Process documented in Jupyter notebook with cell-by-cell execution

## Future Improvements
1. **Feature Selection:** Use recursive feature elimination or importance-based selection
2. **Hyperparameter Tuning:** Grid search or random search for model optimization
3. **Ensemble Methods:** Combine multiple models for improved performance
4. **Class Imbalance Handling:** Implement SMOTE or class weighting
5. **Cross-Validation:** Implement robust CV strategy
6. **Feature Engineering:** Create interaction terms or polynomial features
7. **Model Interpretation:** SHAP values or LIME for explainability
