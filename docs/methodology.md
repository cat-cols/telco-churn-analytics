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
- `TotalCharges` is stored as a string and contains 11 blank values, all belonging to new customers with `tenure = 0`
- Converted `TotalCharges` to numeric with `pd.to_numeric(errors='coerce')`, producing 11 `NaN` values
- Removed the 11 incomplete rows after conversion
- Final dataset: 7,032 complete records (from 7,043 raw)

**Rationale:** Complete case analysis is appropriate here since only 11 rows were affected (0.16% of data).

### 3. Exploratory Data Analysis (EDA)
**Categorical Variable Distribution:**
- Contract: Examined value counts and proportions
  - Month-to-month: 55.0%
  - Two year: 24.1%
  - One year: 20.9%
- PaymentMethod: Examined value counts and proportions
  - Electronic check: 33.6%
  - Mailed check: 22.9%
  - Bank transfer (automatic): 21.9%
  - Credit card (automatic): 21.6%
- InternetService: Fiber optic 44.0%, DSL 34.4%, No 21.7%

**Numerical Variable Statistics:**
- Computed descriptive statistics (mean, std, min, max, quartiles)
- Identified ranges and distributions for all numerical features
- Noted churn rate: 26.54% (indicating class imbalance)

**Churn Rate Breakdown by Key Features (Telco dataset, n=7,043, overall churn 26.54%):**

Segment churn rates computed in `notebooks/01_eda.ipynb` highlight the strongest churn drivers:

*By Contract type:*

| Contract | Count | Churned | Churn Rate |
|----------|------:|--------:|-----------:|
| Month-to-month | 3,875 | 1,655 | 42.71% |
| One year | 1,473 | 166 | 11.27% |
| Two year | 1,695 | 48 | 2.83% |

*By tenure band (months):*

| Tenure Band | Count | Churned | Churn Rate |
|-------------|------:|--------:|-----------:|
| 0-12 | 2,186 | 1,037 | 47.44% |
| 13-24 | 1,024 | 294 | 28.71% |
| 25-48 | 1,594 | 325 | 20.39% |
| 49-72 | 2,239 | 213 | 9.51% |

*By PaymentMethod:*

| Payment Method | Count | Churned | Churn Rate |
|----------------|------:|--------:|-----------:|
| Electronic check | 2,365 | 1,071 | 45.29% |
| Mailed check | 1,612 | 308 | 19.11% |
| Bank transfer (automatic) | 1,544 | 258 | 16.71% |
| Credit card (automatic) | 1,522 | 232 | 15.24% |

**Key takeaways:**
- **Contract** is the strongest signal: month-to-month customers churn at 42.71% vs 2.83% for two-year contracts.
- **Tenure** is strongly inverse to churn: new customers (0-12 months) churn at 47.44%, dropping to 9.51% after 4 years.
- **Electronic check** payers churn at 45.29%, roughly 3x the automatic payment methods (~15-17%).

### 4. Feature Engineering

#### Feature Separation
- **Features (X):** All columns except the target variable
- **Target (y):** Churn column (binary classification)

#### Feature Categorization
**Numerical Features (4 variables):**
- SeniorCitizen (0/1 indicator)
- tenure
- MonthlyCharges
- TotalCharges (after string-to-numeric conversion)

**Categorical Features (15 variables):**
- gender
- Partner
- Dependents
- PhoneService
- MultipleLines
- InternetService
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies
- Contract
- PaperlessBilling
- PaymentMethod

**Identifier:**
- customerID (retained for tracking, excluded from modeling)

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
- Contract → Contract_Month-to-month, Contract_One year, Contract_Two year
- PaymentMethod → PaymentMethod_Electronic check, PaymentMethod_Mailed check, PaymentMethod_Bank transfer (automatic), PaymentMethod_Credit card (automatic)

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
Given the class imbalance (26.54% churn rate):
- **Primary:** AUC-ROC (robust to imbalance)
- **Secondary:** Precision, Recall, F1-Score
- **Additional:** Confusion Matrix, Accuracy

### Cross-Validation Strategy
- K-fold cross-validation (k=5 or k=10)
- Stratified sampling to maintain churn rate distribution
- Repeated runs for stability assessment

## Model Interpretability (SHAP)

To explain *how* the best model (Gradient Boosting) makes its predictions, we apply **SHAP
(SHapley Additive exPlanations)** to the test set in `notebooks/04_results.ipynb`. SHAP assigns
every feature a signed contribution (in log-odds) to each individual prediction; contributions
are additive and sum to the model output. We use `TreeExplainer`, the exact algorithm for tree
ensembles.

**Outputs saved to `results/`:**
- `shap_feature_importance.png` — global importance ranked by mean |SHAP|.
- `shap_summary.png` — beeswarm plot showing magnitude *and* direction of each feature's impact.

### Top 5 Churn Drivers

Ranked by mean |SHAP| across all test customers. *Direction* is the sign of the correlation
between the feature's value and its SHAP value (how the feature pushes the prediction):

| Rank | Feature | Mean \|SHAP\| | Direction |
|-----:|---------|-------------:|-----------|
| 1 | `Contract_Month-to-month` | 0.6602 | Increases churn |
| 2 | `tenure` | 0.4410 | Reduces churn (longer tenure → less churn) |
| 3 | `OnlineSecurity_No` | 0.2645 | Increases churn |
| 4 | `TechSupport_No` | 0.2078 | Increases churn |
| 5 | `MonthlyCharges` | 0.2003 | Increases churn (higher charges → more churn) |

### Business Insights

- **Contract type dominates.** Being on a month-to-month contract is by far the strongest churn
  signal — consistent with the EDA segment rates (42.71% churn for month-to-month vs 2.83% for
  two-year). *Action:* incentivize migration to 1- or 2-year contracts.
- **Tenure protects.** The longer a customer has stayed, the less likely they are to churn; risk
  is concentrated in the first year. *Action:* focus retention/onboarding effort on new customers.
- **Missing add-on services raise risk.** Not having `OnlineSecurity` or `TechSupport` both push
  customers toward churn. *Action:* bundle or promote these services, especially to fiber customers.
- **Higher monthly charges increase churn.** Price sensitivity is real. *Action:* review pricing
  and target discounts/loyalty offers at high-bill, high-risk customers.

These SHAP-derived drivers corroborate the correlational EDA findings while adding a per-feature,
direction-aware, model-grounded explanation suitable for stakeholders.

## Assumptions and Limitations

### Assumptions
1. Data is representative of the customer population
2. Missing values are missing completely at random (MCAR)
3. Feature relationships are consistent across train and test sets
4. CustomerID has no predictive power (used only for tracking)

### Limitations
1. Class imbalance may require specialized techniques (SMOTE, class weights)
2. No temporal information (time-series analysis not possible)
3. customerID is excluded from feature engineering (used only for tracking); gender is included as a categorical feature
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
