# Interview Data Quality Guide: Advanced Topics

> **Purpose:** This guide covers the data quality issues that separate junior from senior data scientists in interviews. These are the "gotcha" questions that test whether you understand the difference between textbook ML and real-world deployment.

## Why This Matters

Data scientists spend 80% of their time on data quality issues. In interviews, employers test your ability to:
- **Prevent** costly mistakes (leakage, bias)
- **Detect** problems before they reach production
- **Debug** issues when models fail unexpectedly

## Table of Contents
1. [Data Leakage (The #1 Interview Killer)](#1-data-leakage-the-1-interview-killer)
2. [Target Leakage & Future Information](#2-target-leakage--future-information)
3. [Multicollinearity & Feature Correlation](#3-multicollinearity--feature-correlation)
4. [High Cardinality Categoricals](#4-high-cardinality-categoricals)
5. [Sampling Bias & Selection Effects](#5-sampling-bias--selection-effects)
6. [Temporal Data Issues](#6-temporal-data-issues)
7. [Data Validation Frameworks](#7-data-validation-frameworks)
8. [Measurement Error & Systematic Bias](#8-measurement-error--systematic-bias)
9. [Schema Evolution](#9-schema-evolution)
10. [Interview Question Patterns](#10-interview-question-patterns)

---

## How to Use This Guide

Each section follows this structure:
- **Concept:** What is it and why does it happen?
- **Detection:** How to find it in your data
- **Examples:** Code showing wrong vs right approaches
- **Impact:** Real-world consequences
- **Interview Angle:** How this appears in questions

---

## 1. Data Leakage (The #1 Interview Killer)

### What is Data Leakage?

Data leakage occurs when information from outside the training dataset is used to create the model. This is the single most common reason models fail in production despite excellent validation metrics.

**The Core Problem:** Your model learns patterns that won't exist in production data. When deployed, it makes predictions based on information it won't actually have.

**Real-World Impact:**
- Model shows 95% accuracy in validation
- Drops to 60% in production
- Team spends weeks debugging algorithm
- Root cause: data leakage in preprocessing
- **Cost:** Delayed deployment, lost revenue, damaged credibility

### Why This Happens

Leakage typically occurs in three stages:
1. **Before split:** Preprocessing on entire dataset
2. **During feature engineering:** Using target statistics
3. **In cross-validation:** Improper pipeline structure

### Types of Leakage

**A. Preprocessing Leakage (Most Common - 90% of Cases)**

```python
# ❌ WRONG: Data Leakage
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fit on ENTIRE dataset

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
# Test data influenced training set statistics!

# ✅ CORRECT: No Leakage
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit ONLY on train
X_test_scaled = scaler.transform(X_test)        # Transform test separately

# Save scaler for production!
import joblib
joblib.dump(scaler, 'scaler.pkl')
```

**Why This Matters - The Math:**

When you fit a scaler on the entire dataset:
```
Training set mean = 50, std = 10
Test set mean = 60, std = 15
Full dataset mean = 55, std = 12.5
```

Your scaler uses mean=55, std=12.5 (full dataset)
- Training points get normalized incorrectly
- Test points are normalized using their own statistics (via the full dataset)
- This gives test data an unfair advantage

**Impact:** Leakage can inflate test accuracy by 10-30%, creating false confidence.

**Production Problem:** When you deploy, new data won't have been part of the "full dataset" your scaler was fit on. The statistics will be different, and performance will drop.

**B. Feature Selection Leakage**

**The Problem:** Feature selection uses relationships between features and target to choose the best ones. If you do this on the full dataset, the selector learns about the test set's target relationships.

**Example:** 
- Feature "X" has 0.8 correlation with target in training set
- Feature "X" has 0.9 correlation with target in test set
- Selector picks "X" because it's strong in BOTH
- Model appears to perform well because test set was already "peaked at"

```python
# ❌ WRONG: Select features before split
from sklearn.feature_selection import SelectKBest

selector = SelectKBest(k=10)
X_selected = selector.fit_transform(X, y)  # Uses full data - SEES TEST SET!

X_train, X_test = train_test_split(X_selected)
# Test performance will be inflated

# ✅ CORRECT: Select features after split, using only training data
X_train, X_test, y_train, y_test = train_test_split(X, y)

selector = SelectKBest(k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
# Only sees training relationships

X_test_selected = selector.transform(X_test)  # Use same features
# Test performance is honest
```

**Key Insight:** Any step that learns from the data (scaling, selection, dimensionality reduction, imputation) must happen AFTER the train/test split, using only training data.

**C. Cross-Validation Leakage**

**The Problem:** In k-fold cross-validation, each fold uses different data for train/test. If you preprocess before CV, information from future validation folds leaks into training.

**Visual Explanation:**
```
Without Pipeline (WRONG):
Fold 1: Scale on ALL data → Train on 1-4, Test on 5 (leaked)
Fold 2: Scale on ALL data → Train on 1-3,5, Test on 4 (leaked)
...

With Pipeline (CORRECT):
Fold 1: Train scaler on 1-4 → Scale 1-4 → Train → Test on 5
Fold 2: Train scaler on 1-3,5 → Scale 1-3,5 → Train → Test on 4
...
```

Each fold learns its own scaling statistics from its training data only.

```python
# ❌ WRONG: Pipeline outside CV - SCALER SEES ALL FOLDS
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fits on entire dataset including all CV folds

cv_scores = cross_val_score(model, X_scaled, y, cv=5)
# Leakage: scaler statistics include validation fold data

# ✅ CORRECT: Pipeline inside CV
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])

cv_scores = cross_val_score(pipeline, X, y, cv=5)
# Each fold fits scaler on training portion only
# No leakage between folds
```

**Pro Tip:** Always use `Pipeline` when doing cross-validation. It's the only way to guarantee no leakage.

### Interview Question Pattern

**Q:** "You build a model with 95% accuracy but it only gets 70% in production. What happened?"

**A:** This classic question tests your debugging process. Here's the systematic approach:

**Step 1: Check for Data Leakage**
```
- Did you fit preprocessing on full data before splitting?
- Did you select features before splitting?
- Any feature that uses target information?
```

**Step 2: Check Train/Test Similarity**
```
- Are train/test from same distribution? (drift detection)
- Is test set representative of production?
- Any duplicate rows between train and test?
```

**Step 3: Check Temporal Issues**
```
- Random split on time-series data? (should use time-based)
- Features using future information?
- Seasonality or trends affecting results?
```

**Step 4: Check Production Pipeline**
```
- Is preprocessing identical in training and production?
- Same imputation strategy?
- Feature engineering code matches?
```

**Most Likely Answer:** "Data leakage in preprocessing is the most common cause. I would verify that all preprocessing steps (scaling, imputation, encoding) were fit only on training data, not the full dataset."

---

## 2. Target Leakage & Future Information

### What is Target Leakage?

Target leakage occurs when features contain information that wouldn't be available at the time you're making the prediction. You're essentially using future information to predict the future.

**The Time Machine Problem:**
Imagine trying to predict if a customer will churn next month, but one of your features is "days since last payment." If a customer has already churned, they won't make payments, so this feature will be very large. The model learns: "Large values = churn" but in production, you won't know if they've stopped paying yet!

### Why This is Subtle

Unlike regular data leakage (where you made a coding mistake), target leakage can seem like a "good" feature:
- It has high correlation with the target
- It improves model performance significantly
- It makes intuitive sense

**The Danger:** It only fails in production when you don't have that future information yet.

### Common Scenarios

### Examples

**A. Time-Travel Problem (Most Common)**

**The Mistake:** Using information that only exists after the event you're trying to predict.

**Scenario:** Predict churn for January on December 31st

```python
# ❌ WRONG: Feature uses future data (January information)
df['days_since_last_payment']  
# If customer churned Jan 15, this would be calculated on Feb 1
# Value = 17 days (very high = strong churn signal)
# But on Dec 31, you don't know they've stopped paying!

df['cancellation_request_date']  
# This IS the churn event itself - perfect predictor!

# ✅ CORRECT: Only use features available at prediction time (Dec 31)
df['days_since_last_payment_as_of_dec_31']
# Uses only December data - valid for predicting January

df['payment_delays_last_3_months']  
# Oct, Nov, Dec delays - available on Dec 31

# Additional valid features
df['tenure_as_of_dec_31']  # Months as customer up to Dec 31
df['total_charges_as_of_dec_31']  # Spending up to Dec 31
```

**Rule of Thumb:** The timestamp of your features must be ≤ the prediction timestamp.

**B. Derived from Target (The GroupBy Trap)**

**The Mistake:** Creating features using groupby on the target variable.

**Why This Leaks:** When you group by the target, you're using the answer to create the question. The feature encodes information about what you're trying to predict.

```python
# ❌ WRONG: Feature calculated using target (LEAKAGE!)
df['avg_monthly_spend_of_churners'] = (
    df.groupby('Churn')['MonthlyCharges'].transform('mean')
)
# This creates:
# - If Churn='Yes': value = average spend of all churners (~$65)
# - If Churn='No': value = average spend of all non-churners (~$75)
# The feature directly encodes the target!

# ✅ CORRECT: Feature doesn't use target information
df['avg_monthly_spend_in_contract_type'] = (
    df.groupby('Contract')['MonthlyCharges'].transform('mean')
)
# This creates:
# - Month-to-month contracts: average spend
# - One year contracts: average spend  
# - Two year contracts: average spend
# Valid because Contract is available before predicting Churn

# Also valid: Segment by demographics (available at prediction time)
df['avg_spend_by_tenure_segment'] = (
    df.groupby(pd.cut(df['tenure'], bins=[0, 12, 24, 100]))['MonthlyCharges']
    .transform('mean')
)
```

**Golden Rule:** Never group by the target variable when creating features.

### Detection Method

**Red Flags for Target Leakage:**
1. Feature correlation > 0.95 with target
2. Feature that's "too good to be true"
3. Feature calculated using timestamps after prediction time
4. Feature derived from groupby on target

**Detection Code:**

```python
def detect_target_leakage(df, target_col, threshold=0.90):
    """
    Detect features with suspiciously high correlation to target.
    These may indicate target leakage.
    """
    from sklearn.preprocessing import LabelEncoder
    
    leakage_suspects = []
    
    for col in df.columns:
        if col == target_col:
            continue
        
        # Convert to numeric for correlation calculation
        if df[col].dtype == 'object':
            le = LabelEncoder()
            correlation = np.corrcoef(
                le.fit_transform(df[col].astype(str)),
                LabelEncoder().fit_transform(df[target_col])
            )[0, 1]
        else:
            correlation = df[col].corr(df[target_col])
        
        if abs(correlation) > threshold:
            leakage_suspects.append({
                'feature': col,
                'correlation': correlation,
                'suspicion': 'HIGH - Investigate for target leakage',
                'action': 'Check if feature uses future information or is derived from target'
            })
    
    return pd.DataFrame(leakage_suspects)

# Run detection
print("Checking for target leakage signals...")
suspects = detect_target_leakage(df, 'Churn', threshold=0.90)
if len(suspects) > 0:
    print("⚠️  WARNING: Potential target leakage detected!")
    print(suspects)
    print("\nInvestigation steps:")
    print("1. Check if feature uses timestamps after prediction date")
    print("2. Verify feature isn't calculated using groupby on target")
    print("3. Confirm feature is available at prediction time")
else:
    print("✅ No high-correlation features detected")

# Additional check: Groupby on target
def check_groupby_leakage(df, target_col):
    """Check if any features were created via groupby on target."""
    # This requires metadata about feature creation
    # In practice, review your feature engineering code for:
    # df.groupby(target_col)...
    pass
```

### Interview Question Pattern

**Q:** "A feature has 0.98 correlation with the target. Is this good?"

**A:** This is a major red flag for target leakage. Here's my diagnostic approach:

**Immediate Questions:**
1. **Temporal Check:** Is this feature available before we need to make the prediction?
2. **Calculation Check:** Does this feature use groupby on the target or information that only exists after the event?
3. **Business Logic:** Does this feature make sense, or is it suspiciously perfect?

**Diagnostic Test:**
```
1. Remove the suspicious feature
2. Retrain the model
3. If accuracy drops from 98% to 70% → Confirmed leakage
4. If accuracy stays at 98% → Might be legitimate strong predictor
```

**Real-World Example:**
- Feature: "days_since_last_login"
- Correlation: 0.96 with churn
- **Problem:** If customer churned yesterday, they haven't logged in recently
- **Leakage:** The "days since" is calculated after we already know they churned
- **Fix:** Use "days_since_last_login_as_of_prediction_date"

**Answer Summary:** "A 0.98 correlation is suspiciously high. I'd investigate whether the feature uses future information or is derived from the target. The diagnostic is to remove it and retrain - if performance drops significantly, we have leakage."

---

## 3. Multicollinearity & Feature Correlation

### What is Multicollinearity?

Multicollinearity occurs when independent features in your dataset are highly correlated with each other. This violates the assumption of independence between predictors and creates problems for model interpretation and stability.

**Why This Happens in Practice:**
- **Derived features:** Creating features that are mathematical transformations of each other
- **Related measurements:** Recording similar business metrics (tenure vs TotalCharges)
- **Data collection:** Sensors measuring the same underlying phenomenon

**Real-World Impact:**
```
Scenario: Your model shows "tenure" has positive impact on churn, 
but "MonthlyCharges" has negative impact.

Problem: These features are correlated (high tenure → high TotalCharges).
When you change one, the other changes too.

Result: Coefficient signs may flip between model runs, 
making interpretation unreliable.
```

### Why Multicollinearity Matters

**1. Unstable Coefficient Estimates**
```
Model Run 1: tenure coefficient = +0.5, TotalCharges = -0.3
Model Run 2: tenure coefficient = +0.3, TotalCharges = -0.1

The model can't decide which feature is "responsible" 
for the prediction because they move together.
```

**2. Inflated Standard Errors**
```
Coefficients appear statistically insignificant (high p-values)
even when the features are actually predictive.
This leads to incorrectly dropping important features.
```

**3. Interpretability Problems**
```
Business asks: "Does tenure or spending predict churn better?"
Answer: "We can't tell reliably because they're related."
```

**Note:** Tree-based models (Random Forest, XGBoost) handle multicollinearity better than linear models because they don't rely on coefficient estimates. However, it still affects:
- Feature importance rankings
- Model stability
- Interpretation

### Detection

**A. Correlation Matrix (First Check)**

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Numerical features only
num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
corr_matrix = df[num_features].corr()

# Visualize
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.show()

# Find high correlations
high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            high_corr.append({
                'feature_1': corr_matrix.columns[i],
                'feature_2': corr_matrix.columns[j],
                'correlation': corr_matrix.iloc[i, j]
            })

pd.DataFrame(high_corr)
```

**B. Variance Inflation Factor (VIF)**

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(X):
    """
    VIF > 5: Moderate multicollinearity
    VIF > 10: Severe multicollinearity
    """
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                       for i in range(X.shape[1])]
    return vif_data.sort_values('VIF', ascending=False)

# Use numerical features
X_numerical = df[['tenure', 'MonthlyCharges', 'TotalCharges']]
vif_results = calculate_vif(X_numerical)
print(vif_results)
```

**Telco Example with Expected Results:**
```python
# Expected correlation matrix for Telco data:
#                  tenure  MonthlyCharges  TotalCharges
# tenure           1.00          0.25          0.83
# MonthlyCharges   0.25          1.00          0.65
# TotalCharges     0.83          0.65          1.00

# Tenure and TotalCharges: ~0.83 correlation (HIGH)
# Reason: TotalCharges ≈ tenure × MonthlyCharges

# Action: Remove one or combine them
```

**B. Variance Inflation Factor (VIF) - The Standard Test**

**What is VIF?**
```
VIF measures how much the variance of a regression coefficient 
increases due to multicollinearity.

VIF = 1 / (1 - R²)

Where R² is from regressing that feature on all other features.

VIF = 1: No correlation
VIF = 5: Moderate correlation (variance 5x larger than if uncorrelated)
VIF = 10: Severe correlation (variance 10x larger)
```

**VIF Thresholds:**
- **VIF < 2:** No multicollinearity concern
- **VIF 2-5:** Moderate, monitor but usually acceptable
- **VIF 5-10:** High multicollinearity, consider action
- **VIF > 10:** Severe, definitely take action

### Solutions

**Strategy Selection Framework:**

| Situation | Recommended Action |
|-----------|-------------------|
| VIF 2-5, tree model | Monitor, likely fine |
| VIF 2-5, linear model | Document, monitor coefficients |
| VIF 5-10, need interpretation | Remove/combine features |
| VIF > 10 | Definitely remove/combine |
| Predictive accuracy focus | Use regularization |

**A. Remove Highly Correlated Features (Simplest)**

When to use: You have redundant features and interpretability matters.

```python
def remove_high_correlation_features(X, threshold=0.8):
    """
    Remove features with correlation above threshold.
    
    Strategy: Keep the first feature in each correlated pair,
    drop subsequent ones.
    """
    corr_matrix = X.corr().abs()
    
    # Upper triangle of correlation matrix
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find features to drop
    to_drop = [column for column in upper.columns 
               if any(upper[column] > threshold)]
    
    print(f"Dropping {len(to_drop)} highly correlated features: {to_drop}")
    print(f"Correlations with dropped features:")
    for col in to_drop:
        high_corr = corr_matrix[col][corr_matrix[col] > threshold]
        print(f"  {col}: {high_corr.to_dict()}")
    
    return X.drop(columns=to_drop), to_drop

# Apply to Telco data
X_reduced, dropped = remove_high_correlation_features(
    X_numerical, 
    threshold=0.8
)
# Likely drops 'TotalCharges' (correlated with tenure and MonthlyCharges)
```

**Decision Framework for Which to Drop:**
1. Drop the less interpretable feature
2. Drop the harder-to-collect feature
3. Drop the one with more missing values
4. Keep the one with stronger individual relationship to target

**B. Combine Features (Most Elegant for Telco)**

When to use: Correlated features represent the same underlying concept.

```python
# Problem: tenure, MonthlyCharges, TotalCharges are all related
# Solution: Create a derived feature that captures the relationship

# Option 1: Average monthly spend (most interpretable)
df['avg_monthly_spend'] = df['TotalCharges'] / (df['tenure'] + 1)
# +1 to avoid division by zero for new customers

# Option 2: Spending velocity
df['monthly_spend_ratio'] = df['MonthlyCharges'] / df['avg_monthly_spend']
# Values > 1: current spend higher than historical average (risk signal?)
# Values < 1: current spend lower than historical average

# Option 3: Tenure-adjusted charges
df['charges_per_year'] = df['TotalCharges'] / (df['tenure'] / 12 + 0.1)

# Drop the now-redundant original
df = df.drop(['TotalCharges'], axis=1)

print("Benefits:")
print("1. Single feature captures relationship between three variables")
print("2. More interpretable: 'average spend' vs 'tenure AND MonthlyCharges AND TotalCharges'")
print("3. Eliminates multicollinearity")
print("4. May be more predictive (customer behavior pattern)")
```

**C. Use Regularization (When Predictive Accuracy Matters Most)**

When to use: You want to keep all features but stabilize their coefficients.

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler

# Must scale first for regularization to work properly
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Ridge (L2): Shrinks coefficients toward zero but keeps all features
# Best for: multicollinearity, when all features have some value
ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_scaled, y)
print("Ridge coefficients:", ridge.coef_)
# All features retained, coefficients are stable

# Lasso (L1): Can drop features entirely by setting coefficients to zero
# Best for: feature selection, sparse solutions
lasso = Lasso(alpha=0.1, random_state=42)
lasso.fit(X_scaled, y)
print("Lasso coefficients:", lasso.coef_)
# Some features may have coefficient = 0 (dropped)

# ElasticNet: Combines L1 and L2
# Best for: high-dimensional data with correlated features
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
elastic.fit(X_scaled, y)

print("\nHow Regularization Solves Multicollinearity:")
print("- Ridge: Shares coefficient magnitude between correlated features")
print("- Lasso: Picks one and drops the other")
print("- Both: Prevents coefficients from becoming unstable/exploding")
```

### Interview Question Pattern

**Q:** "Why might you remove features even if they're predictive?"

**A:** This tests understanding of the bias-variance trade-off and practical constraints:

**1. Multicollinearity (Primary Reason)**
```
"If two features are highly correlated (VIF > 5-10), removing one 
reduces model instability without sacrificing much predictive power. 
The model can't distinguish which feature is driving the prediction anyway."
```

**2. Overfitting Risk**
```
"With limited data, having many correlated features increases 
dimensionality and overfitting risk. Simpler models generalize better."
```

**3. Interpretability**
```
"For business stakeholders, explaining 5 key features is easier 
than explaining 20 correlated ones. The model becomes a communication tool."
```

**4. Computational Efficiency**
```
"Fewer features = faster training, faster inference, lower memory. 
Matters at scale (millions of predictions per hour)."
```

**5. Data Collection Cost**
```
"If one feature requires expensive data collection (surveys, third-party APIs) 
and another free feature captures the same signal, drop the expensive one."
```

**Summary Answer:** 
"I'd remove predictive features when they create multicollinearity (VIF > 5), increase overfitting risk, harm interpretability, or have high collection costs. The goal is the simplest model that captures the signal, not the most complex model with redundant features."

---

## 4. High Cardinality Categoricals

### What is High Cardinality?

Cardinality refers to the number of unique values in a categorical feature. High cardinality means many unique categories.

**Examples by Cardinality:**
- **Low (2-10):** gender, binary flags (Yes/No), contract types
- **Medium (10-100):** states, product categories, age groups
- **High (100-10k+):** ZIP codes, customerIDs, product SKUs, IP addresses

**The Cardinality Trap:**
Data scientists often create features like `customer_id` without realizing they're encoding identifiers, not meaningful categories. The model memorizes: "Customer #1234 always does X" but this doesn't generalize to new customers.

### Problems with High Cardinality

**1. Curse of Dimensionality**
```python
# Problem demonstration
n_samples = 7043
n_unique_ids = 7043

print(f"Samples: {n_samples}")
print(f"Unique customerIDs: {n_unique_ids}")
print(f"One-hot columns: {n_unique_ids}")
print(f"Samples per column: {n_samples / n_unique_ids:.2f}")
# Result: 1 sample per column! Each column has ~1 non-zero value.
```

**2. Overfitting Through Memorization**
```python
# What the model learns:
# customerID_1234 → always predicts what customer 1234 did in training
# customerID_5678 → always predicts what customer 5678 did in training

# Problem in production:
# New customerID_9999 appears
# Model has never seen this ID
# Prediction: random guess (or whatever 'unknown' maps to)
```

**3. Memory and Computation**
- 10,000 unique values → 10,000 one-hot columns
- Sparse matrix storage helps, but still wasteful
- Inference time increases with feature count

**4. Production Breakage**
```python
# Training data: 1000 unique ZIP codes
# Production: New ZIP code appears
# OneHotEncoder: Error! Unknown category!
# Or with handle_unknown='ignore': All-zero vector
```

### Solutions

**Strategy Framework:**

| Cardinality | Strategy | Example |
|-------------|----------|---------|
| 2-10 | One-hot encoding | gender, Yes/No |
| 10-50 | One-hot with rare grouping | Product categories |
| 50-500 | Target encoding | ZIP codes |
| 500+ | Hash encoding | UserIDs, device IDs |
| Identifiers | Exclude | customerID, transactionID |

**A. Exclude Identifier Columns (First Check)**

**When to use:** The feature is a unique identifier, not a meaningful category.

```python
# Identifiers to ALWAYS exclude:
identifiers = [
    'customerID',      # Unique per customer
    'transactionID',   # Unique per transaction
    'orderID',         # Unique per order
    'sessionID',       # Unique per session
    'userID',          # Unique per user
]

# Why exclude?
# 1. No predictive signal (random ID assignment)
# 2. Perfect memorization in training
# 3. Complete failure on new IDs in production

# Check before dropping
print(f"Unique customerIDs: {df['customerID'].nunique()}")
print(f"Total rows: {len(df)}")
print(f"Ratio: {df['customerID'].nunique() / len(df):.2%}")
# If ratio ≈ 100%, it's an identifier

features = df.drop(['customerID'], axis=1)
```

**Exception:** Sometimes IDs encode information (e.g., customerID contains signup year: "2021_001234"). Extract the information, then drop the ID.

**B. Target Encoding (Mean Encoding)**

**When to use:** 50-500 unique values where one-hot is too expensive but grouping loses signal.

**How it works:** Replace each category with the mean target value for that category.

```python
from category_encoders import TargetEncoder

# Example: ZIP codes (hundreds of unique values)
encoder = TargetEncoder(cols=['zip_code'])
X_encoded = encoder.fit_transform(X, y)

# Result:
# ZIP 90210 → 0.35 (35% of customers in 90210 churned)
# ZIP 10001 → 0.12 (12% of customers in 10001 churned)
# ZIP 60601 → 0.28 (28% of customers in 60601 churned)

# The model now learns:
# "High encoded value = high churn ZIP"
# "Low encoded value = low churn ZIP"
```

**The Critical Risk: Target Leakage**

Standard target encoding leaks information! Here's why:

```python
# ❌ WRONG: Standard target encoding (LEAKAGE!)
encoder = TargetEncoder(cols=['zip_code'])
encoder.fit(X, y)  # Uses ALL data
X_encoded = encoder.transform(X)

# Problem: ZIP code means calculated using target from same rows
# ZIP 90210's mean includes whether these customers churned
# This is like telling the model the answer!
```

**Correct Approach: Cross-Validation Target Encoding**

```python
def target_encode_cv(X, y, col, n_splits=5, random_state=42):
    """
    Target encoding with cross-validation to prevent leakage.
    
    Strategy: For each fold, calculate means using only other folds,
    then encode current fold.
    """
    from sklearn.model_selection import KFold
    
    X_encoded = X.copy()
    X_encoded[f'{col}_encoded'] = 0.0
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    for train_idx, val_idx in kf.split(X):
        # Calculate means using ONLY training fold
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        
        means = X_train_fold.groupby(col)[y_train_fold.name].mean()
        
        # Apply to validation fold (the current fold)
        X_val_fold = X.iloc[val_idx]
        encoded_values = X_val_fold[col].map(means)
        
        # Store results
        X_encoded.iloc[val_idx, X_encoded.columns.get_loc(f'{col}_encoded')] = encoded_values
    
    # Handle unseen categories in production
    global_mean = y.mean()
    X_encoded[f'{col}_encoded'].fillna(global_mean, inplace=True)
    
    # Store means for production use
    global_means = X.groupby(col)[y.name].mean()
    
    return X_encoded, global_means

# Usage
X_encoded, means_dict = target_encode_cv(df, df['Churn'], 'zip_code')

# For production: use stored means
# new_data['zip_encoded'] = new_data['zip_code'].map(means_dict).fillna(global_mean)
```

```python
def target_encode_cv(X, y, col, n_splits=5):
    """Target encoding with cross-validation to prevent leakage."""
    from sklearn.model_selection import KFold
    
    X_encoded = X.copy()
    X_encoded[f'{col}_encoded'] = 0
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for train_idx, val_idx in kf.split(X):
        # Calculate means on training fold only
        means = X.iloc[train_idx].groupby(col)[y.name].mean()
        
        # Apply to validation fold
        X_encoded.iloc[val_idx, X_encoded.columns.get_loc(f'{col}_encoded')] = \
            X.iloc[val_idx][col].map(means)
    
    # Fill unseen categories with global mean
    global_mean = y.mean()
    X_encoded[f'{col}_encoded'].fillna(global_mean, inplace=True)
    
    return X_encoded
```

**C. Hash Encoding**

```python
from category_encoders import HashingEncoder

# Fixed number of output columns regardless of unique values
encoder = HashingEncoder(n_components=8, cols=['zip_code'])
X_hashed = encoder.fit_transform(X)
# Always 8 columns, even with 10,000 ZIP codes
```

**D. Binning / Grouping**

```python
# Group rare categories into "Other"
def group_rare_categories(series, threshold=0.01):
    """Group categories representing less than threshold % into 'Other'."""
    value_counts = series.value_counts(normalize=True)
    rare_categories = value_counts[value_counts < threshold].index
    
    return series.replace(rare_categories, 'Other')

df['zip_grouped'] = group_rare_categories(df['zip_code'], threshold=0.01)
```

### Interview Question Pattern

**Q:** "How do you handle a categorical feature with 10,000 unique values?"

**A:** "I'd follow a decision tree based on the nature of the feature:

**Step 1: Identify the Feature Type**
```
Is it an identifier (customerID, sessionID)? 
  → YES: Exclude entirely
  → NO: Continue to encoding strategy
```

**Step 2: Choose Encoding Strategy**
```
If prediction accuracy is priority:
  → Target encoding with cross-validation
  → Pros: Captures category-target relationship
  → Cons: Requires careful leakage prevention

If interpretability is priority:
  → Group rare categories into 'Other'
  → One-hot encode top K categories
  → Pros: Business understands the features
  → Cons: Loses information about rare categories

If fixed dimensionality needed:
  → Hash encoding (8-32 dimensions)
  → Pros: Always same output size
  → Cons: Hash collisions may occur

If using deep learning:
  → Embedding layers
  → Pros: Learned representations
  → Cons: Requires neural network
```

**Step 3: Production Considerations**
```
- Handle unseen categories (new values appear)
- Monitor for category drift
- Document encoding strategy for retraining
```

**My Recommendation for Most Cases:** "Target encoding with cross-validation, because it reduces 10,000 categories to 1 meaningful numeric feature while preserving the relationship to the target."

---

## 5. Sampling Bias & Selection Effects

### What is Sampling Bias?

Sampling bias occurs when your training data doesn't represent the population you'll make predictions on. The model learns patterns from one distribution but must predict on another.

**The Core Problem:**
```
Training Data: Urban customers aged 25-35, high income
Production Data: Mixed urban/rural, ages 18-65, varied income

Model learns: "High income young urbanites behave like X"
But production: "Everyone behaves differently"

Result: Model fails on segments it never saw in training
```

### Types of Sampling Bias

**A. Temporal Bias (Time-Based Distribution Shift)**

**What it is:** Training on historical data, predicting on current/future data, but the world has changed.

**Why This Happens:**
- Customer behavior evolves (new products, competitors, economic conditions)
- Business processes change (new pricing, different onboarding)
- External shocks (pandemics, regulation changes)

```python
# Example: Churn prediction model trained in 2020, deployed 2024
# 2020: Low competition, pandemic pricing discounts
# 2024: High competition, normal pricing

# Detection: Compare churn rates by year
df['year'] = pd.to_datetime(df['signup_date']).dt.year
yearly_churn = df.groupby('year')['Churn'].mean()

print("Churn rate by year:")
print(yearly_churn)

# If you see:
# 2020: 18%
# 2021: 22%
# 2022: 25%
# 2023: 31%
# → Clear temporal trend! Model will be outdated quickly

# Solution: Time-based train/validation split
df = df.sort_values('signup_date')
train = df[df['signup_date'] < '2023-01-01']  # Use historical data
val = df[df['signup_date'] >= '2023-01-01']   # Validate on recent data

# This simulates production: train on past, predict on future
```

**Mitigation Strategies:**
1. **Recent data weighting:** Weight recent samples higher
2. **Rolling window training:** Retrain monthly on last 12 months
3. **Time-based validation:** Always validate on most recent data
4. **Drift detection:** Monitor for when retraining is needed

**B. Selection Bias (Non-Random Subset)**

**What it is:** Your data comes from a non-representative subset of the population.

**Common Scenarios:**
- Survey data (only motivated customers respond)
- App data (only tech-savvy customers use app)
- Support ticket data (only customers with problems)
- Email respondents (different from non-respondents)

```python
# Example: Model trained on survey respondents
# Problem: Survey completers differ from general population

# Detection: Compare survey vs non-survey customers
survey_customers = df[df['completed_survey'] == True]
non_survey = df[df['completed_survey'] == False]

metrics = ['Churn', 'MonthlyCharges', 'tenure', 'satisfaction']
for metric in metrics:
    survey_mean = survey_customers[metric].mean()
    non_survey_mean = non_survey[metric].mean()
    diff_pct = (survey_mean - non_survey_mean) / non_survey_mean * 100
    print(f"{metric}: Survey={survey_mean:.2f}, Non-survey={non_survey_mean:.2f}, Diff={diff_pct:+.1f}%")

# If differences are large → selection bias

# Solutions:
# 1. Weighting (upweight underrepresented segments)
# 2. Data collection (get more non-survey data)
# 3. Acknowledge limitation (model only for survey responders)
```

**C. Survivorship Bias (The WWII Plane Problem)**

**What it is:** Only analyzing "survivors" (those who made it through some process), ignoring those who didn't.

**Classic Example:**
```
WWII Analysts: Looked at planes that returned with bullet holes.
Recommended: Armor the areas with most holes.

Abraham Wald's Insight: Wrong! Armor the areas with NO holes.
Why? Planes hit in those areas never returned!

The data was biased toward survivors.
```

**ML Example:**
```python
# ❌ WRONG: Only current customers
survivors = df[df['Churn'] == 'No']
# Analyzing only survivors:
# "Our customers have high satisfaction!"
# "Our customers rarely complain!"

# Reality: Churned customers had low satisfaction and complained!
# You've filtered out the negative cases.

# ✅ CORRECT: Include all customers from a time period
# Track all customers who joined in 2020
# See who churned by 2023
# Analyze both churned and retained

cohort_2020 = df[df['signup_date'].between('2020-01-01', '2020-12-31')]
churn_rate = cohort_2020['Churn'].mean()
print(f"2020 cohort churn rate: {churn_rate:.1%}")
# This is honest because it includes churned customers
```

**D. Class Imbalance Sampling (Artificial Balancing)**

**What it is:** Artificially balancing classes in training when production is naturally imbalanced.

**The Problem:**
```python
# Production reality: 90% No Churn, 10% Churn
# You want to "help" the model by balancing training data

from sklearn.utils import resample

# ❌ WRONG: Balance training set artificially
majority = df[df['Churn'] == 'No']  # 7000 samples
minority = df[df['Churn'] == 'Yes']  # 1000 samples

# Downsample majority to match minority
majority_downsampled = resample(majority, n_samples=len(minority))  # 1000 samples
balanced = pd.concat([majority_downsampled, minority])  # 2000 samples

# Model trains on 50/50 distribution
# In production, sees 90/10 distribution
# Model will be overconfident about churn (trained on too many churn examples)

# Result: High false positive rate in production
# Lots of unnecessary retention offers sent
# Customer annoyance and cost

# ✅ CORRECT: Train on natural distribution, use appropriate metrics
from sklearn.model_selection import train_test_split

# Stratified split maintains class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y  # Maintains 90/10 ratio in both splits
)

# Model learns real distribution
# Use metrics that handle imbalance: ROC-AUC, PR-AUC
# Adjust prediction threshold based on business costs
```

### Interview Question Pattern

**Q:** "Your model performs well on validation but poorly in A/B test. Why?"

**A:** "This is a classic deployment gap. I investigate systematically:

**Step 1: Data Distribution Check**
```
- Is validation data from same time period as production?
- Do feature distributions match? (drift detection)
- Is there selection bias in training data?
```

**Step 2: Sampling Strategy Review**
```
- Did we use random split when we should have used time-based?
- Did we balance classes artificially?
- Did we exclude any customer segments?
```

**Step 3: Temporal Analysis**
```
- Has customer behavior changed since training?
- Any external events (competitor launch, pricing change)?
- Model decay rate - how often should we retrain?
```

**Most Likely Culprits:**
1. **Temporal bias** (trained on old data, behavior shifted)
2. **Selection bias** (validation subset not representative)
3. **Train/test methodology** (random split instead of time-based)

**Follow-up:** I would implement monitoring to catch this earlier - track prediction distributions and model performance metrics in real-time to detect when the model is going stale."

---

## 6. Temporal Data Issues

### Why Time Matters in ML

**The Core Problem:** Standard ML assumes data is independent and identically distributed (IID). Time-series data violates this - future observations depend on past ones.

**Consequences of Ignoring Time:**
1. **Look-ahead bias:** Using future information to predict the past
2. **Overly optimistic results:** Random split creates leakage
3. **Production failure:** Model can't handle temporal patterns

### Time-Based Train/Test Split

**Why Random Split Fails for Time Data:**

```python
# ❌ WRONG: Random split ignores time
from sklearn.model_selection import train_test_split

# December data might end up in train
# January data might end up in test

X_train, X_test = train_test_split(X, test_size=0.2)

# Problem 1: December includes outcomes from January (impossible!)
# Problem 2: Model "sees" the future in training
# Problem 3: Validation metrics are inflated
```

**Correct Approach:**

```python
# ✅ CORRECT: Time-based split
df = df.sort_values('date')

# 80/20 time split - chronological order
n_samples = len(df)
train_size = int(n_samples * 0.8)

train = df.iloc[:train_size]  # First 80% chronologically
test = df.iloc[train_size:]   # Last 20% chronologically

print(f"Train period: {train['date'].min()} to {train['date'].max()}")
print(f"Test period: {test['date'].min()} to {test['date'].max()}")

# Benefits:
# 1. No future data leaks into training
# 2. Simulates real deployment scenario
# 3. Honest performance estimate
```

**When to Use Time-Based Split:**
- Any data with temporal ordering
- Customer behavior prediction
- Financial time series
- Sequential user actions
- Anything that changes over time

**When Random Split is OK:**
- Image classification (photos don't depend on order)
- Medical imaging (X-rays are independent)
- Static datasets without temporal patterns

### Feature Engineering with Time Windows

```python
# ❌ WRONG: Rolling average includes future information
df['avg_spend_3m'] = df['MonthlyCharges'].rolling(window=3).mean()

# ✅ CORRECT: Only past information
def create_lag_features(df, col, lags=[1, 2, 3]):
    """Create lag features using only past data."""
    for lag in lags:
        df[f'{col}_lag_{lag}'] = df.groupby('customerID')[col].shift(lag)
    return df

df = create_lag_features(df, 'MonthlyCharges', lags=[1, 2, 3])
```

### Walk-Forward Cross-Validation

```python
from sklearn.model_selection import TimeSeriesSplit

# Time series split - respects temporal order
tscv = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Train and evaluate
    # Each fold uses progressively more historical data
```

### Seasonality Detection

```python
def detect_seasonality(df, date_col, target_col):
    """Check for seasonal patterns in target."""
    df['month'] = pd.to_datetime(df[date_col]).dt.month
    
    monthly_churn = df.groupby('month')[target_col].mean()
    
    print("Churn rate by month:")
    print(monthly_churn)
    
    # If variance is high → seasonality present
    cv = monthly_churn.std() / monthly_churn.mean()
    print(f"\nCoefficient of variation: {cv:.2f}")
    
    if cv > 0.1:
        print("Significant seasonality detected!")
        print("Include month as feature or use seasonal models")
    
    return monthly_churn

monthly_pattern = detect_seasonality(df, 'signup_date', 'Churn')
```

### Interview Question Pattern

**Q:** "How is time-series cross-validation different from regular CV?"

**A:** "The fundamental difference is that regular CV shuffles data, which leaks future information into training when dealing with time-series.

**Regular K-Fold CV (WRONG for time data):**
```
Fold 1: Random 80% train, random 20% test
Fold 2: Random 80% train, random 20% test
...
Problem: December data in train, January in test
But January events caused December outcomes!
```

**Time-Series CV (CORRECT):**
```
Fold 1: Train Jan-Mar, Test Apr
Fold 2: Train Jan-Jun, Test Jul
Fold 3: Train Jan-Sep, Test Oct
...
Always: Training period strictly before test period
```

**Key Differences:**
1. **Regular CV:** Shuffles all data randomly
2. **Time-series CV:** Respects chronological order
3. **Regular CV:** Each fold uses same amount of data
4. **Time-series CV:** Later folds use more historical data
5. **Regular CV:** Leaks future information
6. **Time-series CV:** Prevents look-ahead bias

**Practical Impact:**
- Regular CV on time data: 95% accuracy (inflated)
- Time-series CV: 72% accuracy (realistic)
- Production performance: 71% accuracy (matches time-series CV)

**My Rule:** Always use time-based splits when the timestamp matters for the prediction context."

---

## 7. Data Validation Frameworks

### Great Expectations

```python
import great_expectations as ge

# Create context
context = ge.DataContext()

# Define expectations
batch = context.get_batch(
    datasource_name="telco_data",
    data_asset_name="customer_churn"
)

# Expect column to exist
batch.expect_column_to_exist("customerID")

# Expect values in set
batch.expect_column_values_to_be_in_set(
    column="Churn",
    value_set=["Yes", "No"]
)

# Expect numeric range
batch.expect_column_values_to_be_between(
    column="MonthlyCharges",
    min_value=0,
    max_value=200
)

# Expect no nulls
batch.expect_column_values_to_not_be_null("customerID")

# Run validation
results = batch.validate()
print(results)
```

### Pandera (Lightweight)

```python
import pandera as pa
from pandera import Column, Check, DataFrameSchema

# Define schema
schema = DataFrameSchema({
    "customerID": Column(str, checks=Check.str_length(min_value=1)),
    "gender": Column(str, checks=Check.isin(["Male", "Female"])),
    "SeniorCitizen": Column(int, checks=Check.isin([0, 1])),
    "tenure": Column(int, checks=Check.greater_than_or_equal_to(0)),
    "MonthlyCharges": Column(float, checks=[
        Check.greater_than_or_equal_to(0),
        Check.less_than_or_equal_to(200)
    ]),
    "Churn": Column(str, checks=Check.isin(["Yes", "No"]))
}, strict=True)  # Reject unknown columns

# Validate
validated_df = schema(df)
```

### Custom Validation Pipeline

```python
class DataValidator:
    """Production-ready data validation."""
    
    def __init__(self, schema):
        self.schema = schema
        self.validation_history = []
    
    def validate(self, df, dataset_name="production"):
        """Run all validations and return report."""
        errors = []
        warnings = []
        
        # 1. Schema compliance
        for col, rules in self.schema.items():
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
                continue
            
            # Type check
            if rules.get('type') == 'numeric':
                if not pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        pd.to_numeric(df[col])
                    except:
                        errors.append(f"Column {col} not numeric")
            
            # Range check
            if 'min' in rules and df[col].min() < rules['min']:
                errors.append(f"Column {col} below minimum {rules['min']}")
            
            if 'max' in rules and df[col].max() > rules['max']:
                errors.append(f"Column {col} above maximum {rules['max']}")
        
        # 2. Missing values
        missing_pct = df.isnull().mean().mean() * 100
        if missing_pct > 5:
            errors.append(f"Missing values: {missing_pct:.1f}% exceeds 5% threshold")
        elif missing_pct > 1:
            warnings.append(f"Missing values: {missing_pct:.1f}% above 1%")
        
        # 3. Duplicates
        dups = df.duplicated().sum()
        if dups > 0:
            errors.append(f"Found {dups} duplicate rows")
        
        # Report
        report = {
            'dataset': dataset_name,
            'timestamp': pd.Timestamp.now(),
            'passed': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'rows': len(df),
            'columns': len(df.columns)
        }
        
        self.validation_history.append(report)
        return report

# Usage
validator = DataValidator(TELCO_SCHEMA)
report = validator.validate(new_data, dataset_name="production_batch_2024_01")

if not report['passed']:
    raise ValueError(f"Data validation failed: {report['errors']}")
```

### Interview Question Pattern

**Q:** "How do you ensure data quality in production?"

**A:**
1. Schema validation (Great Expectations, Pandera)
2. Statistical checks (drift detection)
3. Unit tests for data pipelines
4. Monitoring dashboards
5. Alerting on validation failures
6. Circuit breakers (stop prediction if data invalid)

---

## 8. Measurement Error & Systematic Bias

### Types of Measurement Error

**A. Random Error (Noise)**
```python
# Unpredictable variations
# Example: Customer satisfaction scores vary due to mood

# Detection: High variance relative to range
noise_ratio = df['satisfaction_score'].std() / df['satisfaction_score'].mean()
if noise_ratio > 0.3:
    print("High measurement noise detected")
```

**B. Systematic Bias**
```python
# Consistent over/under estimation
# Example: All satisfaction scores inflated by 0.5 points

# Detection: Compare to external benchmark
print(f"Mean satisfaction: {df['satisfaction_score'].mean():.2f}")
print(f"Industry benchmark: 3.5")
# If significantly different → systematic bias
```

**C. Reporting Bias**
```python
# People report what makes them look good
# Example: Self-reported income vs actual

# Detection: Compare to objective measures
df['income_reported'] = df['income_survey']
df['income_actual'] = df['income_verified']  # From tax records

bias = (df['income_reported'] - df['income_actual']).mean()
print(f"Reporting bias: ${bias:,.2f}")
```

### Correcting for Bias

```python
def calibrate_measurement(series, bias_estimate):
    """Subtract known bias from measurements."""
    return series - bias_estimate

# Apply calibration
df['satisfaction_calibrated'] = calibrate_measurement(
    df['satisfaction_score'], 
    bias_estimate=0.3
)
```

### Interview Question Pattern

**Q:** "Your feature has high variance. How do you handle it?"

**A:**
1. Check if it's measurement noise or real signal
2. Smoothing/aggregation (rolling averages)
3. Feature engineering (binning)
4. Robust scalers (less sensitive to outliers)
5. Models that handle noise well (tree-based)

---

## 9. Schema Evolution

### Handling New Categories

```python
class RobustEncoder:
    """Handle unseen categories in production."""
    
    def __init__(self):
        self.encoder = OneHotEncoder(
            handle_unknown='ignore',  # Key: ignore unseen categories
            sparse_output=False
        )
        self.known_categories = {}
    
    def fit(self, X, categorical_cols):
        """Fit on training data."""
        self.categorical_cols = categorical_cols
        
        for col in categorical_cols:
            self.known_categories[col] = set(X[col].unique())
        
        self.encoder.fit(X[categorical_cols])
        return self
    
    def transform(self, X):
        """Transform with handling for new categories."""
        X = X.copy()
        
        # Check for new categories
        for col in self.categorical_cols:
            new_cats = set(X[col].unique()) - self.known_categories[col]
            if new_cats:
                print(f"Warning: New categories in {col}: {new_cats}")
                # Replace with 'Unknown' or mode
                X[col] = X[col].apply(
                    lambda x: x if x in self.known_categories[col] else 'Unknown'
                )
        
        return self.encoder.transform(X[self.categorical_cols])
```

### Versioning Strategy

```python
# Track schema versions
SCHEMA_VERSIONS = {
    'v1': {
        'columns': ['customerID', 'gender', 'tenure', 'MonthlyCharges', 'Churn'],
        'date_range': '2020-01-01 to 2023-12-31'
    },
    'v2': {
        'columns': ['customerID', 'gender', 'tenure', 'MonthlyCharges', 'PaymentMethod', 'Churn'],
        'date_range': '2024-01-01 to present',
        'notes': 'Added PaymentMethod feature'
    }
}

def validate_schema_version(df, expected_version='v2'):
    """Check if data matches expected schema version."""
    expected_cols = set(SCHEMA_VERSIONS[expected_version]['columns'])
    actual_cols = set(df.columns)
    
    missing = expected_cols - actual_cols
    extra = actual_cols - expected_cols
    
    if missing or extra:
        print(f"Schema mismatch!")
        print(f"Missing: {missing}")
        print(f"Extra: {extra}")
        return False
    
    return True
```

### Interview Question Pattern

**Q:** "A new product category appears in production data. What happens?"

**A:**
1. `OneHotEncoder(handle_unknown='ignore')` creates all-zero vector
2. Model can still make prediction (may be less confident)
3. Log the new category for monitoring
4. Retrain model with new category when sufficient data collected
5. Consider fallback strategies (use similar category, default prediction)

---

## 10. Interview Question Patterns

### Category 1: Leakage Detection

**Q:** "How do you know if you have data leakage?"

**A:**
- Suspiciously high performance (95%+ accuracy)
- Feature with correlation >0.9 to target
- Train/test performance gap is small but production performance drops
- Pipeline fitted on full data before split

### Category 2: Validation Strategy

**Q:** "How do you validate model performance?"

**A:**
1. **Holdout set**: Time-based, not random
2. **Cross-validation**: Stratified for imbalanced, time-series for temporal
3. **A/B test**: Small production rollout
4. **Monitoring**: Track prediction distributions and performance over time

### Category 3: Production Issues

**Q:** "Model works in dev but fails in production. Debug steps?"

**A:**
1. Check data distributions (drift detection)
2. Verify preprocessing is identical
3. Check for leakage in training
4. Examine schema differences
5. Review sampling strategy
6. Check for temporal effects

### Category 4: Feature Engineering

**Q:** "When do you create new features vs use raw data?"

**A:**
- **Create features** when: domain knowledge suggests relationship, capturing interactions
- **Use raw data** when: interpretability needed, feature importance tracking, simple baseline

### Category 5: Trade-offs

**Q:** "Accuracy vs interpretability - which do you choose?"

**A:**
- **Regulated industries** (finance, healthcare): Interpretability
- **High-volume consumer**: Accuracy (with monitoring)
- **Debugging phase**: Interpretability
- **Production phase**: Depends on cost of error vs cost of explanation

---

## Quick Reference: Interview Checklist

| Topic | Key Point | Code/Example |
|-------|-----------|--------------|
| Leakage | Fit on train only | `scaler.fit(X_train)` not `X` |
| Target Leakage | Feature uses future info | Check correlation >0.95 |
| Multicollinearity | VIF > 5 problematic | `variance_inflation_factor()` |
| High Cardinality | 10k+ unique values | Target encoding, hashing |
| Sampling Bias | Train ≠ Production | Time-based split |
| Temporal | Respect time order | `TimeSeriesSplit` |
| Validation | Great Expectations | `expect_column_values_to_be_in_set()` |
| Schema Evolution | New categories | `handle_unknown='ignore'` |

---

*This guide covers the advanced data quality topics commonly tested in data science interviews.*
