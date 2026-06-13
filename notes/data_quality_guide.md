# Data Quality Guide: Telco Churn Analytics

## Table of Contents
1. [The TotalCharges Whitespace Crisis](#1-the-totalcharges-whitespace-crisis)
2. [Missing Value Patterns](#2-missing-value-patterns)
3. [Data Type Mismatches](#3-data-type-mismatches)
4. [Categorical Inconsistencies](#4-categorical-inconsistencies)
5. [Numerical Outliers](#5-numerical-outliers)
6. [Duplicate Records](#6-duplicate-records)
7. [Class Imbalance](#7-class-imbalance)
8. [Data Drift Detection](#8-data-drift-detection)
9. [Quality Monitoring Framework](#9-quality-monitoring-framework)

---

## 1. The TotalCharges Whitespace Crisis

### The Problem

The `TotalCharges` column appears numeric but is stored as **string/object** type. Upon inspection:

```python
import pandas as pd

df = pd.read_csv('data/raw/telco_customer_churn.csv')
print(df['TotalCharges'].dtype)  # object (string)
print(df['TotalCharges'].head(10))
```

**Output:**
```
0       29.85
1      1889.5
2       108.15
3    1840.75
4    151.65
5         
6         
7         
8    3046.05
9    3487.95
```

Notice rows 5, 6, 7 contain **whitespace-only strings** (not empty strings, not NaN).

### Root Cause Analysis

```python
# Identify problematic rows
whitespace_mask = df['TotalCharges'].str.strip() == ''
problematic = df[whitespace_mask]

print(f"Rows with whitespace-only TotalCharges: {len(problematic)}")
print(problematic[['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges']])
```

**Findings:**
- 11 customers affected (0.16% of dataset)
- All have `tenure = 0` (brand new customers)
- Likely ETL issue: TotalCharges calculated as `tenure * MonthlyCharges`, but for tenure=0, system wrote whitespace instead of "0"

### Solutions

**Option A: Convert and Coerce (Current Implementation)**
```python
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df_clean = df.dropna()
```
- **Pros:** Simple, consistent with business logic (new customers have $0 total charges)
- **Cons:** Loses 11 records

**Option B: Impute with Calculation**
```python
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# For missing TotalCharges, estimate as tenure * MonthlyCharges
mask = df['TotalCharges'].isna()
df.loc[mask, 'TotalCharges'] = df.loc[mask, 'tenure'] * df.loc[mask, 'MonthlyCharges']
```
- **Pros:** Retains all records
- **Cons:** For tenure=0, still get $0 (same as dropping)

**Option C: Flag and Analyze Separately**
```python
df['TotalCharges_Numeric'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['Is_New_Customer'] = df['TotalCharges_Numeric'].isna().astype(int)
# Keep new customers for separate analysis
```
- **Pros:** Preserves information about customer newness
- **Cons:** Adds complexity

**Recommendation:** Use Option A for production simplicity. 0.16% data loss is acceptable.

---

## 2. Missing Value Patterns

### Detection Script

```python
def analyze_missing_values(df):
    """Comprehensive missing value analysis."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    report = pd.DataFrame({
        'Missing_Count': missing,
        'Missing_Percent': missing_pct
    }).sort_values('Missing_Percent', ascending=False)
    
    print("Missing Value Report:")
    print(report[report['Missing_Count'] > 0])
    
    # Check for patterns in missing values
    print("\nMissing Value Patterns:")
    missing_rows = df[df.isnull().any(axis=1)]
    if len(missing_rows) > 0:
        print(f"Rows with any missing: {len(missing_rows)}")
        print("Common missing combinations:")
        print(missing_rows.isnull().sum(axis=1).value_counts().head())
    
    return report

# Run analysis
analyze_missing_values(df)
```

### Missing Value Patterns in Telco Data

| Pattern | Count | Interpretation |
|---------|-------|----------------|
| TotalCharges only | 11 | New customers (tenure=0) |
| No missing | 7,032 | Normal records |

### Handling Strategy

**Rule:** If TotalCharges is missing, check tenure. If tenure=0, safe to drop or set to 0.

```python
def handle_missing_totalcharges(df):
    """Intelligent missing value handling."""
    # Convert first
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Check correlation with tenure
    missing_mask = df['TotalCharges'].isna()
    if missing_mask.sum() > 0:
        print(f"Missing TotalCharges by tenure:")
        print(df[missing_mask]['tenure'].value_counts())
        
        # Verify all missing have tenure=0
        if df[missing_mask]['tenure'].eq(0).all():
            print("All missing TotalCharges are new customers - safe to drop")
            return df.dropna()
        else:
            print("WARNING: Some missing TotalCharges have tenure > 0 - investigate!")
            return df
    
    return df
```

---

## 3. Data Type Mismatches

### Automatic Type Detection

```python
def detect_type_issues(df):
    """Identify columns that should be numeric but are objects."""
    issues = []
    
    for col in df.select_dtypes(include=['object']).columns:
        # Try to convert to numeric
        try:
            converted = pd.to_numeric(df[col], errors='coerce')
            non_null_pct = converted.notna().mean()
            
            if non_null_pct > 0.9:  # >90% convertible
                issues.append({
                    'column': col,
                    'current_type': 'object',
                    'suggested_type': 'numeric',
                    'convertible_pct': non_null_pct * 100,
                    'sample_values': df[col].dropna().head(3).tolist()
                })
        except:
            pass
    
    return pd.DataFrame(issues)

# Run detection
type_issues = detect_type_issues(df)
print(type_issues)
```

### Validation Schema

```python
TELCO_SCHEMA = {
    'customerID': {'type': 'string', 'nullable': False, 'unique': True},
    'gender': {'type': 'categorical', 'allowed': ['Male', 'Female']},
    'SeniorCitizen': {'type': 'numeric', 'range': [0, 1]},
    'Partner': {'type': 'categorical', 'allowed': ['Yes', 'No']},
    'Dependents': {'type': 'categorical', 'allowed': ['Yes', 'No']},
    'tenure': {'type': 'numeric', 'range': [0, 100]},
    'PhoneService': {'type': 'categorical', 'allowed': ['Yes', 'No']},
    'MultipleLines': {'type': 'categorical', 'allowed': ['Yes', 'No', 'No phone service']},
    'InternetService': {'type': 'categorical', 'allowed': ['DSL', 'Fiber optic', 'No']},
    'OnlineSecurity': {'type': 'categorical', 'allowed': ['Yes', 'No', 'No internet service']},
    'Contract': {'type': 'categorical', 'allowed': ['Month-to-month', 'One year', 'Two year']},
    'PaperlessBilling': {'type': 'categorical', 'allowed': ['Yes', 'No']},
    'PaymentMethod': {'type': 'categorical', 'allowed': ['Electronic check', 'Mailed check', 
                                                          'Bank transfer (automatic)', 
                                                          'Credit card (automatic)']},
    'MonthlyCharges': {'type': 'numeric', 'range': [0, 200]},
    'TotalCharges': {'type': 'numeric', 'range': [0, 10000]},
    'Churn': {'type': 'categorical', 'allowed': ['Yes', 'No']}
}

def validate_schema(df, schema):
    """Validate data against schema and report violations."""
    violations = []
    
    for col, rules in schema.items():
        if col not in df.columns:
            violations.append({'column': col, 'issue': 'Missing column'})
            continue
        
        # Check nullability
        if not rules.get('nullable', True) and df[col].isnull().any():
            violations.append({'column': col, 'issue': 'Contains nulls'})
        
        # Check categoricals
        if rules['type'] == 'categorical' and 'allowed' in rules:
            invalid = ~df[col].isin(rules['allowed'])
            if invalid.any():
                invalid_values = df.loc[invalid, col].unique()
                violations.append({
                    'column': col, 
                    'issue': f'Invalid values: {invalid_values}'
                })
        
        # Check numeric ranges
        if rules['type'] == 'numeric' and 'range' in rules:
            min_val, max_val = rules['range']
            out_of_range = (df[col] < min_val) | (df[col] > max_val)
            if out_of_range.any():
                violations.append({
                    'column': col,
                    'issue': f'Values outside range [{min_val}, {max_val}]'
                })
    
    return pd.DataFrame(violations)

# Run validation
violations = validate_schema(df, TELCO_SCHEMA)
if len(violations) > 0:
    print("Schema Violations Found:")
    print(violations)
else:
    print("All columns pass schema validation")
```

---

## 4. Categorical Inconsistencies

### Detection

```python
def analyze_categoricals(df):
    """Analyze all categorical columns for inconsistencies."""
    cat_cols = df.select_dtypes(include=['object']).columns
    
    for col in cat_cols:
        values = df[col].value_counts()
        print(f"\n{col}:")
        print(values)
        
        # Check for case inconsistencies
        if df[col].dtype == 'object':
            lower_unique = df[col].str.lower().nunique() if hasattr(df[col], 'str') else len(values)
            if lower_unique < len(values):
                print(f"  WARNING: Possible case inconsistencies!")

analyze_categoricals(df)
```

### Fixes

```python
# Trailing whitespace
df['PaymentMethod'] = df['PaymentMethod'].str.strip()

# Case standardization
df['gender'] = df['gender'].str.title()

# Synonym mapping
mapping = {
    'E-check': 'Electronic check',
    'Electronic Check': 'Electronic check',
    'Credit Card': 'Credit card (automatic)'
}
df['PaymentMethod'] = df['PaymentMethod'].replace(mapping)
```

---

## 5. Numerical Outliers

### Detection

```python
def detect_outliers_iqr(df, column):
    """Detect outliers using Interquartile Range method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    print(f"{column} Outliers (IQR method):")
    print(f"  Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"  Count: {len(outliers)}")
    
    return outliers

# Check each numerical column
for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
    detect_outliers_iqr(df, col)
```

### Telco-Specific Outlier Rules

| Column | Cap | Rationale |
|--------|-----|-----------|
| tenure | 72 months | 6 years maximum reasonable |
| MonthlyCharges | $150 | Beyond premium tier |
| TotalCharges | $10,000 | Long-term high-value customer |

### Handling

```python
def handle_outliers(df, strategy='clip'):
    """Handle outliers based on business rules."""
    df = df.copy()
    caps = {
        'tenure': 72,
        'MonthlyCharges': 150,
        'TotalCharges': 10000
    }
    
    if strategy == 'clip':
        for col, cap in caps.items():
            if col in df.columns:
                df[col] = df[col].clip(upper=cap)
    
    elif strategy == 'flag':
        for col, cap in caps.items():
            if col in df.columns:
                df[f'{col}_is_outlier'] = (df[col] > cap).astype(int)
    
    return df
```

---

## 6. Duplicate Records

### Detection

```python
def detect_duplicates(df):
    """Comprehensive duplicate analysis."""
    
    # Exact duplicates
    exact_dups = df.duplicated().sum()
    print(f"Exact duplicate rows: {exact_dups}")
    
    # Duplicate customerIDs
    id_dups = df['customerID'].duplicated().sum()
    print(f"Duplicate customerIDs: {id_dups}")
    
    if id_dups > 0:
        print(df[df['customerID'].duplicated(keep=False)].sort_values('customerID'))
    
    return {'exact': exact_dups, 'id_duplicates': id_dups}

dup_report = detect_duplicates(df)
```

### Resolution

```python
def resolve_duplicates(df, strategy='keep_first'):
    """Remove duplicates based on strategy."""
    if strategy == 'keep_first':
        return df.drop_duplicates(subset=['customerID'], keep='first')
    elif strategy == 'keep_last':
        return df.drop_duplicates(subset=['customerID'], keep='last')
    
    return df

df_clean = resolve_duplicates(df, strategy='keep_first')
```

---

## 7. Class Imbalance

### Analysis

```python
def analyze_class_distribution(df, target='Churn'):
    """Analyze target variable distribution."""
    dist = df[target].value_counts(normalize=True)
    counts = df[target].value_counts()
    
    print("Class Distribution:")
    print(counts)
    print(f"\nPercentages: {dist * 100}")
    
    ratio = dist.max() / dist.min()
    print(f"Imbalance ratio: {ratio:.2f}:1")
    
    if ratio < 1.5:
        print("Status: Balanced")
    elif ratio < 4:
        print("Status: Moderate imbalance")
    else:
        print("Status: Severe imbalance")
    
    return dist

dist = analyze_class_distribution(df)
```

**Telco Results:** 73.5% No Churn, 26.5% Churn, 2.78:1 ratio (moderate)

### Impact Assessment

```python
def demonstrate_imbalance_issue():
    """Show what happens with naive model on imbalanced data."""
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import classification_report
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Always predict majority class
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X, y)
    y_pred = dummy.predict(X)
    
    print("Naive 'Always Predict Majority' Classifier:")
    print(classification_report(y, y_pred))

demonstrate_imbalance_issue()
```

---

## 8. Data Drift Detection

### Drift Detector Implementation

```python
from scipy.stats import ks_2samp, chi2_contingency
import numpy as np

class DataDriftDetector:
    """Detect drift between reference and current data."""
    
    def __init__(self, reference_data: pd.DataFrame):
        self.reference = reference_data.copy()
        self.numerical_cols = reference_data.select_dtypes(
            include=[np.number]).columns.tolist()
        self.categorical_cols = reference_data.select_dtypes(
            include=['object']).columns.tolist()
    
    def detect_numerical_drift(self, current_data: pd.DataFrame, 
                               threshold: float = 0.05) -> pd.DataFrame:
        """KS test for numerical features."""
        drift_results = []
        
        for col in self.numerical_cols:
            if col not in current_data.columns:
                continue
            
            ref_values = self.reference[col].dropna()
            cur_values = current_data[col].dropna()
            
            statistic, p_value = ks_2samp(ref_values, cur_values)
            
            drift_results.append({
                'feature': col,
                'drift_detected': p_value < threshold,
                'p_value': p_value,
                'type': 'numerical'
            })
        
        return pd.DataFrame(drift_results)
    
    def detect_categorical_drift(self, current_data: pd.DataFrame,
                                  threshold: float = 0.05) -> pd.DataFrame:
        """Chi-square test for categorical features."""
        drift_results = []
        
        for col in self.categorical_cols:
            if col not in current_data.columns:
                continue
            
            ref_counts = self.reference[col].value_counts()
            cur_counts = current_data[col].value_counts()
            
            all_categories = set(ref_counts.index) | set(cur_counts.index)
            ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
            cur_aligned = [cur_counts.get(cat, 0) for cat in all_categories]
            
            contingency = np.array([ref_aligned, cur_aligned])
            
            try:
                chi2, p_value, dof, expected = chi2_contingency(contingency)
                drift_results.append({
                    'feature': col,
                    'drift_detected': p_value < threshold,
                    'p_value': p_value,
                    'type': 'categorical'
                })
            except ValueError:
                drift_results.append({
                    'feature': col,
                    'drift_detected': False,
                    'p_value': 1.0,
                    'type': 'categorical'
                })
        
        return pd.DataFrame(drift_results)
    
    def generate_report(self, current_data: pd.DataFrame) -> dict:
        """Complete drift detection report."""
        num_drift = self.detect_numerical_drift(current_data)
        cat_drift = self.detect_categorical_drift(current_data)
        
        all_drift = pd.concat([num_drift, cat_drift], ignore_index=True)
        drifted_features = all_drift[all_drift['drift_detected'] == True]
        
        return {
            'total_features': len(all_drift),
            'drifted_features_count': len(drifted_features),
            'drifted_features': drifted_features['feature'].tolist(),
            'drift_percentage': len(drifted_features) / len(all_drift) * 100,
            'details': all_drift
        }

# Usage
detector = DataDriftDetector(reference_data=df_train)
report = detector.generate_report(current_data=df_new)
print(f"Drift detected in {report['drift_percentage']:.1f}% of features")
```

---

## 9. Quality Monitoring Framework

### Automated Quality Checks

```python
class DataQualityMonitor:
    """Automated data quality monitoring system."""
    
    def __init__(self, schema: dict):
        self.schema = schema
        self.quality_log = []
    
    def run_checks(self, df: pd.DataFrame, dataset_name: str = "unnamed") -> dict:
        """Run all quality checks and return report."""
        report = {
            'dataset': dataset_name,
            'timestamp': pd.Timestamp.now(),
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'checks': {}
        }
        
        # Check 1: Schema compliance
        schema_violations = validate_schema(df, self.schema)
        report['checks']['schema'] = {
            'passed': len(schema_violations) == 0,
            'violations': schema_violations.to_dict('records') if len(schema_violations) > 0 else []
        }
        
        # Check 2: Missing values (<5% threshold)
        total_missing = df.isnull().sum().sum()
        missing_pct = (total_missing / (len(df) * len(df.columns))) * 100
        report['checks']['missing_values'] = {
            'passed': missing_pct < 5,
            'missing_percentage': missing_pct
        }
        
        # Check 3: Duplicates
        dup_count = df.duplicated().sum()
        report['checks']['duplicates'] = {
            'passed': dup_count == 0,
            'duplicate_count': dup_count
        }
        
        # Check 4: Class balance (<10:1 ratio)
        if 'Churn' in df.columns:
            dist = df['Churn'].value_counts(normalize=True)
            ratio = dist.max() / dist.min()
            report['checks']['class_balance'] = {
                'passed': ratio < 10,
                'imbalance_ratio': ratio
            }
        
        # Overall quality score
        passed_checks = sum(1 for check in report['checks'].values() 
                          if check.get('passed', True))
        total_checks = len(report['checks'])
        report['quality_score'] = passed_checks / total_checks
        report['passed_all'] = passed_checks == total_checks
        
        self.quality_log.append(report)
        return report

# Initialize and run
monitor = DataQualityMonitor(TELCO_SCHEMA)
quality_report = monitor.run_checks(df, dataset_name="telco_churn")

if quality_report['passed_all']:
    print("All quality checks passed!")
else:
    print("Quality issues detected:")
    for check, result in quality_report['checks'].items():
        if not result.get('passed', True):
            print(f"  - {check}: FAILED")
```

---

## Quick Reference: Common Issues & Solutions

| Issue | Detection | Solution |
|-------|-----------|----------|
| TotalCharges whitespace | `pd.to_numeric(errors='coerce')` | Drop rows or set to 0 |
| Missing values | `df.isnull().sum()` | Drop if <5%, else impute |
| Type mismatch | Try conversion, check % success | Cast to correct type |
| Categorical variations | Value counts, case analysis | Standardize mappings |
| Outliers | IQR or Z-score method | Clip to business limits |
| Duplicates | `df.duplicated()` | Drop or merge |
| Class imbalance | Value counts ratio | Use ROC-AUC, adjust threshold |
| Data drift | KS test, Chi-square | Retrain model, investigate cause |

---

*This guide provides the foundation for detecting, understanding, and resolving data quality issues in the Telco Churn project.*
