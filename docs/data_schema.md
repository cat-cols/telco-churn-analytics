# Data Schema

## Technical Schema Definition

### Raw Data Schema

**File**: `customer_churn_dataset-training-master.csv`
**Format**: CSV
**Encoding**: UTF-8
**Delimiter**: Comma (`,`)

### Column Specifications

| Column Name | Data Type | Nullable | Constraints | Description |
|-------------|-----------|----------|-------------|-------------|
| CustomerID | float64 | No | Unique identifier | Customer unique ID |
| Age | float64 | No | Range: 18-65 | Customer age in years |
| Gender | object | No | Values: Male, Female | Customer gender |
| Tenure | float64 | No | Range: 1-60 | Months with company |
| Usage Frequency | float64 | No | Range: 1-30 | Service usage frequency |
| Support Calls | float64 | No | Range: 0-10 | Number of support calls |
| Payment Delay | float64 | No | Range: 0-30 | Days payment delayed |
| Subscription Type | object | No | Values: Basic, Standard, Premium | Subscription plan |
| Contract Length | object | No | Values: Monthly, Quarterly, Annual | Contract duration |
| Total Spend | float64 | No | Range: 100-1000 | Total amount spent |
| Last Interaction | float64 | No | Range: 1-30 | Days since last interaction |
| Churn | float64 | No | Values: 0, 1 | Target variable (0=No, 1=Yes) |

### Processed Data Schema

**File**: `train.parquet`, `test.parquet`
**Format**: Apache Parquet
**Compression**: Snappy (default)

#### Processed Features

**Numerical Features (StandardScaler Applied)**
- Age (scaled)
- Tenure (scaled)
- Usage Frequency (scaled)
- Support Calls (scaled)
- Payment Delay (scaled)
- Total Spend (scaled)
- Last Interaction (scaled)

**Categorical Features (OneHotEncoder Applied)**
- Subscription Type_Basic
- Subscription Type_Standard
- Subscription Type_Premium
- Contract Length_Annual
- Contract Length_Quarterly
- Contract Length_Monthly

**Identifier**
- CustomerID (original, not scaled)

### Preprocessing Objects

**Scaler**: StandardScaler
- Parameters: mean and standard deviation from training data
- File: `scaler.pkl`

**Encoder**: OneHotEncoder
- Parameters: category mappings from training data
- File: `encoder.pkl`

### Data Relationships

```
Customer (1) ────── (1) Churn Status
Customer (1) ────── (1) Subscription
Customer (1) ────── (1) Contract
Customer (1) ────── (N) Support Calls
Customer (1) ────── (N) Usage Events
```

### Data Integrity Rules

1. **Primary Key**: CustomerID should be unique
2. **Referential Integrity**: All CustomerIDs should be valid
3. **Domain Constraints**: Values must fall within specified ranges
4. **Null Handling**: No null values allowed in processed data
5. **Data Consistency**: Churn values must be 0 or 1

### Schema Evolution

**Version**: 1.0
**Last Updated**: 2024

Changes from raw to processed:
- Numerical features standardized (mean=0, std=1)
- Categorical features one-hot encoded
- Missing values removed
- Train/test split applied (80/20)
