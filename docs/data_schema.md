# Data Schema

## Technical Schema Definition

### Raw Data Schema

**File**: `data/raw/telco_customer_churn.csv`
**Format**: CSV
**Encoding**: UTF-8
**Delimiter**: Comma (`,`)

### Column Specifications

| Column Name | Data Type | Nullable | Constraints | Description |
|-------------|-----------|----------|-------------|-------------|
| customerID | str | No | Unique identifier | Customer unique ID |
| gender | str | No | Values: Male, Female | Customer gender |
| SeniorCitizen | int64 | No | Values: 0, 1 | Senior citizen status |
| Partner | str | No | Values: Yes, No | Partner status |
| Dependents | str | No | Values: Yes, No | Dependents status |
| tenure | int64 | No | Range: 0-72 | Months with company |
| PhoneService | str | No | Values: Yes, No | Phone service status |
| MultipleLines | str | No | Values: Yes, No, No phone service | Multiple lines status |
| InternetService | str | No | Values: DSL, Fiber optic, No | Internet service type |
| OnlineSecurity | str | No | Values: Yes, No, No internet service | Online security status |
| OnlineBackup | str | No | Values: Yes, No, No internet service | Online backup status |
| DeviceProtection | str | No | Values: Yes, No, No internet service | Device protection status |
| TechSupport | str | No | Values: Yes, No, No internet service | Tech support status |
| StreamingTV | str | No | Values: Yes, No, No internet service | Streaming TV status |
| StreamingMovies | str | No | Values: Yes, No, No internet service | Streaming movies status |
| Contract | str | No | Values: Month-to-month, One year, Two year | Contract type |
| PaperlessBilling | str | No | Values: Yes, No | Paperless billing status |
| PaymentMethod | str | No | Values: Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic) | Payment method |
| MonthlyCharges | float64 | No | Range: 18.25-118.75 | Monthly charges amount |
| TotalCharges | str | No | String (needs conversion) | Total charges amount |
| Churn | str | No | Values: Yes, No | Target variable |

### Processed Data Schema

**File**: `train.parquet`, `test.parquet`
**Format**: Apache Parquet
**Compression**: Snappy (default)

#### Processed Features

**Numerical Features (StandardScaler Applied)**
- SeniorCitizen (scaled)
- tenure (scaled)
- MonthlyCharges (scaled)
- TotalCharges (scaled, after string conversion)

**Categorical Features (OneHotEncoder Applied)**
- gender_Female, gender_Male
- Partner_No, Partner_Yes
- Dependents_No, Dependents_Yes
- PhoneService_No, PhoneService_Yes
- MultipleLines_No, MultipleLines_No phone service, MultipleLines_Yes
- InternetService_DSL, InternetService_Fiber optic, InternetService_No
- OnlineSecurity_No, OnlineSecurity_No internet service, OnlineSecurity_Yes
- OnlineBackup_No, OnlineBackup_No internet service, OnlineBackup_Yes
- DeviceProtection_No, DeviceProtection_No internet service, DeviceProtection_Yes
- TechSupport_No, TechSupport_No internet service, TechSupport_Yes
- StreamingTV_No, StreamingTV_No internet service, StreamingTV_Yes
- StreamingMovies_No, StreamingMovies_No internet service, StreamingMovies_Yes
- Contract_Month-to-month, Contract_One year, Contract_Two year
- PaperlessBilling_No, PaperlessBilling_Yes
- PaymentMethod_Bank transfer (automatic), PaymentMethod_Credit card (automatic), PaymentMethod_Electronic check, PaymentMethod_Mailed check

**Identifier**
- customerID (original, not scaled)

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

1. **Primary Key**: customerID should be unique
2. **Referential Integrity**: All customerIDs should be valid
3. **Domain Constraints**: Values must fall within specified ranges
4. **Null Handling**: No null values allowed in processed data
5. **Data Consistency**: Churn values must be Yes or No (converted to 0/1 for modeling)

### Schema Evolution

**Version**: 1.0
**Last Updated**: 2024

Changes from raw to processed:
- TotalCharges converted from string to numeric
- Numerical features standardized (mean=0, std=1)
- Categorical features one-hot encoded
- Churn converted from Yes/No to 1/0
- Train/test split applied (80/20)
