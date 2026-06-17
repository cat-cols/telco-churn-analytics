# Data Dictionary

## Overview
This document describes the IBM Telco Customer Churn dataset, including all columns, data types, and feature descriptions.

## Dataset Information
- **Source**: IBM Telco Customer Churn Dataset
- **Total Records**: 7,043
- **Target Variable**: Churn (binary classification)
- **File Format**: CSV

## Feature Description

### Customer Information

| Column Name | Data Type | Description | Range/Values |
|-------------|-----------|-------------|--------------|
| customerID | str | Unique identifier for each customer | String (e.g., "7590-VHVEG") |
| gender | str | Customer gender | Male, Female |
| SeniorCitizen | int64 | Whether the customer is a senior citizen | 0 (No), 1 (Yes) |
| Partner | str | Whether the customer has a partner | Yes, No |
| Dependents | str | Whether the customer has dependents | Yes, No |

### Service Information

| Column Name | Data Type | Description | Range/Values |
|-------------|-----------|-------------|--------------|
| tenure | int64 | Number of months the customer has been with the company | 0 - 72 months |
| PhoneService | str | Whether the customer has phone service | Yes, No |
| MultipleLines | str | Whether the customer has multiple lines | Yes, No, No phone service |
| InternetService | str | Type of internet service | DSL, Fiber optic, No |
| OnlineSecurity | str | Whether the customer has online security | Yes, No, No internet service |
| OnlineBackup | str | Whether the customer has online backup | Yes, No, No internet service |
| DeviceProtection | str | Whether the customer has device protection | Yes, No, No internet service |
| TechSupport | str | Whether the customer has tech support | Yes, No, No internet service |
| StreamingTV | str | Whether the customer has streaming TV | Yes, No, No internet service |
| StreamingMovies | str | Whether the customer has streaming movies | Yes, No, No internet service |

### Account Information

| Column Name | Data Type | Description | Range/Values |
|-------------|-----------|-------------|--------------|
| Contract | str | Contract type | Month-to-month, One year, Two year |
| PaperlessBilling | str | Whether the customer has paperless billing | Yes, No |
| PaymentMethod | str | Payment method | Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic) |
| MonthlyCharges | float64 | Monthly charges amount | $18.25 - $118.75 |
| TotalCharges | str | Total charges amount (stored as string) | String (needs conversion to numeric) |

### Target Variable

| Column Name | Data Type | Description | Range/Values |
|-------------|-----------|-------------|--------------|
| Churn | str | Whether the customer churned | Yes, No |

## Data Quality Notes

### Missing Values
- No missing values in the dataset
- TotalCharges is stored as string and may need conversion to numeric
- Some TotalCharges values may be empty strings for new customers

### Data Distribution

#### Categorical Variables
- **gender**: Approximately balanced
- **SeniorCitizen**: 16.2% are senior citizens
- **Partner**: 48.3% have partners
- **Dependents**: 30.0% have dependents
- **Contract**: Month-to-month (55%), One year (21%), Two year (24%)
- **PaymentMethod**: Electronic check (33.6%), Mailed check (22.8%), Bank transfer (21.9%), Credit card (21.6%)

#### Numerical Variables
- **tenure**: Mean 32.4 months, Std 24.6
- **MonthlyCharges**: Mean $64.76, Std $30.09

#### Class Balance Analysis
- **Churn Rate**: 26.54% (1,869 customers)
- **Non-Churn Rate**: 73.46% (5,174 customers)
- **Imbalance Ratio**: 2.77:1 (Non-churn to churn)
- **Total Records**: 7,043 customers
- **Missing Values**: 0 in Churn column

## Feature Categories

### Numerical Features (StandardScaler Applied)
- SeniorCitizen
- tenure
- MonthlyCharges
- TotalCharges (after string conversion)

### Categorical Features (OneHotEncoder Applied)
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

### Identifier
- customerID (used for tracking, not in modeling)

## Preprocessing Applied
1. **String Conversion**: TotalCharges converted from string to numeric
2. **Train/Test Split**: 80% training, 20% testing (random_state=42)
3. **Numerical Scaling**: StandardScaler (mean=0, std=1)
4. **Categorical Encoding**: OneHotEncoder (binary columns for each category)
5. **Target Encoding**: Churn converted to binary (Yes=1, No=0)
