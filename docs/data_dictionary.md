# Data Dictionary

## Overview
This document describes the IBM Telco Customer Churn dataset, including all columns, data types, and feature descriptions.

## Dataset Information
- **Source**: IBM Telco Customer Churn Dataset
- **Total Records**: 440,832 (after cleaning)
- **Target Variable**: Churn (binary classification)
- **File Format**: CSV/Excel

## Feature Description

### Customer Information

| Column Name | Data Type | Description | Range/Values |
|-------------|-----------|-------------|--------------|
| CustomerID | float64 | Unique identifier for each customer | Numeric (2 - 449,999) |
| Age | float64 | Customer age in years | 18 - 65 |
| Gender | object | Customer gender | Male, Female |
| Tenure | float64 | Number of months the customer has been with the company | 1 - 60 months |

### Usage Behavior

| Column Name | Data Type | Description | Range/Values |
|-------------|-----------|-------------|--------------|
| Usage Frequency | float64 | How frequently the customer uses the service | 1 - 30 |
| Support Calls | float64 | Number of customer support calls made | 0 - 10 |
| Payment Delay | float64 | Number of days payment was delayed | 0 - 30 days |
| Last Interaction | float64 | Days since last customer interaction | 1 - 30 days |

### Subscription Details

| Column Name | Data Type | Description | Range/Values |
|-------------|-----------|-------------|--------------|
| Subscription Type | object | Type of subscription plan | Basic, Standard, Premium |
| Contract Length | object | Duration of contract | Monthly, Quarterly, Annual |
| Total Spend | float64 | Total amount spent by customer | $100 - $1,000 |

### Target Variable

| Column Name | Data Type | Description | Range/Values |
|-------------|-----------|-------------|--------------|
| Churn | float64 | Whether the customer churned (1) or not (0) | 0 (No), 1 (Yes) |

## Data Quality Notes

### Missing Values
- Original dataset contained 1 row with all null values (row index 199,295)
- This row was removed during data cleaning
- Final dataset: 440,832 complete records with no missing values

### Data Distribution

#### Categorical Variables
- **Subscription Type**: Balanced distribution (~33% each for Basic, Standard, Premium)
- **Contract Length**: Annual (40.2%), Quarterly (40.0%), Monthly (19.8%)
- **Gender**: Mixed distribution (specific percentages available in EDA)

#### Numerical Variables
- **Age**: Mean 39.4 years, Std 12.4
- **Tenure**: Mean 31.3 months, Std 17.3
- **Total Spend**: Mean $631.6, Std $240.8
- **Churn Rate**: 56.7% (imbalanced dataset)

## Feature Categories

### Numerical Features (StandardScaler Applied)
- Age
- Tenure
- Usage Frequency
- Support Calls
- Payment Delay
- Total Spend
- Last Interaction

### Categorical Features (OneHotEncoder Applied)
- Subscription Type
- Contract Length

### Identifier
- CustomerID (used for tracking, not in modeling)

## Preprocessing Applied
1. **Null Handling**: Removed single row with all null values
2. **Train/Test Split**: 80% training, 20% testing (random_state=42)
3. **Numerical Scaling**: StandardScaler (mean=0, std=1)
4. **Categorical Encoding**: OneHotEncoder (binary columns for each category)
