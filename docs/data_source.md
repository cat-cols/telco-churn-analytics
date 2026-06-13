# Data Source

## Dataset Information

**Source**: IBM Telco Customer Churn Dataset
**Format**: CSV
**Size**: 7,043 records × 21 features
**License**: Available for research and educational purposes
**File Location**: `data/raw/telco_customer_churn.csv`

## Dataset Description

The IBM Telco Customer Churn dataset contains customer information from a telecommunications company, including demographics, service usage patterns, subscription details, and churn status.

## Data Fields

The dataset includes the following categories of information:

### Customer Demographics
- customerID: Unique customer identifier
- gender: Customer gender (Male/Female)
- SeniorCitizen: Whether customer is a senior citizen (0=No, 1=Yes)
- Partner: Whether customer has a partner (Yes/No)
- Dependents: Whether customer has dependents (Yes/No)

### Service Information
- tenure: Number of months with the company (0-72)
- PhoneService: Whether customer has phone service (Yes/No)
- MultipleLines: Whether customer has multiple lines (Yes/No/No phone service)
- InternetService: Type of internet service (DSL/Fiber optic/No)
- OnlineSecurity: Whether customer has online security (Yes/No/No internet service)
- OnlineBackup: Whether customer has online backup (Yes/No/No internet service)
- DeviceProtection: Whether customer has device protection (Yes/No/No internet service)
- TechSupport: Whether customer has tech support (Yes/No/No internet service)
- StreamingTV: Whether customer has streaming TV (Yes/No/No internet service)
- StreamingMovies: Whether customer has streaming movies (Yes/No/No internet service)

### Account Information
- Contract: Contract type (Month-to-month/One year/Two year)
- PaperlessBilling: Whether customer has paperless billing (Yes/No)
- PaymentMethod: Payment method (Electronic check/Mailed check/Bank transfer/Credit card)
- MonthlyCharges: Monthly charges amount ($18.25-$118.75)
- TotalCharges: Total charges amount (stored as string, needs conversion)

### Target Variable
- Churn: Whether the customer churned (Yes/No)

## Data Quality

- **Missing Values**: No missing values in the dataset
- **Duplicate Records**: None detected
- **Data Types**: Most features are appropriate types, TotalCharges stored as string
- **Outliers**: Within expected ranges for telecommunications data
- **Data Issues**: TotalCharges column stored as string, requires conversion to numeric

## Data Collection Method

The data was collected from a telecommunications company's customer relationship management (CRM) system over a specified period. It represents a snapshot of customer behavior and churn status.

## Data Usage Rights

This dataset is provided for educational and research purposes. When using this dataset, please:
- Cite the original source (IBM)
- Use it responsibly and ethically
- Do not attempt to identify individual customers
- Follow applicable data protection regulations

## Data Limitations

- Temporal information is not included (time-series analysis not possible)
- Geographic information is not available
- Some features may be correlated (e.g., MonthlyCharges and tenure)
- Class imbalance exists (26.5% churn rate)
- TotalCharges stored as string requires preprocessing

## Data Access

The dataset should be placed in the `data/raw` directory as `telco_customer_churn.csv` for the preprocessing pipeline to work correctly.
