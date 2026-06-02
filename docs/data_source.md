# Data Source

## Dataset Information

**Source**: IBM Telco Customer Churn Dataset
**Format**: CSV
**Size**: 440,832 records × 12 features
**License**: Available for research and educational purposes

## Dataset Description

The IBM Telco Customer Churn dataset contains customer information from a telecommunications company, including demographics, service usage patterns, subscription details, and churn status.

## Data Fields

The dataset includes the following categories of information:

### Customer Demographics
- CustomerID: Unique customer identifier
- Age: Customer age (18-65)
- Gender: Customer gender (Male/Female)

### Service Usage
- Tenure: Number of months with the company (1-60)
- Usage Frequency: How frequently the customer uses the service (1-30)
- Support Calls: Number of customer support calls made (0-10)
- Last Interaction: Days since last customer interaction (1-30)

### Subscription Details
- Subscription Type: Type of subscription plan (Basic/Standard/Premium)
- Contract Length: Duration of contract (Monthly/Quarterly/Annual)
- Total Spend: Total amount spent by customer ($100-$1,000)

### Behavioral Metrics
- Payment Delay: Number of days payment was delayed (0-30)

### Target Variable
- Churn: Whether the customer churned (0=No, 1=Yes)

## Data Quality

- **Missing Values**: 1 row with all null values was removed during preprocessing
- **Duplicate Records**: None detected
- **Data Types**: Appropriate types for each feature
- **Outliers**: Within expected ranges for telecommunications data

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
- Some features may be correlated (e.g., Total Spend and Tenure)
- Class imbalance exists (56.7% churn rate)

## Data Access

The dataset should be placed in the `data/` directory as `customer_churn_dataset-training-master.csv` for the preprocessing pipeline to work correctly.
