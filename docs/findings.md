# Key Findings

## Dataset Overview

- **Total Records**: 7,043 customers (7,032 after cleaning)
- **Churn Rate**: 26.54% (1,869 churned vs. 5,174 retained)
- **Data Quality**: High — 11 rows removed (blank `TotalCharges`, all `tenure = 0` new customers)

## Customer Demographics

### Gender Distribution
- Male: 50.5%, Female: 49.5% (balanced)
- Minimal churn difference: Female 26.9% vs. Male 26.2%

### Senior Citizens
- 16.2% of customers are senior citizens
- Senior citizens churn at **41.7%** vs. **23.6%** for non-seniors

### Household
- Partner: 48.3% have a partner; customers without a partner churn more (33.0% vs. 19.7%)
- Dependents: 30.0% have dependents; customers without dependents churn more (31.3% vs. 15.5%)

## Service Usage Patterns

### Tenure Analysis
- Mean tenure: 32.4 months, Std: 24.6, Range: 0-72 months
- Churn is strongly inverse to tenure:
  - 0-12 months: **47.4%**
  - 13-24 months: 28.7%
  - 25-48 months: 20.4%
  - 49-72 months: **9.5%**

### Internet Service
- Fiber optic: 44.0%, DSL: 34.4%, No internet: 21.7%
- Fiber optic customers churn at **41.9%**, vs. DSL 19.0% and No internet 7.4%

### Phone Service
- 90.3% have phone service (little churn difference)

## Subscription Analysis

### Contract Type
- **Month-to-month**: 55.0% of customers — churn **42.7%**
- **One year**: 20.9% — churn 11.3%
- **Two year**: 24.1% — churn **2.8%**
- Contract type is the single strongest churn signal

### Payment Method
- **Electronic check**: 33.6% — churn **45.3%**
- Mailed check: 22.9% — churn 19.1%
- Bank transfer (automatic): 21.9% — churn 16.7%
- Credit card (automatic): 21.6% — churn 15.2%
- Manual electronic check payers churn ~3x more than automatic payers

## Financial Metrics

### Charges
- MonthlyCharges: Mean $64.76, Std $30.09, Range $18.25-$118.75
- TotalCharges: Mean $2,283.30, Std $2,266.77, Range $18.80-$8,684.80
- Higher monthly charges (often fiber optic) are associated with higher churn

## Churn Risk Factors

### High Risk Indicators
1. **Month-to-month contract** (42.7% churn)
2. **Short tenure** (< 12 months, 47.4% churn)
3. **Electronic check** payment (45.3% churn)
4. **Fiber optic** internet service (41.9% churn)
5. **Senior citizen** status (41.7% churn)

### Protective Factors
1. **Two-year contract** (2.8% churn)
2. **Long tenure** (> 48 months, 9.5% churn)
3. **Automatic payment** methods (15-17% churn)
4. **No internet / DSL** service (7.4% / 19.0% churn)
5. **Has partner and dependents**

## Business Implications

### Customer Segmentation
Distinct churn segments emerge from the data:
1. **High Risk**: Month-to-month, short tenure, electronic check, fiber optic
2. **Medium Risk**: One-year contract, mid tenure, DSL
3. **Low Risk**: Two-year contract, long tenure, automatic payment

### Retention Opportunities
- Early intervention programs for new customers (first 12 months)
- Contract conversion incentives (month-to-month → one/two year)
- Migrate electronic-check payers to automatic payment methods
- Investigate fiber optic experience/pricing driving elevated churn

## Data Quality Insights

- **Missing Data**: 11 blank `TotalCharges` values (new customers, tenure = 0)
- **Type Issue**: `TotalCharges` stored as string, requires numeric conversion
- **Duplicates**: None
- **Completeness**: 99.84% complete records

## Recommendations for Modeling

1. **Address class imbalance** through class weights (applied) or SMOTE
2. **Feature engineering** opportunities:
   - Create tenure-based segments / bands
   - Flag electronic-check and fiber-optic customers
   - Derive charges-per-tenure ratios
3. **Consider temporal aspects** if time-series data becomes available
4. **Explore interaction effects** between contract type and payment method
