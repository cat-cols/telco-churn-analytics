# 📊 Data Analyst Portfolio - Telco Churn Project

## 🎯 **Project Overview**

**Role**: Data Analyst  
**Industry**: Telecommunications  
**Dataset**: 7,043 customers, 21 attributes  
**Business Problem**: Customer churn analysis and retention strategy  

---

## 📈 **Business Questions Answered**

### **1. Who are our highest-risk customers?**
- **Segmented customers** into risk levels (Low/Medium/High)
- **Identified key characteristics** of each segment
- **Created actionable profiles** for targeted marketing

### **2. What drives customer churn?**
- **Statistical analysis** revealed 4 primary drivers
- **Quantified impact** of each factor
- **Validated significance** with chi-square tests

### **3. How can we reduce churn?**
- **Developed 8 specific recommendations**
- **Calculated ROI** for each strategy
- **Prioritized by business impact**

---

## 🔧 **Technical Skills Demonstrated**

### **Data Analysis**
```python
# Statistical significance testing
import scipy.stats as stats
chi2, p_value = stats.chi2 contingency_table

# Customer segmentation analysis
customer_segments = df.groupby(['Contract', 'tenure_group'])['Churn'].mean()

# Correlation analysis
correlation_matrix = df.corr()
```

### **Business Intelligence**
- **KPI Development**: Churn rate, customer lifetime value, retention ROI
- **Executive Reporting**: Clear summaries with actionable insights
- **Data Visualization**: Charts and dashboards for stakeholders

### **Data Quality**
- **Missing value handling**: 11 new customers with zero tenure
- **Data validation**: Consistency checks and outlier detection
- **Data preparation**: SQL-ready format for enterprise systems

---

## 💼 **Projected Business Impact**

### **Quantified Results**
- **Identified 77% of at-risk customers** before churn
- **Projected $351K annual savings** from retention programs
- **4.2x ROI** on targeted retention strategies

### **Strategic Value**
- **Evidence-based decision making** for retention programs
- **Customer segmentation** for personalized marketing
- **Financial modeling** for business case development

---

## 📊 **Analysis Process**

### **1. Data Exploration**
```python
# Initial data assessment
df.info()
df.describe()
df.isnull().sum()

# Churn rate analysis
overall_churn_rate = df['Churn'].value_counts(normalize=True)['Yes']
```

### **2. Statistical Analysis**
```python
# Contract type impact
contract_churn = df.groupby('Contract')['Churn'].value_counts(normalize=True)
chi2_test = stats.chi2_contingency(contract_table)

# Tenure analysis
df['tenure_group'] = pd.cut(df['tenure'], bins=[0,12,24,60,72])
tenure_churn = df.groupby('tenure_group')['Churn'].mean()
```

### **3. Visualization & Reporting**
```python
# Churn by contract type
sns.barplot(data=df, x='Contract', y='Churn', hue='Churn')

# Customer tenure distribution
sns.histplot(data=df, x='tenure', hue='Churn', multiple='stack')
```

---

## 🎯 **Key Findings Summary**

| Factor | Impact | Statistical Significance |
|--------|---------|-------------------------|
| Contract Type | 43% vs 3% churn | p < 0.001 |
| Payment Method | 45% vs 15% churn | p < 0.001 |
| Customer Tenure | 47% vs 10% churn | p < 0.001 |
| Internet Service | 42% vs 19% churn | p < 0.001 |

---

## 💡 **Business Recommendations**

### **High Impact, Low Effort**
1. **Auto-pay incentives** - 3x churn reduction
2. **Contract upgrade offers** - 4x churn reduction
3. **New customer onboarding** - 2x churn reduction

### **Medium Impact, Medium Effort**
1. **Customer success program** for high-risk segments
2. **Service quality improvements** for fiber optic
3. **Retention scoring system** implementation

---

## 🔍 **Data Analyst Skills Highlighted**

### **Technical Skills**
- **Python**: pandas, numpy, scipy for statistical analysis
- **SQL**: Churn-segment analysis in DuckDB — GROUP BY aggregates, CASE WHEN cohorts, window functions (`sql/churn_analysis.sql`)
- **Excel**: Business calculations and reporting
- **Visualization**: matplotlib, seaborn, Tableau-ready formats

### **Business Skills**
- **Statistical Analysis**: Significance testing, correlation analysis
- **Business Intelligence**: KPI development, executive reporting
- **Data Storytelling**: Clear communication of insights
- **ROI Analysis**: Financial modeling and business cases

### **Soft Skills**
- **Problem Solving**: Translating business questions into analysis
- **Communication**: Executive summaries and presentations
- **Attention to Detail**: Data quality and validation
- **Business Acumen**: Understanding customer retention economics

---

## 📈 **Portfolio Value for Job Search**

### **What Recruiters Look For**
✅ **End-to-end analysis** from data to recommendations  
✅ **Business impact** with quantified results  
✅ **Statistical rigor** with significance testing  
✅ **Clear communication** of complex findings  
✅ **SQL-ready** data preparation  

### **Competitive Advantages**
- **Real business problem** (not just academic exercise)
- **Quantified ROI** and financial impact
- **Executive-ready** reporting format
- **Statistical validation** of insights
- **Actionable recommendations** with prioritization

---

## 🎯 **Next Steps for Recruiters**

1. **Review the full analysis** in `BUSINESS_ANALYSIS.md`
2. **Examine the code** in `notebooks/` for technical depth
3. **Check the visualizations** in `outputs/` for communication skills
4. **Contact me** to discuss similar business challenges

---

**📧 Let's discuss how I can bring similar data-driven insights to your team!**
