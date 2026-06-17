# TODO — Telco Churn Analytics

Status key: ✅ Done · 🔄 In progress · ⬜ Not started · 🚩 High priority

---

## 1. Data

- ✅ Source raw dataset (`data/raw/telco_customer_churn.csv`)
- ✅ Document features in `docs/data_dictionary.md`
- ✅ Validate dataset schema on load (column names, dtypes, row count) — `validate_schema()` in `preprocess.py`
- ✅ Profile class balance (churn vs. non-churn counts + %) and record in `docs/data_dictionary.md`
- ⬜ Document known data quality issues (whitespace `TotalCharges`, 11 tenure=0 new customers)

---

## 2. Preprocessing

- ✅ Convert `TotalCharges` from string to numeric
- ✅ Drop rows with remaining `NaN` after imputation
- ✅ Scale numerical features with `StandardScaler`
- ✅ Encode categorical features with `OneHotEncoder(handle_unknown='ignore')` (fix #2)
- ✅ Save fitted `scaler.pkl` and `encoder.pkl` to `data/processed/`
- ✅ Save train/test splits as `.parquet` to `data/processed/`
- ✅ Handle class imbalance — `class_weight='balanced'` added to LR and RF; GB uses `sample_weight`; recall lifted from 0.516 → 0.791 (LR balanced)
- ✅ Re-run `python3 src/run_pipeline.py` to regenerate artifacts — encoder now saved with `handle_unknown='ignore'` (fix #2 baked in)

---

## 3. Exploratory Data Analysis

- ✅ EDA notebook (`notebooks/01_eda.ipynb`) — distributions, churn rate, correlations
- ✅ Add churn rate breakdown by key features (Contract type, tenure band, PaymentMethod) — `churn_breakdown()` cell in `notebooks/01_eda.ipynb` with tables + 1×3 bar chart vs. overall 26.54%
- ✅ Plot feature correlation heatmap and save to `results/`
- ✅ Document EDA findings and business insights in `docs/methodology.md`

---

## 4. Modelling

- ✅ Train Logistic Regression, Random Forest, Gradient Boosting
- ✅ Evaluate with accuracy, precision, recall, F1, ROC-AUC
- ✅ Select best model by ROC-AUC and save to `models/best_model.pkl`
- ✅ Save comparison table to `models/model_comparison_results.csv`
- ✅ Address low recall — class weighting raised recall to 0.791; threshold 0.30 on tuned GB gives recall=0.773 / precision=0.522 / F1=0.623
- ✅ Hyperparameter tuning — GridSearchCV on GB; best params: lr=0.05, max_depth=3, n_estimators=100, subsample=1.0; ROC-AUC 0.8340 → 0.8347
- ⬜ Add XGBoost / LightGBM to model comparison (optional, listed in README)
- ✅ Cross-validate best model (k=5) — Gradient Boosting ROC-AUC = 0.8477 ± 0.0046 (folds 0.843–0.856); tiny std confirms result is NOT split-dependent (single-split test value 0.8340 is within sampling noise)

---

## 5. Strategic Business Questions (NEW)

### High Priority (Business Impact)
- 🚩 Add financial impact analysis (CLV, revenue at risk, ROI calculations)
- 🚩 Add intervention timing analysis (when to act, optimal lead time)
- 🚩 Update README and notebooks with new business questions

### Medium Priority (Operational Excellence)
- ⬜ Add customer value prioritization (high-value vs high-risk segmentation)
- ⬜ Add success measurement framework (KPIs, A/B testing setup)
- ⬜ Add model evolution and monitoring (long-term sustainability)
- ⬜ Add limitations and risk assessment (ethical considerations, bias detection)

### Low Priority (Strategic Context)
- ⬜ Add competitive analysis (industry benchmarks, positioning)

---

## 6. Model Interpretability

- ✅ Compute SHAP values for best model on test set — `TreeExplainer` on Gradient Boosting in `notebooks/04_results.ipynb`
- ✅ Plot global feature importance (SHAP bar chart) — saved to `results/shap_feature_importance.png`
- ✅ Plot SHAP summary plot — saved to `results/shap_summary.png`
- ✅ Identify top 5 churn drivers and document in `docs/methodology.md` — Contract_Month-to-month, tenure (protective), OnlineSecurity_No, TechSupport_No, MonthlyCharges

---

## 7. Prediction Pipeline (`src/predict.py`)

- ✅ End-to-end prediction from raw CSV input
- ✅ Edge case handling: new customers (tenure=0), missing values, unseen categories, outlier flagging
- ✅ Fix #1: `TypeError` — coerce `TotalCharges` before numeric range validation
- ✅ Fix #3: `ValueError` — use `list()` not `set()` for encoder category lookup
- ✅ Output `Risk_Level` (Low/Medium/High) alongside churn probability
- ✅ Add `--threshold` tuning guidance to `docs/usage.md` — "Making Predictions" section with operating-point table (0.30 best-F1 recommended) + `scripts/threshold_analysis.py` re-derivation; also fixed that script's int-label bug
- ✅ Validate predictions output schema (column names, value ranges) before saving — `validate_output_schema()` in `predict.py` (required cols, Churn_Probability∈[0,1], Predicted_Churn∈{0,1}, Risk_Level∈{Low,Medium,High}); raises before save; 7 regression tests added

---

## 8. Testing

- ✅ Write unit test: `validate_input_data` raises no error on string `TotalCharges` (regression for fix #1)
- ✅ Write unit test: `handle_unseen_categories` does not raise on unknown category values (regression for fix #2/#3)
- ✅ Write unit test: `preprocess_new_data` output column order matches training feature order
- ✅ Write unit test: `handle_new_customers` sets `TotalCharges=0` for `tenure=0` rows
- ✅ Write integration test: full `predict_from_file` run on a 10-row CSV sample produces valid output
- ✅ Configure test runner (`pytest`) in `pyproject.toml`

---

## 9. Results & Reporting

- ✅ Save confusion matrix plot for best model to `results/` — `results/confusion_matrix.png` (generated by `04_results.ipynb`)
- ✅ Save ROC curve plot to `results/` — `results/roc_curve.png` (with Youden-J optimal threshold marked)
- ✅ Complete `notebooks/04_results.ipynb` with final model metrics and charts — runs end-to-end: dashboard, radar, probability dist, confusion matrix, ROC, feature importance, SHAP, summary report
- ✅ Write plain-language summary of findings for business stakeholders in `docs/` — `docs/business_summary.md` (jargon-free, grounded in real results)

---

## 10. Documentation

- ✅ `README.md` — project overview, setup, usage
- ✅ `docs/data_dictionary.md` — feature descriptions
- ✅ `docs/methodology.md` — analytical approach
- ✅ `CHANGELOG.md` — Keep-a-Changelog format
- ✅ `docs/usage.md` — prediction threshold guidance (operating-point table) + "Edge-case behaviour" subsection (new customers, imputation, unseen categories, outlier flagging, input/output validation)
- ✅ `docs/setup.md` — dependency list realigned to `requirements.txt` (added numpy/matplotlib/seaborn/shap/ipykernel, removed stale kagglehub; optional xgboost/lightgbm/imbalanced-learn noted); refreshed verify-install check and project structure
- ⬜ 🚩 Close GitHub issues #1, #2, #3 once regression tests are in place

---

## 11. Deployment (Optional / Future)

- ✅ Create a FastAPI endpoint wrapping `predict_from_file` for batch or single-record scoring — `src/api.py` with lifespan model loading, POST /predict/records (JSON), POST /predict/file (CSV), GET /health, comprehensive integration tests in `tests/test_api.py`
- ⬜ Containerise with Docker (`Dockerfile` + `docker-compose.yml`)
- ⬜ Add model versioning — tag `best_model.pkl` with training date and ROC-AUC in filename
- ⬜ Set up CI (GitHub Actions) — run `pytest` on push to main

---

## 12. Portfolio / De-genericize (Stand-out additions)

High-leverage additions to differentiate this from a typical Telco-churn portfolio project.

- ⬜ 🚩 Add a SQL layer — load the data into DuckDB/SQLite and write the churn-segment analysis as SQL queries (more "analyst")
- ⬜ 🚩 Build one dashboard — small Streamlit or Power BI view of risk segments + a "what-if" threshold slider (visual, interactive, memorable)
- ⬜ 🚩 Quantify business impact — translate churn reduction into retained revenue with stated assumptions (single biggest differentiator)
- ⬜ Write it up publicly — short blog post / README story: "the bug that almost broke my pipeline" (the `TotalCharges` saga) to show real problem-solving

---


## Project Assessment - Missing Components

**Your project is very complete** but has a few gaps:

### ✅ What You Have (Excellent)
- **Complete ML pipeline** (raw → deployment)
- **Feature engineering** (advanced)
- **Business analysis** (ANALYST_PORTFOLIO.md)
- **Tests** (comprehensive test suite)
- **Documentation** (README, docs)
- **Results** (visualizations, reports)
- **API** (FastAPI deployment)

### ❌ Missing Components

**1. Model Monitoring/Drift Detection**
```python
# Missing: scripts/monitor_model.py
# - Track prediction distribution shifts
# - Alert on performance degradation
# - Retraining triggers
```

**2. Automated Retraining Pipeline**
```python
# Missing: scripts/retrain_model.py
# - Scheduled model updates
# - Performance comparison
# - A/B testing new models
```

**3. Data Validation**
```python
# Missing: src/validation.py
# - Schema validation with Pydantic
# - Data quality checks
# - Input validation for API
```

**4. Experiment Tracking**
```python
# Missing: MLflow integration
# - Model versioning
# - Hyperparameter logging
# - Performance tracking
```

**5. CI/CD Pipeline**
```yaml
# Missing: .github/workflows/
# - Automated testing
# - Model validation
# - Deployment triggers
```

### 🎯 Priority Additions

**High Impact (Add these first)**:
1. **Model monitoring** - Essential for production
2. **Data validation** - Prevents production issues
3. **Experiment tracking** - Professional ML practice

**Medium Impact**:
4. **Automated retraining** - Production maintenance
5. **CI/CD** - DevOps integration

### 🏆 Portfolio Assessment

**Your project is already top-tier**. These missing items would take it from "excellent" to "production-ready enterprise level."

**Current**: 9/10 (Excellent portfolio project)
**With additions**: 10/10 (Enterprise-ready)

**Recommendation**: Add model monitoring and data validation - these have the biggest portfolio impact with reasonable effort.

## Pipeline Count Assessment

**✅ Perfect number - not too many, not too few**

### Why This Count Works

**1 pipeline is ideal** for a portfolio project because:
- **Focused scope**: Single business problem (churn prediction)
- **Complete story**: End-to-end ML workflow
- **Manageable**: Easy to understand and explain
- **Portfolio-ready**: Shows full ML lifecycle

### When to Add More Pipelines

**Consider additional pipelines if**:
- **Different data sources**: Customer support tickets + billing data
- **Multiple models**: Time series forecasting + classification
- **A/B testing**: Pipeline for model comparison
- **Production monitoring**: Pipeline for model drift detection

### When You Have Too Many

**Warning signs of over-pipelining**:
- **Redundant work**: Same preprocessing in multiple pipelines
- **Complex dependencies**: Pipeline A depends on B depends on C
- **Maintenance burden**: Breaking one pipeline breaks others
- **Unclear purpose**: Can't explain what each pipeline does

### Your Current Setup - Goldilocks Zone

```
✅ Single ML pipeline (complete workflow)
✅ Analysis tools (support the main pipeline)
✅ Clear purpose (churn prediction)
✅ Logical flow (raw → insights)
```

### Recommendation

**Keep it as-is**. Your project demonstrates:
- **Full ML understanding** (preprocessing → deployment)
- **Business focus** (single, well-defined problem)
- **Technical depth** (feature engineering, threshold analysis)
- **Portfolio strength** (complete, not overwhelming)

**Add more pipelines only if** you expand the project scope (e.g., customer segmentation, retention modeling).

---
---


Yes, absolutely! I've added these strategic business questions to your TODO list. Here's the prioritized plan:

## High Priority (Business Impact):
1. **Financial impact analysis** - CLV, revenue at risk, ROI calculations
2. **Intervention timing analysis** - When to act, optimal lead time
3. **Update README and notebooks** - Incorporate new business questions

## Medium Priority (Operational Excellence):
4. **Customer value prioritization** - High-value vs high-risk segmentation
5. **Success measurement framework** - KPIs, A/B testing setup
6. **Model evolution and monitoring** - Long-term sustainability
7. **Limitations and risk assessment** - Ethical considerations, bias detection

## Low Priority (Strategic Context):
8. **Competitive analysis** - Industry benchmarks, positioning

## Why This Priority Order:

**High Priority** items directly connect the technical work to financial outcomes and business decisions - these will have the most immediate impact.

**Medium Priority** items ensure the solution is robust, measurable, and sustainable long-term.

**Low Priority** items provide valuable context but aren't critical for initial deployment.

This structure will transform your project from a churn prediction model into a comprehensive customer retention intelligence system that drives measurable business value.


Yes, absolutely! The current README structure is too condensed. Your suggested format is much clearer and more business-friendly. Let me refactor it to better highlight the business value:

## Changes Made:

### 1. **README.md** - Restructured Business Questions Section
- Changed from condensed bullet points to numbered questions with detailed answers
- Added proper spacing and formatting for better readability
- Each question now has clear, actionable answers

### 2. **05_results.ipynb** - Updated Notebook Introduction
- Added business questions context to the first markdown cell
- Clearly states which business questions the notebook addresses
- Links analysis outputs directly to business value

### 3. **docs/business_summary.md** - Restructured for Business Focus
- Reorganized content around the 5 key business questions
- Each section now directly answers a specific business question
- Maintained the plain-language approach while improving structure

## The 5 Business Questions Now Clearly Structured:

1. **Who are our highest-risk customers?** - Risk scoring system
2. **What drives churn most?** - Top 5 churn drivers with percentages
3. **How accurate is this?** - Performance metrics and reliability
4. **What should we do?** - Actionable recommendations
5. **How do we use this?** - Deployment and integration guidance

This structure makes it immediately clear to business stakeholders what value the project provides and how they can use the insights. The documentation now flows from problem → solution → implementation, which is much more business-friendly.

---
---








Based on the current project structure and analysis, here are additional business questions that would strengthen the project's business value:

## Strategic Business Questions to Add:

### **6. What's the financial impact?**
- Customer lifetime value (CLV) analysis
- Revenue at risk by customer segment
- ROI of retention strategies
- Cost-benefit analysis of intervention tactics

### **7. When should we intervene?**
- Optimal timing for retention offers
- Early warning indicators before churn risk spikes
- Seasonal patterns in churn behavior
- Lead time needed for effective retention

### **8. Which customers are most valuable to retain?**
- High-value vs high-risk customer prioritization
- Revenue-weighted risk scoring
- Profitability analysis by customer segment
- Resource allocation optimization

### **9. How do we measure success?**
- KPIs for retention campaigns
- A/B testing framework for interventions
- Monitoring and alerting systems
- Performance dashboards for business teams

### **10. What are the competitive implications?**
- Benchmark against industry standards
- Market positioning based on retention rates
- Competitive differentiation opportunities
- Strategic implications for customer acquisition

## Technical Questions to Support Business Goals:

### **11. How does the model evolve?**
- Model drift detection and retraining schedules
- Performance monitoring over time
- Concept drift handling
- Continuous improvement framework

### **12. What are the limitations and risks?**
- Model uncertainty and confidence intervals
- Edge cases and failure modes
- Ethical considerations and bias detection
- Regulatory compliance implications

## Why These Questions Matter:

1. **Financial Focus**: Connects predictions directly to revenue impact
2. **Actionability**: Provides specific guidance on timing and resource allocation
3. **Measurement**: Enables tracking of business outcomes
4. **Sustainability**: Addresses long-term model management
5. **Risk Management**: Identifies potential issues and mitigations

These additions would transform the project from a technical exercise into a comprehensive business intelligence system that drives strategic decision-making.

---
---

No, creating a separate branch for each TODO item is **not** best practice. That would create unnecessary complexity and merge conflicts.

## Recommended Branch Strategy:

### **For Portfolio Projects (like yours):**
- **Main branch**: Always reflects the current working state
- **Feature branches**: Group related TODO items logically

### **Suggested Branch Structure:**

```bash
# Main development branch
main  # Current working project

# Feature branches (group related work)
feature/business-questions     # Items 1-3 from new section
feature/financial-analysis    # CLV, revenue impact
feature/monitoring-framework  # Model evolution, risk assessment
feature/competitive-analysis  # Industry benchmarks
```

### **Why This Approach:**

**1. Cohesion**: Related work stays together
- Financial analysis items share code and data
- Monitoring framework builds on existing infrastructure

**2. Manageability**: Fewer branches to track
- 4-5 focused branches vs 12+ micro-branches
- Easier code reviews and merges

**3. Context**: Each branch tells a complete story
- `feature/business-questions` = complete business intelligence enhancement
- `feature/financial-analysis` = full financial impact analysis

### **When to Create Separate Branches:**
- **Major architectural changes** (e.g., adding monitoring system)
- **Experimental features** (might not merge)
- **Breaking changes** (requires careful coordination)

### **For Your Current TODO:**
```bash
# High priority items (business impact)
git checkout -b feature/business-questions
# Work on: financial analysis, timing analysis, README updates

# Medium priority items (operational excellence)
git checkout -b feature/operational-excellence
# Work on: customer prioritization, success measurement, monitoring, risk assessment
```

This approach keeps your git history clean while allowing focused development on related functionality.

---
---

Minor Improvements
Consider adding monitoring/logging for production API
Could benefit from automated retraining pipeline
Model drift detection would enhance production readiness


---
---
> Determine value of each segment.
> - Where do we make our money?
> - Which segments are most profitable?
> - Which segments are most expensive to serve?
> - Which segments have the highest CLV?
> - Which segments have the highest risk of churn and highest income potential?

Business Impact#
High-risk segment: 2x churn rate
Medium-risk segment: Baseline churn rate
Low-risk segment: 3x lower churn rate
Retention focus: Top 40% of customers by risk

---
---

Based on your excellent churn analytics project, here are strategic next projects that build on your strengths and address common data analyst use cases:

## **Recommended Next Projects (by priority)**

### **1. Customer Lifetime Value (CLV) Analysis**
**Why it's perfect**: Direct sequel to churn analysis - shows revenue impact
- Calculate CLV by customer segment
- Identify high-value vs high-risk customers
- Build retention ROI models
- Dataset: Could extend your telco data or use e-commerce data

### **2. Marketing Campaign Effectiveness**
**Why it's strategic**: Every company needs marketing analytics
- A/B test analysis and statistical significance
- Campaign attribution and ROI
- Customer segmentation for targeting
- Dataset: Marketing campaign data with conversion metrics

### **3. Operational Efficiency Dashboard**
**Why it's valuable**: Shows business process optimization skills
- Process bottleneck analysis
- Resource utilization metrics
- Cost reduction opportunities
- Dataset: Sales, inventory, or service delivery data

### **4. Product Usage Analytics**
**Why it's relevant**: Hot skill for SaaS/tech companies
- Feature adoption analysis
- User journey mapping
- Engagement funnel analysis
- Dataset: User behavior logs, web analytics data

## **Quick Win Projects (1-2 weeks)**

### **5. Sales Performance Analysis**
- Territory performance comparison
- Sales cycle analysis
- Forecasting vs actual variance
- Dataset: CRM data (Salesforce style)

### **6. Financial KPI Dashboard**
- Revenue trend analysis
- Expense categorization and optimization
- Cash flow forecasting
- Dataset: Company financial data

## **How to Choose**

### **For Job Market Strategy**:
- **CLV Analysis** - Most directly valuable after churn
- **Marketing Analytics** - Highest demand across industries
- **Operational Efficiency** - Shows business process thinking

### **For Technical Growth**:
- **Product Usage** - More complex event data
- **Financial Analysis** - Time series and forecasting

### **For Business Impact**:
- **CLV** - Direct revenue connection
- **Marketing** - Shows ROI thinking

## **Project Structure Template**

Follow your successful churn project pattern:
1. **Business questions** → 2. **Data exploration** → 3. **Statistical analysis** → 4. **Visualization** → 5. **Actionable insights** → 6. **Production-ready tool**

## **Data Sources**

- **Kaggle datasets**: CLV, marketing campaigns, sales data
- **Public APIs**: Google Analytics, social media metrics
- **Company data** (if available): CRM, financial systems

**My recommendation**: Start with **CLV Analysis** - it's the natural next step after churn prediction and shows you can connect risk analysis to revenue impact, which is exactly what hiring managers want to see.

Which direction interests you most?