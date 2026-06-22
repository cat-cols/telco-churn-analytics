# Telco Churn Analytics

📊 **Data analysis project identifying customer churn drivers** with actionable business insights. Built with Python, pandas, and statistical analysis for customer retention strategies.

## 📊 Business Impact

**27% of customers churn** - we can identify **~82% of them before they leave** (recall at a 0.25 threshold) with our prediction system. This enables proactive retention that can save significant revenue.

### Key Business Questions Answered:

**1. Who are our highest-risk customers?**

Provides risk scoring with Low/Medium/High levels
Identifies specific customers most likely to churn

**2. What drives churn most?**

Identifies top churn drivers: Month-to-month contracts, new customers, payment method
Quantifies impact of each factor

**3. How accurate is this?**

~84% ROC-AUC performance (0.835 for the best model)
Stable across data samples
~82% recall at the 0.25 operating threshold

**4. What should we do?**

Specific, data-driven retention strategies
Actionable recommendations for each risk segment

**5. How do we use this?**

Ready-to-deploy API for integration
Threshold tuning for business needs
Production-ready prediction system

## 🎯 Quick Wins for Business

Based on our analysis, these actions will have the biggest impact:

1. **Target month-to-month customers** - 43% churn vs 3% for 2-year contracts
2. **Focus on new customers** - 47% churn in first year vs 10% after 4 years  
3. **Move customers to auto-pay** - Electronic check payers churn 3× more
4. **Investigate fiber experience** - 42% churn vs 19% for DSL

## 📈 Analysis Results

Our **statistical analysis** reveals:
- **0.835 ROC-AUC** for ranking at-risk customers (79% raw accuracy)
- **~82% of potential churners** identified at the 0.25 operating threshold
- **~49% precision** at that threshold (65% precision at the default 0.50)
- **Statistically significant findings** across customer segments

![Analysis Results](results/model_performance_dashboard.png)

## 🏗️ Technical Overview

This data analysis project uses the IBM Telco Customer Churn dataset (7,043 customers, 21 features) with comprehensive analysis for business insights:

- ✅ **Data Cleaning**: Handles missing values, new customers, data quality issues
- ✅ **Statistical Analysis**: Chi-square tests, correlation analysis, significance testing
- ✅ **Data Visualization**: Charts, dashboards, and business intelligence reports
- ✅ **Segmentation Analysis**: Customer cohorts and behavioral patterns
- ✅ **Business Insights**: Actionable recommendations with ROI estimates
- ✅ **SQL Analysis Layer**: Churn-segment analysis written as DuckDB SQL (`sql/churn_analysis.sql`)

## 🗂️ Project Structure

```
telco-churn-analytics/
├── 📊 results/                     # Business insights & visualizations
│   ├── model_performance_dashboard.png  # Key performance metrics
│   ├── shap_*.png                  # Feature importance analysis
│   ├── confusion_matrix.png        # Prediction accuracy
│   └── *.csv                       # Detailed results
├── 📁 data/
│   ├── raw/                        # Raw dataset (telco_customer_churn.csv)
│   └── processed/                  # Scaled/encoded parquet splits + .pkl artifacts
├── 📚 docs/                        # Business & technical documentation
│   ├── business_summary.md         # 🎯 Non-technical business insights
│   ├── data_dictionary.md          # Feature descriptions
│   ├── methodology.md              # Analytical approach
│   ├── setup.md                    # Installation guide
│   └── usage.md                    # Usage and threshold guidance
├── 🤖 models/
│   ├── best_model.pkl              # Trained prediction model
│   └── model_comparison_results.csv # All model variants
├── 📓 notebooks/                   # Exploratory analysis
│   ├── 01_eda.ipynb                # Data exploration
│   ├── 02_preprocessing.ipynb      # Data preparation
│   ├── 03_feature_engineering.ipynb # Feature engineering
│   ├── 04_modeling.ipynb           # Model training
│   └── 05_results.ipynb            # Results visualization
├── 🗄️ sql/                         # SQL analysis layer (DuckDB)
│   └── churn_analysis.sql          # Churn-segment queries (GROUP BY, CASE, window fns)
├── ⚙️ scripts/                     # Utility scripts
│   ├── threshold_analysis.py       # Precision/recall analysis
│   └── sql_analysis.py             # Runs sql/churn_analysis.sql via DuckDB
├── 🚀 src/                         # Production pipeline
│   ├── api.py                      # 🌐 FastAPI deployment service
│   ├── config.py                   # Configuration constants
│   ├── utils.py                    # Helper functions
│   ├── preprocess.py               # Data preprocessing
│   ├── train.py                    # Model training
│   ├── predict.py                  # Prediction service
│   └── run_pipeline.py             # End-to-end orchestrator
├── 🧪 tests/                       # Test suite
│   ├── test_api.py                 # API integration tests
│   ├── test_predict.py             # Prediction tests
│   └── test_*.py                   # Other test modules
└── 📝 notes/                       # Technical notes
    └── 07_FastAPI.md               # API implementation guide
```

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/cat-cols/telco-churn-analytics
cd telco-churn-analytics

python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -e .
```

> **No data download required.** The dataset (`data/raw/telco_customer_churn.csv`, ~955 KB, public IBM Telco Customer Churn data) ships with the repo, so the full pipeline runs end-to-end straight from a fresh clone.

### 2. Run Full Pipeline
```bash
# Complete pipeline: preprocessing → training → predictions
python3 src/run_pipeline.py
```

### 3. Deploy API Server
```bash
# Start prediction API service
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

# Access interactive docs at: http://localhost:8000/docs
```

### 4. Make Predictions
```bash
# Batch predictions
python3 src/predict.py --input data/new_customers.csv --output predictions.csv

# Interactive threshold analysis for business tuning
python3 scripts/interactive_threshold.py

# Detailed precision/recall analysis
python3 scripts/threshold_analysis.py
```

## 🎯 Using the Results

### For Business Teams:
- **Read**: `docs/business_summary.md` - Non-technical insights and recommendations
- **View**: `results/` folder - Visualizations and performance charts
- **Use**: API service - Integrate with your retention systems

### For Technical Teams:
- **Explore**: `notebooks/` - Full analysis and model development
- **Deploy**: `src/api.py` - Production-ready FastAPI service
- **Test**: `python3 -m pytest tests/` - Comprehensive test suite

## 📊 Key Visualizations

![Model Performance Dashboard](results/model_performance_dashboard.png)
*Comprehensive model comparison across all metrics*

![SHAP Feature Importance](results/shap_feature_importance.png)
*Top churn drivers: Month-to-month contracts, tenure, lack of support services*

![Interactive 3D Segmentation](results/interactive_3d_segmentation.html)
*Customer risk segments in 3D space - explore interactively*

![Model Performance](results/model_radar_chart.png)
*Model comparison across key metrics*

![Risk Distribution](results/probability_distribution.png)
*Customer risk segmentation*

![Feature Correlations](results/correlation_heatmap.png)
*Relationships between customer features and churn patterns*

## 🧪 Testing & Quality

```bash
# Run all tests
python3 -m pytest tests/ -v

# Test API endpoints
python3 -m pytest tests/test_api.py -v

# Test prediction logic
python3 -m pytest tests/test_predict.py -v

# Code style checking
flake8 src/ tests/

# Security vulnerability scanning
bandit -r src/
```

## 🔧 Advanced Usage

### Interactive Analysis Tools
```bash
# Interactive threshold tuning with visual feedback
python3 scripts/interactive_threshold.py

# Class imbalance analysis and recommendations
python3 scripts/analyze_class_balance.py

# Comprehensive threshold sweep analysis
python3 scripts/threshold_analysis.py --step 0.02 --output results/threshold_sweep.csv
```

### SQL Analysis Layer
Run the churn-segment analysis as SQL (DuckDB queries the raw CSV directly — no server):
```bash
pip install -e ".[sql]"

# Print all query results
python3 scripts/sql_analysis.py

# Export each result to results/sql/<name>.csv
python3 scripts/sql_analysis.py --save

# Run a single named query
python3 scripts/sql_analysis.py --query churn_by_contract
```
Queries live in `sql/churn_analysis.sql` and cover churn rates by contract, payment method, internet service, and tenure cohorts, a compound high-risk segment, a window-function risk ranking, and revenue-at-risk.

### 3D Customer Segmentation
Explore customer segments interactively in your browser:
```bash
# Open interactive 3D visualization
open results/interactive_3d_segmentation.html
```

### Custom Threshold Tuning
```bash
# Fine-grained threshold analysis
python3 scripts/threshold_analysis.py --step 0.02 --output results/threshold_sweep.csv
```

### Individual Pipeline Stages
```bash
python3 src/preprocess.py                    # Data preprocessing
python3 src/train.py                         # Model training
python3 src/predict.py --input file.csv     # Make predictions
```

### Jupyter Exploration
```bash
jupyter notebook notebooks/
# Run in order: 01_eda → 02_preprocessing → 03_feature_engineering → 04_modeling → 05_results
```

## 📋 Dataset Overview

**IBM Telco Customer Churn dataset** — 7,043 customers, 21 features:

### Customer Features:
- **Demographics**: gender, SeniorCitizen, Partner, Dependents
- **Services**: PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- **Account**: Contract, PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges
- **Target**: Churn (Yes / No) — **27% positive rate**

### Data Quality Handled:
- ✅ Missing values in `TotalCharges` for new customers
- ✅ 11 new customers (tenure=0) properly flagged and handled
- ✅ Class imbalance addressed with balanced models

## 🤖 Model Performance

### Best Model: **Gradient Boosting (tuned)**
- **0.835 ROC-AUC** - Strong discrimination ability
- **~82% recall** at the 0.25 operating threshold - catches most churners
- **~49% precision** at 0.25 (65% at the default 0.50 threshold)
- **79% accuracy** at the default threshold
- **Stable performance** - Cross-validated and reliable

### Performance Tradeoffs:
| Metric | Conservative | **Balanced** | Aggressive |
|---|---|---|---|
| **Threshold** | 0.50 | **0.25** | 0.20 |
| **Recall** | 48% | **82%** | 85% |
| **Precision** | 65% | **49%** | 46% |
| **F1 Score** | 55% | **62%** | 60% |

**Recommendation**: Use balanced threshold (0.25) for optimal business impact.

## 📚 Documentation

### 🎯 For Business Stakeholders:
- **[Business Summary](docs/business_summary.md)** - Plain-language insights and recommendations
- **[Executive Walkthrough](docs/executive_walkthrough.md)** - High-level overview and action plan

### 🔧 For Technical Teams:
- **[Data Dictionary](docs/data_dictionary.md)** - Feature descriptions and types
- **[Setup Guide](docs/setup.md)** - Installation and configuration
- **[Usage Guide](docs/usage.md)** - Prediction pipeline and threshold tuning
- **[Methodology](docs/methodology.md)** - Analytical approach and decisions
- **[FastAPI Guide](notes/07_FastAPI.md)** - API deployment and patterns

### 🧪 For Development:
- **[Contributing Guide](docs/contributing.md)** - How to contribute
- **[Technical Implementation](notes/technical_implementation_guide.md)** - Architecture and design
- **[Debugging Log](notes/debugging_log.md)** - Issues and solutions
- **Code Quality**: Uses flake8 for style checking and bandit for security scanning

## 🌟 Business Value & Technical Skills

### ✅ What We Deliver:
- **Risk Scoring**: Low/Medium/High risk levels for every customer
- **Actionable Insights**: Clear drivers of churn with specific recommendations
- **Production Ready**: REST API service for integration with existing systems
- **Explainable AI**: SHAP analysis shows why customers are flagged
- **Data Visualization**: Comprehensive charts and business intelligence dashboards

### 💰 Expected Business Impact:
- **Identify 77% of at-risk customers** before they churn
- **Focus retention efforts** on high-impact segments
- **Reduce customer acquisition costs** through better retention
- **Data-driven decision making** for retention strategies

### 🔧 Technical Skills Demonstrated:
- **Machine Learning**: Classification, feature engineering, model validation
- **Data Science**: Statistical analysis, pandas data manipulation, data visualization
- **Software Engineering**: FastAPI development, REST API design, testing with pytest
- **MLOps**: Model deployment, production ML, API integration
- **Business Intelligence**: Customer analytics, churn prediction, retention strategies

## 🚀 Deployment Options

### 1. **API Service** (Recommended)
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```
- Real-time predictions
- Interactive documentation at `/docs`
- Batch processing support
- Health monitoring

### 2. **Docker** (Containerized API)
```bash
# Build the image (bundles the trained model + preprocessors)
docker build -t telco-churn-api .

# Run the service
docker run --rm -p 8000:8000 telco-churn-api

# Verify it's healthy, then explore docs at http://localhost:8000/docs
curl http://localhost:8000/health
```
- Self-contained, reproducible runtime
- Built-in `HEALTHCHECK` against `/health`
- Lean image (serving dependencies only)

### 3. **Batch Processing**
```bash
python3 src/predict.py --input customers.csv --output predictions.csv
```
- Large dataset processing
- Summary statistics
- Error reporting

### 4. **Integration Ready**
- REST API for web applications
- CSV import for business tools
- Python package for data science workflows

## 🤝 Contributing

We welcome contributions! See [docs/contributing.md](docs/contributing.md) for guidelines.

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **IBM** for the Telco Customer Churn dataset
- **scikit-learn** for excellent ML tools
- **FastAPI** for modern API development
- **SHAP** for model explainability

---

## 💼 **Hire Me - Data Analyst**

🔥 **Available for immediate hire** - Turning data into actionable business insights

### **What I Deliver:**
- **Business Intelligence**: Data-driven insights that drive strategic decisions
- **Customer Analytics**: Deep understanding of customer behavior and retention
- **Statistical Analysis**: Rigorous testing and significance validation
- **Data Visualization**: Clear charts and dashboards for stakeholders
- **SQL & Database**: Ready for enterprise data environments

### **Technical Stack:**
- **Data Analysis**: Python, pandas, statistical testing, Excel
- **Visualization**: Matplotlib, Seaborn, Tableau-ready formats
- **Database**: SQL queries, data warehousing concepts
- **Business Tools**: ROI analysis, KPI tracking, reporting
- **Communication**: Executive summaries, presentations

### **Business Impact:**
- **Customer Retention**: Identified 77% of at-risk customers before churn
- **Cost Savings**: Data-driven retention strategies reduce acquisition costs
- **Revenue Growth**: Actionable insights for customer lifetime value
- **Strategic Planning**: Evidence-based recommendations for leadership

**GitHub**: [cat-cols](https://github.com/cat-cols) | **LinkedIn**: [https://www.linkedin.com/in/brandon-hardison/](https://www.linkedin.com/in/brandon-hardison/) | **Email**: [brandon.hardison@gmail.com](mailto:brandon.hardison@gmail.com)

---

## 📞 Next Steps

1. **Review Business Summary** - Understand the insights and recommendations
2. **Try the API** - Start the server and test with your data
3. **Integrate** - Connect to your retention systems
4. **Monitor** - Track performance and business impact

**Ready to reduce churn?** Start with the Quick Start guide above! 🚀