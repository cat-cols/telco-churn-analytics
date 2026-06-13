# Telco Churn Analytics

Machine learning project for predicting customer churn in telecommunications using Python and scikit-learn.

## Overview

This project uses the IBM Telco Customer Churn dataset (7,043 customers, 21 features) to build predictive models that identify customers at risk of churning. It follows a hybrid approach: Jupyter notebooks for exploration and production-ready Python scripts for deployment.

The pipeline handles real-world data quality issues including whitespace-encoded missing values, new customers with zero tenure, unseen categorical values at inference time, and class imbalance (~73% non-churn / 27% churn).

## Business Problem

Predict which customers are likely to churn (cancel their service) to enable proactive retention strategies. The model outputs a churn probability and risk level (Low / Medium / High) for each customer, with a configurable decision threshold to tune the precision/recall tradeoff.

## Project Structure

```
telco-churn-analytics/
├── data/
│   ├── raw/                        # Raw dataset (telco_customer_churn.csv)
│   └── processed/                  # Scaled/encoded parquet splits + .pkl artifacts
├── docs/                           # Project documentation
│   ├── data_dictionary.md          # Feature descriptions and types
│   ├── methodology.md              # Analytical approach
│   ├── setup.md                    # Installation guide
│   └── usage.md                    # Usage and threshold guidance
├── models/
│   ├── best_model.pkl              # Saved best estimator
│   └── model_comparison_results.csv # All variants: baseline, balanced, tuned
├── notebooks/                      # Jupyter notebooks for exploration
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   ├── 02_preprocessing.ipynb      # Preprocessing walkthrough
│   ├── 03_modeling.ipynb           # Model training and evaluation
│   └── 04_results.ipynb            # Results visualisation
├── scripts/
│   └── threshold_analysis.py       # Precision/recall sweep CLI tool
├── src/                            # Production pipeline
│   ├── config.py                   # All configuration and column lists
│   ├── utils.py                    # Shared I/O helpers
│   ├── preprocess.py               # Preprocessing + schema validation
│   ├── train.py                    # Training: baseline, balanced, GridSearchCV
│   ├── predict.py                  # Inference with edge case handling
│   └── run_pipeline.py             # End-to-end orchestrator
├── tests/
│   ├── fixtures/sample_customers.csv  # 10-row test fixture
│   ├── test_predict.py             # Unit tests for predict.py
│   ├── test_preprocess.py          # Unit tests for preprocess.py
│   └── test_integration.py         # Integration tests (full pipeline)
└── notes/                          # Internal technical notes
```

## Installation

### Prerequisites
- Python 3.10+
- pip

### Setup
```bash
git clone https://github.com/cat-cols/telco-churn-analytics
cd telco-churn-analytics

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

### Run the full pipeline
```bash
# Preprocessing → training (all variants) → predictions
python src/run_pipeline.py

# Skip stages if artifacts are already fresh
python src/run_pipeline.py --skip-preprocessing
python src/run_pipeline.py --skip-preprocessing --skip-training
```

### Run individual stages
```bash
python src/preprocess.py                                          # preprocess + save artifacts
python src/train.py                                               # train all variants + GridSearchCV
python src/predict.py --input data/new_customers.csv --output predictions.csv
```

### Threshold analysis
```bash
# Print precision/recall/F1 sweep at default step (0.05)
python scripts/threshold_analysis.py

# Finer sweep, save to CSV
python scripts/threshold_analysis.py --step 0.02 --output results/threshold_sweep.csv
```

### Run tests
```bash
python -m pytest tests/ -v
```

### Explore with notebooks
```bash
jupyter notebook notebooks/
```
Run in order: `01_eda` → `02_preprocessing` → `03_modeling` → `04_results`

## Dataset

IBM Telco Customer Churn dataset — 7,043 customers, 21 features:
- **Demographics:** gender, SeniorCitizen, Partner, Dependents
- **Services:** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- **Account:** Contract, PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges
- **Target:** Churn (Yes / No) — ~27% positive rate

Known data quality issues handled by the pipeline:
- `TotalCharges` stored as string; whitespace values for new customers (tenure = 0)
- 11 customers with tenure = 0 flagged as `Is_New_Customer` and assigned TotalCharges = 0

## Models

`train.py` runs three stages and saves the overall best model by ROC-AUC:

1. **Baseline** — Logistic Regression, Random Forest, Gradient Boosting (default settings)
2. **Balanced** — same models with `class_weight='balanced'` (or `sample_weight` for GB) to address class imbalance
3. **Tuned** — `GridSearchCV` (5-fold, ROC-AUC scoring) on the best pre-tuning model

Full results are saved to `models/model_comparison_results.csv`.

## Performance

Best saved model: **Gradient Boosting (tuned)**

| Metric | Default threshold (0.50) | Recommended threshold (0.35) |
|---|---|---|
| ROC-AUC | **0.8347** | 0.8347 |
| Recall | 0.476 | **0.687** |
| Precision | 0.647 | 0.549 |
| F1 | 0.549 | 0.610 |

Best recall variant: `logistic_regression_balanced` — Recall **0.791**, ROC-AUC 0.8317.

Run `python scripts/threshold_analysis.py` to see the full precision/recall tradeoff for the current model.

## Documentation

- [Data Dictionary](docs/data_dictionary.md) — Feature descriptions and types
- [Setup Guide](docs/setup.md) — Detailed installation instructions
- [Usage Guide](docs/usage.md) — Prediction pipeline and threshold guidance
- [Methodology](docs/methodology.md) — Analytical approach and preprocessing decisions
- [Technical Implementation Guide](notes/technical_implementation_guide.md) — Architecture, design decisions, testing strategy, artifact lifecycle
- [Debugging Log](notes/debugging_log.md) — All bugs found and fixed with root cause analysis

## Contributing

See [docs/contributing.md](docs/contributing.md). Please read before submitting pull requests.

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- IBM for the Telco Customer Churn dataset
- scikit-learn for machine learning tools
- The open-source community