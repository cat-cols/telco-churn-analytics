# Telco Churn Analytics

Machine learning project for predicting customer churn in telecommunications using Python and scikit-learn.

## Overview

This project uses the IBM Telco Customer Churn dataset to build predictive models that identify customers at risk of churning. The project follows a hybrid approach with Jupyter notebooks for exploration and production-ready Python scripts for deployment.

## Business Problem

Predict which customers are likely to churn (cancel their service) to enable proactive retention strategies and reduce customer attrition.

## Project Structure

```
telco-churn-analytics/
├── data/
│   ├── raw/                    # Raw dataset files
│   └── processed/              # Processed/transformed data
├── docs/                       # Documentation
│   ├── data_dictionary.md      # Feature descriptions
│   ├── setup.md                # Installation guide
│   ├── usage.md                # Usage instructions
│   └── methodology.md          # Analytical methodology
├── notebooks/                  # Jupyter notebooks for exploration
│   ├── 01_EDA.ipynb           # Exploratory data analysis
│   ├── 02_preprocessing.ipynb # Data preprocessing pipeline
│   ├── 03_modeling.ipynb      # Model training and evaluation
│   └── 04_results.ipynb       # Results visualization
├── src/                        # Production scripts
│   ├── config.py              # Configuration parameters
│   ├── utils.py               # Utility functions
│   ├── preprocess.py          # Data preprocessing
│   ├── train.py               # Model training
│   └── predict.py             # Prediction/inference
└── models/                     # Saved models and artifacts
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd telco-churn-analytics

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Using Notebooks (Development)
```bash
jupyter notebook notebooks/
```

Run notebooks in order:
1. `01_EDA.ipynb` - Exploratory data analysis
2. `02_preprocessing.ipynb` - Data preprocessing
3. `03_modeling.ipynb` - Model training
4. `04_results.ipynb` - Results visualization

### Using Production Scripts (Deployment)

# Run complete pipeline (preprocessing → training → predictions)
python src/run_pipeline.py

# Skip preprocessing if data already processed
python src/run_pipeline.py --skip-preprocessing

# Skip training if model already trained
python src/run_pipeline.py --skip-training

# Skip both (just generate predictions)
python src/run_pipeline.py --skip-preprocessing --skip-training

---

```bash
# Preprocess data
python src/preprocess.py

# Train models
python src/train.py

# Make predictions
python src/predict.py --input data/new_customers.csv --output predictions.csv
```

## Dataset

The project uses the IBM Telco Customer Churn dataset with 440,832 customer records and 12 features including:
- Customer demographics (Age, Gender)
- Service usage (Tenure, Usage Frequency, Support Calls)
- Subscription details (Subscription Type, Contract Length, Total Spend)
- Behavioral metrics (Payment Delay, Last Interaction)

## Models

Multiple machine learning algorithms are evaluated:
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost (optional)
- LightGBM (optional)

## Performance

Models are evaluated using:
- ROC-AUC (primary metric for imbalanced data)
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Feature Importance Analysis

## Documentation

- [Data Dictionary](docs/data_dictionary.md) - Complete feature documentation
- [Setup Guide](docs/setup.md) - Detailed installation instructions
- [Usage Guide](docs/usage.md) - How to use the project
- [Methodology](docs/methodology.md) - Analytical approach and preprocessing

## Contributing

Contributions are welcome! Please read the contributing guidelines before submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- IBM for the Telco Customer Churn dataset
- scikit-learn for machine learning tools
- The open-source community