# Setup Guide

## Prerequisites
- Python 3.11 or higher (developed and tested on Python 3.12)
- Git (for cloning the repository)
- Virtual environment tool (venv or conda)

## Installation Steps

### 1. Clone the Repository
```bash
git clone <repository-url>
cd telco-churn-analytics
```

### 2. Create Virtual Environment
Using venv:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Using conda:
```bash
conda create -n telco-churn python=3.12
conda activate telco-churn
```

### 3. Install Dependencies

Install the project in editable mode from `pyproject.toml`:

```bash
pip install -e .
```

This installs the core runtime dependencies (versions pinned in `pyproject.toml`):

| Package | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data manipulation |
| `scikit-learn` | Modeling, preprocessing, metrics |
| `pyarrow` | Reading/writing Parquet files |
| `matplotlib`, `seaborn` | Visualization |
| `shap` | Model interpretability (SHAP analysis in `04_results.ipynb`) |
| `jupyter`, `ipykernel` | Running the notebooks |
| `fastapi`, `uvicorn` | Prediction API |

**Optional extras** (install only if needed):

```bash
pip install -e ".[dev]"       # pytest, flake8, black, bandit
pip install -e ".[advanced]"  # xgboost, lightgbm
pip install -e ".[imbalance]" # imbalanced-learn (SMOTE)
pip install -e ".[interactive]" # plotly
pip install -e ".[sql]"        # duckdb
pip install -e ".[pipeline]"   # papermill (automated notebook execution)
```

> Note: `kagglehub`/`kaggle` are **not** project dependencies. They are only needed if
> you choose to download the dataset via the Kaggle API (see Step 4).

### 4. Download the Dataset
The IBM Telco Customer Churn dataset should be placed in the `data/` directory.

**Dataset Details:**
- **File Name**: `telco_customer_churn.csv`
- **Source**: IBM Telco Customer Churn Dataset
- **Format**: CSV
- **Size**: 7,043 records × 21 columns (19 features + customerID + Churn target)

**How to Obtain:**

**Option 1: Download from IBM**
1. Visit the IBM dataset source or Kaggle
2. Download the IBM Telco Customer Churn dataset
3. Rename the file to `telco_customer_churn.csv`
4. Place it in the `data/raw/` directory

**Option 2: Using Kaggle (if available)**
```bash
# Install kaggle package
pip install kaggle

# Download dataset (requires Kaggle API credentials)
# Note: Check Kaggle for the exact dataset name for IBM Telco Churn
kaggle datasets download -d <dataset-name>
```

**Expected File Structure:**
```
data/
└── raw/
    └── telco_customer_churn.csv
```

For more details about the dataset, see [data_source.md](data_source.md).

### 5. Verify Installation
Run the following to verify your setup:
```bash
python3 -c "import pandas, numpy, sklearn, pyarrow, matplotlib, seaborn, shap; print('All dependencies installed successfully')"
```

## Project Structure
```
telco-churn-analytics/
├── data/
│   ├── raw/              # Raw dataset files
│   └── processed/        # Processed data + fitted scaler/encoder
├── docs/                 # Documentation files
├── models/               # Trained model artifacts (.pkl) + comparison CSV
├── notebooks/            # Jupyter notebooks (01_eda → 04_results)
├── notes/                # In-depth learning/reference guides
├── outputs/              # Generated plots, reports, predictions
├── scripts/              # Standalone analysis scripts (e.g. threshold_analysis.py)
├── src/                  # Pipeline source (preprocess, train, predict, config)
├── tests/                # Pytest unit & integration tests
└── README.md             # Project overview
```

## IDE Setup
### VS Code
1. Install the Python extension
2. Select the virtual environment as interpreter:
   - Command Palette → Python: Select Interpreter
   - Choose `.venv/bin/python`
3. Install Jupyter extension for notebook support

### Jupyter Notebook
Launch Jupyter from the project directory:
```bash
jupyter notebook
```

## Troubleshooting

### Common Issues

**Issue**: ModuleNotFoundError
- **Solution**: Ensure virtual environment is activated and dependencies are installed

**Issue**: Dataset not found
- **Solution**: Verify dataset is in the correct `data/` directory path

**Issue**: Permission errors
- **Solution**: Check file permissions on data directory

**Issue**: Memory errors with large dataset
- **Solution**: Use chunked processing or increase available RAM

## Environment Variables (Optional)
If using Kaggle API:
```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

## Next Steps
After setup, refer to [usage.md](usage.md) for instructions on running the analysis notebooks.
