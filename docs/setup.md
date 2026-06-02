# Setup Guide

## Prerequisites
- Python 3.8 or higher
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
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Using conda:
```bash
conda create -n telco-churn python=3.8
conda activate telco-churn
```

### 3. Install Dependencies
The project uses the following key packages:
- pandas
- scikit-learn
- pyarrow
- kagglehub
- jupyter

Install dependencies:
```bash
pip install pandas scikit-learn pyarrow kagglehub jupyter
```

Or if using a requirements file (if available):
```bash
pip install -r requirements.txt
```

### 4. Download the Dataset
The IBM Telco Customer Churn dataset should be placed in the `data/` directory.

**Dataset Details:**
- **File Name**: `customer_churn_dataset-training-master.csv`
- **Source**: IBM Telco Customer Churn Dataset
- **Format**: CSV
- **Size**: 440,832 records × 12 features

**How to Obtain:**

**Option 1: Download from IBM**
1. Visit the IBM dataset source or Kaggle
2. Download the IBM Telco Customer Churn dataset
3. Rename the file to `customer_churn_dataset-training-master.csv`
4. Place it in the `data/` directory

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
└── customer_churn_dataset-training-master.csv
```

For more details about the dataset, see [data_source.md](data_source.md).

### 5. Verify Installation
Run the following to verify your setup:
```bash
python -c "import pandas; import sklearn; import pyarrow; print('All dependencies installed successfully')"
```

## Project Structure
```
telco-churn-analytics/
├── data/
│   ├── raw/              # Raw dataset files
│   └── processed/        # Processed/transformed data
├── docs/                 # Documentation files
├── notebooks/            # Jupyter notebooks for analysis
├── src/                  # Source code (if any)
└── README.md            # Project overview
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
