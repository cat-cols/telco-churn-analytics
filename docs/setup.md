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

**Option 1: Using Kaggle (if available)**
```bash
# Install kaggle package
pip install kaggle

# Download dataset (requires Kaggle API credentials)
kaggle datasets download -d <dataset-name>
```

**Option 2: Manual Download**
1. Download the dataset from the source
2. Place the file in `data/` directory
3. Expected format: CSV or Excel file

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
