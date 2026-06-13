# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Fixed

- **[#1](https://github.com/cat-cols/telco-churn-analytics/issues/1) `TypeError` in `validate_input_data`** (`src/predict.py`)
  `TotalCharges` arrives as a raw string column from CSV. Comparing it directly against
  numeric `VALIDATION_RULES` bounds raised `TypeError: '<' not supported between instances
  of 'str' and 'int'`. Fixed by coercing with `pd.to_numeric(..., errors='coerce')` inline
  before the range check, excluding `NaN` rows via `.notna()`.

- **[#2](https://github.com/cat-cols/telco-churn-analytics/issues/2) `ValueError` on unseen encoder categories** (`src/preprocess.py`)
  `handle_unseen_categories()` in `predict.py` mapped unknown values to the string
  `'Unknown'`, but the encoder was trained without `handle_unknown='ignore'` and rejected
  it at transform time. Fixed by adding `handle_unknown='ignore'` to the `OneHotEncoder`
  initialisation in `preprocess.py` so unseen categories are silently zeroed out.

- **[#3](https://github.com/cat-cols/telco-churn-analytics/issues/3) `ValueError: truth value of array ambiguous`** (`src/predict.py`)
  `encoder.categories_` entries are numpy arrays. Wrapping one in `set()` before passing
  to `pandas.Series.isin()` triggered numpy's ambiguous truth-value error. Fixed by using
  `list()` instead of `set()`.

### Changed

- `validate_input_data` now coerces validation columns to numeric before range-checking,
  making it safe to call on raw CSV input before preprocessing.
- `OneHotEncoder` in `preprocess.py` now trained with `handle_unknown='ignore'`; existing
  `encoder.pkl` artifact must be regenerated with `python3 src/run_pipeline.py`.
- Regenerated all artifacts via `python3 src/run_pipeline.py` — `encoder.pkl` now persists
  `handle_unknown='ignore'`; best model remains Gradient Boosting (ROC-AUC: 0.8340).
- Extended `train.py` to run 6 variants (3 baseline + 3 balanced) then GridSearchCV on the
  best; saves full comparison table with `stage` column to `models/model_comparison_results.csv`.
- Added `MODEL_CONFIGS_BALANCED` and `PARAM_GRIDS` to `config.py` for class-weight and
  hyperparameter tuning configuration.
- Best saved model updated to `gradient_boosting_tuned`
  (ROC-AUC: 0.8347, Recall: 0.476 at default threshold; Recall: 0.773 at threshold=0.30).

---

## [0.1.0] — 2026-06-13

### Added

- Initial project structure: hybrid notebook (`notebooks/`) + production scripts (`src/`)
- Exploratory data analysis notebook (`01_eda.ipynb`)
- Data preprocessing pipeline (`02_preprocessing.ipynb`, `src/preprocess.py`)
- Model training and evaluation (`03_modeling.ipynb`, `src/train.py`)
- Results visualisation notebook (`04_results.ipynb`)
- Production scripts: `config.py`, `utils.py`, `preprocess.py`, `train.py`, `predict.py`,
  `run_pipeline.py`
- Robust edge-case handling in `predict.py`: new customers, missing values, unseen
  categories, outlier flagging
- Core documentation: `data_dictionary.md`, `setup.md`, `usage.md`, `methodology.md`
- Contributing guidelines and MIT licence
