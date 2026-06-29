# Project Build Order

Chronological order in which files should be created to complete this project from scratch.
Files marked ✅ already exist. Files marked ⬜ still need to be created or completed.

---

## Stage 1 — Project Scaffolding

These files define the structure and configuration before any code is written.

| # | File | Status | Purpose |
|---|---|---|---|
| 1 | `README.md` | ✅ | Project overview, setup instructions, usage |
| 2 | `LICENSE` | ✅ | MIT licence |
| 3 | `requirements.txt` | ✅ | Python dependencies |
| 4 | `pyproject.toml` | ✅ | Project metadata, tool config (pytest, ruff, etc.) |
| 5 | `pyrightconfig.json` | ✅ | Static type checker configuration |
| 6 | `.gitignore` | ✅ | Ignore venv, data, model artifacts, caches |
| 7 | `CHANGELOG.md` | ✅ | Keep-a-Changelog formatted change history |
| 8 | `TODO.md` | ✅ | Prioritised task list |

---

## Stage 2 — Data

Raw data must exist before any code can run against it.

| # | File | Status | Purpose |
|---|---|---|---|
| 9 | `data/raw/telco_customer_churn.csv` | ✅ | Raw source dataset (7,043 rows, 21 columns) |
| 10 | `docs/data_dictionary.md` | ✅ | Feature descriptions, dtypes, known quality issues |
| 11 | `docs/data_schema.md` | ✅ | Formal schema specification |
| 12 | `docs/data_source.md` | ✅ | Dataset provenance and licence |

---

## Stage 3 — Configuration & Utilities

Shared code that every other script depends on — must exist before any pipeline scripts.

| # | File | Status | Purpose |
|---|---|---|---|
| 13 | `src/config.py` | ✅ | All paths, column lists, model parameters |
| 14 | `src/utils.py` | ✅ | Load/save helpers, model evaluation function |

---

## Stage 4 — Exploratory Data Analysis

EDA informs preprocessing and modelling decisions. Do this before writing the pipeline.

| # | File | Status | Purpose |
|---|---|---|---|
| 15 | `notebooks/01_eda.ipynb` | ✅ | Distributions, churn rate, correlations |
| 16 | `docs/findings.md` | ✅ | Written summary of EDA insights for stakeholders |

---

## Stage 5 — Preprocessing Pipeline

| # | File | Status | Purpose |
|---|---|---|---|
| 17 | `notebooks/02_preprocessing.ipynb` | ✅ | Interactive walkthrough of preprocessing steps |
| 18 | `src/preprocess.py` | ✅ | Production preprocessing: schema validation, scaling, encoding |

Running `python3 src/preprocess.py` produces:

| # | File | Status | Purpose |
|---|---|---|---|
| 19 | `data/processed/train.parquet` | ✅ | Processed training features |
| 20 | `data/processed/test.parquet` | ✅ | Processed test features |
| 21 | `data/processed/train_labels.parquet` | ✅ | Training labels |
| 22 | `data/processed/test_labels.parquet` | ✅ | Test labels |
| 23 | `data/processed/scaler.pkl` | ✅ | Fitted StandardScaler |
| 24 | `data/processed/encoder.pkl` | ✅ | Fitted OneHotEncoder |

---

## Stage 6 — Model Training

| # | File | Status | Purpose |
|---|---|---|---|
| 25 | `notebooks/03_modeling.ipynb` | ✅ | Interactive model training and comparison |
| 26 | `src/train.py` | ✅ | Production training: fits LR, RF, GB; selects best by ROC-AUC |

Running `python3 src/train.py` produces:

| # | File | Status | Purpose |
|---|---|---|---|
| 27 | `models/best_model.pkl` | ✅ | Serialised best model |
| 28 | `models/model_comparison_results.csv` | ✅ | Metrics for all trained models |

---

## Stage 7 — Prediction Pipeline

| # | File | Status | Purpose |
|---|---|---|---|
| 29 | `src/predict.py` | ✅ | Inference with edge case handling, risk scoring |
| 30 | `src/run_pipeline.py` | ✅ | End-to-end orchestrator (preprocess → train → predict) |

Running the pipeline produces:

| # | File | Status | Purpose |
|---|---|---|---|
| 31 | `outputs/test_predictions.csv` | ✅ | Churn predictions on full dataset |
| 32 | `outputs/test_predictions_summary.txt` | ✅ | Plain-text prediction summary |

---

## Stage 8 — Testing

Write tests after the pipeline is stable so you know what behaviour to lock in.

| # | File | Status | Purpose |
|---|---|---|---|
| 33 | `tests/test_predict.py` | ⬜ | Unit tests: `validate_input_data`, `handle_unseen_categories`, `handle_new_customers` |
| 34 | `tests/test_preprocess.py` | ⬜ | Unit tests: `validate_schema`, column order, dtype handling |
| 35 | `tests/test_integration.py` | ⬜ | Integration test: full `predict_from_file` run on a 10-row sample |
| 36 | `tests/fixtures/sample_customers.csv` | ⬜ | Small test fixture CSV (10 rows) covering edge cases |

---

## Stage 9 — Results & Reporting

| # | File | Status | Purpose |
|---|---|---|---|
| 37 | `notebooks/04_results.ipynb` | ⬜ | Final metrics, confusion matrix, ROC curve, SHAP plots |
| 38 | `outputs/confusion_matrix.png` | ⬜ | Confusion matrix for best model |
| 39 | `outputs/roc_curve.png` | ⬜ | ROC curve for all models |
| 40 | `outputs/shap_summary.png` | ⬜ | SHAP feature importance plot |
| 41 | `docs/model_evaluation.md` | ✅ | Model evaluation framework and metric explanations |
| 42 | `docs/executive_walkthrough.md` | ✅ | Non-technical summary for business stakeholders |

---

## Stage 10 — Documentation Completion

Fill in documentation that depends on knowing the final results.

| # | File | Status | Purpose |
|---|---|---|---|
| 43 | `docs/methodology.md` | ✅ | Analytical approach — update with final EDA findings and SHAP insights |
| 44 | `docs/usage.md` | ✅ | Usage guide — add threshold tuning guidance |
| 45 | `docs/setup.md` | ✅ | Setup guide — verify against current `requirements.txt` |
| 46 | `docs/contributing.md` | ✅ | Contribution guidelines |
| 47 | `notes/debugging_log.md` | ✅ | Bug log — keep updated as issues are found |

---

## Stage 11 — Deployment (Optional / Future)

| # | File | Status | Purpose |
|---|---|---|---|
| 48 | `src/api.py` | ⬜ | FastAPI endpoint wrapping `predict_from_file` |
| 49 | `Dockerfile` | ⬜ | Container definition |
| 50 | `docker-compose.yml` | ⬜ | Local container orchestration |
| 51 | `.github/workflows/ci.yml` | ⬜ | GitHub Actions CI: run pytest on push to main |
