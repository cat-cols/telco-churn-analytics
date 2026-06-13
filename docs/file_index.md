# File Index

All tracked files in the repository, excluding `.venv/`, `__pycache__/`, and `.git/`.

---

## Root

| File | Description |
|---|---|
| `README.md` | Project overview, setup, and usage instructions |
| `TODO.md` | Prioritised task list across all project phases |
| `CHANGELOG.md` | Keep-a-Changelog formatted history of notable changes |
| `_CHANGELOG.md` | ⚠️ Appears to be a duplicate — review and remove if redundant |
| `LICENSE` | MIT licence |
| `requirements.txt` | Python package dependencies |
| `pyproject.toml` | Project metadata and tool configuration |
| `pyrightconfig.json` | Pyright static type checker configuration |

---

## `src/` — Production Scripts

| File | Description |
|---|---|
| `src/config.py` | Centralised configuration: paths, column lists, model parameters |
| `src/utils.py` | Shared utilities: load/save CSV, parquet, pickle; model evaluation |
| `src/preprocess.py` | Data preprocessing pipeline: schema validation, scaling, encoding |
| `src/train.py` | Model training: fits LR, RF, GB; selects best by ROC-AUC |
| `src/predict.py` | Inference engine: edge case handling, risk scoring, CSV output |
| `src/run_pipeline.py` | End-to-end orchestrator: preprocessing → training → predictions |

---

## `notebooks/` — Exploration & Documentation

| File | Description |
|---|---|
| `notebooks/01_eda.ipynb` | Exploratory data analysis: distributions, churn rate, correlations |
| `notebooks/02_preprocessing.ipynb` | Preprocessing walkthrough |
| `notebooks/03_modeling.ipynb` | Model training and evaluation |
| `notebooks/04_results.ipynb` | Results visualisation |

---

## `data/` — Dataset Files (gitignored)

| File | Description |
|---|---|
| `data/raw/telco_customer_churn.csv` | Raw IBM Telco dataset (7,043 rows, 21 columns) |
| `data/raw/customer_churn.zip` | Zipped source archive |
| `data/processed/train.parquet` | Processed training features |
| `data/processed/test.parquet` | Processed test features |
| `data/processed/train_labels.parquet` | Training labels |
| `data/processed/test_labels.parquet` | Test labels |
| `data/processed/scaler.pkl` | Fitted `StandardScaler` artifact |
| `data/processed/encoder.pkl` | Fitted `OneHotEncoder` artifact |

---

## `models/` — Model Artifacts (gitignored)

| File | Description |
|---|---|
| `models/best_model.pkl` | Serialised best model (Gradient Boosting, ROC-AUC 0.834) |
| `models/model_comparison_results.csv` | Metrics table for all trained models |

---

## `results/` — Pipeline Outputs (gitignored)

| File | Description |
|---|---|
| `results/test_predictions.csv` | Churn predictions on raw dataset (7,043 rows) |
| `results/test_predictions_summary.txt` | Plain-text summary: churn rate, risk distribution |

---

## `docs/` — Project Documentation

| File | Description |
|---|---|
| `docs/data_dictionary.md` | Feature descriptions and data types |
| `docs/data_schema.md` | Schema specification |
| `docs/data_source.md` | Dataset provenance and licence |
| `docs/methodology.md` | Analytical approach and preprocessing rationale |
| `docs/model_evaluation.md` | Model evaluation framework and metrics |
| `docs/findings.md` | Results and business insights |
| `docs/executive_walkthrough.md` | Non-technical summary for stakeholders |
| `docs/setup.md` | Installation and environment setup guide |
| `docs/usage.md` | How to run the pipeline and make predictions |
| `docs/contributing.md` | Contribution guidelines |
| `docs/file_index.md` | This file |
| `docs/project_build_order.md` | Chronological order in which all project files should be created |

---

## `notes/` — Development Notes

| File | Description |
|---|---|
| `notes/debugging_log.md` | Chronological log of bugs encountered and fixes applied |
| `notes/technical_implementation_guide.md` | Component-level implementation explanations and design decisions |
| `notes/BEGINNER_GUIDE.md` | Getting started guide for new contributors |
| `notes/data_quality_guide.md` | Data quality issues, detection, and remediation |
| `notes/error_handling_guide.md` | Error handling patterns used across the codebase |
| `notes/interview_data_quality_guide.md` | Advanced data quality topics |
| `notes/dataset-intention.md` | Notes on dataset purpose and intended use |

---

## `archive/` — Archived Work

| File | Description |
|---|---|
| `archive/notebooks/customer_churn.ipynb` | Original exploratory notebook (superseded by `notebooks/`) |

---

## `.vscode/` — Editor Configuration

| File | Description |
|---|---|
| `.vscode/settings.json` | Workspace settings |
| `.vscode/extensions.json` | Recommended extensions |
