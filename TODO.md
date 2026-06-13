# TODO — Telco Churn Analytics

Status key: ✅ Done · 🔄 In progress · ⬜ Not started · 🚩 High priority

---

## 1. Data

- ✅ Source raw dataset (`data/raw/telco_customer_churn.csv`)
- ✅ Document features in `docs/data_dictionary.md`
- ✅ Validate dataset schema on load (column names, dtypes, row count) — `validate_schema()` in `preprocess.py`
- ⬜ Profile class balance (churn vs. non-churn counts + %) and record in `docs/data_dictionary.md`
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
- ⬜ Add churn rate breakdown by key features (Contract type, tenure band, PaymentMethod)
- ⬜ Plot feature correlation heatmap and save to `results/`
- ⬜ Document EDA findings and business insights in `docs/methodology.md`

---

## 4. Modelling

- ✅ Train Logistic Regression, Random Forest, Gradient Boosting
- ✅ Evaluate with accuracy, precision, recall, F1, ROC-AUC
- ✅ Select best model by ROC-AUC and save to `models/best_model.pkl`
- ✅ Save comparison table to `models/model_comparison_results.csv`
- ✅ Address low recall — class weighting raised recall to 0.791; threshold 0.30 on tuned GB gives recall=0.773 / precision=0.522 / F1=0.623
- ✅ Hyperparameter tuning — GridSearchCV on GB; best params: lr=0.05, max_depth=3, n_estimators=100, subsample=1.0; ROC-AUC 0.8340 → 0.8347
- ⬜ Add XGBoost / LightGBM to model comparison (optional, listed in README)
- ⬜ Cross-validate best model (k=5) to confirm ROC-AUC is not split-dependent

---

## 5. Model Interpretability

- ⬜ Compute SHAP values for best model on test set
- ⬜ Plot global feature importance (SHAP bar chart) — save to `results/`
- ⬜ Plot SHAP summary plot — save to `results/`
- ⬜ Identify top 5 churn drivers and document in `docs/methodology.md`

---

## 6. Prediction Pipeline (`src/predict.py`)

- ✅ End-to-end prediction from raw CSV input
- ✅ Edge case handling: new customers (tenure=0), missing values, unseen categories, outlier flagging
- ✅ Fix #1: `TypeError` — coerce `TotalCharges` before numeric range validation
- ✅ Fix #3: `ValueError` — use `list()` not `set()` for encoder category lookup
- ✅ Output `Risk_Level` (Low/Medium/High) alongside churn probability
- ⬜ Add `--threshold` tuning guidance to `docs/usage.md`
- ⬜ Validate predictions output schema (column names, value ranges) before saving

---

## 7. Testing

- ✅ Write unit test: `validate_input_data` raises no error on string `TotalCharges` (regression for fix #1)
- ✅ Write unit test: `handle_unseen_categories` does not raise on unknown category values (regression for fix #2/#3)
- ✅ Write unit test: `preprocess_new_data` output column order matches training feature order
- ✅ Write unit test: `handle_new_customers` sets `TotalCharges=0` for `tenure=0` rows
- ✅ Write integration test: full `predict_from_file` run on a 10-row CSV sample produces valid output
- ✅ Configure test runner (`pytest`) in `pyproject.toml`

---

## 8. Results & Reporting

- ⬜ Save confusion matrix plot for best model to `results/`
- ⬜ Save ROC curve plot to `results/`
- ⬜ Complete `notebooks/04_results.ipynb` with final model metrics and charts
- ⬜ Write plain-language summary of findings for business stakeholders in `docs/`

---

## 9. Documentation

- ✅ `README.md` — project overview, setup, usage
- ✅ `docs/data_dictionary.md` — feature descriptions
- ✅ `docs/methodology.md` — analytical approach
- ✅ `CHANGELOG.md` — Keep-a-Changelog format
- ⬜ `docs/usage.md` — fill in prediction threshold guidance, edge case behaviour
- ⬜ `docs/setup.md` — verify setup steps still match current `requirements.txt`
- ⬜ 🚩 Close GitHub issues #1, #2, #3 once regression tests are in place

---

## 10. Deployment (Optional / Future)

- ⬜ Create a FastAPI endpoint wrapping `predict_from_file` for batch or single-record scoring
- ⬜ Containerise with Docker (`Dockerfile` + `docker-compose.yml`)
- ⬜ Add model versioning — tag `best_model.pkl` with training date and ROC-AUC in filename
- ⬜ Set up CI (GitHub Actions) — run `pytest` on push to main
