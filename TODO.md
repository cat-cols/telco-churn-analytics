# TODO — Telco Churn Analytics

Status key: ✅ Done · 🔄 In progress · ⬜ Not started · 🚩 High priority

---

## 1. Data

- ✅ Source raw dataset (`data/raw/telco_customer_churn.csv`)
- ✅ Document features in `docs/data_dictionary.md`
- ✅ Validate dataset schema on load (column names, dtypes, row count) — `validate_schema()` in `preprocess.py`
- ✅ Profile class balance (churn vs. non-churn counts + %) and record in `docs/data_dictionary.md`
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
- ✅ Add churn rate breakdown by key features (Contract type, tenure band, PaymentMethod) — `churn_breakdown()` cell in `notebooks/01_eda.ipynb` with tables + 1×3 bar chart vs. overall 26.54%
- ✅ Plot feature correlation heatmap and save to `results/`
- ✅ Document EDA findings and business insights in `docs/methodology.md`

---

## 4. Modelling

- ✅ Train Logistic Regression, Random Forest, Gradient Boosting
- ✅ Evaluate with accuracy, precision, recall, F1, ROC-AUC
- ✅ Select best model by ROC-AUC and save to `models/best_model.pkl`
- ✅ Save comparison table to `models/model_comparison_results.csv`
- ✅ Address low recall — class weighting raised recall to 0.791; threshold 0.30 on tuned GB gives recall=0.773 / precision=0.522 / F1=0.623
- ✅ Hyperparameter tuning — GridSearchCV on GB; best params: lr=0.05, max_depth=3, n_estimators=100, subsample=1.0; ROC-AUC 0.8340 → 0.8347
- ⬜ Add XGBoost / LightGBM to model comparison (optional, listed in README)
- ✅ Cross-validate best model (k=5) — Gradient Boosting ROC-AUC = 0.8477 ± 0.0046 (folds 0.843–0.856); tiny std confirms result is NOT split-dependent (single-split test value 0.8340 is within sampling noise)

---

## 5. Strategic Business Questions (NEW)

### High Priority (Business Impact)
- 🚩 Add financial impact analysis (CLV, revenue at risk, ROI calculations)
- 🚩 Add intervention timing analysis (when to act, optimal lead time)
- 🚩 Update README and notebooks with new business questions

### Medium Priority (Operational Excellence)
- ⬜ Add customer value prioritization (high-value vs high-risk segmentation)
- ⬜ Add success measurement framework (KPIs, A/B testing setup)
- ⬜ Add model evolution and monitoring (long-term sustainability)
- ⬜ Add limitations and risk assessment (ethical considerations, bias detection)

### Low Priority (Strategic Context)
- ⬜ Add competitive analysis (industry benchmarks, positioning)

---

## 6. Model Interpretability

- ✅ Compute SHAP values for best model on test set — `TreeExplainer` on Gradient Boosting in `notebooks/04_results.ipynb`
- ✅ Plot global feature importance (SHAP bar chart) — saved to `results/shap_feature_importance.png`
- ✅ Plot SHAP summary plot — saved to `results/shap_summary.png`
- ✅ Identify top 5 churn drivers and document in `docs/methodology.md` — Contract_Month-to-month, tenure (protective), OnlineSecurity_No, TechSupport_No, MonthlyCharges

---

## 7. Prediction Pipeline (`src/predict.py`)

- ✅ End-to-end prediction from raw CSV input
- ✅ Edge case handling: new customers (tenure=0), missing values, unseen categories, outlier flagging
- ✅ Fix #1: `TypeError` — coerce `TotalCharges` before numeric range validation
- ✅ Fix #3: `ValueError` — use `list()` not `set()` for encoder category lookup
- ✅ Output `Risk_Level` (Low/Medium/High) alongside churn probability
- ✅ Add `--threshold` tuning guidance to `docs/usage.md` — "Making Predictions" section with operating-point table (0.30 best-F1 recommended) + `scripts/threshold_analysis.py` re-derivation; also fixed that script's int-label bug
- ✅ Validate predictions output schema (column names, value ranges) before saving — `validate_output_schema()` in `predict.py` (required cols, Churn_Probability∈[0,1], Predicted_Churn∈{0,1}, Risk_Level∈{Low,Medium,High}); raises before save; 7 regression tests added

---

## 8. Testing

- ✅ Write unit test: `validate_input_data` raises no error on string `TotalCharges` (regression for fix #1)
- ✅ Write unit test: `handle_unseen_categories` does not raise on unknown category values (regression for fix #2/#3)
- ✅ Write unit test: `preprocess_new_data` output column order matches training feature order
- ✅ Write unit test: `handle_new_customers` sets `TotalCharges=0` for `tenure=0` rows
- ✅ Write integration test: full `predict_from_file` run on a 10-row CSV sample produces valid output
- ✅ Configure test runner (`pytest`) in `pyproject.toml`

---

## 9. Results & Reporting

- ✅ Save confusion matrix plot for best model to `results/` — `results/confusion_matrix.png` (generated by `04_results.ipynb`)
- ✅ Save ROC curve plot to `results/` — `results/roc_curve.png` (with Youden-J optimal threshold marked)
- ✅ Complete `notebooks/04_results.ipynb` with final model metrics and charts — runs end-to-end: dashboard, radar, probability dist, confusion matrix, ROC, feature importance, SHAP, summary report
- ✅ Write plain-language summary of findings for business stakeholders in `docs/` — `docs/business_summary.md` (jargon-free, grounded in real results)

---

## 10. Documentation

- ✅ `README.md` — project overview, setup, usage
- ✅ `docs/data_dictionary.md` — feature descriptions
- ✅ `docs/methodology.md` — analytical approach
- ✅ `CHANGELOG.md` — Keep-a-Changelog format
- ✅ `docs/usage.md` — prediction threshold guidance (operating-point table) + "Edge-case behaviour" subsection (new customers, imputation, unseen categories, outlier flagging, input/output validation)
- ✅ `docs/setup.md` — dependency list realigned to `requirements.txt` (added numpy/matplotlib/seaborn/shap/ipykernel, removed stale kagglehub; optional xgboost/lightgbm/imbalanced-learn noted); refreshed verify-install check and project structure
- ⬜ 🚩 Close GitHub issues #1, #2, #3 once regression tests are in place

---

## 11. Deployment (Optional / Future)

- ✅ Create a FastAPI endpoint wrapping `predict_from_file` for batch or single-record scoring — `src/api.py` with lifespan model loading, POST /predict/records (JSON), POST /predict/file (CSV), GET /health, comprehensive integration tests in `tests/test_api.py`
- ⬜ Containerise with Docker (`Dockerfile` + `docker-compose.yml`)
- ⬜ Add model versioning — tag `best_model.pkl` with training date and ROC-AUC in filename
- ⬜ Set up CI (GitHub Actions) — run `pytest` on push to main

---

## 12. Portfolio / De-genericize (Stand-out additions)

High-leverage additions to differentiate this from a typical Telco-churn portfolio project.

- ⬜ 🚩 Add a SQL layer — load the data into DuckDB/SQLite and write the churn-segment analysis as SQL queries (more "analyst")
- ⬜ 🚩 Build one dashboard — small Streamlit or Power BI view of risk segments + a "what-if" threshold slider (visual, interactive, memorable)
- ⬜ 🚩 Quantify business impact — translate churn reduction into retained revenue with stated assumptions (single biggest differentiator)
- ⬜ Write it up publicly — short blog post / README story: "the bug that almost broke my pipeline" (the `TotalCharges` saga) to show real problem-solving

---
