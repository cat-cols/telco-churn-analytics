# Telco Churn Analytics — End-to-End Interview Walkthrough

A deep, narrative explanation of how this project works, written as if I were
walking an interviewer through it. It covers the *what*, the *why*, and the
*engineering trade-offs* at every stage — not just the happy path.

---

## 1. The 30-second pitch

> "It's a production-shaped machine-learning system that predicts which telecom
> customers are about to churn. It takes raw customer data, runs a reproducible
> preprocessing → training → prediction pipeline, selects the best of several
> models, and serves predictions through a FastAPI REST API. It ships with the
> dataset, committed model artifacts, 56 passing tests, CI, and a Dockerfile, so
> anyone can clone it and reproduce the results end-to-end."

**Business framing:** Churn is expensive — retaining an existing customer is far
cheaper than acquiring a new one. The model produces a per-customer churn
*probability* and a *risk tier* (Low/Medium/High) so the business can target
retention spend at the customers most likely to leave.

---

## 2. The data

- **Source:** IBM Telco Customer Churn dataset — 7,043 customers, 21 columns.
- **Location:** `data/raw/telco_customer_churn.csv` (ships with the repo, ~955 KB).
- **Target:** `Churn` (Yes/No), with a **~27% positive rate** — a class imbalance
  that drives several design decisions later.
- **Feature groups** (defined in `src/config.py`):
  - **Numerical:** `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`
  - **Categorical (15):** `gender`, `Partner`, `Contract`, `PaymentMethod`,
    `InternetService`, etc.
  - **Identifier:** `customerID` (carried through but never used as a feature).

**The famous data-quality gotcha:** `TotalCharges` is stored as a *string* in the
raw CSV, and new customers (tenure = 0) have a blank/whitespace value. Naively
casting it to float throws; silently coercing creates NaNs. Handling this
correctly is a recurring theme (see §4 and §6).

---

## 3. Architecture & repository layout

The project deliberately separates **interactive exploration** (notebooks) from
the **production pipeline** (`src/`), so the notebooks tell the story and the
`src/` modules are the tested, importable code.

```
src/
  config.py        # Single source of truth: paths, columns, model params
  utils.py         # I/O helpers (pickle/parquet) + evaluate_model()
  preprocess.py    # Raw CSV -> validated, scaled, encoded train/test splits
  train.py         # Trains model families, tunes the winner, saves best
  predict.py       # Inference with heavy edge-case handling
  api.py           # FastAPI service wrapping predict()
  run_pipeline.py  # Orchestrates preprocess -> train -> predict end-to-end
notebooks/         # 01_eda -> 02_preprocessing -> ... -> 05_results
tests/             # 56 tests across preprocess/predict/api/integration
models/            # Committed artifacts: best_model.pkl, comparison CSV
data/              # raw/ (committed CSV) + processed/ (parquet + pkl)
outputs/           # Predictions, plots, SHAP, reports
```

**Why a `config.py`?** Every path is derived from
`PROJECT_ROOT = Path(__file__).parent.parent`, so the project is **relocatable**
— there are no hardcoded absolute paths. This is also why moving the project
folder didn't break anything.

---

## 4. Stage 1 — Preprocessing (`src/preprocess.py`)

This converts the raw CSV into model-ready train/test splits and **fits the
preprocessing objects that must be reused at inference time.**

**Step by step:**

1. **Load** the raw CSV.
2. **Schema validation** (`validate_schema`) — *fail fast* before any
   transformation. It checks:
   - all required columns are present,
   - at least 100 rows exist,
   - key dtypes match (e.g. `tenure` is int64).
   If anything is wrong it raises a `ValueError` listing every problem at once.
   *Why:* a bad input file should fail loudly at the front door, not produce
   silently wrong predictions downstream.
3. **Fix `TotalCharges`** — `pd.to_numeric(..., errors="coerce")` turns the
   string column numeric and blanks into NaN.
4. **Clean** — drop rows with NaN.
5. **Train/test split** — 80/20, `random_state=42` for reproducibility, done
   *before* fitting any transformer to avoid data leakage.
6. **Scale numerical features** — `StandardScaler` **fit on train only**, then
   applied to both splits.
7. **Encode categoricals** — `OneHotEncoder(handle_unknown="ignore")`. The
   `handle_unknown="ignore"` is deliberate: at inference time an unseen category
   becomes an all-zero vector instead of crashing.
8. **Persist** the processed parquet files **and** the fitted `scaler.pkl` /
   `encoder.pkl`.

**The single most important idea here:** the scaler and encoder are *fitted on
training data and serialized*, so inference applies the **exact same
transformation**. Re-fitting at predict time would be a classic train/serve skew
bug.

**Interview soundbite:** *"I split before I scale, and I save the fitted
transformers — those two choices prevent the two most common leakage/skew bugs."*

---

## 5. Stage 2 — Training (`src/train.py`)

Training is a **4-stage tournament**, not a single `.fit()`:

1. **Baseline models** — Logistic Regression, Random Forest, Gradient Boosting
   with default-ish configs.
2. **Balanced variants** — the same families with `class_weight="balanced"` to
   fight the 27% imbalance. Gradient Boosting has no `class_weight` param, so I
   pass `sample_weight=compute_sample_weight("balanced", y_train)` instead — a
   detail that shows I understand *how* class weighting actually works.
3. **Hyperparameter tuning** — take the best family by ROC-AUC and run
   `GridSearchCV` (5-fold, `scoring="roc_auc"`, `n_jobs=-1`) over a curated grid
   from `config.PARAM_GRIDS`.
4. **Selection** — pick the overall best by ROC-AUC, save it as
   `models/best_model.pkl`, and write the full per-model metrics table to
   `models/model_comparison_results.csv`.

**Evaluation** (`utils.evaluate_model`) reports accuracy, precision, recall, F1,
and ROC-AUC, carefully converting the "Yes"/"No" string labels to binary and
pulling the probability of the *positive* class via `model.classes_`.

### The honest results (from the committed comparison CSV)

| Model | ROC-AUC | Recall | Precision |
|---|---|---|---|
| gradient_boosting (baseline) | 0.834 | 0.50 | 0.63 |
| gradient_boosting_tuned | **0.835** | 0.48 | 0.65 |
| logistic_regression_balanced | 0.832 | **0.79** | 0.50 |
| gradient_boosting_balanced | 0.832 | 0.78 | 0.50 |

**This table is the most interesting part of the interview.** The pipeline
selects `gradient_boosting_tuned` because it maximizes ROC-AUC (0.835) — but its
**recall is only ~0.48**, meaning it misses over half of real churners. The
*balanced* models trade precision for ~0.79 recall.

**The key insight:** for churn, **recall usually matters more than raw accuracy
or ROC-AUC** — missing a customer who's about to leave (false negative) is
costlier than wasting a small retention offer on someone who'd have stayed (false
positive). So "best ROC-AUC" is not automatically "best for the business." That's
exactly why I built **threshold tuning** (§7) and kept the balanced variants in
the comparison rather than hiding them.

**Interview soundbite:** *"The model that wins on ROC-AUC isn't the one I'd deploy
for retention — I'd either deploy a balanced variant or lower the decision
threshold, because recall is the business-relevant metric here."*

---

## 6. Stage 3 — Prediction & edge cases (`src/predict.py`)

This is the most defensively engineered module, because **inference data is
messier than training data.** The `predict()` flow:

1. **`validate_input_data`** — checks required columns, flags values outside
   business ranges (e.g. tenure 0–72, MonthlyCharges 0–200), and detects
   duplicate customer IDs. Issues are *logged* but don't crash the run.
2. **`handle_new_customers`** — tenure = 0 customers get `TotalCharges = 0` and an
   `Is_New_Customer` flag.
3. **`handle_missing_values`** — coerces `TotalCharges` numeric, imputes with
   training-derived constants (median tenure 29, median MonthlyCharges 65),
   and drops rows missing *critical* features.
4. **`handle_unseen_categories`** — logs unknown category values; the encoder's
   `handle_unknown="ignore"` absorbs them safely.
5. **`detect_and_flag_outliers`** — flags (does **not** drop) extreme values for
   business review.
6. **Scale + encode** using the **loaded** scaler/encoder (never re-fit).
7. **Predict** probabilities, apply the threshold, and bucket into risk tiers via
   `pd.cut(prob, bins=[0, 0.3, 0.7, 1.0])` → Low/Medium/High.
8. **`validate_output_schema`** — before saving, assert the output has the right
   columns, probabilities in [0, 1], `Predicted_Churn` strictly 0/1, and valid
   risk levels. *Input validation and output validation* — both ends are guarded.

**Interview soundbite:** *"Three real bugs are documented in the code as `Fix #1/2/3`
— a string-comparison crash on `TotalCharges`, the encoder failing on unseen
categories, and a numpy truth-value error from using `set()` with `isin()`. I
keep those annotations so the fixes don't regress."*

---

## 7. Threshold tuning (`scripts/`)

Because the default 0.5 threshold under-detects churners, the project includes
threshold-analysis tooling (`scripts/threshold_analysis.py`,
`scripts/interactive_threshold.py`). Lowering the threshold raises recall at the
cost of precision; the right operating point depends on the **cost ratio** of a
missed churner vs. a wasted retention offer. This turns a static model into a
*business-tunable* decision tool — and the API exposes `threshold` as a parameter
so the business can pick the operating point per use case.

---

## 8. Stage 4 — Serving (`src/api.py`)

A **FastAPI** service wraps the prediction logic for real-time use.

- **Startup loading via `lifespan`:** the model, scaler, and encoder are loaded
  **once** at startup into `app.state.artifacts` (a frozen dataclass), not per
  request — this is the difference between a toy and a service.
- **Readiness guard:** `_get_artifacts` returns **503** if artifacts aren't
  loaded, so the service never serves on a half-initialized state.
- **Endpoints:**
  - `GET /health` — liveness/readiness check.
  - `POST /predict/records` — JSON batch of customer records.
  - `POST /predict/file` — CSV upload (with a 5 MiB chunked-read limit → 413 if
    exceeded).
- **Strict input contracts via Pydantic:** every categorical field is a `Literal`
  type (e.g. `Contract` ∈ {Month-to-month, One year, Two year}), so invalid
  values are rejected with a **422** *before* touching the model.
- **Cross-field business rules** (`validate_service_consistency`): e.g.
  `MultipleLines` must be "No phone service" when `PhoneService` is "No", and
  `TotalCharges` must be 0 when `tenure` is 0. The API enforces domain logic, not
  just types.
- **Output schema re-validated** before returning, mirroring the batch path.

**Interview soundbite:** *"The API rejects bad input at the edge with 422s, loads
the model once at startup, and returns 503 until it's ready — it behaves like a
service, not a notebook with a web wrapper."*

---

## 9. Orchestration (`src/run_pipeline.py`)

`run_pipeline.py` chains preprocess → train → predict with `--skip-preprocessing`
and `--skip-training` flags (so you can re-run just the parts you need). It writes
`outputs/test_predictions.csv` and logs a final performance summary. This is the
single command that proves the whole thing works end-to-end.

---

## 10. Testing & CI

- **56 tests** across `test_preprocess.py`, `test_predict.py`, `test_api.py`, and
  `test_integration.py`, plus a small fixture CSV of edge-case customers.
- **API tests** use FastAPI's `TestClient` inside a context manager so the
  `lifespan` startup actually runs and loads artifacts (otherwise every endpoint
  would return 503 — a real bug I fixed).
- **CI** (`.github/workflows/ci.yml`) runs pytest with coverage on Python 3.11 and
  3.12, plus flake8 + black lint.

---

## 11. Reproducibility & deployment

- **Data ships with the repo** — no download or Kaggle account needed.
- **Committed artifacts** — `best_model.pkl`, processed parquet, and results mean
  you can serve predictions immediately, or retrain from scratch with one command.
- **Pinned dependencies** — `scikit-learn==1.9.0` is pinned because the model was
  serialized with it; a version mismatch triggers `InconsistentVersionWarning` or
  subtle behavior changes on unpickle.
- **Dockerfile** — installs only the serving subset, copies code + artifacts, adds
  a `HEALTHCHECK`, and runs `uvicorn`. A reviewer with only Docker can run it.

**Reproduce from a clean clone:**
```bash
git clone https://github.com/cat-cols/telco-churn-analytics
cd telco-churn-analytics
python -m venv .venv && source .venv/bin/activate
pip install -e .
python src/run_pipeline.py          # full pipeline
uvicorn src.api:app --reload        # or serve the API
```

---

## 12. Anticipated interview questions (and my answers)

**Q: Why is your "best" model's recall so low?**
Because selection optimizes ROC-AUC, which rewards ranking quality, not the 0.5
decision threshold. For deployment I'd lower the threshold or use a balanced
variant — recall is the business-relevant metric for churn, and I expose
`threshold` in the API specifically to tune that operating point.

**Q: How do you prevent data leakage?**
Split before fitting transformers; fit the scaler/encoder on train only; persist
and reuse them at inference. No test data ever influences the transforms.

**Q: How do you handle a customer with a service plan you've never seen?**
`OneHotEncoder(handle_unknown="ignore")` maps it to an all-zero vector; the
predict path logs it; and the API would reject truly invalid categorical values
up front with a 422.

**Q: What would you add next?**
Model monitoring / drift detection, an automated retraining trigger, and
experiment tracking (MLflow). Possibly a SQL/DuckDB analytics layer and a small
Streamlit dashboard with a what-if threshold slider for non-technical
stakeholders.

**Q: What's the single biggest engineering decision?**
Treating the fitted preprocessing objects as first-class, versioned artifacts.
That one decision is what makes training and serving consistent and the whole
project reproducible.

---

## 13. One-paragraph summary

This project ingests the IBM Telco dataset, validates and transforms it with
leakage-safe, serialized preprocessing, trains and tunes a tournament of model
families while explicitly addressing class imbalance, and serves the winner
through a hardened FastAPI service with strict input/output contracts. It's fully
reproducible from a clean clone, tested in CI, and containerized — and I can
defend every metric, including the deliberate tension between ROC-AUC-based
selection and recall-driven business value.
```
