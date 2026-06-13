# Error Handling Guide: Telco Churn Prediction

This guide documents the error handling philosophy, patterns, and conventions used throughout the Telco Churn Prediction codebase.

---

## Philosophy

**Fail fast, log clearly, recover gracefully.**

- **Development-time errors** (type mismatches, missing files): Raise immediately with clear messages
- **Runtime data issues** (invalid values, missing columns): Log warnings and apply sensible defaults
- **Critical failures** (model not found, data corruption): Raise after logging context

---

## Logging Standards

### Logger Configuration

Every module uses a module-level logger:

```python
import logging

logger = logging.getLogger(__name__)
```

### Log Levels

| Level | When to Use | Example |
|-------|-------------|---------|
| `INFO` | Normal operation milestones | "Loaded data with shape: (1000, 21)" |
| `WARNING` | Data issues that were auto-corrected | "Column 'X': 5 unknown values replaced" |
| `ERROR` | Failures that prevent completion | "Failed to load model: File not found" |
| `DEBUG` | Detailed diagnostics (verbose mode) | Feature shapes, intermediate values |

### Log Format

CLI scripts configure logging based on verbosity:

```python
if args.verbose:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
else:
    logging.basicConfig(
        level=logging.WARNING,
        format='%(levelname)s: %(message)s'
    )
```

---

## Pattern 1: Validation with Error Collection

**Use case:** Input validation where multiple issues may exist.

```python
def validate_input_data(data: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate input data before processing.

    Returns:
        (is_valid, error_messages)
    """
    errors = []

    # Check required columns exist
    required_cols = set(NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS + [IDENTIFIER_COLUMN])
    missing_cols = required_cols - set(data.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # Check for extreme values
    for col, rules in VALIDATION_RULES.items():
        if col in data.columns:
            invalid = data[(data[col] < rules['min']) | (data[col] > rules['max'])]
            if len(invalid) > 0:
                errors.append(f"{col}: {len(invalid)} values outside range")
                logger.warning(f"Extreme values in {col}: {invalid[col].tolist()}")

    return len(errors) == 0, errors
```

**Principle:** Collect all validation errors before failing, so users can fix multiple issues at once.

---

## Pattern 2: Try/Except with Context Logging

**Use case:** File I/O, external dependencies, or model loading.

```python
def predict_from_file(input_file: str, output_file: str | None = None):
    """Load data from file, make predictions, and save results."""
    logger.info(f"Loading data from {input_file}")

    try:
        new_data = pd.read_csv(input_file)
        logger.info(f"Loaded {len(new_data)} rows from {input_file}")
    except Exception as e:
        logger.error(f"Failed to load input file: {e}")
        raise

    # Load model and preprocessors
    try:
        model, scaler, encoder = load_model_and_preprocessors()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # ... continue processing
```

**Principle:** Log the operation before attempting it. If it fails, log the error with context, then re-raise.

---

## Pattern 3: Graceful Degradation with Warnings

**Use case:** Data edge cases that shouldn't halt execution.

```python
def handle_unseen_categories(data: pd.DataFrame, encoder) -> pd.DataFrame:
    """Handle unseen categories by replacing with most frequent known category."""
    data = data.copy()

    for col in CATEGORICAL_COLUMNS:
        if col not in data.columns:
            continue

        known_categories = list(encoder.categories_[
            CATEGORICAL_COLUMNS.index(col)
        ])

        unknown_mask = ~data[col].isin(known_categories)
        unknown_count = unknown_mask.sum()

        if unknown_count > 0:
            unknown_values = data.loc[unknown_mask, col].unique()
            logger.warning(
                f"Column '{col}': {unknown_count} unknown values {unknown_values}. "
                f"Using most frequent category."
            )

            # Replace with most frequent known category
            most_frequent = known_categories[0]
            data.loc[unknown_mask, col] = most_frequent

    return data
```

**Principle:** Warn about data issues, apply reasonable defaults, continue processing.

---

## Pattern 4: Safe Attribute Access

**Use case:** Dictionary access where values might be `None`.

```python
# Before (type checker error):
count = results.get('Is_New_Customer', pd.Series([0])).sum()
# Error: "sum" is not a known attribute of "None"

# After (explicit None check):
is_new = results.get('Is_New_Customer', pd.Series([0]))
new_customer_count = is_new.sum() if is_new is not None else 0
```

**Principle:** Explicitly handle `None` cases to satisfy static analysis and prevent runtime errors.

---

## Pattern 5: Conditional Attribute Checking

**Use case:** ML models with optional methods.

```python
def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)

    if hasattr(model, 'predict_proba'):
        # Get probability of positive class
        classes = model.classes_
        if 'Yes' in classes:
            yes_index = list(classes).index('Yes')
            y_proba = model.predict_proba(X_test)[:, yes_index]
        else:
            y_proba = (y_pred == 'Yes').astype(int)
    else:
        y_proba = (y_pred == 'Yes').astype(int)

    # ... calculate metrics
```

**Principle:** Check for optional attributes before using them; provide fallback behavior.

---

## Pattern 6: Defensive Data Type Conversion

**Use case:** Data that may have incorrect types (e.g., numeric stored as string).

```python
# Convert TotalCharges from string to numeric
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
logger.info("Converted TotalCharges from string to numeric")

# Handle coercion results
data_clean = data.dropna()
logger.info(f"Cleaned data shape: {data_clean.shape}")
```

**Principle:** Use `errors='coerce'` to convert invalid values to `NaN`, then handle missing data explicitly.

---

## Critical Error Scenarios

### Missing Model Artifacts

```python
def load_model_and_preprocessors():
    """Load model with existence checks."""
    for file_path, name in [
        (BEST_MODEL_FILE, 'model'),
        (SCALER_FILE, 'scaler'),
        (ENCODER_FILE, 'encoder')
    ]:
        if not file_path.exists():
            raise FileNotFoundError(f"{name} file not found: {file_path}")

    # ... load objects
```

### Fitted Artifact Verification

```python
encoder = load_pickle(ENCODER_FILE)

if not hasattr(encoder, 'categories_'):
    raise ValueError("Encoder not fitted! Run training first.")

scaler = load_pickle(SCALER_FILE)

if not hasattr(scaler, 'mean_'):
    raise ValueError("Scaler not fitted! Run training first.")
```

---

## Business Rule Validation

Centralize business constraints in validation dictionaries:

```python
VALIDATION_RULES = {
    'tenure': {'min': 0, 'max': 72},
    'MonthlyCharges': {'min': 0, 'max': 200},
    'TotalCharges': {'min': 0, 'max': 15000}
}

IMPUTATION_VALUES = {
    'TotalCharges': 0,  # New customers
    'MonthlyCharges': 65.0,  # Median from training
    'tenure': 29  # Median from training
}
```

---

## Error Message Conventions

### Be Specific

```python
# Good
errors.append(f"Missing required columns: {missing_cols}")

# Bad
errors.append("Some columns are missing")
```

### Include Context

```python
# Good
logger.warning(
    f"Column '{col}': {unknown_count} unknown values {unknown_values}. "
    f"Using most frequent category."
)

# Bad
logger.warning("Unknown values found")
```

### Distinguish User vs. System Errors

```python
# User error (bad input)
raise ValueError(f"Missing required columns: {missing_cols}")

# System error (missing file)
raise FileNotFoundError(f"Model file not found: {model_path}")
```

---

## Testing Error Handling

When writing tests, verify both success and failure paths:

```python
def test_validate_input_data_missing_columns():
    """Test validation catches missing columns."""
    data = pd.DataFrame({'wrong_column': [1, 2, 3]})
    is_valid, errors = validate_input_data(data)

    assert not is_valid
    assert any("Missing required columns" in e for e in errors)

def test_handle_unseen_categories():
    """Test unknown categories are replaced with warning."""
    data = pd.DataFrame({'gender': ['Male', 'UnknownCategory']})
    # ... setup encoder

    with pytest.warns(UserWarning, match="unknown values"):
        result = handle_unseen_categories(data, encoder)

    assert result['gender'].iloc[1] != 'UnknownCategory'
```

---

## Quick Reference: When to Raise vs. Log

| Scenario | Action | Rationale |
|----------|--------|-----------|
| Missing required columns | **Raise** | Cannot proceed without data |
| Unknown category values | **Log + Fix** | Can substitute reasonable value |
| Extreme outliers | **Log** | Model can handle; flag for review |
| Missing model file | **Raise** | System configuration error |
| Failed pickle load | **Raise** | Data corruption or version mismatch |
| Type conversion fails | **Log + Coerce** | Convert to NaN, handle downstream |
| Duplicate customerIDs | **Log + Continue** | May be intentional batch processing |

---

## Related Documentation

- `debugging_log.md` — Specific bugs encountered and their fixes
- `data_dictionary.md` — Column definitions and constraints
- `data_schema.md` — Expected data types and ranges

---

*Last updated: 2024-01-13*
*Applies to: All Python modules in `src/`*
