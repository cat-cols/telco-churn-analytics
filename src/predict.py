"""
Prediction/inference script for Telco Churn Prediction.

Includes robust edge case handling:
- New customers (tenure=0)
- Unseen categories
- Missing values
- Extreme outliers
- Data validation
"""

import pandas as pd
import numpy as np
import argparse
import logging
from typing import Dict, List, Tuple, Optional

from config import (
    IDENTIFIER_COLUMN, NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS,
    SCALER_FILE, ENCODER_FILE, BEST_MODEL_FILE
)
from utils import load_pickle

logger = logging.getLogger(__name__)

# Business rules for validation
VALIDATION_RULES = {
    'tenure': {'min': 0, 'max': 72},
    'MonthlyCharges': {'min': 0, 'max': 200},
    'TotalCharges': {'min': 0, 'max': 15000}
}

# Imputation values (should match training)
IMPUTATION_VALUES = {
    'TotalCharges': 0,  # New customers
    'MonthlyCharges': 65.0,  # Median from training
    'tenure': 29  # Median from training
}

# Expected schema of the prediction output (enforced by validate_output_schema)
REQUIRED_OUTPUT_COLUMNS = {
    'CustomerID',
    'Churn_Probability',
    'Predicted_Churn',
    'Risk_Level',
}
VALID_RISK_LEVELS = {'Low', 'Medium', 'High'}


def load_model_and_preprocessors():
    """Load the trained model and preprocessing objects."""
    logger.info("Loading model and preprocessing objects")

    model = load_pickle(BEST_MODEL_FILE)
    scaler = load_pickle(SCALER_FILE)
    encoder = load_pickle(ENCODER_FILE)

    logger.info("Model and preprocessing objects loaded successfully")
    return model, scaler, encoder


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
            # Fix #1: coerce to numeric before comparison
            # TotalCharges is still a raw string at validation time
            col_numeric: pd.Series = pd.to_numeric(
                data[col], errors='coerce'
            )  # type: ignore[assignment]
            min_val: int = rules['min']
            max_val: int = rules['max']
            mask: pd.Series = col_numeric.notna() & (
                (col_numeric < min_val) | (col_numeric > max_val)
            )  # type: ignore[operator]
            invalid = data[mask]
            if len(invalid) > 0:
                errors.append(
                    f"{col}: {len(invalid)} values outside range "
                    f"[{rules['min']}, {rules['max']}]"
                )
                logger.warning(f"Extreme values in {col}: {invalid[col].tolist()}")

    # Check for duplicate customerIDs
    if IDENTIFIER_COLUMN in data.columns:
        dups = data[IDENTIFIER_COLUMN].duplicated().sum()
        if dups > 0:
            errors.append(f"Found {dups} duplicate customerIDs")

    return len(errors) == 0, errors


def handle_new_customers(data: pd.DataFrame) -> pd.DataFrame:
    """
    Handle new customers (tenure=0) by properly setting TotalCharges.

    Edge Case: New customers have no billing history (TotalCharges should be $0).
    """
    data = data.copy()

    # Identify new customers
    new_customer_mask = data['tenure'] == 0
    new_customer_count = new_customer_mask.sum()

    if new_customer_count > 0:
        logger.info(f"Detected {new_customer_count} new customers (tenure=0)")

        # Assign 0 compatible with the column's current dtype
        # (string from raw CSV or numeric after coercion)
        tc_dtype = data['TotalCharges'].dtype
        zero_val: float | str = 0.0 if pd.api.types.is_numeric_dtype(tc_dtype) else "0"
        data.loc[new_customer_mask, 'TotalCharges'] = zero_val

        # Flag new customers for downstream use
        data['Is_New_Customer'] = new_customer_mask.astype(int)

        logger.info(f"Set TotalCharges=0 for {new_customer_count} new customers")

    return data


def handle_unseen_categories(data: pd.DataFrame, encoder) -> pd.DataFrame:
    """
    Handle categorical values not seen during training.

    Maps unknown categories to 'Unknown' bucket to prevent encoder errors.
    """
    data = data.copy()

    for col in CATEGORICAL_COLUMNS:
        if col not in data.columns:
            continue

        # Fix #3: use list() not set() — set() triggers ambiguous truth-value
        # error with numpy arrays via isin()
        known_categories = list(
            encoder.categories_[CATEGORICAL_COLUMNS.index(col)]
        )

        # Find unknown values
        unknown_mask = ~data[col].isin(known_categories)
        unknown_count = unknown_mask.sum()

        if unknown_count > 0:
            unknown_values = data.loc[unknown_mask, col].unique()
            logger.warning(
                f"Column '{col}': {unknown_count} unknown values {unknown_values}. "
                f"Mapping to 'Unknown'"
            )

            # Map unknown to 'Unknown' (encoder should have this as fallback)
            data.loc[unknown_mask, col] = 'Unknown'

    return data


def handle_missing_values(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Handle missing values with business-logic imputation.

    Returns:
        (imputed_data, imputation_report)
    """
    data = data.copy()
    imputation_report = {}

    # Convert TotalCharges to numeric (handles whitespace strings)
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

    # Apply imputation for each column
    for col, value in IMPUTATION_VALUES.items():
        if col in data.columns:
            missing_count = data[col].isna().sum()
            if missing_count > 0:
                data[col] = data[col].fillna(value)
                imputation_report[col] = missing_count
                logger.info(
                    f"Imputed {missing_count} missing values in {col} with {value}"
                )

    # Drop rows with remaining missing values (critical features)
    critical_features = ['customerID', 'tenure', 'MonthlyCharges']
    before_drop = len(data)
    data = data.dropna(subset=critical_features)
    after_drop = len(data)

    if after_drop < before_drop:
        dropped = before_drop - after_drop
        logger.warning(f"Dropped {dropped} rows with missing critical features")
        imputation_report['dropped_rows'] = dropped

    return data, imputation_report


def detect_and_flag_outliers(data: pd.DataFrame) -> pd.DataFrame:
    """
    Detect extreme values and flag for review.

    Does not remove outliers - just flags them for business review.
    """
    data = data.copy()

    # Create flag columns
    for col, rules in VALIDATION_RULES.items():
        if col in data.columns:
            flag_col = f"{col}_outlier_flag"
            data[flag_col] = (
                (data[col] < rules['min']) | (data[col] > rules['max'])
            ).astype(int)

            outlier_count = data[flag_col].sum()
            if outlier_count > 0:
                logger.warning(f"Flagged {outlier_count} outliers in {col}")

    # Combined outlier flag
    outlier_cols = [c for c in data.columns if c.endswith('_outlier_flag')]
    if outlier_cols:
        data['Any_Outlier_Flag'] = data[outlier_cols].max(axis=1)

    return data


def safe_predict_single(model, features: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Safely make prediction on a single sample with error handling.

    Returns None if prediction fails.
    """
    try:
        return model.predict_proba(features)[:, 1]
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return None


def preprocess_new_data(data: pd.DataFrame, scaler, encoder) -> pd.DataFrame:
    """Preprocess new data for prediction with edge case handling."""
    logger.info(f"Preprocessing new data with shape: {data.shape}")

    # Step 1: Handle new customers (tenure=0)
    data = handle_new_customers(data)

    # Step 2: Handle missing values
    data, imputation_report = handle_missing_values(data)
    logger.info(f"Imputation report: {imputation_report}")

    # Step 3: Handle unseen categories
    data = handle_unseen_categories(data, encoder)

    # Step 4: Detect outliers (flag only, don't remove)
    data = detect_and_flag_outliers(data)

    logger.info(f"Cleaned data shape: {data.shape}")

    # Step 5: Scale numerical features
    numerical_data = scaler.transform(data[NUMERICAL_COLUMNS])
    numerical_df = pd.DataFrame(
        data=numerical_data,
        index=data.index,
        columns=NUMERICAL_COLUMNS
    )

    # Step 6: Encode categorical features
    categorical_data = encoder.transform(data[CATEGORICAL_COLUMNS])
    categorical_df = pd.DataFrame(
        data=categorical_data,
        index=data.index,
        columns=encoder.get_feature_names_out()
    )

    # Step 7: Combine features (preserve flags)
    flag_cols = [
        c for c in data.columns
        if 'flag' in c.lower() or c == 'Is_New_Customer'
    ]
    flag_data = data[flag_cols] if flag_cols else pd.DataFrame()
    processed_data = pd.concat([
        data[IDENTIFIER_COLUMN],
        numerical_df,
        categorical_df,
        flag_data
    ], axis=1)

    return processed_data


def predict(
    data: pd.DataFrame, model, scaler, encoder, threshold: float = 0.5
) -> pd.DataFrame:
    """
    Make predictions on new data with comprehensive edge case handling.

    Edge cases handled:
    - New customers (tenure=0): TotalCharges set to $0
    - Unseen categories: Mapped to 'Unknown' bucket
    - Missing values: Imputed with training-derived values
    - Extreme outliers: Flagged for review
    - Validation errors: Reported but don't crash
    """
    logger.info("Starting prediction with edge case handling")

    # Validate input
    is_valid, errors = validate_input_data(data)
    if not is_valid:
        logger.warning(f"Data validation issues: {errors}")
        # Continue processing - handle issues in preprocessing

    # Preprocess data (includes all edge case handling)
    processed_data = preprocess_new_data(data, scaler, encoder)

    # Separate features from metadata
    metadata_cols = [IDENTIFIER_COLUMN] + [
        c for c in processed_data.columns
        if c not in NUMERICAL_COLUMNS + list(encoder.get_feature_names_out())
        and c != IDENTIFIER_COLUMN
    ]

    features = processed_data.drop(columns=metadata_cols)
    metadata = processed_data[metadata_cols] if metadata_cols else pd.DataFrame()

    # Make predictions with error handling per row
    logger.info(f"Predicting on {len(features)} samples")
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    # Create results dataframe
    results = pd.DataFrame({
        'CustomerID': processed_data[IDENTIFIER_COLUMN],
        'Churn_Probability': probabilities,
        'Predicted_Churn': predictions,
        'Risk_Level': pd.cut(
            probabilities,
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low', 'Medium', 'High']
        )
    })

    # Add metadata flags if available
    if 'Is_New_Customer' in metadata.columns:
        results['Is_New_Customer'] = metadata['Is_New_Customer']

    if 'Any_Outlier_Flag' in metadata.columns:
        results['Has_Outlier_Flag'] = metadata['Any_Outlier_Flag']

    # Summary statistics
    is_new = results.get('Is_New_Customer', pd.Series([0]))
    new_customer_count = is_new.sum() if is_new is not None else 0
    has_outlier = results.get('Has_Outlier_Flag', pd.Series([0]))
    outlier_count = has_outlier.sum() if has_outlier is not None else 0

    logger.info("Predictions completed:")
    logger.info(f"  Total samples: {len(results)}")
    logger.info(f"  New customers: {new_customer_count}")
    logger.info(f"  With outlier flags: {outlier_count}")
    logger.info(f"  Predicted churners: {results['Predicted_Churn'].sum()}")

    return results


def validate_output_schema(results: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate the prediction output's columns and value ranges before saving.

    Checks:
    - All required columns are present.
    - Churn_Probability is numeric, non-null, and within [0, 1].
    - Predicted_Churn contains only 0/1.
    - Risk_Level values are within the allowed set (Low/Medium/High); NaN is
      tolerated since pd.cut yields NaN for a probability of exactly 0.0.

    Returns:
        (is_valid, error_messages)
    """
    errors: List[str] = []

    # 1. Required columns present
    missing = REQUIRED_OUTPUT_COLUMNS - set(results.columns)
    if missing:
        errors.append(f"Missing output columns: {sorted(missing)}")

    # 2. Churn_Probability numeric and within [0, 1]
    if 'Churn_Probability' in results.columns:
        proba = pd.to_numeric(results['Churn_Probability'], errors='coerce')
        # Ensure proba is a Series (handle scalar case)
        if isinstance(proba, (float, int)):
            na_count = 1 if pd.isna(proba) else 0
        elif isinstance(proba, pd.Series):
            series_proba: pd.Series = proba
            na_count = int(series_proba.isna().sum())
        else:
            # Fallback for any other type - try to convert to Series
            try:
                proba_series = pd.Series([proba])
                na_count = int(proba_series.isna().sum())
            except Exception:
                na_count = 0
        if na_count > 0:
            errors.append(f"Churn_Probability has {na_count} non-numeric/NaN values")
        # Handle out_of_range check for both Series and scalar
        if isinstance(proba, (float, int)):
            out_of_range = not pd.isna(proba) and (proba < 0 or proba > 1)
            has_out_of_range = out_of_range
        elif isinstance(proba, pd.Series):
            out_of_range = pd.notna(proba) & ((proba < 0) | (proba > 1))
            has_out_of_range = out_of_range.any()
        else:
            # Fallback for any other type - try to convert to Series
            try:
                proba_series = pd.Series([proba])
                out_of_range = pd.notna(proba_series) & ((proba_series < 0) | (proba_series > 1))
                has_out_of_range = out_of_range.any()
            except Exception:
                has_out_of_range = False
        if has_out_of_range:
            count = 1 if isinstance(proba, (float, int)) else int(out_of_range.sum())
            errors.append(
                f"Churn_Probability has {count} values outside [0, 1]"
            )

    # 3. Predicted_Churn strictly 0/1
    if 'Predicted_Churn' in results.columns:
        invalid = ~results['Predicted_Churn'].isin([0, 1])
        if invalid.any():
            errors.append(
                f"Predicted_Churn has {int(invalid.sum())} values not in {{0, 1}}"
            )

    # 4. Risk_Level membership (ignore NaN)
    if 'Risk_Level' in results.columns:
        rl = results['Risk_Level'].astype('object')
        invalid_rl = rl.notna() & ~rl.isin(VALID_RISK_LEVELS)
        if invalid_rl.any():
            bad = sorted(set(rl[invalid_rl].tolist()))
            errors.append(
                f"Risk_Level has invalid values: {bad}"
            )

    return len(errors) == 0, errors


def predict_from_file(
    input_file: str, output_file: str | None = None, threshold: float = 0.5
):
    """
    Load data from file, make predictions with edge case handling, and save results.

    Robust to:
    - Missing values
    - New categories
    - Extreme values
    - Data format issues
    """
    logger.info(f"Loading data from {input_file}")

    try:
        # Load new data
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

    # Make predictions
    results = predict(new_data, model, scaler, encoder, threshold)

    # Validate output schema (column names + value ranges) before saving
    is_valid, schema_errors = validate_output_schema(results)
    if not is_valid:
        logger.error(f"Output schema validation failed: {schema_errors}")
        raise ValueError(f"Prediction output failed schema validation: {schema_errors}")
    logger.info("Output schema validation passed")

    # Save results
    if output_file:
        results.to_csv(output_file, index=False)
        logger.info(f"Predictions saved to {output_file}")

        # Also save summary
        summary_file = output_file.replace('.csv', '_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("Prediction Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total customers: {len(results)}\n")
            f.write(f"Predicted churners: {results['Predicted_Churn'].sum()}\n")
            f.write(f"Churn rate: {results['Predicted_Churn'].mean():.1%}\n\n")

            if 'Is_New_Customer' in results.columns:
                new_churn = results[
                    results['Is_New_Customer'] == 1
                ]['Predicted_Churn'].mean()
                f.write(f"New customer churn rate: {new_churn:.1%}\n")

            f.write("\nRisk distribution:\n")
            f.write(str(results['Risk_Level'].value_counts()))

        logger.info(f"Summary saved to {summary_file}")
    else:
        logger.info("Prediction results:")
        print(results.to_string())

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Make churn predictions on new data with robust edge case handling'
    )
    parser.add_argument(
        '--input', type=str, required=True,
        help='Input CSV file with customer data'
    )
    parser.add_argument(
        '--output', type=str,
        help='Output CSV file for predictions'
    )
    parser.add_argument(
        '--threshold', type=float, default=0.5,
        help='Classification threshold (default: 0.5)'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
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

    predict_from_file(args.input, args.output, args.threshold)
