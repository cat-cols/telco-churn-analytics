"""
Unit tests for src/predict.py

Covers:
- validate_input_data  (regression fix #1)
- handle_new_customers (contract test)
- handle_unseen_categories (regression fix #3)
- handle_missing_values (contract test)
"""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder

from src.predict import (
    handle_missing_values,
    handle_new_customers,
    handle_unseen_categories,
    validate_input_data,
    validate_output_schema,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_row(**overrides) -> dict:
    """Return a dict representing one valid customer row with all required columns."""
    base = {
        "customerID": "T0001",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 24,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "Yes",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 55.90,
        "TotalCharges": "1341.60",
        "Churn": "No",
    }
    base.update(overrides)
    return base


def _df(*rows) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def _fitted_encoder(categories: list[list[str]]) -> OneHotEncoder:
    """Return a fitted OneHotEncoder for use in handle_unseen_categories tests."""
    from config import CATEGORICAL_COLUMNS
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    sample = pd.DataFrame({col: [cats[0]] for col, cats in zip(CATEGORICAL_COLUMNS, categories)})
    encoder.fit(sample)
    return encoder


# ---------------------------------------------------------------------------
# validate_input_data
# ---------------------------------------------------------------------------

class TestValidateInputData:

    def test_valid_data_passes(self):
        data = _df(_minimal_row())
        is_valid, errors = validate_input_data(data)
        assert is_valid is True
        assert errors == []

    def test_regression_fix1_string_total_charges_no_type_error(self):
        """Regression for fix #1 — TypeError when TotalCharges is string dtype."""
        data = _df(_minimal_row(TotalCharges="1341.60"))
        # Must not raise TypeError
        is_valid, errors = validate_input_data(data)
        assert is_valid is True

    def test_missing_column_reported(self):
        data = _df(_minimal_row())
        data = data.drop(columns=["tenure"])
        is_valid, errors = validate_input_data(data)
        assert is_valid is False
        assert any("tenure" in e for e in errors)

    def test_out_of_range_monthly_charges_reported(self):
        data = _df(_minimal_row(MonthlyCharges=9999))
        is_valid, errors = validate_input_data(data)
        assert is_valid is False
        assert any("MonthlyCharges" in e for e in errors)

    def test_duplicate_customer_ids_reported(self):
        data = _df(_minimal_row(customerID="DUP"), _minimal_row(customerID="DUP"))
        is_valid, errors = validate_input_data(data)
        assert is_valid is False
        assert any("duplicate" in e.lower() for e in errors)

    def test_whitespace_total_charges_not_flagged_as_out_of_range(self):
        """Whitespace TotalCharges coerces to NaN — should not be counted as out-of-range."""
        data = _df(_minimal_row(TotalCharges=" "))
        is_valid, errors = validate_input_data(data)
        assert not any("TotalCharges" in e for e in errors)


# ---------------------------------------------------------------------------
# handle_new_customers
# ---------------------------------------------------------------------------

class TestHandleNewCustomers:

    def test_tenure_zero_sets_total_charges_to_zero(self):
        data = _df(
            _minimal_row(customerID="NEW", tenure=0, TotalCharges=" "),
            _minimal_row(customerID="OLD", tenure=12, TotalCharges="450.00"),
        )
        result = handle_new_customers(data)
        new_row = result[result["customerID"] == "NEW"]
        assert float(pd.Series(new_row["TotalCharges"]).values[0]) == 0.0

    def test_existing_customer_total_charges_unchanged(self):
        data = _df(_minimal_row(customerID="OLD", tenure=12, TotalCharges="450.00"))
        result = handle_new_customers(data)
        assert result["TotalCharges"].iloc[0] == "450.00"

    def test_new_customer_flag_added(self):
        data = _df(_minimal_row(tenure=0, TotalCharges=" "))
        result = handle_new_customers(data)
        assert "Is_New_Customer" in result.columns
        assert result["Is_New_Customer"].iloc[0] == 1

    def test_no_flag_column_when_no_new_customers(self):
        data = _df(_minimal_row(tenure=12, TotalCharges="450.00"))
        result = handle_new_customers(data)
        assert "Is_New_Customer" not in result.columns

    def test_input_not_mutated(self):
        data = _df(_minimal_row(tenure=0, TotalCharges=" "))
        original_len = len(data)
        handle_new_customers(data)
        assert len(data) == original_len
        assert "Is_New_Customer" not in data.columns


# ---------------------------------------------------------------------------
# handle_unseen_categories
# ---------------------------------------------------------------------------

class TestHandleUnseenCategories:

    def _make_encoder_with_gender(self) -> OneHotEncoder:
        from config import CATEGORICAL_COLUMNS
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        # Fit on two rows so both 'Male' and 'Female' are known categories
        row = {col: "Yes" for col in CATEGORICAL_COLUMNS}
        row["gender"] = "Male"
        row2 = dict(row)
        row2["gender"] = "Female"
        encoder.fit(pd.DataFrame([row, row2]))
        return encoder

    def test_regression_fix3_unknown_category_no_value_error(self):
        """Regression for fix #3 — ValueError from set() vs list() in isin()."""
        encoder = self._make_encoder_with_gender()
        data = _df(_minimal_row(gender="Unknown_Gender"))
        # Must not raise ValueError
        result = handle_unseen_categories(data, encoder)
        assert result is not None

    def test_known_category_unchanged(self):
        encoder = self._make_encoder_with_gender()
        data = _df(_minimal_row(gender="Male"))
        result = handle_unseen_categories(data, encoder)
        assert result["gender"].iloc[0] == "Male"

    def test_unknown_category_mapped(self):
        encoder = self._make_encoder_with_gender()
        data = _df(_minimal_row(gender="Nonbinary"))
        result = handle_unseen_categories(data, encoder)
        assert result["gender"].iloc[0] == "Unknown"

    def test_input_not_mutated(self):
        encoder = self._make_encoder_with_gender()
        data = _df(_minimal_row(gender="Nonbinary"))
        handle_unseen_categories(data, encoder)
        assert data["gender"].iloc[0] == "Nonbinary"


# ---------------------------------------------------------------------------
# handle_missing_values
# ---------------------------------------------------------------------------

class TestValidateOutputSchema:

    def _valid_output(self, **overrides) -> pd.DataFrame:
        base = pd.DataFrame({
            "CustomerID": ["T0001", "T0002"],
            "Churn_Probability": [0.12, 0.84],
            "Predicted_Churn": [0, 1],
            "Risk_Level": pd.Categorical(["Low", "High"],
                                         categories=["Low", "Medium", "High"]),
        })
        for col, vals in overrides.items():
            base[col] = vals
        return base

    def test_valid_output_passes(self):
        is_valid, errors = validate_output_schema(self._valid_output())
        assert is_valid is True
        assert errors == []

    def test_missing_required_column_reported(self):
        data = self._valid_output().drop(columns=["Risk_Level"])
        is_valid, errors = validate_output_schema(data)
        assert is_valid is False
        assert any("Risk_Level" in e for e in errors)

    def test_probability_out_of_range_reported(self):
        data = self._valid_output(Churn_Probability=[1.5, -0.2])
        is_valid, errors = validate_output_schema(data)
        assert is_valid is False
        assert any("Churn_Probability" in e and "[0, 1]" in e for e in errors)

    def test_nan_probability_reported(self):
        data = self._valid_output(Churn_Probability=[float("nan"), 0.5])
        is_valid, errors = validate_output_schema(data)
        assert is_valid is False
        assert any("Churn_Probability" in e for e in errors)

    def test_predicted_churn_not_binary_reported(self):
        data = self._valid_output(Predicted_Churn=[0, 2])
        is_valid, errors = validate_output_schema(data)
        assert is_valid is False
        assert any("Predicted_Churn" in e for e in errors)

    def test_invalid_risk_level_reported(self):
        data = self._valid_output()
        data["Risk_Level"] = ["Low", "Critical"]
        is_valid, errors = validate_output_schema(data)
        assert is_valid is False
        assert any("Risk_Level" in e for e in errors)

    def test_nan_risk_level_tolerated(self):
        """pd.cut yields NaN for probability exactly 0.0 — should not fail validation."""
        data = self._valid_output()
        data["Risk_Level"] = pd.Categorical(["Low", None],
                                            categories=["Low", "Medium", "High"])
        is_valid, errors = validate_output_schema(data)
        assert is_valid is True


class TestHandleMissingValues:

    def test_whitespace_total_charges_imputed(self):
        data = _df(_minimal_row(TotalCharges=" "))
        result, report = handle_missing_values(data)
        assert pd.to_numeric(result["TotalCharges"].iloc[0], errors="coerce") == 0.0

    def test_nan_monthly_charges_imputed_with_median(self):
        data = _df(_minimal_row(MonthlyCharges=float("nan")))
        result, report = handle_missing_values(data)
        assert result["MonthlyCharges"].iloc[0] == 65.0

    def test_imputation_report_records_count(self):
        data = _df(_minimal_row(TotalCharges=" "), _minimal_row(TotalCharges=" "))
        _, report = handle_missing_values(data)
        assert report.get("TotalCharges", 0) == 2

    def test_row_dropped_when_critical_feature_missing(self):
        # customerID is a critical feature not in IMPUTATION_VALUES, so NaN won't be filled
        data = _df(_minimal_row())
        data["customerID"] = None
        result, report = handle_missing_values(data)
        assert len(result) == 0
        assert report.get("dropped_rows", 0) == 1
