"""
Unit tests for src/preprocess.py

Covers:
- validate_schema: required columns, min row count, dtype checks
"""

import pathlib
import sys

from typing import Any

import pandas as pd
import pytest

from src.preprocess import validate_schema
from src.config import NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS, IDENTIFIER_COLUMN, TARGET_COLUMN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_df(n_rows: int = 200) -> pd.DataFrame:
    """Return a minimal valid DataFrame that passes schema validation."""
    all_cols = [IDENTIFIER_COLUMN] + NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS + [TARGET_COLUMN]
    data: dict[str, list[Any]] = {col: ["placeholder"] * n_rows for col in all_cols}
    data["SeniorCitizen"] = [0] * n_rows
    data["tenure"] = [12] * n_rows
    data["MonthlyCharges"] = [55.90] * n_rows
    data["TotalCharges"] = ["1341.60"] * n_rows
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# validate_schema
# ---------------------------------------------------------------------------

class TestValidateSchema:

    def test_valid_dataframe_passes(self):
        df = _valid_df()
        validate_schema(df)  # must not raise

    def test_missing_column_raises(self):
        df = _valid_df().drop(columns=["tenure"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_schema(df)

    def test_multiple_missing_columns_reported_together(self):
        df = _valid_df().drop(columns=["tenure", "MonthlyCharges"])
        with pytest.raises(ValueError) as exc_info:
            validate_schema(df)
        msg = str(exc_info.value)
        assert "tenure" in msg
        assert "MonthlyCharges" in msg

    def test_extra_columns_do_not_cause_failure(self):
        df = _valid_df()
        df["extra_column"] = "foo"
        validate_schema(df)  # must not raise

    def test_too_few_rows_raises(self):
        df = _valid_df(n_rows=5)
        with pytest.raises(ValueError, match="rows"):
            validate_schema(df)

    def test_exactly_min_rows_passes(self):
        from preprocess import MIN_EXPECTED_ROWS
        df = _valid_df(n_rows=MIN_EXPECTED_ROWS)
        validate_schema(df)  # must not raise

    def test_wrong_dtype_senior_citizen_raises(self):
        df = _valid_df()
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)  # force string
        with pytest.raises(ValueError, match="SeniorCitizen"):
            validate_schema(df)

    def test_wrong_dtype_tenure_raises(self):
        df = _valid_df()
        df["tenure"] = df["tenure"].astype(str)
        with pytest.raises(ValueError, match="tenure"):
            validate_schema(df)

    def test_total_charges_as_string_does_not_raise(self):
        """TotalCharges is intentionally excluded from dtype checks — arrives as string from CSV."""
        df = _valid_df()
        df["TotalCharges"] = "1341.60"  # string
        validate_schema(df)  # must not raise

    def test_error_message_lists_all_problems(self):
        """All failures collected and raised together, not one at a time."""
        df = _valid_df(n_rows=5).drop(columns=["tenure"])
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)
        with pytest.raises(ValueError) as exc_info:
            validate_schema(df)
        msg = str(exc_info.value)
        assert "tenure" in msg
        assert "rows" in msg
        assert "SeniorCitizen" in msg
