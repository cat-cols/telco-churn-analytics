"""
Integration tests for the full prediction pipeline.

Requires fitted artifacts in data/processed/ and models/.
Run: python3 src/run_pipeline.py  — before running these tests.

Skip gracefully if artifacts are missing so unit tests can still run in CI
without the full model chain.
"""

import pathlib
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from config import BEST_MODEL_FILE, ENCODER_FILE, SCALER_FILE

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "sample_customers.csv"

artifacts_missing = not all(p.exists() for p in [BEST_MODEL_FILE, SCALER_FILE, ENCODER_FILE])

pytestmark = pytest.mark.skipif(
    artifacts_missing,
    reason="Model artifacts not found — run python3 src/run_pipeline.py first",
)


# ---------------------------------------------------------------------------
# predict_from_file
# ---------------------------------------------------------------------------

class TestPredictFromFile:

    def test_output_file_created(self, tmp_path):
        from predict import predict_from_file
        output = tmp_path / "predictions.csv"
        predict_from_file(str(FIXTURE), str(output))
        assert output.exists()

    def test_output_schema_correct(self, tmp_path):
        from predict import predict_from_file
        output = tmp_path / "predictions.csv"
        predict_from_file(str(FIXTURE), str(output))
        result = pd.read_csv(output)
        for col in ["CustomerID", "Churn_Probability", "Predicted_Churn", "Risk_Level"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_row_count_matches_input(self, tmp_path):
        from predict import predict_from_file
        output = tmp_path / "predictions.csv"
        predict_from_file(str(FIXTURE), str(output))
        result = pd.read_csv(output)
        input_rows = len(pd.read_csv(FIXTURE))
        assert len(result) == input_rows

    def test_churn_probability_in_range(self, tmp_path):
        from predict import predict_from_file
        output = tmp_path / "predictions.csv"
        predict_from_file(str(FIXTURE), str(output))
        result = pd.read_csv(output)
        assert result["Churn_Probability"].between(0.0, 1.0).all()

    def test_predicted_churn_binary(self, tmp_path):
        from predict import predict_from_file
        output = tmp_path / "predictions.csv"
        predict_from_file(str(FIXTURE), str(output))
        result = pd.read_csv(output)
        assert bool(result["Predicted_Churn"].isin([0, 1]).all())

    def test_risk_level_valid_values(self, tmp_path):
        from predict import predict_from_file
        output = tmp_path / "predictions.csv"
        predict_from_file(str(FIXTURE), str(output))
        result = pd.read_csv(output)
        assert bool(result["Risk_Level"].isin(["Low", "Medium", "High"]).all())

    def test_new_customer_rows_handled(self, tmp_path):
        """Fixture contains 2 rows with tenure=0 — pipeline must not crash."""
        from predict import predict_from_file
        output = tmp_path / "predictions.csv"
        predict_from_file(str(FIXTURE), str(output))
        result = pd.read_csv(output)
        # Both new-customer rows (T0002, T0010) should appear in output
        assert "T0002" in result["CustomerID"].values
        assert "T0010" in result["CustomerID"].values

    def test_summary_file_created(self, tmp_path):
        from predict import predict_from_file
        output = tmp_path / "predictions.csv"
        predict_from_file(str(FIXTURE), str(output))
        summary = tmp_path / "predictions_summary.txt"
        assert summary.exists()
