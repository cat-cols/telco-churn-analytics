"""
Integration tests for the FastAPI churn prediction API.

These tests verify that the API endpoints work correctly with the actual
model artifacts and provide end-to-end validation of the service.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import app

# Create test client
client = TestClient(app)


class TestHealthEndpoint:
    """Test the health check endpoint."""
    
    def test_health_check_success(self):
        """Test health check when model is loaded successfully."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["scaler_loaded"] is True
        assert data["encoder_loaded"] is True


class TestPredictRecordsEndpoint:
    """Test the /predict/records endpoint for JSON input."""
    
    def test_single_record_prediction(self):
        """Test prediction for a single customer record."""
        # Create a valid customer record
        record = {
            "customerID": "test-customer-001",
            "SeniorCitizen": 0,
            "tenure": 12,
            "MonthlyCharges": 75.50,
            "TotalCharges": 906.00,
            "gender": "Female",
            "Partner": "Yes",
            "Dependents": "No",
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check"
        }
        
        request_data = {
            "records": [record],
            "threshold": 0.5
        }
        
        response = client.post("/predict/records", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "predictions" in data
        assert "summary" in data
        
        # Check predictions structure
        predictions = data["predictions"]
        assert len(predictions) == 1
        
        prediction = predictions[0]
        assert "CustomerID" in prediction
        assert "Churn_Probability" in prediction
        assert "Predicted_Churn" in prediction
        assert "Risk_Level" in prediction
        assert prediction["CustomerID"] == "test-customer-001"
        
        # Check summary structure
        summary = data["summary"]
        assert summary["total_customers"] == 1
        assert "predicted_churners" in summary
        assert "churn_rate" in summary
        assert summary["threshold_used"] == 0.5
        assert "risk_distribution" in summary
    
    def test_batch_prediction(self):
        """Test prediction for multiple customer records."""
        # Create multiple records
        records = [
            {
                "customerID": "test-customer-001",
                "SeniorCitizen": 0,
                "tenure": 12,
                "MonthlyCharges": 75.50,
                "TotalCharges": 906.00,
                "gender": "Female",
                "Partner": "Yes",
                "Dependents": "No",
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check"
            },
            {
                "customerID": "test-customer-002",
                "SeniorCitizen": 1,
                "tenure": 60,
                "MonthlyCharges": 95.00,
                "TotalCharges": 5700.00,
                "gender": "Male",
                "Partner": "No",
                "Dependents": "Yes",
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "Yes",
                "DeviceProtection": "Yes",
                "TechSupport": "Yes",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Two year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Mailed check"
            }
        ]
        
        request_data = {
            "records": records,
            "threshold": 0.3
        }
        
        response = client.post("/predict/records", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        predictions = data["predictions"]
        summary = data["summary"]
        
        assert len(predictions) == 2
        assert summary["total_customers"] == 2
        assert summary["threshold_used"] == 0.3
    
    def test_custom_threshold(self):
        """Test prediction with custom threshold."""
        record = {
            "customerID": "test-customer-003",
            "SeniorCitizen": 0,
            "tenure": 6,
            "MonthlyCharges": 50.00,
            "TotalCharges": 300.00,
            "gender": "Female",
            "Partner": "No",
            "Dependents": "No",
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check"
        }
        
        request_data = {
            "records": [record],
            "threshold": 0.8  # High threshold
        }
        
        response = client.post("/predict/records", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        summary = data["summary"]
        assert summary["threshold_used"] == 0.8
    
    def test_invalid_threshold(self):
        """Test prediction with invalid threshold values."""
        record = {
            "customerID": "test-customer-004",
            "SeniorCitizen": 0,
            "tenure": 12,
            "MonthlyCharges": 75.50,
            "TotalCharges": 906.00,
            "gender": "Female",
            "Partner": "Yes",
            "Dependents": "No",
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check"
        }
        
        # Test threshold > 1.0
        request_data = {
            "records": [record],
            "threshold": 1.5
        }
        
        response = client.post("/predict/records", json=request_data)
        assert response.status_code == 422  # Validation error
        
        # Test threshold < 0.0
        request_data["threshold"] = -0.1
        
        response = client.post("/predict/records", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_empty_records_list(self):
        """Test prediction with empty records list."""
        request_data = {
            "records": [],
            "threshold": 0.5
        }
        
        response = client.post("/predict/records", json=request_data)
        # This should work but return empty results
        assert response.status_code == 200
        
        data = response.json()
        assert data["predictions"] == []
        assert data["summary"]["total_customers"] == 0


class TestPredictFileEndpoint:
    """Test the /predict/file endpoint for CSV file uploads."""
    
    def test_csv_file_upload(self):
        """Test prediction with CSV file upload."""
        # Create test CSV data
        test_data = pd.DataFrame({
            'customerID': ['test-001', 'test-002'],
            'SeniorCitizen': [0, 1],
            'tenure': [12, 60],
            'MonthlyCharges': [75.50, 95.00],
            'TotalCharges': [906.00, 5700.00],
            'gender': ['Female', 'Male'],
            'Partner': ['Yes', 'No'],
            'Dependents': ['No', 'Yes'],
            'PhoneService': ['Yes', 'Yes'],
            'MultipleLines': ['No', 'Yes'],
            'InternetService': ['Fiber optic', 'DSL'],
            'OnlineSecurity': ['No', 'Yes'],
            'OnlineBackup': ['Yes', 'Yes'],
            'DeviceProtection': ['No', 'Yes'],
            'TechSupport': ['No', 'Yes'],
            'StreamingTV': ['Yes', 'No'],
            'StreamingMovies': ['No', 'No'],
            'Contract': ['Month-to-month', 'Two year'],
            'PaperlessBilling': ['Yes', 'No'],
            'PaymentMethod': ['Electronic check', 'Mailed check']
        })
        
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data.to_csv(f.name, index=False)
            temp_file_path = f.name
        
        try:
            # Upload file
            with open(temp_file_path, 'rb') as f:
                response = client.post(
                    "/predict/file",
                    files={"file": ("test.csv", f, "text/csv")}
                )
            
            assert response.status_code == 200
            
            data = response.json()
            predictions = data["predictions"]
            summary = data["summary"]
            
            assert len(predictions) == 2
            assert summary["total_customers"] == 2
            assert summary["threshold_used"] == 0.5  # Default threshold
            
        finally:
            # Clean up temporary file
            Path(temp_file_path).unlink()
    
    def test_invalid_file_type(self):
        """Test upload with non-CSV file."""
        # Create a text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is not a CSV file")
            temp_file_path = f.name
        
        try:
            # Try to upload as CSV
            with open(temp_file_path, 'rb') as f:
                response = client.post(
                    "/predict/file",
                    files={"file": ("test.txt", f, "text/plain")}
                )
            
            assert response.status_code == 400
            assert "Only CSV files are supported" in response.json()["detail"]
            
        finally:
            Path(temp_file_path).unlink()
    
    def test_empty_csv_file(self):
        """Test upload with empty CSV file."""
        # Create empty CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("")  # Empty file
            temp_file_path = f.name
        
        try:
            # Upload empty file
            with open(temp_file_path, 'rb') as f:
                response = client.post(
                    "/predict/file",
                    files={"file": ("empty.csv", f, "text/csv")}
                )
            
            assert response.status_code == 400
            assert "empty or invalid" in response.json()["detail"]
            
        finally:
            Path(temp_file_path).unlink()
    
    def test_malformed_csv_file(self):
        """Test upload with malformed CSV file."""
        # Create malformed CSV (missing columns)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("customerID,gender\n")  # Missing required columns
            f.write("test-001,Female\n")
            temp_file_path = f.name
        
        try:
            # Upload malformed file
            with open(temp_file_path, 'rb') as f:
                response = client.post(
                    "/predict/file",
                    files={"file": ("malformed.csv", f, "text/csv")}
                )
            
            # This should return an error due to missing columns
            assert response.status_code == 500
            
        finally:
            Path(temp_file_path).unlink()


class TestAPIIntegration:
    """Integration tests that verify API behavior with edge cases."""
    
    def test_new_customer_handling(self):
        """Test that new customers (tenure=0) are handled correctly."""
        record = {
            "customerID": "new-customer-001",
            "SeniorCitizen": 0,
            "tenure": 0,  # New customer
            "MonthlyCharges": 0.00,  # Should be set to 0
            "TotalCharges": 0.00,  # Should be set to 0
            "gender": "Female",
            "Partner": "No",
            "Dependents": "No",
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check"
        }
        
        request_data = {
            "records": [record],
            "threshold": 0.5
        }
        
        response = client.post("/predict/records", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        summary = data["summary"]
        
        # Should have new customer statistics
        assert "new_customers" in summary
        assert summary["new_customers"] == 1
        assert "new_customer_churn_rate" in summary
    
    def test_unseen_category_handling(self):
        """Test that unseen categorical values are handled gracefully."""
        record = {
            "customerID": "unseen-category-001",
            "SeniorCitizen": 0,
            "tenure": 12,
            "MonthlyCharges": 75.50,
            "TotalCharges": 906.00,
            "gender": "Non-binary",  # Unseen category
            "Partner": "Maybe",     # Unseen category
            "Dependents": "No",
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check"
        }
        
        request_data = {
            "records": [record],
            "threshold": 0.5
        }
        
        response = client.post("/predict/records", json=request_data)
        # Should handle unseen categories gracefully
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__])
