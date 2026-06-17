"""
FastAPI application for telco churn prediction.

This module provides REST API endpoints for making predictions on customer data.
It wraps the existing prediction logic from src/predict.py and loads the model
once at startup for efficiency.
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from config import (
    BEST_MODEL_FILE,
    ENCODER_FILE,
    SCALER_FILE,
)
from predict import predict, validate_output_schema
from utils import load_pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and preprocessors
model = None
scaler = None
encoder = None


# Load model and preprocessors at import time for testing
def _load_model_artifacts():
    """Load model and preprocessors - called at import time."""
    global model, scaler, encoder
    try:
        logger.info(f"Loading model from {BEST_MODEL_FILE}")
        model = load_pickle(BEST_MODEL_FILE)

        logger.info(f"Loading scaler from {SCALER_FILE}")
        scaler = load_pickle(SCALER_FILE)

        logger.info(f"Loading encoder from {ENCODER_FILE}")
        encoder = load_pickle(ENCODER_FILE)

        logger.info("Model and preprocessors loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model artifacts: {e}")
        # Don't raise here - let the API handle missing models gracefully


# Load artifacts at import time
_load_model_artifacts()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and preprocessors at startup, cleanup at shutdown."""
    global model, scaler, encoder

    logger.info("Starting up FastAPI application...")

    try:
        # Load model and preprocessors
        logger.info(f"Loading model from {BEST_MODEL_FILE}")
        model = load_pickle(BEST_MODEL_FILE)

        logger.info(f"Loading scaler from {SCALER_FILE}")
        scaler = load_pickle(SCALER_FILE)

        logger.info(f"Loading encoder from {ENCODER_FILE}")
        encoder = load_pickle(ENCODER_FILE)

        logger.info("Model and preprocessors loaded successfully")

        yield

    except Exception as e:
        logger.error(f"Failed to load model or preprocessors: {e}")
        raise
    finally:
        logger.info("Shutting down FastAPI application...")


# Pydantic model for input validation
class CustomerRecord(BaseModel):
    """Pydantic model for a single customer record."""
    customerID: str = Field(..., description="Unique customer identifier")

    # Numerical fields
    SeniorCitizen: int = Field(
        ..., ge=0, le=1,
        description="Whether the customer is a senior citizen (0 or 1)"
    )
    tenure: int = Field(
        ..., ge=0,
        description="Number of months the customer has been with the company"
    )
    MonthlyCharges: float = Field(
        ..., ge=0, description="Monthly amount charged to the customer"
    )
    TotalCharges: float = Field(
        ..., ge=0, description="Total amount charged to the customer"
    )

    # Categorical fields
    gender: str = Field(..., description="Customer gender (Male/Female)")
    Partner: str = Field(
        ..., description="Whether the customer has a partner (Yes/No)"
    )
    Dependents: str = Field(
        ..., description="Whether the customer has dependents (Yes/No)"
    )
    PhoneService: str = Field(
        ..., description="Whether the customer has phone service (Yes/No)"
    )
    MultipleLines: str = Field(
        ..., description="Whether the customer has multiple lines "
        "(Yes/No/No phone service)"
    )
    InternetService: str = Field(
        ..., description="Internet service type (DSL/Fiber optic/No)"
    )
    OnlineSecurity: str = Field(
        ..., description="Whether the customer has online security "
        "(Yes/No/No internet service)"
    )
    OnlineBackup: str = Field(
        ..., description="Whether the customer has online backup "
        "(Yes/No/No internet service)"
    )
    DeviceProtection: str = Field(
        ..., description="Whether the customer has device protection "
        "(Yes/No/No internet service)"
    )
    TechSupport: str = Field(
        ..., description="Whether the customer has tech support "
        "(Yes/No/No internet service)"
    )
    StreamingTV: str = Field(
        ..., description="Whether the customer has streaming TV "
        "(Yes/No/No internet service)"
    )
    StreamingMovies: str = Field(
        ..., description="Whether the customer has streaming movies "
        "(Yes/No/No internet service)"
    )
    Contract: str = Field(
        ..., description="Contract type (Month-to-month/One year/Two year)"
    )
    PaperlessBilling: str = Field(
        ..., description="Whether the customer has paperless billing (Yes/No)"
    )
    PaymentMethod: str = Field(
        ..., description="Payment method (Electronic check/Mailed check/"
        "Bank transfer/Credit card)"
    )


class PredictionRequest(BaseModel):
    """Request model for batch prediction with customer records."""
    records: List[CustomerRecord] = Field(
        ..., description="List of customer records to predict"
    )
    threshold: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Classification threshold for churn prediction"
    )


class PredictionResponse(BaseModel):
    """Response model for prediction results."""
    predictions: List[Dict[str, Any]] = Field(
        ..., description="Prediction results for each customer"
    )
    summary: Dict[str, Any] = Field(
        ..., description="Summary statistics of predictions"
    )


# Create FastAPI app
app = FastAPI(
    title="Telco Churn Prediction API",
    description="API for predicting customer churn in telecommunications",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """Health check endpoint to verify the API is running and model is loaded."""
    if model is None or scaler is None or encoder is None:
        raise HTTPException(status_code=503, detail="Model or preprocessors not loaded")

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "encoder_loaded": encoder is not None
    }


@app.post("/predict/records", response_model=PredictionResponse)
async def predict_records(request: PredictionRequest):
    """
    Predict churn for customer records provided as JSON.

    Args:
        request: PredictionRequest containing customer records and optional threshold

    Returns:
        PredictionResponse with predictions and summary statistics
    """
    if model is None or scaler is None or encoder is None:
        raise HTTPException(status_code=503, detail="Model or preprocessors not loaded")

    try:
        # Handle empty records list
        if not request.records:
            logger.info("Received empty records list")
            return PredictionResponse(
                predictions=[],
                summary={
                    "total_customers": 0,
                    "predicted_churners": 0,
                    "churn_rate": 0.0,
                    "threshold_used": request.threshold,
                    "risk_distribution": {}
                }
            )

        # Convert Pydantic models to DataFrame
        records_data = [record.model_dump() for record in request.records]
        df = pd.DataFrame(records_data)

        logger.info(f"Received {len(df)} records for prediction")

        # Make predictions using the core predict function
        results = predict(df, model, scaler, encoder, threshold=request.threshold)

        # Validate output schema
        is_valid, schema_errors = validate_output_schema(results)
        if not is_valid:
            logger.error(f"Output schema validation failed: {schema_errors}")
            raise HTTPException(
                status_code=500,
                detail=f"Prediction output validation failed: {schema_errors}"
            )

        # Convert DataFrame to JSON-serializable format
        # predictions = json.loads(results.to_json(orient='records'))
        json_str = results.to_json(orient='records')
        if json_str is None:
            predictions = []
        else:
            predictions = json.loads(json_str)

        # Create summary statistics
        summary = {
            "total_customers": len(results),
            "predicted_churners": int(results['Predicted_Churn'].sum().item()),
            "churn_rate": float(results['Predicted_Churn'].mean().item()),
            "threshold_used": request.threshold,
            "risk_distribution": results['Risk_Level'].value_counts().to_dict()
        }

        # Add new customer stats if available
        if 'Is_New_Customer' in results.columns:
            new_customers = results['Is_New_Customer'].sum()
            if new_customers > 0:
                new_churn_rate = results[
                    results['Is_New_Customer'] == 1
                ]['Predicted_Churn'].mean()
                summary.update({
                    "new_customers": int(new_customers.item()),
                    "new_customer_churn_rate": float(
                        new_churn_rate.item() if not pd.isna(new_churn_rate).any() else 0.0
                    )
                })

        logger.info(
            f"Prediction completed: {summary['predicted_churners']} churners "
            f"out of {summary['total_customers']} customers"
        )

        return PredictionResponse(predictions=predictions, summary=summary)

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/file", response_model=PredictionResponse)
async def predict_from_uploaded_file(file: UploadFile = File(...)):
    """
    Predict churn for customers from an uploaded CSV file.

    Args:
        file: CSV file containing customer data

    Returns:
        PredictionResponse with predictions and summary statistics
    """
    if model is None or scaler is None or encoder is None:
        raise HTTPException(status_code=503, detail="Model or preprocessors not loaded")

    # Validate file type
    if not file.filename or not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    temp_file_path = None
    try:
        # Save uploaded file to temporary location
        with NamedTemporaryFile(mode='w+b', suffix='.csv', delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        logger.info(f"Processing uploaded file: {file.filename}")

        # Load data from temporary file
        df = pd.read_csv(temp_file_path)
        logger.info(f"Loaded {len(df)} records from uploaded file")

        # Make predictions using the core predict function
        results = predict(df, model, scaler, encoder)

        # Validate output schema
        is_valid, schema_errors = validate_output_schema(results)
        if not is_valid:
            logger.error(f"Output schema validation failed: {schema_errors}")
            raise HTTPException(
                status_code=500,
                detail=f"Prediction output validation failed: {schema_errors}"
            )

        # Convert DataFrame to JSON-serializable format
        json_str = results.to_json(orient='records')
        if json_str is None:
            predictions = []
        else:
            predictions = json.loads(json_str)

        # Create summary statistics
        summary = {
            "total_customers": len(results),
            "predicted_churners": int(results['Predicted_Churn'].sum().item()),
            "churn_rate": float(results['Predicted_Churn'].mean().item()),
            "threshold_used": 0.5,  # Default threshold for file upload
            "risk_distribution": results['Risk_Level'].value_counts().to_dict()
        }

        # Add new customer stats if available
        if 'Is_New_Customer' in results.columns:
            new_customers = results['Is_New_Customer'].sum()
            if new_customers > 0:
                new_churn_rate = results[
                    results['Is_New_Customer'] == 1
                ]['Predicted_Churn'].mean()
                summary.update({
                    "new_customers": int(new_customers.item()),
                    "new_customer_churn_rate": float(
                        new_churn_rate.item() if not pd.isna(new_churn_rate).any() else 0.0
                    )
                })

        logger.info(
            f"File prediction completed: {summary['predicted_churners']} churners "
            f"out of {summary['total_customers']} customers"
        )

        # Clean up temporary file
        if temp_file_path:
            Path(temp_file_path).unlink(missing_ok=True)

        return PredictionResponse(predictions=predictions, summary=summary)

    except pd.errors.EmptyDataError:
        logger.error("Uploaded CSV file is empty or invalid")
        # Clean up temporary file if it exists
        if temp_file_path:
            Path(temp_file_path).unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="Uploaded CSV file is empty or invalid"
        )
    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}")
        # Clean up temporary file if it exists
        if temp_file_path:
            Path(temp_file_path).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
