"""
FastAPI application for telco churn prediction.

This module provides REST API endpoints for making predictions on customer data.
It wraps the existing prediction logic from src/predict.py and loads the model
once at startup for efficiency.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from io import BytesIO
from typing import Any, AsyncIterator, List, Dict, Union, Literal

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field, model_validator

from config import BEST_MODEL_FILE, ENCODER_FILE, SCALER_FILE
from predict import predict, validate_output_schema
from utils import load_pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MiB
UPLOAD_READ_CHUNK_BYTES = 1024 * 1024  # 1 MiB

YesNoType = Literal["Yes", "No"]
GenderType = Literal["Male", "Female"]
MultipleLinesType = Literal["Yes", "No", "No phone service"]
InternetServiceType = Literal["DSL", "Fiber optic", "No"]
InternetDependentServiceType = Literal["Yes", "No", "No internet service"]
ContractType = Literal["Month-to-month", "One year", "Two year"]
PaymentMethodType = Literal[
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]


@dataclass(frozen=True)
class ModelArtifacts:
    """Loaded model and preprocessing artifacts used by the prediction pipeline."""

    model: Any
    scaler: Any
    encoder: Any


def _load_model_artifacts() -> ModelArtifacts:
    """Load model and preprocessors once during app startup."""
    logger.info("Loading model artifacts")
    artifacts = ModelArtifacts(
        model=load_pickle(BEST_MODEL_FILE),
        scaler=load_pickle(SCALER_FILE),
        encoder=load_pickle(ENCODER_FILE),
    )
    logger.info("Model and preprocessors loaded successfully")
    return artifacts


def _get_artifacts(request: Request) -> ModelArtifacts:
    """Return loaded artifacts or raise a 503 response if the app is not ready."""
    artifacts = getattr(request.app.state, "artifacts", None)
    if not isinstance(artifacts, ModelArtifacts):
        raise HTTPException(status_code=503, detail="Model or preprocessors not loaded")
    return artifacts


def _empty_response(threshold: float) -> PredictionResponse:
    """Build a consistent response for an empty input batch."""
    return PredictionResponse(
        predictions=[],
        summary={
            "total_customers": 0,
            "predicted_churners": 0,
            "churn_rate": 0.0,
            "threshold_used": threshold,
            "risk_distribution": {},
        },
    )


def _predictions_to_json_records(results: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert a prediction DataFrame into JSON-serializable records."""
    json_str = results.to_json(orient="records")
    loaded_json = [] if json_str is None else json.loads(json_str)
    if not isinstance(loaded_json, list):
        raise ValueError("Prediction output could not be converted to JSON records")
    return loaded_json


def _numeric_series(series: pd.Series) -> pd.Series:
    """Coerce a Series to numeric values with invalid values treated as zero."""
    numeric_series = pd.Series(pd.to_numeric(series, errors="coerce"))
    return numeric_series.fillna(0)


def _series_sum_as_int(series: pd.Series) -> int:
    """Return a scalar integer sum without relying on pandas Series.sum typing."""
    values = _numeric_series(series).to_numpy()
    return int(values.sum())


def _series_mean_as_float(series: pd.Series) -> float:
    """Return a scalar float mean without returning NaN for empty/all-invalid data."""
    if series.empty:
        return 0.0
    numeric = _numeric_series(series)
    mean_value = numeric.mean()
    return 0.0 if pd.isna(mean_value) else float(mean_value)


def _build_summary(results: pd.DataFrame, threshold: float) -> Dict[str, Any]:
    """Build summary metrics while safely handling empty outputs."""
    total_customers = len(results)
    predicted_churners = (
        _series_sum_as_int(pd.Series(results["Predicted_Churn"])) if total_customers else 0
    )
    churn_rate = (
        _series_mean_as_float(pd.Series(results["Predicted_Churn"])) if total_customers else 0.0
    )

    summary: Dict[str, Any] = {
        "total_customers": total_customers,
        "predicted_churners": predicted_churners,
        "churn_rate": churn_rate,
        "threshold_used": threshold,
        "risk_distribution": results["Risk_Level"].value_counts().to_dict()
        if total_customers
        else {},
    }

    if "Is_New_Customer" in results.columns and total_customers:
        new_customers = _series_sum_as_int(pd.Series(results["Is_New_Customer"]))
        if new_customers > 0:
            new_customer_mask = _numeric_series(pd.Series(results["Is_New_Customer"])) == 1
            new_churn_rate = _series_mean_as_float(
                results.loc[new_customer_mask, "Predicted_Churn"]
            )
            summary.update(
                {
                    "new_customers": new_customers,
                    "new_customer_churn_rate": new_churn_rate,
                }
            )

    return summary


def _predict_dataframe(
    df: pd.DataFrame,
    artifacts: ModelArtifacts,
    threshold: float,
) -> PredictionResponse:
    """Run the core prediction pipeline and shape the API response."""
    if df.empty:
        return _empty_response(threshold)

    results = predict(
        df,
        artifacts.model,
        artifacts.scaler,
        artifacts.encoder,
        threshold=threshold,
    )

    is_valid, schema_errors = validate_output_schema(results)
    if not is_valid:
        logger.error("Output schema validation failed: %s", schema_errors)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction output validation failed: {schema_errors}",
        )

    predictions = _predictions_to_json_records(results)
    summary = _build_summary(results, threshold)
    return PredictionResponse(predictions=predictions, summary=summary)


async def _read_upload_with_limit(file: UploadFile) -> bytes:
    """Read an uploaded file in chunks and reject files over MAX_UPLOAD_BYTES."""
    content = bytearray()

    while True:
        chunk = await file.read(UPLOAD_READ_CHUNK_BYTES)
        if not chunk:
            break

        content.extend(chunk)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Uploaded CSV exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB limit",
            )

    return bytes(content)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load model and preprocessors at startup, cleanup at shutdown."""
    logger.info("Starting up FastAPI application...")
    app.state.artifacts = _load_model_artifacts()
    try:
        yield
    finally:
        app.state.artifacts = None
        logger.info("Shutting down FastAPI application...")


class CustomerRecord(BaseModel):
    """Pydantic model for a single customer record."""

    customerID: str = Field(..., min_length=1, description="Unique customer identifier")

    SeniorCitizen: int = Field(
        ..., ge=0, le=1, description="Whether the customer is a senior citizen (0 or 1)"
    )
    tenure: int = Field(
        ...,
        ge=0,
        le=120,
        description="Number of months the customer has been with the company",
    )
    MonthlyCharges: float = Field(
        ...,
        ge=0,
        le=10_000,
        description="Monthly amount charged to the customer",
    )
    TotalCharges: float = Field(
        ...,
        ge=0,
        le=1_000_000,
        description="Total amount charged to the customer",
    )

    gender: GenderType = Field(..., description="Customer gender")
    Partner: YesNoType = Field(..., description="Whether the customer has a partner")
    Dependents: YesNoType = Field(..., description="Whether the customer has dependents")
    PhoneService: YesNoType = Field(..., description="Whether the customer has phone service")
    MultipleLines: MultipleLinesType = Field(
        ..., description="Whether the customer has multiple lines"
    )
    InternetService: InternetServiceType = Field(..., description="Internet service type")
    OnlineSecurity: InternetDependentServiceType = Field(
        ..., description="Whether the customer has online security"
    )
    OnlineBackup: InternetDependentServiceType = Field(
        ..., description="Whether the customer has online backup"
    )
    DeviceProtection: InternetDependentServiceType = Field(
        ..., description="Whether the customer has device protection"
    )
    TechSupport: InternetDependentServiceType = Field(
        ..., description="Whether the customer has tech support"
    )
    StreamingTV: InternetDependentServiceType = Field(
        ..., description="Whether the customer has streaming TV"
    )
    StreamingMovies: InternetDependentServiceType = Field(
        ..., description="Whether the customer has streaming movies"
    )
    Contract: ContractType = Field(..., description="Contract type")
    PaperlessBilling: YesNoType = Field(
        ..., description="Whether the customer has paperless billing"
    )
    PaymentMethod: PaymentMethodType = Field(..., description="Payment method")

    @model_validator(mode="after")
    def validate_service_consistency(self) -> CustomerRecord:
        """Reject internally inconsistent service combinations early."""
        if self.PhoneService == "No" and self.MultipleLines != "No phone service":
            raise ValueError("MultipleLines must be 'No phone service' when PhoneService is 'No'")

        if self.PhoneService == "Yes" and self.MultipleLines == "No phone service":
            raise ValueError("MultipleLines cannot be 'No phone service' when PhoneService is 'Yes'")

        internet_fields = [
            self.OnlineSecurity,
            self.OnlineBackup,
            self.DeviceProtection,
            self.TechSupport,
            self.StreamingTV,
            self.StreamingMovies,
        ]
        if self.InternetService == "No" and any(
            value != "No internet service" for value in internet_fields
        ):
            raise ValueError(
                "Internet-dependent services must be 'No internet service' when InternetService is 'No'"
            )

        if self.InternetService != "No" and any(
            value == "No internet service" for value in internet_fields
        ):
            raise ValueError(
                "Internet-dependent services cannot be 'No internet service' when InternetService is active"
            )

        if self.tenure == 0 and self.TotalCharges > 0:
            raise ValueError("TotalCharges must be 0 when tenure is 0")

        return self


class PredictionRequest(BaseModel):
    """Request model for batch prediction with customer records."""

    records: List[CustomerRecord] = Field(
        ..., description="List of customer records to predict"
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Classification threshold for churn prediction",
    )


class PredictionResponse(BaseModel):
    """Response model for prediction results."""

    predictions: List[Dict[str, Any]] = Field(
        ..., description="Prediction results for each customer"
    )
    summary: Dict[str, Any] = Field(
        ..., description="Summary statistics of predictions"
    )


app = FastAPI(
    title="Telco Churn Prediction API",
    description="API for predicting customer churn in telecommunications",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check(request: Request) -> Dict[str, Union[bool, str]]:
    """Health check endpoint to verify the API is running and model is loaded."""
    _get_artifacts(request)
    return {
        "status": "healthy",
        "model_loaded": True,
        "scaler_loaded": True,
        "encoder_loaded": True,
    }


@app.post("/predict/records", response_model=PredictionResponse)
async def predict_records(
    request_body: PredictionRequest,
    request: Request,
) -> PredictionResponse:
    """Predict churn for customer records provided as JSON."""
    artifacts = _get_artifacts(request)

    if not request_body.records:
        logger.info("Received empty records list")
        return _empty_response(request_body.threshold)

    try:
        records_data = [record.model_dump() for record in request_body.records]
        df = pd.DataFrame(records_data)
        logger.info("Received %s records for prediction", len(df))

        response = _predict_dataframe(df, artifacts, request_body.threshold)
        logger.info(
            "Prediction completed: %s churners out of %s customers",
            response.summary["predicted_churners"],
            response.summary["total_customers"],
        )
        return response
    except HTTPException:
        raise
    except (ValueError, KeyError, TypeError, RuntimeError) as exc:
        logger.exception("Error during prediction")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.post("/predict/file", response_model=PredictionResponse)
async def predict_from_uploaded_file(
    request: Request,
    file: UploadFile = File(...),
    threshold: float = Query(
        0.5,
        ge=0.0,
        le=1.0,
        description="Classification threshold for churn prediction",
    ),
) -> PredictionResponse:
    """Predict churn for customers from an uploaded CSV file."""
    artifacts = _get_artifacts(request)

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    try:
        content = await _read_upload_with_limit(file)
        if not content.strip():
            raise HTTPException(
                status_code=400, detail="Uploaded CSV file is empty or invalid"
            )

        logger.info("Processing uploaded file: %s", file.filename)
        df = pd.read_csv(BytesIO(content))
        logger.info("Loaded %s records from uploaded file", len(df))

        response = _predict_dataframe(df, artifacts, threshold)
        logger.info(
            "File prediction completed: %s churners out of %s customers",
            response.summary["predicted_churners"],
            response.summary["total_customers"],
        )
        return response
    except HTTPException:
        raise
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        logger.warning("Uploaded CSV file is empty or invalid: %s", exc)
        raise HTTPException(
            status_code=400, detail="Uploaded CSV file is empty or invalid"
        ) from exc
    except (ValueError, KeyError, TypeError, RuntimeError) as exc:
        logger.exception("Error processing uploaded file")
        raise HTTPException(
            status_code=500, detail=f"File processing failed: {exc}"
        ) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
