"""
Prediction/inference script for Telco Churn Prediction.
"""

import pandas as pd
import numpy as np
import argparse
import logging

from config import (
    PROCESSED_DATA_DIR, MODELS_DIR, IDENTIFIER_COLUMN,
    NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS,
    SCALER_FILE, ENCODER_FILE, BEST_MODEL_FILE
)
from utils import load_pickle

logger = logging.getLogger(__name__)


def load_model_and_preprocessors():
    """Load the trained model and preprocessing objects."""
    logger.info("Loading model and preprocessing objects")
    
    model = load_pickle(BEST_MODEL_FILE)
    scaler = load_pickle(SCALER_FILE)
    encoder = load_pickle(ENCODER_FILE)
    
    logger.info("Model and preprocessing objects loaded successfully")
    return model, scaler, encoder


def preprocess_new_data(data: pd.DataFrame, scaler, encoder) -> pd.DataFrame:
    """Preprocess new data for prediction."""
    logger.info(f"Preprocessing new data with shape: {data.shape}")
    
    # Clean data - handle TotalCharges conversion and missing values
    data_clean = data.copy()
    data_clean['TotalCharges'] = pd.to_numeric(data_clean['TotalCharges'], errors='coerce')
    data_clean = data_clean.dropna()
    logger.info(f"Cleaned data shape: {data_clean.shape}")
    
    # Scale numerical features
    numerical_data = scaler.transform(data_clean[NUMERICAL_COLUMNS])
    numerical_df = pd.DataFrame(
        data=numerical_data,
        index=data_clean.index,
        columns=NUMERICAL_COLUMNS
    )
    
    # Encode categorical features
    categorical_data = encoder.transform(data_clean[CATEGORICAL_COLUMNS])
    categorical_df = pd.DataFrame(
        data=categorical_data,
        index=data_clean.index,
        columns=encoder.get_feature_names_out()
    )
    
    # Combine features
    processed_data = pd.concat([
        data_clean[IDENTIFIER_COLUMN],
        numerical_df,
        categorical_df
    ], axis=1)
    
    return processed_data


def predict(data: pd.DataFrame, model, scaler, encoder, threshold: float = 0.5) -> pd.DataFrame:
    """Make predictions on new data."""
    logger.info("Making predictions")
    
    # Preprocess data
    processed_data = preprocess_new_data(data, scaler, encoder)
    
    # Remove identifier for prediction
    features = processed_data.drop(columns=[IDENTIFIER_COLUMN])
    
    # Make predictions
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    
    # Create results dataframe using the cleaned data
    results = pd.DataFrame({
        'CustomerID': processed_data[IDENTIFIER_COLUMN],
        'Churn_Probability': probabilities,
        'Predicted_Churn': (probabilities >= threshold).astype(int)
    })
    
    logger.info(f"Predictions completed for {len(results)} samples")
    return results


def predict_from_file(input_file: str, output_file: str | None = None, threshold: float = 0.5):
    """Load data from file, make predictions, and save results."""
    logger.info(f"Loading data from {input_file}")
    
    # Load new data
    new_data = pd.read_csv(input_file)
    
    # Load model and preprocessors
    model, scaler, encoder = load_model_and_preprocessors()
    
    # Make predictions
    results = predict(new_data, model, scaler, encoder, threshold)
    
    # Save results
    if output_file:
        results.to_csv(output_file, index=False)
        logger.info(f"Predictions saved to {output_file}")
    else:
        logger.info("Prediction results:")
        print(results)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make churn predictions on new data')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file with customer data')
    parser.add_argument('--output', type=str, help='Output CSV file for predictions')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold (default: 0.5)')
    
    args = parser.parse_args()
    
    predict_from_file(args.input, args.output, args.threshold)
