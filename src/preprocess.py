"""
Data preprocessing pipeline for Telco Churn Prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging

from config import (
    DATA_PATH, PROCESSED_DATA_DIR, NUMERICAL_COLUMNS,
    CATEGORICAL_COLUMNS, IDENTIFIER_COLUMN, TARGET_COLUMN,
    RANDOM_STATE, TEST_SIZE, TRAIN_DATA_FILE, TEST_DATA_FILE,
    TRAIN_LABELS_FILE, TEST_LABELS_FILE, SCALER_FILE, ENCODER_FILE
)
from utils import load_data, save_parquet, save_pickle

logger = logging.getLogger(__name__)


def preprocess_data():
    """Execute the complete data preprocessing pipeline."""
    logger.info("Starting data preprocessing pipeline")

    # Load data
    data = load_data(DATA_PATH)
    logger.info(f"Loaded data with shape: {data.shape}")

    # Convert TotalCharges from string to numeric
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    logger.info("Converted TotalCharges from string to numeric")

    # Clean data
    data_clean = data.dropna()
    logger.info(f"Cleaned data shape: {data_clean.shape}")

    # Separate features and target
    features = data_clean.drop(columns=[TARGET_COLUMN])
    labels = data_clean[TARGET_COLUMN]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    # Scale numerical features
    scaler = StandardScaler()
    X_train_numerical = X_train[NUMERICAL_COLUMNS]
    scaler.fit(X_train_numerical)

    X_train_scaled = scaler.transform(X_train_numerical)
    X_train_scaled_df = pd.DataFrame(
        data=X_train_scaled,
        index=X_train.index,
        columns=NUMERICAL_COLUMNS
    )

    X_test_scaled = scaler.transform(X_test[NUMERICAL_COLUMNS])
    X_test_scaled_df = pd.DataFrame(
        data=X_test_scaled,
        index=X_test.index,
        columns=NUMERICAL_COLUMNS
    )

    # Encode categorical features
    encoder = OneHotEncoder(sparse_output=False)
    X_train_categorical = X_train[CATEGORICAL_COLUMNS]
    encoder.fit(X_train_categorical)

    X_train_encoded = encoder.transform(X_train_categorical)
    X_train_encoded_df = pd.DataFrame(
        data=X_train_encoded,
        index=X_train.index,
        columns=encoder.get_feature_names_out()
    )

    X_test_encoded = encoder.transform(X_test[CATEGORICAL_COLUMNS])
    X_test_encoded_df = pd.DataFrame(
        data=X_test_encoded,
        index=X_test.index,
        columns=encoder.get_feature_names_out()
    )

    # Combine processed features
    X_train_processed = pd.concat([
        X_train[IDENTIFIER_COLUMN],
        X_train_scaled_df,
        X_train_encoded_df
    ], axis=1)

    X_test_processed = pd.concat([
        X_test[IDENTIFIER_COLUMN],
        X_test_scaled_df,
        X_test_encoded_df
    ], axis=1)

    # Save processed data
    save_parquet(X_train_processed, TRAIN_DATA_FILE)
    save_parquet(X_test_processed, TEST_DATA_FILE)
    save_parquet(y_train.to_frame(), TRAIN_LABELS_FILE)
    save_parquet(y_test.to_frame(), TEST_LABELS_FILE)

    # Save preprocessing objects
    save_pickle(scaler, SCALER_FILE)
    save_pickle(encoder, ENCODER_FILE)

    logger.info("Data preprocessing pipeline completed successfully")
    logger.info(f"Processed data saved to {PROCESSED_DATA_DIR}")

    return X_train_processed, X_test_processed, y_train, y_test


if __name__ == "__main__":
    preprocess_data()
