"""
End-to-end pipeline orchestration script.
Runs the complete workflow: preprocessing → training → results.
"""

import argparse
import logging

from preprocess import preprocess_data
from train import train_models
from utils import load_parquet, load_pickle
import pandas as pd
from config import (
    MODELS_DIR,
    RESULTS_FILE,
    PROCESSED_DATA_DIR,
    IDENTIFIER_COLUMN,
    OUTPUTS_DIR,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_full_pipeline(skip_preprocessing: bool = False, skip_training: bool = False):
    """Run the complete pipeline from raw data to results."""
    logger.info("=" * 60)
    logger.info("STARTING END-TO-END PIPELINE")
    logger.info("=" * 60)

    # Step 1: Preprocessing
    if not skip_preprocessing:
        logger.info("\n[STEP 1/3] DATA PREPROCESSING")
        logger.info("-" * 60)
        X_train, X_test, y_train, y_test = preprocess_data()
        logger.info("✓ Preprocessing completed")
    else:
        logger.info("\n[STEP 1/3] SKIPPING PREPROCESSING (using existing data)")

    # Step 2: Model Training
    if not skip_training:
        logger.info("\n[STEP 2/3] MODEL TRAINING")
        logger.info("-" * 60)
        best_model, results_df = train_models()
        logger.info("✓ Model training completed")
    else:
        logger.info("\n[STEP 2/3] SKIPPING TRAINING (using existing model)")
        best_model = load_pickle(MODELS_DIR / "best_model.pkl")
        results_df = pd.read_csv(RESULTS_FILE)

    # Step 3: Generate Predictions on Test Set
    logger.info("\n[STEP 3/3] GENERATING PREDICTIONS")
    logger.info("-" * 60)

    X_test = load_parquet(PROCESSED_DATA_DIR / "test.parquet")
    y_test = load_parquet(PROCESSED_DATA_DIR / "test_labels.parquet").squeeze()

    X_test_features = X_test.drop(columns=[IDENTIFIER_COLUMN])
    y_pred = best_model.predict(X_test_features)
    y_proba = best_model.predict_proba(X_test_features)[:, 1]

    # Create predictions dataframe
    predictions_df = pd.DataFrame(
        {
            "CustomerID": X_test[IDENTIFIER_COLUMN],
            "Actual_Churn": y_test,
            "Predicted_Churn": y_pred,
            "Churn_Probability": y_proba,
        }
    )

    # Save predictions
    predictions_df.to_csv(OUTPUTS_DIR / "test_predictions.csv", index=False)
    logger.info(f"✓ Predictions saved to {OUTPUTS_DIR / 'test_predictions.csv'}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info("\nBest Model Performance:")
    logger.info(f"  Accuracy: {results_df['accuracy'].max():.4f}")
    logger.info(f"  ROC-AUC: {results_df['roc_auc'].max():.4f}")
    logger.info(f"  Precision: {results_df['precision'].max():.4f}")
    logger.info(f"  Recall: {results_df['recall'].max():.4f}")
    logger.info(f"  F1-Score: {results_df['f1'].max():.4f}")

    logger.info("\nGenerated Files:")
    logger.info(f"  - Processed data: {PROCESSED_DATA_DIR}")
    logger.info(f"  - Trained models: {MODELS_DIR}")
    logger.info(f"  - Outputs: {OUTPUTS_DIR / 'test_predictions.csv'}")
    logger.info(f"  - Model comparison: {MODELS_DIR / 'model_comparison_results.csv'}")

    return best_model, results_df, predictions_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the complete churn prediction pipeline"
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip preprocessing step (use existing processed data)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training step (use existing model)",
    )

    args = parser.parse_args()

    run_full_pipeline(
        skip_preprocessing=args.skip_preprocessing, skip_training=args.skip_training
    )
