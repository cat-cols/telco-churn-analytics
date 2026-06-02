"""
Model training script for Telco Churn Prediction.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import logging

from config import (
    PROCESSED_DATA_DIR, MODELS_DIR, IDENTIFIER_COLUMN,
    RANDOM_STATE, MODEL_CONFIGS, TRAIN_DATA_FILE, TEST_DATA_FILE,
    TRAIN_LABELS_FILE, TEST_LABELS_FILE, BEST_MODEL_FILE, RESULTS_FILE
)
from utils import load_parquet, save_pickle, evaluate_model

logger = logging.getLogger(__name__)


def train_models():
    """Train multiple models and select the best one."""
    logger.info("Starting model training")

    # Load processed data
    X_train = load_parquet(TRAIN_DATA_FILE)
    X_test = load_parquet(TEST_DATA_FILE)
    y_train = load_parquet(TRAIN_LABELS_FILE).squeeze()
    y_test = load_parquet(TEST_LABELS_FILE).squeeze()

    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Test data shape: {X_test.shape}")

    # Remove identifier
    X_train_features = X_train.drop(columns=[IDENTIFIER_COLUMN])
    X_test_features = X_test.drop(columns=[IDENTIFIER_COLUMN])
    
    # Initialize models
    models = {
        'logistic_regression': LogisticRegression(**MODEL_CONFIGS['logistic_regression']),
        'random_forest': RandomForestClassifier(**MODEL_CONFIGS['random_forest']),
        'gradient_boosting': GradientBoostingClassifier(**MODEL_CONFIGS['gradient_boosting'])
    }
    
    # Train and evaluate models
    results = []
    trained_models = {}
    
    for model_name, model in models.items():
        logger.info(f"Training {model_name}")
        
        model.fit(X_train_features, y_train)
        metrics = evaluate_model(model, X_test_features, y_test)
        metrics['model'] = model_name
        
        results.append(metrics)
        trained_models[model_name] = model
        
        logger.info(f"{model_name} - ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Select best model
    results_df = pd.DataFrame(results)
    best_model_name = results_df.loc[results_df['roc_auc'].idxmax(), 'model']
    best_model = trained_models[best_model_name]
    
    logger.info(f"Best model: {best_model_name} with ROC-AUC: {results_df['roc_auc'].max():.4f}")
    
    # Save best model
    save_pickle(best_model, BEST_MODEL_FILE)
    logger.info(f"Best model saved to {BEST_MODEL_FILE}")
    
    # Save results
    results_df.to_csv(RESULTS_FILE, index=False)
    logger.info(f"Results saved to {RESULTS_FILE}")
    
    logger.info("Model training completed successfully")
    
    return best_model, results_df


if __name__ == "__main__":
    train_models()
