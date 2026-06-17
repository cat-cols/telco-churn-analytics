"""
Model training script for Telco Churn Prediction.

Stages:
1. Train baseline models (no class weighting)
2. Train balanced variants (class_weight='balanced' or sample_weight for GB)
3. GridSearchCV on the best model from stages 1+2
4. Select overall best by ROC-AUC; save artifact and full comparison table
"""

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight
import logging

from config import (
    IDENTIFIER_COLUMN, PARAM_GRIDS, RANDOM_STATE,
    MODEL_CONFIGS, MODEL_CONFIGS_BALANCED,
    TRAIN_DATA_FILE, TEST_DATA_FILE,
    TRAIN_LABELS_FILE, TEST_LABELS_FILE,
    BEST_MODEL_FILE, RESULTS_FILE
)
from utils import load_parquet, save_pickle, evaluate_model

logger = logging.getLogger(__name__)


def _build_models(configs: dict) -> dict:
    """Instantiate sklearn estimators from a config dict."""
    mapping = {
        'logistic_regression':          LogisticRegression,
        'logistic_regression_balanced': LogisticRegression,
        'random_forest':                RandomForestClassifier,
        'random_forest_balanced':       RandomForestClassifier,
        'gradient_boosting':            GradientBoostingClassifier,
        'gradient_boosting_balanced':   GradientBoostingClassifier,
    }
    return {name: mapping[name](**params) for name, params in configs.items()}


def train_models():
    """Train all model variants, tune the best, and save the winner."""
    logger.info("Starting model training")

    # Load processed data
    X_train = load_parquet(TRAIN_DATA_FILE)
    X_test = load_parquet(TEST_DATA_FILE)
    y_train = load_parquet(TRAIN_LABELS_FILE).squeeze()
    y_test = load_parquet(TEST_LABELS_FILE).squeeze()

    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Test data shape: {X_test.shape}")

    X_train_f = X_train.drop(columns=[IDENTIFIER_COLUMN])
    X_test_f = X_test.drop(columns=[IDENTIFIER_COLUMN])

    # Sample weights for GB balanced (GradientBoosting has no class_weight param)
    sample_weights = compute_sample_weight('balanced', y_train)

    results: list[dict] = []
    trained_models: dict = {}

    # --- Stage 1 & 2: baseline + balanced variants ---
    all_configs = {**MODEL_CONFIGS, **MODEL_CONFIGS_BALANCED}
    models = _build_models(all_configs)

    for name, model in models.items():
        logger.info(f"Training {name}")
        fit_kwargs: dict = {}
        if name == 'gradient_boosting_balanced':
            fit_kwargs['sample_weight'] = sample_weights

        model.fit(X_train_f, y_train, **fit_kwargs)
        metrics = evaluate_model(model, X_test_f, y_test)
        metrics['model'] = name
        metrics['stage'] = 'balanced' if 'balanced' in name else 'baseline'
        results.append(metrics)
        trained_models[name] = model
        logger.info(
            f"  {name}: ROC-AUC={metrics['roc_auc']:.4f}  "
            f"Recall={metrics['recall']:.4f}"
        )

    # Pick best from stages 1+2 to tune
    stage_df = pd.DataFrame(results)
    best_pre_tune = stage_df.loc[stage_df['roc_auc'].idxmax(), 'model']
    logger.info(f"Best pre-tuning: {best_pre_tune}")

    # --- Stage 3: GridSearchCV on the base model family of the winner ---
    base_family = best_pre_tune.replace('_balanced', '')
    param_grid = PARAM_GRIDS.get(base_family, {})

    if param_grid:
        logger.info(f"Running GridSearchCV on {base_family} ...")
        base_estimator_map = {
            'logistic_regression': LogisticRegression(
                random_state=RANDOM_STATE, max_iter=1000
            ),
            'random_forest': RandomForestClassifier(random_state=RANDOM_STATE),
            'gradient_boosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
        }
        grid_search = GridSearchCV(
            base_estimator_map[base_family],
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            refit=True
        )
        grid_search.fit(X_train_f, y_train)
        tuned_model = grid_search.best_estimator_
        logger.info(f"  Best params: {grid_search.best_params_}")

        tuned_metrics = evaluate_model(tuned_model, X_test_f, y_test)
        tuned_metrics['model'] = f'{base_family}_tuned'
        tuned_metrics['stage'] = 'tuned'
        results.append(tuned_metrics)
        trained_models[f'{base_family}_tuned'] = tuned_model
        logger.info(
            f"  Tuned: ROC-AUC={tuned_metrics['roc_auc']:.4f}  "
            f"Recall={tuned_metrics['recall']:.4f}"
        )

    # --- Stage 4: Select overall best and save ---
    results_df = pd.DataFrame(results)
    best_name = results_df.loc[results_df['roc_auc'].idxmax(), 'model']
    best_model = trained_models[best_name]

    recall_value = results_df.loc[results_df['model'] == best_name, 'recall'].values[0]
    logger.info(
        f"Overall best: {best_name}  "
        f"ROC-AUC={results_df['roc_auc'].max():.4f}  "
        f"Recall={recall_value:.4f}"
    )

    save_pickle(best_model, BEST_MODEL_FILE)
    logger.info(f"Best model saved to {BEST_MODEL_FILE}")

    results_df.to_csv(RESULTS_FILE, index=False)
    logger.info(f"Full comparison saved to {RESULTS_FILE}")

    logger.info("Model training completed successfully")
    return best_model, results_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_models()
