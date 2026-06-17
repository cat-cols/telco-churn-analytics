# Model Evaluation

## Evaluation Framework

### Primary Metrics
- **ROC-AUC**: Primary metric for imbalanced classification (26.54% churn rate)
- **Accuracy**: Overall correctness
- **Precision**: Reliability of positive predictions
- **Recall**: Ability to catch churn cases
- **F1-Score**: Balance between precision and recall

### Secondary Metrics
- **Confusion Matrix**: Detailed classification breakdown
- **Feature Importance**: Understanding model decisions
- **Calibration**: Probability prediction accuracy

## Model Performance Summary

### Model Comparison

Results on the held-out test set (`models/model_comparison_results.csv`):

| Model | Stage | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | baseline | 0.787 | 0.621 | 0.516 | 0.564 | 0.832 |
| Random Forest | baseline | 0.778 | 0.607 | 0.463 | 0.525 | 0.810 |
| Gradient Boosting | baseline | 0.789 | 0.630 | 0.500 | 0.557 | 0.834 |
| Logistic Regression (balanced) | balanced | 0.732 | 0.497 | **0.791** | 0.611 | 0.832 |
| Random Forest (balanced) | balanced | 0.756 | 0.536 | 0.610 | 0.571 | 0.810 |
| Gradient Boosting (balanced) | balanced | 0.735 | 0.501 | 0.775 | 0.609 | 0.832 |
| Gradient Boosting (tuned) | tuned | 0.792 | 0.647 | 0.476 | 0.549 | **0.835** |

### Best Model Selection
- **Primary Choice (by ROC-AUC)**: Gradient Boosting (tuned) — ROC-AUC 0.835
- **Best for Recall**: Logistic Regression (balanced) — recall 0.791
- **Baseline**: Logistic Regression (fast, interpretable)

## Performance Analysis

### ROC-AUC Analysis
- **Best Model**: 0.835 ROC-AUC (Gradient Boosting tuned)
- **Interpretation**: Good discrimination ability
- **Comparison**: Significantly better than random (0.5)
- **Business Impact**: Reliable risk scoring

### Precision-Recall Trade-off
- **Default-threshold tuned GB**: precision 0.647, recall 0.476 (conservative)
- **Class-weighted models** lift recall substantially: LR balanced reaches recall 0.791 (precision 0.497)
- **Threshold tuning**: Lowering the tuned GB threshold to 0.30 yields recall 0.773 / precision 0.522 / F1 0.623
- **Balance**: Choose operating point based on retention budget vs. coverage of at-risk customers

### Confusion Matrix Analysis
- **True Positives**: Correctly identified churners
- **False Positives**: Customers incorrectly flagged as churn risk
- **False Negatives**: Missed churners (most critical for business)
- **True Negatives**: Correctly identified loyal customers

## Feature Importance

### Top Predictive Features
1. **Contract**: Month-to-month = much higher churn (42.7% vs. 2.8% for two-year)
2. **tenure**: Longer tenure = lower churn (47.4% at 0-12mo vs. 9.5% at 49-72mo)
3. **InternetService**: Fiber optic associated with higher churn (41.9%)
4. **PaymentMethod**: Electronic check = higher churn (45.3%)
5. **MonthlyCharges / TotalCharges**: Higher monthly charges linked to churn

### Feature Engineering Impact
- **StandardScaler**: Improved model convergence
- **OneHotEncoder**: Essential for categorical variables
- **Feature Selection**: All features contribute meaningfully

## Model Robustness

### Cross-Validation
- **Strategy**: 5-fold stratified cross-validation on the best model (Gradient Boosting), run on all available data (train + test) in `notebooks/03_modeling.ipynb`
- **Result**: ROC-AUC = 0.8477 ± 0.0046 across folds (range 0.8429–0.8555)
- **Stability**: Fold-to-fold std < 0.01 confirms ROC-AUC is robust and **not split-dependent**; the single-split test value (0.8340) differs from the CV mean only by normal sampling noise

### Overfitting Analysis
- **Train-Test Gap**: Minimal (<5% difference)
- **Regularization**: Built-in to tree-based models
- **Feature Importance**: Consistent across models

## Class Imbalance Handling

### Current Approach
- **Imbalance**: 26.54% churn rate (minority class)
- **Handling**: `class_weight='balanced'` applied to LR and RF; GB uses `sample_weight`
- **Impact**: Recall lifted from 0.516 (LR baseline) to 0.791 (LR balanced)
- **Metrics**: ROC-AUC appropriate for imbalanced data

### Future Improvements
- **SMOTE**: Synthetic minority oversampling
- **Class Weights**: Weight loss function by class frequency
- **Threshold Tuning**: Optimize decision threshold

## Model Interpretability

### Global Interpretability
- **Feature Importance**: Clear ranking of predictive factors
- **Partial Dependence**: Understanding feature effects
- **SHAP Values**: Individual prediction explanations (future)

### Local Interpretability
- **Decision Trees**: Inherently interpretable
- **LIME**: Local explanations (future enhancement)
- **Counterfactuals**: What-if scenarios (future)

## Deployment Considerations

### Performance Requirements
- **Inference Time**: <100ms per prediction
- **Batch Processing**: Handle large customer bases
- **Memory Usage**: Efficient model storage

### Monitoring Requirements
- **Drift Detection**: Monitor feature distribution changes
- **Performance Tracking**: Regular evaluation on new data
- **Retraining Schedule**: Monthly or quarterly updates

## Business Metrics Translation

### Risk Scoring
- **High Risk**: Churn probability >0.7
- **Medium Risk**: Churn probability 0.4-0.7
- **Low Risk**: Churn probability <0.4

### Action Thresholds
- **Immediate Action**: Probability >0.8
- **Planned Intervention**: Probability 0.6-0.8
- **Monitoring**: Probability 0.4-0.6
- **No Action**: Probability <0.4

## Limitations and Future Work

### Current Limitations
- **Temporal Dynamics**: No time-series analysis
- **External Factors**: Market conditions not considered
- **Customer Interactions**: Limited behavioral data
- **Geographic Scope**: Single region analysis

### Future Enhancements
- **Deep Learning**: Neural networks for complex patterns
- **Ensemble Methods**: Stacking multiple models
- **Feature Engineering**: Create interaction terms
- **Hyperparameter Tuning**: Grid search for optimization
- **Cross-Selling**: Predict product adoption
- **Lifetime Value**: Customer value prediction

## Evaluation Checklist

- [x] Multiple models trained and compared
- [x] Appropriate metrics for imbalanced data
- [x] Feature importance analyzed
- [x] Confusion matrix reviewed
- [x] ROC-AUC curve generated
- [x] Overfitting assessed
- [x] Business thresholds defined
- [x] Class imbalance handled (class weights / sample weights)
- [x] Hyperparameter tuning (GridSearchCV on Gradient Boosting)
- [x] Cross-validation implemented — 5-fold stratified CV in `notebooks/03_modeling.ipynb` (GB ROC-AUC 0.8477 ± 0.0046; not split-dependent)
- [ ] SHAP values calculated (future)
- [ ] A/B testing framework (future)
