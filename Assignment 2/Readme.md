# Assignment 2: Data and Model Version Control

This assignment demonstrates an end-to-end reproducible ML workflow.

## Data Version Control (DVC)
- Raw data saved as `raw_data.csv`
- Train/Validation/Test splits created
- Two versions generated using different random seeds (42 and 123)
- Old and updated versions checked out using DVC
- Target distribution compared across versions

## Experiment Tracking (MLflow)
- Experiment: SMS_Spam_Models
- Three benchmark models:
  - Logistic Regression
  - Naive Bayes
  - Random Forest
- Metric used: AUCPR
- All runs logged with parameters and metrics
- Models registered in MLflow Model Registry

## Result
Random Forest achieved the highest AUCPR and was selected as the best model.

This workflow ensures reproducibility using DVC for data and MLflow for experiment tracking.
