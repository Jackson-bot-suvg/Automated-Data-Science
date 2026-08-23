# Technical Report for spaceship_titanic

## Executive Summary

This workflow trained a binary_classification solution for target `Transported`. The selected model is GradientBoosting, evaluated primarily with accuracy=0.7852.

## Problem Definition

- Task summary: Binary classification task predicting which passengers of the Spaceship Titanic were transported to an alternate dimension based on ship records. User task: predict whether transported
- Task family: `binary_classification`
- Target mode: `single_target`
- Target columns: `Transported`
- Prediction kind: `probability`
- Primary metric: `accuracy`
- Candidate metrics: `accuracy`, `roc_auc`, `f1`

## Dataset Snapshot

- Training shape: 8693 x 14
- Test shape: 4277 x 13
- Raw dataset overview: 8693 rows and 14 columns.
- Column type coverage: 14 columns profiled.
- Most visible missingness: `Cabin` (33), `ShoppingMall` (32), `HomePlanet` (27).
- Approximate raw feature count: 12.

## Data Preparation Summary

- drop_columns on `Name`, `Cabin`
- fill_missing on `VRDeck` with strategy `median`
- clip_outliers on `VRDeck`
- EDA highlights: dataset_shape.rows=8693; dataset_shape.columns=14; target_variable=Transported
- In-depth EDA highlights: target_analysis.Transported.variable=Transported; target_analysis.Transported.model_type=binary_classification; target_analysis.Transported.summary={'count': 8693, 'unique': 2, 'top': True, 'freq': 4378}

## Feature Engineering Summary

- Selected features: 13
- Dropped features: 0
- Feature set preview: `CryoSleep`, `Age`, `VIP`, `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck` ...
- create_interaction on `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck` creating `TotalSpending` via `+`
- encode_categorical on `Destination` using `onehot`
- Leakage assessment: (not available); (not available) ...

## Modeling Summary

- Candidate models: `LogisticRegression`, `RandomForest`, `GradientBoosting`
- Selected model: `GradientBoosting`
- Primary metric focus: `accuracy`
- Best-model validation metrics: cv_accuracy_mean=0.7913; cv_accuracy_std=0.0103; cv_roc_auc_mean=0.8753; cv_roc_auc_std=0.0029; cv_f1_mean=0.801; cv_f1_std=0.0111
- Baseline metrics: accuracy=0.8097; f1=0.8194; precision=0.7848; recall=0.8572; roc_auc=0.893 ...
- Best hyperparameters: learning_rate=0.1; max_depth=3; n_estimators=100

## Prediction Outputs

- Prediction file: `artifacts/predictions/submission.csv`
- Output columns: `Transported`
- Submission column order: `PassengerId`, `Transported`
- Sample submission reference: `/Users/a99/Downloads/5703-CS33-2/data/spaceship_titanic/sample_submission.csv`
- Prediction semantics: `probability`
- Serialized model artifact: `artifacts/models/best_model.pkl`
- Model results artifact: `artifacts/models/model_results.json`

## Business Translation Inputs

- Main target description: target `Transported`
- Delivery-ready metric: `accuracy` with selected model `GradientBoosting`
- Prediction output type: `probability `
- Caveats to carry into the business report: Feature importance is unavailable, so driver-level interpretation is limited.

## Artifacts

- `best_model`: `artifacts/models/best_model.pkl`
- `business_report`: `artifacts/reports/business_report.md`
- `cleaned_test_data`: `artifacts/data/cleaned_test.csv`
- `cleaned_train_data`: `artifacts/data/cleaned_train.csv`
- `cleaning_report`: `artifacts/data/cleaning_report.md`
- `feature_summary`: `artifacts/features/feature_summary.json`
- `indepth_eda_report`: `artifacts/eda/in_depth_eda.md`
- `model_results`: `artifacts/models/model_results.json`
- `prediction_file`: `artifacts/predictions/submission.csv`
- `preliminary_eda_report`: `artifacts/eda/preliminary_eda.md`
- `technical_report`: `artifacts/reports/technical_report.md`
- `figure_1`: `artifacts/figures/eda_missingness.png`
- `figure_2`: `artifacts/figures/target_analysis_Transported_spaceship_titanic-cli-run.png`
- `figure_3`: `artifacts/figures/correlation_matrix_spaceship_titanic-cli-run.png`
- `figure_4`: `artifacts/figures/outlier_analysis_spaceship_titanic-cli-run.png`

## Limitations

- Feature importance is unavailable, so driver-level interpretation is limited.
