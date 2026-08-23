# Cross-Framework Matrix Comparison

- Generated: 2026-05-05 06:15:52 UTC
- Commit: `3b981a9`
- Dataset count: `8`
- Main timeout: `2400s`
- AIDE timeout: `1800s`
- AIDE steps: `2`

## Snapshot

- Comparable tasks so far: `8/8`
- Win count: `main=6`, `aide=2`, `tie=0`
- Total runtime so far: `main=1907.2035s`, `aide=1106.6661s`
- Stability signals so far: `main_feature_retries=0`, `aide_buggy_nodes=4`

## Summary

| Dataset | Primary Metric | Main | AIDE | Better | Advantage | Main Time (s) | AIDE Time (s) | Main FE Retries | AIDE Buggy Nodes |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `spaceship_titanic` | `accuracy` | 0.7964 | 0.7952 | `main` | 0.0012 | 159.5029 | 115.8226 | 0 | 0 |
| `academic_success` | `f1_macro` | 0.7881 | 0.7921 | `aide` | 0.0040 | 507.8703 | 147.5690 | 0 | 0 |
| `house_prices` | `neg_root_mean_squared_error` | -31606.2974 | -32195.9950 | `main` | 589.6976 | 177.4710 | 102.4763 | 0 | 1 |
| `plate_defect` | `roc_auc_macro_ovr` | 0.8748 | 0.8597 | `main` | 0.0151 | 223.7770 | 167.8008 | 0 | 1 |
| `titanic_autokaggle` | `roc_auc` | 0.8745 | 0.8660 | `main` | 0.0085 | 110.9870 | 114.8955 | 0 | 0 |
| `bank_churn` | `roc_auc` | 0.8881 | 0.8893 | `aide` | 0.0012 | 410.4611 | 257.2572 | 0 | 1 |
| `ghouls` | `f1_macro` | 0.7548 | 0.7045 | `main` | 0.0503 | 104.5780 | 111.6102 | 0 | 0 |
| `obesity_risks` | `f1_macro` | 0.8884 | 0.8878 | `main` | 0.0006 | 212.5562 | 89.2345 | 0 | 1 |

## 1. `spaceship_titanic`

- Description: `single-target binary classification`
- Primary metric: `accuracy`
- Better framework: `main`
- Metric advantage: `0.0012`

### Main
- Status: `PASS`
- Metric: `0.7964`
- Runtime (s): `159.5029`
- Feature-engineering retries: `0`
- Technical report: `/Users/a99/Downloads/5703-CS33-2/artifacts/runs/spaceship_titanic_20260505_151040_b0b3176c/reports/technical_report.md`
- Prediction file: `artifacts/runs/spaceship_titanic_20260505_151040_b0b3176c/predictions/submission.csv`

### AIDE
- Status: `PASS`
- Metric: `0.7952`
- Runtime (s): `115.8226`
- Buggy nodes: `0` / total nodes `2`
- Best solution LOC: `65`
- Prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/aide_eval/aideml/workspaces/2-quaint-tacky-quoll/working/submission.csv`
- AIDE best-node analysis: The code executed successfully without errors. A RandomForestClassifier with 300 estimators and max_depth=15 was trained and evaluated using 5-fold cross-validation, achieving a mean validation accuracy of 79.52%. The model processed the data in 2 seconds and generated a submission file. This represents a reasonable baseline performance for the Space Titanic classification task.

## 2. `academic_success`

- Description: `single-target multiclass classification`
- Primary metric: `f1_macro`
- Better framework: `aide`
- Metric advantage: `0.0040`

### Main
- Status: `PASS`
- Metric: `0.7881`
- Runtime (s): `507.8703`
- Feature-engineering retries: `0`
- Technical report: `/Users/a99/Downloads/5703-CS33-2/artifacts/runs/classification_with_an_academic_success_dataset_20260505_151630_c8277cc2/reports/technical_report.md`
- Prediction file: `artifacts/runs/classification_with_an_academic_success_dataset_20260505_151630_c8277cc2/predictions/submission.csv`

### AIDE
- Status: `PASS`
- Metric: `0.7921`
- Runtime (s): `147.5690`
- Buggy nodes: `0` / total nodes `2`
- Best solution LOC: `67`
- Prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/aide_eval/aideml/workspaces/2-intrepid-arrogant-chachalaca/working/submission.csv`
- AIDE best-node analysis: The Random Forest model achieved a stable 5-fold CV F1 macro score of 0.792056 (+/- 0.002491). The low standard deviation indicates consistent performance across all folds. The model used 300 estimators with max_depth=20 and class_weight="balanced" to handle any class imbalance. Submission file was successfully created.

## 3. `house_prices`

- Description: `single-target regression`
- Primary metric: `neg_root_mean_squared_error`
- Better framework: `main`
- Metric advantage: `589.6976`

### Main
- Status: `PASS`
- Metric: `-31606.2974`
- Runtime (s): `177.4710`
- Feature-engineering retries: `0`
- Technical report: `/Users/a99/Downloads/5703-CS33-2/artifacts/runs/house_prices_20260505_152833_82bd7d09/reports/technical_report.md`
- Prediction file: `artifacts/runs/house_prices_20260505_152833_82bd7d09/predictions/submission.csv`

### AIDE
- Status: `PASS`
- Metric: `-32195.9950`
- Runtime (s): `102.4763`
- Buggy nodes: `1` / total nodes `2`
- Best solution LOC: `56`
- Prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/aide_eval/aideml/workspaces/2-rich-divergent-puffin/working/submission.csv`
- AIDE best-node analysis: Code executed successfully without bugs. The Ridge regression pipeline with preprocessing achieved a mean cross-validation negative RMSE of -32195.99 (RMSE ≈ $32,196), which is reasonable for house price prediction. The submission file was generated correctly.

## 4. `plate_defect`

- Description: `multi-target multilabel classification`
- Primary metric: `roc_auc_macro_ovr`
- Better framework: `main`
- Metric advantage: `0.0151`

### Main
- Status: `PASS`
- Metric: `0.8748`
- Runtime (s): `223.7770`
- Feature-engineering retries: `0`
- Technical report: `/Users/a99/Downloads/5703-CS33-2/artifacts/runs/plate_defect_20260505_153427_3d1cf377/reports/technical_report.md`
- Prediction file: `artifacts/runs/plate_defect_20260505_153427_3d1cf377/predictions/submission.csv`

### AIDE
- Status: `PASS`
- Metric: `0.8597`
- Runtime (s): `167.8008`
- Buggy nodes: `1` / total nodes `2`
- Best solution LOC: `79`
- Prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/aide_eval/aideml/workspaces/2-sympathetic-elfish-serpent/working/submission.csv`
- AIDE best-node analysis: The logistic regression baseline pipeline executed successfully with a Macro ROC AUC (OVR) of 0.859695. The 5-fold cross-validation properly computed out-of-fold predictions for all 7 target columns (Pastry, Z_Scratch, K_Scatch, Stains, Dirtiness, Bumps, Other_Faults), and final models trained on full data generated test predictions saved to submission.csv.

## 5. `titanic_autokaggle`

- Description: `single-target binary classification smoke dataset`
- Primary metric: `roc_auc`
- Better framework: `main`
- Metric advantage: `0.0085`

### Main
- Status: `PASS`
- Metric: `0.8745`
- Runtime (s): `110.9870`
- Feature-engineering retries: `0`
- Technical report: `/Users/a99/Downloads/5703-CS33-2/artifacts/runs/titanic_autokaggle_20260505_154202_d35ee0ab/reports/technical_report.md`
- Prediction file: `artifacts/runs/titanic_autokaggle_20260505_154202_d35ee0ab/predictions/submission.csv`

### AIDE
- Status: `PASS`
- Metric: `0.8660`
- Runtime (s): `114.8955`
- Buggy nodes: `0` / total nodes `2`
- Best solution LOC: `146`
- Prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/aide_eval/aideml/workspaces/2-phenomenal-quick-ladybug/working/submission.csv`
- AIDE best-node analysis: The model executed successfully with a 5-fold cross-validation ROC AUC of 0.866. The implementation includes comprehensive feature engineering (title extraction, family size, age/fare bins, feature interactions) and uses Logistic Regression with regularization (C=0.5). The submission file with predicted probabilities was saved successfully.

## 6. `bank_churn`

- Description: `single-target binary classification smoke dataset`
- Primary metric: `roc_auc`
- Better framework: `aide`
- Metric advantage: `0.0012`

### Main
- Status: `PASS`
- Metric: `0.8881`
- Runtime (s): `410.4611`
- Feature-engineering retries: `0`
- Technical report: `/Users/a99/Downloads/5703-CS33-2/artifacts/runs/bank_churn_20260505_154702_d4016d50/reports/technical_report.md`
- Prediction file: `artifacts/runs/bank_churn_20260505_154702_d4016d50/predictions/submission.csv`

### AIDE
- Status: `PASS`
- Metric: `0.8893`
- Runtime (s): `257.2572`
- Buggy nodes: `1` / total nodes `2`
- Best solution LOC: `49`
- Prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/aide_eval/aideml/workspaces/2-flamingo-of-eternal-intensity/working/submission.csv`
- AIDE best-node analysis: The GradientBoostingClassifier model achieved a mean ROC AUC of 0.889314 with a very stable std of 0.001237 across 5 folds. The model processed the data correctly, created valid probability predictions for all 110,023 test samples, and saved a properly formatted submission file. No bugs or issues detected.

## 7. `ghouls`

- Description: `single-target multiclass classification smoke dataset`
- Primary metric: `f1_macro`
- Better framework: `main`
- Metric advantage: `0.0503`

### Main
- Status: `PASS`
- Metric: `0.7548`
- Runtime (s): `104.5780`
- Feature-engineering retries: `0`
- Technical report: `/Users/a99/Downloads/5703-CS33-2/artifacts/runs/ghouls_goblins_and_ghosts_boo_20260505_155916_b163acf9/reports/technical_report.md`
- Prediction file: `artifacts/runs/ghouls_goblins_and_ghosts_boo_20260505_155916_b163acf9/predictions/submission.csv`

### AIDE
- Status: `PASS`
- Metric: `0.7045`
- Runtime (s): `111.6102`
- Buggy nodes: `0` / total nodes `2`
- Best solution LOC: `70`
- Prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/aide_eval/aideml/workspaces/2-inscrutable-topaz-rottweiler/working/submission.csv`
- AIDE best-node analysis: The Random Forest classifier with grid search hyperparameter tuning completed successfully. The best model uses 300 estimators with max_depth=None, min_samples_leaf=2, and min_samples_split=5. The model achieved a macro F1 score of 0.7045 on 5-fold cross-validation. The submission file was created successfully.

## 8. `obesity_risks`

- Description: `single-target multiclass classification smoke dataset`
- Primary metric: `f1_macro`
- Better framework: `main`
- Metric advantage: `0.0006`

### Main
- Status: `PASS`
- Metric: `0.8884`
- Runtime (s): `212.5562`
- Feature-engineering retries: `0`
- Technical report: `/Users/a99/Downloads/5703-CS33-2/artifacts/runs/obesity_risks_20260505_160915_1f4ec9b9/reports/technical_report.md`
- Prediction file: `artifacts/runs/obesity_risks_20260505_160915_1f4ec9b9/predictions/submission.csv`

### AIDE
- Status: `PASS`
- Metric: `0.8878`
- Runtime (s): `89.2345`
- Buggy nodes: `1` / total nodes `2`
- Best solution LOC: `104`
- Prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/aide_eval/aideml/workspaces/2-inventive-sparrow-of-competition/working/submission.csv`
- AIDE best-node analysis: The code executed successfully with no bugs. A Random Forest classifier with balanced class weights was trained using 5-fold stratified cross-validation. The model achieved a mean F1 Macro score of 0.8878 (std: 0.0051), indicating stable and consistent performance across all folds. The submission file was generated with 13,840 predictions distributed across 7 obesity risk classes.
