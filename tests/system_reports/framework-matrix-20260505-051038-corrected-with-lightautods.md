# Cross-Framework Matrix Comparison (Augmented With LightAutoDS And AutoKaggle)

- Generated: 2026-05-15 08:30:57 UTC
- Refreshed from baseline: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/framework-matrix-20260505-051038-corrected-with-lightautods.json`
- Main, AIDE, and LightAutoDS rows are reused from the prior report; only AutoKaggle was newly executed.

## Snapshot

- Comparable tasks: `8/8`
- Win count: `main=2`, `aide=0`, `lightautods_lama=3`, `autokaggle=3`
- Total runtime: `main=1127.4196s`, `aide=1106.6661s`, `lightautods_lama=443.8051s`, `autokaggle=1274.7969s`
- AutoKaggle route: original multi-agent workflow with a held-out validation split exposed as the temporary competition test set, then externally rescored on the main-project benchmark metric.
- AutoKaggle provider: `MiniMax 2.7` through the repo's OpenAI-compatible API path, with compatibility patches limited to the isolated `_external/autokaggle_eval` copy.

## Summary

| Dataset | Primary Metric | Main | AIDE | LightAutoDS | AutoKaggle | Better | Advantage | Main Time (s) | AIDE Time (s) | LightAutoDS Time (s) | AutoKaggle Time (s) |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `spaceship_titanic` | `accuracy` | 0.8050 | 0.7952 | 0.7964 | 0.7792 | `main` | 0.0086 | 102.7315 | 115.8226 | 10.8035 | 27.4039 |
| `academic_success` | `f1_macro` | 0.7925 | 0.7921 | 0.7929 | 0.7909 | `lightautods_lama` | 0.0004 | 223.2278 | 147.5690 | 109.4121 | 266.0120 |
| `house_prices` | `neg_root_mean_squared_error` | -30976.5130 | -32195.9950 | -30231.3257 | -27967.9664 | `autokaggle` | 2263.3593 | 41.3362 | 102.4763 | 22.9973 | 46.8023 |
| `plate_defect` | `roc_auc_macro_ovr` | 0.8748 | 0.8597 | 0.8855 | 0.8748 | `lightautods_lama` | 0.0107 | 223.7770 | 167.8008 | 143.9201 | 250.9362 |
| `titanic_autokaggle` | `roc_auc` | 0.8794 | 0.8660 | 0.7037 | 0.8522 | `main` | 0.0134 | 24.0602 | 114.8955 | 17.2567 | 31.0128 |
| `bank_churn` | `roc_auc` | 0.8890 | 0.8893 | 0.8771 | 0.8907 | `autokaggle` | 0.0014 | 177.9309 | 257.2572 | 47.1595 | 454.0694 |
| `ghouls` | `f1_macro` | 0.7639 | 0.7045 | 0.8366 | 0.7857 | `lightautods_lama` | 0.0510 | 140.8697 | 111.6102 | 17.7597 | 39.6847 |
| `obesity_risks` | `f1_macro` | 0.8914 | 0.8878 | 0.8862 | 0.8932 | `autokaggle` | 0.0018 | 193.4863 | 89.2345 | 74.4962 | 158.8756 |

## 1. `spaceship_titanic`

- Description: `single-target binary classification`
- Primary metric: `accuracy`
- Better framework: `main`
- Metric advantage: `0.0086`

### Main
- Status: `PASS`
- Metric: `0.8050`
- Runtime (s): `102.7315`
- Technical report: `/Users/a99/Downloads/5703-CS33-2/artifacts/runs/spaceship_titanic_20260506_025223_46389b3b/reports/technical_report.md`
- Prediction file: `artifacts/runs/spaceship_titanic_20260506_025223_46389b3b/predictions/submission.csv`
- Notes: `Hybrid planner main refresh used deterministic data_cleaning + indepth_eda and retained planner-guided feature engineering/model building. HistGradientBoosting improved cv_accuracy_mean to 0.8050 while reducing total runtime to 102.7315s.`

### AIDE
- Status: `PASS`
- Metric: `0.7952`
- Runtime (s): `115.8226`
- Prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/aide_eval/aideml/workspaces/2-quaint-tacky-quoll/working/submission.csv`

### LightAutoDS
- Status: `PASS`
- Metric: `0.7964`
- Runtime (s): `10.8035`
- Route: `LAMA`
- Internal config: `{"task_type": "binary", "target": "Transported", "task_metric": "accuracy"}`
- Validation predictions: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/tmp/lads-lama-spaceship_titanic-20260505-102152/models/validation_predictions.csv`
- Prediction file: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/tmp/lads-lama-spaceship_titanic-20260505-102152/models/submission.csv`

### AutoKaggle
- Status: `PASS`
- Metric: `0.7792`
- Runtime (s): `27.4039`
- Route: `multi_agent_validation_harness`
- Model: `MiniMax-M2.7`
- Valid output: `True`
- Prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/autokaggle_eval/AutoKaggle/multi_agents/competition/autokaggle-spaceship_titanic-20260513-094339/submission.csv`
- Log file: `/Users/a99/Downloads/5703-CS33-2/_external/autokaggle_eval/AutoKaggle/multi_agents/competition/autokaggle-spaceship_titanic-20260513-094339/autokaggle-spaceship_titanic-20260513-094339.log`
- Notes: `king.
Developer updates internal memory in Phase: Feature Engineering.
2026-05-13 17:43:48,322 - agent_base - INFO - Agent reviewer is created with model MiniMax-M2.7.
2026-05-13 17:43:48,322 - agent_base - INFO - State Feature Engineering `

## 2. `academic_success`

- Description: `single-target multiclass classification`
- Primary metric: `f1_macro`
- Better framework: `lightautods_lama`
- Metric advantage: `0.0004`

### Main
- Status: `PASS`
- Metric: `0.7925`
- Runtime (s): `223.2278`
- Technical report: `/Users/a99/Downloads/5703-CS33-2/artifacts/runs/classification_with_an_academic_success_dataset_20260505_224727_b1bc04b0/reports/technical_report.md`
- Prediction file: `artifacts/runs/classification_with_an_academic_success_dataset_20260505_224727_b1bc04b0/predictions/submission.csv`
- Notes: `
- `cleaned_test_data`: `artifacts/runs/classification_with_an_academic_success_dataset_20260505_224727_b1bc04b0/data/cleaned_test.csv`
- `cleaned_train_data`: `artifacts/runs/classification_with_an_academic_success_dataset_20260505_224727_`

### AIDE
- Status: `PASS`
- Metric: `0.7921`
- Runtime (s): `147.5690`
- Prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/aide_eval/aideml/workspaces/2-intrepid-arrogant-chachalaca/working/submission.csv`

### LightAutoDS
- Status: `PASS`
- Metric: `0.7929`
- Runtime (s): `109.4121`
- Route: `LAMA`
- Internal config: `{"task_type": "multiclass", "target": "Target", "task_metric": "f1_macro"}`
- Validation predictions: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/tmp/lads-lama-academic_success-20260505-102204/models/validation_predictions.csv`
- Prediction file: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/tmp/lads-lama-academic_success-20260505-102204/models/submission.csv`

### AutoKaggle
- Status: `PASS`
- Metric: `0.7909`
- Runtime (s): `266.0120`
- Route: `multi_agent_validation_harness`
- Model: `MiniMax-M2.7`
- Valid output: `True`
- Prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/autokaggle_eval/AutoKaggle/multi_agents/competition/autokaggle-academic_success-20260515-082310/submission.csv`
- Log file: `/Users/a99/Downloads/5703-CS33-2/_external/autokaggle_eval/AutoKaggle/multi_agents/competition/autokaggle-academic_success-20260515-082310/autokaggle-academic_success-20260515-082310.log`
- Notes: `working.
Developer updates internal memory in Phase: Feature Engineering.
2026-05-15 16:23:24,663 - agent_base - INFO - Agent reviewer is created with model MiniMax-M2.7.
2026-05-15 16:23:24,663 - agent_base - INFO - State Feature Engineeri`

## 3. `house_prices`

- Description: `single-target regression`
- Primary metric: `neg_root_mean_squared_error`
- Better framework: `autokaggle`
- Metric advantage: `2263.3593`

### Main
- Status: `PASS`
- Metric: `-30976.5130`
- Runtime (s): `41.3362`
- Technical report: `/Users/a99/Downloads/5703-CS33-2/artifacts/runs/house_prices_20260506_022052_6f9fbf69/reports/technical_report.md`
- Prediction file: `artifacts/runs/house_prices_20260506_022052_6f9fbf69/predictions/submission.csv`
- Notes: `Deterministic no-planner refresh preserved the benchmark regression metric and selected GradientBoosting with cv_neg_root_mean_squared_error_mean=-30976.5130. The refreshed run completed in 41.3362s.`

### AIDE
- Status: `PASS`
- Metric: `-32195.9950`
- Runtime (s): `102.4763`
- Prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/aide_eval/aideml/workspaces/2-rich-divergent-puffin/working/submission.csv`

### LightAutoDS
- Status: `PASS`
- Metric: `-30231.3257`
- Runtime (s): `22.9973`
- Route: `LAMA`
- Internal config: `{"task_type": "reg", "target": "SalePrice", "task_metric": "mse"}`
- Validation predictions: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/tmp/lads-lama-house_prices-20260505-102354/models/validation_predictions.csv`
- Prediction file: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/tmp/lads-lama-house_prices-20260505-102354/models/submission.csv`

### AutoKaggle
- Status: `PASS`
- Metric: `-27967.9664`
- Runtime (s): `46.8023`
- Route: `multi_agent_validation_harness`
- Model: `MiniMax-M2.7`
- Valid output: `True`
- Prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/autokaggle_eval/AutoKaggle/multi_agents/competition/autokaggle-house_prices-20260513-094418/submission.csv`
- Log file: `/Users/a99/Downloads/5703-CS33-2/_external/autokaggle_eval/AutoKaggle/multi_agents/competition/autokaggle-house_prices-20260513-094418/autokaggle-house_prices-20260513-094418.log`
- Notes: `er finishes working.
Developer updates internal memory in Phase: Feature Engineering.
2026-05-13 17:44:29,543 - agent_base - INFO - Agent reviewer is created with model MiniMax-M2.7.
2026-05-13 17:44:29,543 - agent_base - INFO - State Featu`

## 4. `plate_defect`

- Description: `multi-target multilabel classification`
- Primary metric: `roc_auc_macro_ovr`
- Better framework: `lightautods_lama`
- Metric advantage: `0.0107`

### Main
- Status: `PASS`
- Metric: `0.8748`
- Runtime (s): `223.7770`
- Technical report: `/Users/a99/Downloads/5703-CS33-2/artifacts/runs/plate_defect_20260505_153427_3d1cf377/reports/technical_report.md`
- Prediction file: `artifacts/runs/plate_defect_20260505_153427_3d1cf377/predictions/submission.csv`
- Notes: `260505_153427_3d1cf377/data/cleaning_report.md`
- `engineered_test_data`: `artifacts/runs/plate_defect_20260505_153427_3d1cf377/features/engineered_test.csv`
- `engineered_train_data`: `artifacts/runs/plate_defect_20260505_153427_3d1cf377/f`

### AIDE
- Status: `PASS`
- Metric: `0.8597`
- Runtime (s): `167.8008`
- Prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/aide_eval/aideml/workspaces/2-sympathetic-elfish-serpent/working/submission.csv`

### LightAutoDS
- Status: `PASS`
- Metric: `0.8855`
- Runtime (s): `143.9201`
- Route: `LAMA`
- Internal config: `{"task_type": "multilabel", "target": "Pastry,Z_Scratch,K_Scatch,Stains,Dirtiness,Bumps,Other_Faults", "task_metric": "logloss"}`
- Validation predictions: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/tmp/lads-lama-plate_defect-20260505-102418/models/validation_predictions.csv`
- Prediction file: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/tmp/lads-lama-plate_defect-20260505-102418/models/submission.csv`

### AutoKaggle
- Status: `PASS`
- Metric: `0.8748`
- Runtime (s): `250.9362`
- Route: `multi_agent_validation_harness`
- Model: `MiniMax-M2.7`
- Valid output: `True`
- Prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/autokaggle_eval/AutoKaggle/multi_agents/competition/autokaggle-plate_defect-20260513-094505/submission.csv`
- Log file: `/Users/a99/Downloads/5703-CS33-2/_external/autokaggle_eval/AutoKaggle/multi_agents/competition/autokaggle-plate_defect-20260513-094505/autokaggle-plate_defect-20260513-094505.log`
- Notes: `ry in Phase: Model Building, Validation, and Prediction.
2026-05-13 17:46:43,162 - root - INFO - Current State: State: Model Building, Validation, and Prediction, Current Step: 0, Current Agent: Planner, Finished: False
2026-05-13 17:46:43,`

## 5. `titanic_autokaggle`

- Description: `single-target binary classification smoke dataset`
- Primary metric: `roc_auc`
- Better framework: `main`
- Metric advantage: `0.0134`

### Main
- Status: `PASS`
- Metric: `0.8794`
- Runtime (s): `24.0602`
- Technical report: `/Users/a99/Downloads/5703-CS33-2/artifacts/runs/titanic_autokaggle_20260506_022609_b86782ec/reports/technical_report.md`
- Prediction file: `artifacts/runs/titanic_autokaggle_20260506_022609_b86782ec/predictions/submission.csv`
- Notes: `Deterministic no-planner refresh selected RandomForest and improved cv_roc_auc_mean to 0.8794 while cutting runtime to 24.0602s. Submission artifacts were generated successfully.`

### AIDE
- Status: `PASS`
- Metric: `0.8660`
- Runtime (s): `114.8955`
- Prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/aide_eval/aideml/workspaces/2-phenomenal-quick-ladybug/working/submission.csv`

### LightAutoDS
- Status: `PASS`
- Metric: `0.7037`
- Runtime (s): `17.2567`
- Route: `LAMA`
- Internal config: `{"task_type": "binary", "target": "Survived", "task_metric": "auc"}`
- Validation predictions: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/tmp/lads-lama-titanic_autokaggle-20260505-102642/models/validation_predictions.csv`
- Prediction file: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/tmp/lads-lama-titanic_autokaggle-20260505-102642/models/submission.csv`

### AutoKaggle
- Status: `PASS`
- Metric: `0.8522`
- Runtime (s): `31.0128`
- Route: `multi_agent_validation_harness`
- Model: `MiniMax-M2.7`
- Valid output: `True`
- Prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/autokaggle_eval/AutoKaggle/multi_agents/competition/autokaggle-titanic_autokaggle-20260513-094916/submission.csv`
- Log file: `/Users/a99/Downloads/5703-CS33-2/_external/autokaggle_eval/AutoKaggle/multi_agents/competition/autokaggle-titanic_autokaggle-20260513-094916/autokaggle-titanic_autokaggle-20260513-094916.log`
- Notes: `g.
Developer updates internal memory in Phase: Feature Engineering.
2026-05-13 17:49:30,632 - agent_base - INFO - Agent reviewer is created with model MiniMax-M2.7.
2026-05-13 17:49:30,632 - agent_base - INFO - State Feature Engineering - A`

## 6. `bank_churn`

- Description: `single-target binary classification smoke dataset`
- Primary metric: `roc_auc`
- Better framework: `autokaggle`
- Metric advantage: `0.0014`

### Main
- Status: `PASS`
- Metric: `0.8890`
- Runtime (s): `177.9309`
- Technical report: `/Users/a99/Downloads/5703-CS33-2/artifacts/runs/bank_churn_20260506_022748_4a824b11/reports/technical_report.md`
- Prediction file: `artifacts/runs/bank_churn_20260506_022748_4a824b11/predictions/submission.csv`
- Notes: `Deterministic no-planner refresh matched the prior main cv_roc_auc_mean at 0.8890 with HistGradientBoosting and reduced runtime to 177.9309s.`

### AIDE
- Status: `PASS`
- Metric: `0.8893`
- Runtime (s): `257.2572`
- Prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/aide_eval/aideml/workspaces/2-flamingo-of-eternal-intensity/working/submission.csv`

### LightAutoDS
- Status: `PASS`
- Metric: `0.8771`
- Runtime (s): `47.1595`
- Route: `LAMA`
- Internal config: `{"task_type": "binary", "target": "Exited", "task_metric": "auc"}`
- Validation predictions: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/tmp/lads-lama-bank_churn-20260505-102700/models/validation_predictions.csv`
- Prediction file: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/tmp/lads-lama-bank_churn-20260505-102700/models/submission.csv`

### AutoKaggle
- Status: `PASS`
- Metric: `0.8907`
- Runtime (s): `454.0694`
- Route: `multi_agent_validation_harness`
- Model: `MiniMax-M2.7`
- Valid output: `True`
- Prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/autokaggle_eval/AutoKaggle/multi_agents/competition/autokaggle-bank_churn-20260513-094948/submission.csv`
- Log file: `/Users/a99/Downloads/5703-CS33-2/_external/autokaggle_eval/AutoKaggle/multi_agents/competition/autokaggle-bank_churn-20260513-094948/autokaggle-bank_churn-20260513-094948.log`
- Notes: `eveloper finishes working.
Developer updates internal memory in Phase: Feature Engineering.
2026-05-13 17:50:09,282 - agent_base - INFO - Agent reviewer is created with model MiniMax-M2.7.
2026-05-13 17:50:09,282 - agent_base - INFO - State`

## 7. `ghouls`

- Description: `single-target multiclass classification smoke dataset`
- Primary metric: `f1_macro`
- Better framework: `lightautods_lama`
- Metric advantage: `0.0510`

### Main
- Status: `PASS`
- Metric: `0.7639`
- Runtime (s): `140.8697`
- Technical report: `/Users/a99/Downloads/5703-CS33-2/artifacts/runs/ghouls_goblins_and_ghosts_boo_20260506_013632_8bc7cada/reports/technical_report.md`
- Prediction file: `artifacts/runs/ghouls_goblins_and_ghosts_boo_20260506_013632_8bc7cada/predictions/submission.csv`
- Notes: `The expanded multiclass candidate pool evaluated RandomForest, HistGradientBoosting, LogisticRegression, and SVC. LogisticRegression remained the best model and improved the primary validation metric to cv_f1_macro_mean=0.7639. Artifacts we`

### AIDE
- Status: `PASS`
- Metric: `0.7045`
- Runtime (s): `111.6102`
- Prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/aide_eval/aideml/workspaces/2-inscrutable-topaz-rottweiler/working/submission.csv`

### LightAutoDS
- Status: `PASS`
- Metric: `0.8366`
- Runtime (s): `17.7597`
- Route: `LAMA`
- Internal config: `{"task_type": "multiclass", "target": "type", "task_metric": "f1_macro"}`
- Validation predictions: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/tmp/lads-lama-ghouls-20260505-102748/models/validation_predictions.csv`
- Prediction file: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/tmp/lads-lama-ghouls-20260505-102748/models/submission.csv`

### AutoKaggle
- Status: `PASS`
- Metric: `0.7857`
- Runtime (s): `39.6847`
- Route: `multi_agent_validation_harness`
- Model: `MiniMax-M2.7`
- Valid output: `True`
- Prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/autokaggle_eval/AutoKaggle/multi_agents/competition/autokaggle-ghouls-20260515-082737/submission.csv`
- Log file: `/Users/a99/Downloads/5703-CS33-2/_external/autokaggle_eval/AutoKaggle/multi_agents/competition/autokaggle-ghouls-20260515-082737/autokaggle-ghouls-20260515-082737.log`
- Notes: `ng - Agent developer finishes working.
Developer updates internal memory in Phase: Feature Engineering.
2026-05-15 16:27:53,636 - agent_base - INFO - Agent reviewer is created with model MiniMax-M2.7.
2026-05-15 16:27:53,636 - agent_base - `

## 8. `obesity_risks`

- Description: `single-target multiclass classification smoke dataset`
- Primary metric: `f1_macro`
- Better framework: `autokaggle`
- Metric advantage: `0.0018`

### Main
- Status: `PASS`
- Metric: `0.8914`
- Runtime (s): `193.4863`
- Technical report: `/Users/a99/Downloads/5703-CS33-2/artifacts/runs/obesity_risks_20260505_232529_6e9cceaf/reports/technical_report.md`
- Prediction file: `artifacts/runs/obesity_risks_20260505_232529_6e9cceaf/predictions/submission.csv`
- Notes: `Obeyesdad`
- Sample submission reference: `/Users/a99/Downloads/5703-CS33-2/data/obesity_risks/sample_submission.csv`
- Prediction semantics: `label`
- Serialized model artifact: `artifacts/runs/obesity_risks_20260505_232529_6e9cceaf/models`

### AIDE
- Status: `PASS`
- Metric: `0.8878`
- Runtime (s): `89.2345`
- Prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/aide_eval/aideml/workspaces/2-inventive-sparrow-of-competition/working/submission.csv`

### LightAutoDS
- Status: `PASS`
- Metric: `0.8862`
- Runtime (s): `74.4962`
- Route: `LAMA`
- Internal config: `{"task_type": "multiclass", "target": "NObeyesdad", "task_metric": "f1_macro"}`
- Validation predictions: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/tmp/lads-lama-obesity_risks-20260505-102806/models/validation_predictions.csv`
- Prediction file: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/tmp/lads-lama-obesity_risks-20260505-102806/models/submission.csv`

### AutoKaggle
- Status: `PASS`
- Metric: `0.8932`
- Runtime (s): `158.8756`
- Route: `multi_agent_validation_harness`
- Model: `MiniMax-M2.7`
- Valid output: `True`
- Prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/autokaggle_eval/AutoKaggle/multi_agents/competition/autokaggle-obesity_risks-20260515-082818/submission.csv`
- Log file: `/Users/a99/Downloads/5703-CS33-2/_external/autokaggle_eval/AutoKaggle/multi_agents/competition/autokaggle-obesity_risks-20260515-082818/autokaggle-obesity_risks-20260515-082818.log`
- Notes: `finishes working.
Developer updates internal memory in Phase: Feature Engineering.
2026-05-15 16:28:35,433 - agent_base - INFO - Agent reviewer is created with model MiniMax-M2.7.
2026-05-15 16:28:35,434 - agent_base - INFO - State Feature `
