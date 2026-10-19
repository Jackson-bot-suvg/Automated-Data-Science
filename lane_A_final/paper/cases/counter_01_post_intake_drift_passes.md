# counter_01_post_intake_drift_passes

- run_id: `A1__DEV-BIN-SPACESHIP__F05__F0_ORIGINAL__sb1101`
- verdict of the oracle: **S5** — fit target != gold
- why this case: unfavourable: target_spec rewritten after intake; every later gate validates against the rewritten contract; wrong target delivered (S5).
- C_gold targets: see private oracle; C_read targets: `Transported`; C_used fit targets: `CryoSleep`
- output provenance at serialization: `{"PassengerId": "test_id", "Transported": "prediction"}`
- raw trace: `logs/events/A1__DEV-BIN-SPACESHIP__F05__F0_ORIGINAL__sb1101.jsonl`

| seq | event_id | event | actor | phase | key fields |
|---|---|---|---|---|---|
| 4 | `8cd86b3a412d` | node_end | reader | reader | C_read targets=['Transported'] family=binary_classification out=['Transported'] ids=['PassengerId'] |
| 6 | `cf76ae4e51e5` | gate_check | background_reviewer | reader | computed= / f0_status=approved |
| 7 | `f23250edeb3c` | gate_verdict | background_reviewer | reader | verdict=approved -> planner |
| 11 | `8373761a2e8d` | node_end | planner | preliminary_eda | planned_phase=preliminary_eda candidate_hash=3fc8cf900465 |
| 15 | `a27cf34b246e` | gate_check | reviewer | preliminary_eda | computed= / f0_status=approved |
| 16 | `50fa1fb2888f` | gate_verdict | reviewer | preliminary_eda | verdict=approved -> planner(next_phase) |
| 20 | `257892de2f70` | node_end | planner | data_cleaning | planned_phase=data_cleaning candidate_hash=da0c645e2afb |
| 24 | `45df895a48b4` | gate_check | reviewer | data_cleaning | computed= / f0_status=approved |
| 25 | `7d29616cddb1` | gate_verdict | reviewer | data_cleaning | verdict=approved -> planner(next_phase) |
| 29 | `7e5e1c8ae6e2` | node_end | planner | indepth_eda | planned_phase=indepth_eda candidate_hash=7842728b6448 |
| 33 | `395c1a100683` | gate_check | reviewer | indepth_eda | computed= / f0_status=approved |
| 34 | `a41ea2cdd850` | gate_verdict | reviewer | indepth_eda | verdict=approved -> planner(next_phase) |
| 39 | `cd0e1866f2ab` | node_end | planner | feature_engineering | planned_phase=feature_engineering candidate_hash=52acbdebe523 |
| 43 | `66eb763363dc` | gate_check | reviewer | feature_engineering | computed= / f0_status=approved |
| 44 | `165879761c7c` | gate_verdict | reviewer | feature_engineering | verdict=approved -> planner(next_phase) |
| 49 | `474fe29e8199` | injection | lane_a_mutator | model_building | {"template_id": "F05", "before_target_columns": ["Transported"], "after_target_columns": ["CryoSleep"]} |
| 50 | `66cad7c17a86` | node_end | planner | model_building | planned_phase=model_building candidate_hash=b5192e55c7f2 |
| 52 | `1c4939af365e` | fit_start | model_building | model_building | y=['CryoSleep'] family=binary_classification metric=accuracy actual_fit_target_hash=c59154d4b1ba |
| 53 | `211ec2ce09d7` | fit_end | model_building | model_building | {"best_model": "HistGradientBoosting", "fit_seconds": 17.165, "results": {"ExtraTrees": {"cv_accuracy_mean": 0.9237}, "HistGradientBoosting": {"cv_accuracy_mean |
| 54 | `4eb4db6dc885` | predict | model_building | model_building | {"classes_order": ["False", "True"], "hook_file": "/Users/jacksonzhang/Desktop/26S1_CS33-2/lane_A/work/runs/A1__DEV-BIN-SPACESHIP__F05__F0_ORIGINAL__sb1101/hook |
| 55 | `6ba49b69aa9d` | serialize | model_building | model_building | provenance={'PassengerId': 'test_id', 'Transported': 'prediction'} mapping_digest=dd8677417ec8 |
| 56 | `4327d54fb069` | artifact_write | model_building | model_building | {"columns": ["PassengerId", "Transported"], "paths": {"best_model": "artifacts/runs/c00_20261019_141651_08ce8523/models/best_model.pkl", "model_results": "artif |
| 59 | `1cf7d25e50a9` | gate_check | reviewer | model_building | computed= / f0_status=approved |
| 60 | `c5f9207ea6e4` | gate_verdict | reviewer | model_building | verdict=approved -> summarizer |
| 65 | `5fa8f6abb187` | run_end | lane_a_runner |  | outcome=completed_graph review=approved completed=True prediction=True |
