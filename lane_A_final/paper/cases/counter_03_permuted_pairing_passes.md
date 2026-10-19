# counter_03_permuted_pairing_passes

- run_id: `A1__DEV-BIN-SPACESHIP__F10__F0_ORIGINAL__sb1101`
- verdict of the oracle: **S5** — output mapping invalid: Transported: delivered values != predict() output row-for-row
- why this case: unfavourable: ID->prediction pairing permuted after predict(); columns/rows/nulls pass the model_building gate (S5).
- C_gold targets: see private oracle; C_read targets: `Transported`; C_used fit targets: `Transported`
- output provenance at serialization: `{"PassengerId": "test_id", "Transported": "prediction_reordered_or_altered"}`
- raw trace: `logs/events/A1__DEV-BIN-SPACESHIP__F10__F0_ORIGINAL__sb1101.jsonl`

| seq | event_id | event | actor | phase | key fields |
|---|---|---|---|---|---|
| 4 | `4ccc0f516365` | node_end | reader | reader | C_read targets=['Transported'] family=binary_classification out=['Transported'] ids=['PassengerId'] |
| 6 | `a29f02444134` | gate_check | background_reviewer | reader | computed= / f0_status=approved |
| 7 | `f2fe53008779` | gate_verdict | background_reviewer | reader | verdict=approved -> planner |
| 11 | `543c563b509f` | node_end | planner | preliminary_eda | planned_phase=preliminary_eda candidate_hash=3fc8cf900465 |
| 15 | `b63dabca7abf` | gate_check | reviewer | preliminary_eda | computed= / f0_status=approved |
| 16 | `d2b0e4846360` | gate_verdict | reviewer | preliminary_eda | verdict=approved -> planner(next_phase) |
| 20 | `7ed4b59bd6f8` | node_end | planner | data_cleaning | planned_phase=data_cleaning candidate_hash=da0c645e2afb |
| 24 | `c02d2e6daf19` | gate_check | reviewer | data_cleaning | computed= / f0_status=approved |
| 25 | `d4e1e008c0e2` | gate_verdict | reviewer | data_cleaning | verdict=approved -> planner(next_phase) |
| 29 | `839da01f6675` | node_end | planner | indepth_eda | planned_phase=indepth_eda candidate_hash=7842728b6448 |
| 33 | `120bf898a94c` | gate_check | reviewer | indepth_eda | computed= / f0_status=approved |
| 34 | `1e46b7b98e15` | gate_verdict | reviewer | indepth_eda | verdict=approved -> planner(next_phase) |
| 39 | `d78ab2dac95b` | node_end | planner | feature_engineering | planned_phase=feature_engineering candidate_hash=32c00a1a0339 |
| 43 | `358bf0a6cd2d` | gate_check | reviewer | feature_engineering | computed= / f0_status=approved |
| 44 | `9b6fd33ea274` | gate_verdict | reviewer | feature_engineering | verdict=approved -> planner(next_phase) |
| 49 | `fb3c56722904` | node_end | planner | model_building | planned_phase=model_building candidate_hash=4f3f82f3fd77 |
| 51 | `a054855507ff` | fit_start | model_building | model_building | y=['Transported'] family=binary_classification metric=accuracy actual_fit_target_hash=9ef7e79995e0 |
| 52 | `df51025c8bc9` | fit_end | model_building | model_building | {"best_model": "HistGradientBoosting", "fit_seconds": 18.467, "results": {"ExtraTrees": {"cv_accuracy_mean": 0.7795}, "HistGradientBoosting": {"cv_accuracy_mean |
| 53 | `00b3a422a8ff` | predict | model_building | model_building | {"classes_order": ["False", "True"], "hook_file": "/Users/jacksonzhang/Desktop/26S1_CS33-2/lane_A/work/runs/A1__DEV-BIN-SPACESHIP__F10__F0_ORIGINAL__sb1101/hook |
| 54 | `3984e8b32b70` | injection | lane_a_mutator | model_building | {"template_id": "F10", "fixed_points": 0} |
| 55 | `959cc773dcd6` | serialize | model_building | model_building | provenance={'PassengerId': 'test_id', 'Transported': 'prediction_reordered_or_altered'} mapping_digest=373ea0d15069 |
| 56 | `92dc59d24201` | artifact_write | model_building | model_building | {"columns": ["PassengerId", "Transported"], "paths": {"best_model": "artifacts/runs/c00_20261019_141435_96299d62/models/best_model.pkl", "model_results": "artif |
| 59 | `d3782c621185` | gate_check | reviewer | model_building | computed= / f0_status=approved |
| 60 | `101b57342b83` | gate_verdict | reviewer | model_building | verdict=approved -> summarizer |
| 65 | `b403055c4be4` | run_end | lane_a_runner |  | outcome=completed_graph review=approved completed=True prediction=True |
