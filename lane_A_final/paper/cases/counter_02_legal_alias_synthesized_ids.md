# counter_02_legal_alias_synthesized_ids

- run_id: `A1__DEV-REG-HOUSE__B01__F0_ORIGINAL__sb1101`
- verdict of the oracle: **S5** — output mapping invalid: sale_price_prediction: delivered values != predict() output row-for-row
- why this case: unfavourable: a legal output alias is not aligned; the alias column is filled with synthesized range ids and approved (S5).
- C_gold targets: see private oracle; C_read targets: `SalePrice`; C_used fit targets: `SalePrice`
- output provenance at serialization: `{"Id": "test_id", "sale_price_prediction": "synthesized_range_id"}`
- raw trace: `logs/events/A1__DEV-REG-HOUSE__B01__F0_ORIGINAL__sb1101.jsonl`

| seq | event_id | event | actor | phase | key fields |
|---|---|---|---|---|---|
| 4 | `3f4b67cf85a3` | node_end | reader | reader | C_read targets=['SalePrice'] family=regression out=[] ids=['Id', 'sale_price_prediction'] |
| 6 | `90ce37d031be` | gate_check | background_reviewer | reader | computed= / f0_status=approved |
| 7 | `9814d52132e1` | gate_verdict | background_reviewer | reader | verdict=approved -> planner |
| 11 | `fdb2d406ce15` | node_end | planner | preliminary_eda | planned_phase=preliminary_eda candidate_hash=3fc8cf900465 |
| 15 | `ff088d501ae4` | gate_check | reviewer | preliminary_eda | computed= / f0_status=approved |
| 16 | `d2a91753e8bb` | gate_verdict | reviewer | preliminary_eda | verdict=approved -> planner(next_phase) |
| 20 | `a28ace5a4773` | node_end | planner | data_cleaning | planned_phase=data_cleaning candidate_hash=a9b1e6ec314c |
| 24 | `d3209806ffff` | gate_check | reviewer | data_cleaning | computed= / f0_status=approved |
| 25 | `083c9d1e7138` | gate_verdict | reviewer | data_cleaning | verdict=approved -> planner(next_phase) |
| 29 | `fd504b1bedb4` | node_end | planner | indepth_eda | planned_phase=indepth_eda candidate_hash=b26f577c4df5 |
| 33 | `db96245d4341` | gate_check | reviewer | indepth_eda | computed= / f0_status=approved |
| 34 | `b551d22877d0` | gate_verdict | reviewer | indepth_eda | verdict=approved -> planner(next_phase) |
| 39 | `33e251727946` | node_end | planner | feature_engineering | planned_phase=feature_engineering candidate_hash=2395f0dadb6b |
| 43 | `0a65af72908d` | gate_check | reviewer | feature_engineering | computed= / f0_status=approved |
| 44 | `1e9aa93d7afa` | gate_verdict | reviewer | feature_engineering | verdict=approved -> planner(next_phase) |
| 49 | `531d447cba14` | node_end | planner | model_building | planned_phase=model_building candidate_hash=0a6c566e3035 |
| 51 | `7f79a86f01ce` | fit_start | model_building | model_building | y=['SalePrice'] family=regression metric=neg_root_mean_squared_error actual_fit_target_hash=bb9db78db7d4 |
| 52 | `fe1d673c6201` | fit_end | model_building | model_building | {"best_model": "Ridge", "fit_seconds": 25.648, "results": {"ExtraTrees": {"cv_neg_root_mean_squared_error_mean": -27243.9813}, "GradientBoosting": {"cv_neg_root |
| 53 | `1564c16bf23f` | predict | model_building | model_building | {"classes_order": null, "hook_file": "/Users/jacksonzhang/Desktop/26S1_CS33-2/lane_A/work/runs/A1__DEV-REG-HOUSE__B01__F0_ORIGINAL__sb1101/hooks/predict_output. |
| 54 | `26fbe9f2e7de` | serialize | model_building | model_building | provenance={'Id': 'test_id', 'sale_price_prediction': 'synthesized_range_id'} mapping_digest=8294b1a859f4 |
| 55 | `70159746c03d` | artifact_write | model_building | model_building | {"columns": ["Id", "sale_price_prediction"], "paths": {"best_model": "artifacts/runs/b01_20261019_141730_ebecd124/models/best_model.pkl", "model_results": "arti |
| 58 | `5a7cfc68fb65` | gate_check | reviewer | model_building | computed= / f0_status=approved |
| 59 | `9063c78818bd` | gate_verdict | reviewer | model_building | verdict=approved -> summarizer |
| 64 | `c9d7d385ce19` | run_end | lane_a_runner |  | outcome=completed_graph review=approved completed=True prediction=True |
