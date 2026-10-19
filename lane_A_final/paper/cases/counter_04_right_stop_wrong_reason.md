# counter_04_right_stop_wrong_reason

- run_id: `A0P__DEV-ML-STEEL__F01__F0_ORIGINAL__sb1101__r1`
- verdict of the oracle: **S3** — stopped on contract violation
- why this case: mixed: Steel with all 7 targets removed; the reader's fuzzy hint fallback picks a feature as target and the gate stops on 'leakage' of that feature (S3 for the wrong reason).
- C_gold targets: see private oracle; C_read targets: `Steel_Plate_Thickness`; C_used fit targets: ``
- output provenance at serialization: `{}`
- raw trace: `logs/events/A0P__DEV-ML-STEEL__F01__F0_ORIGINAL__sb1101__r1.jsonl`

| seq | event_id | event | actor | phase | key fields |
|---|---|---|---|---|---|
| 4 | `9d65f0737bfe` | node_end | reader | reader | C_read targets=['Steel_Plate_Thickness'] family=regression out=[] ids=['id', 'Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults'] |
| 6 | `b71a4a439aea` | gate_check | background_reviewer | reader | computed=[S] target leakage: test dataset contains target columns Steel_Plate_Thickness / f0_status=retry |
| 7 | `18dc1c06ccf8` | gate_verdict | background_reviewer | reader | verdict=retry -> reader(retry) |
| 12 | `e6f4129b6eb8` | node_end | reader | reader | C_read targets=['Steel_Plate_Thickness'] family=regression out=[] ids=['id', 'Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults'] |
| 14 | `caf6bfb3b0de` | gate_check | background_reviewer | reader | computed=[S] target leakage: test dataset contains target columns Steel_Plate_Thickness / f0_status=retry |
| 15 | `de4534436d0a` | gate_verdict | background_reviewer | reader | verdict=retry -> reader(retry) |
| 20 | `e8ec6c7f4cb6` | node_end | reader | reader | C_read targets=['Steel_Plate_Thickness'] family=regression out=[] ids=['id', 'Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults'] |
| 22 | `73a6d66a4371` | gate_check | background_reviewer | reader | computed=[S] target leakage: test dataset contains target columns Steel_Plate_Thickness / f0_status=fatal |
| 23 | `37388d027155` | gate_verdict | background_reviewer | reader | verdict=fatal -> END(aborted) |
| 26 | `dbbeafc63c00` | run_end | lane_a_runner |  | outcome=completed_graph review=fatal completed=False prediction=False |
