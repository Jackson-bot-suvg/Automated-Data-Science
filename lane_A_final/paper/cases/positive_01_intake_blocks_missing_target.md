# positive_01_intake_blocks_missing_target

- run_id: `A1__DEV-BIN-SPACESHIP__F01__F0_ORIGINAL__sb1101`
- verdict of the oracle: **S3** — stopped on contract violation
- why this case: favourable: F0's intake gate stops an unrecoverable missing-target package before any fit (expected S3).
- C_gold targets: see private oracle; C_read targets: `Transported`; C_used fit targets: ``
- output provenance at serialization: `{}`
- raw trace: `logs/events/A1__DEV-BIN-SPACESHIP__F01__F0_ORIGINAL__sb1101.jsonl`

| seq | event_id | event | actor | phase | key fields |
|---|---|---|---|---|---|
| 4 | `d529e7347165` | node_end | reader | reader | C_read targets=['Transported'] family=binary_classification out=['Transported'] ids=['PassengerId'] |
| 6 | `c9bbdd018be3` | gate_check | background_reviewer | reader | computed=[S] target unresolved: training dataset is missing target column 'Transported' / f0_status=retry |
| 7 | `013eb8898229` | gate_verdict | background_reviewer | reader | verdict=retry -> reader(retry) |
| 12 | `2cbb6121e193` | node_end | reader | reader | C_read targets=['Transported'] family=binary_classification out=['Transported'] ids=['PassengerId'] |
| 14 | `08476c8d8e3c` | gate_check | background_reviewer | reader | computed=[S] target unresolved: training dataset is missing target column 'Transported' / f0_status=retry |
| 15 | `afdf66f8211f` | gate_verdict | background_reviewer | reader | verdict=retry -> reader(retry) |
| 20 | `0c292ef5c6db` | node_end | reader | reader | C_read targets=['Transported'] family=binary_classification out=['Transported'] ids=['PassengerId'] |
| 22 | `55f7380f6c11` | gate_check | background_reviewer | reader | computed=[S] target unresolved: training dataset is missing target column 'Transported' / f0_status=fatal |
| 23 | `38ac3d4373ac` | gate_verdict | background_reviewer | reader | verdict=fatal -> END(aborted) |
| 26 | `7d09eb62eae6` | run_end | lane_a_runner |  | outcome=completed_graph review=fatal completed=False prediction=False |
