# positive_02_intake_blocks_wrong_hint

- run_id: `A1__DEV-MC-GHOULS__F03__F0_ORIGINAL__sb1101`
- verdict of the oracle: **S3** — stopped on contract violation
- why this case: favourable: F0's intake gate stops a misleading explicit target hint (expected S3).
- C_gold targets: see private oracle; C_read targets: `hair_length`; C_used fit targets: ``
- output provenance at serialization: `{}`
- raw trace: `logs/events/A1__DEV-MC-GHOULS__F03__F0_ORIGINAL__sb1101.jsonl`

| seq | event_id | event | actor | phase | key fields |
|---|---|---|---|---|---|
| 4 | `a15ff5df8515` | node_end | reader | reader | C_read targets=['hair_length'] family=regression out=[] ids=['id', 'type'] |
| 6 | `8c79ba25cf31` | gate_check | background_reviewer | reader | computed=[S] target leakage: test dataset contains target columns hair_length / f0_status=retry |
| 7 | `aad3b8d91691` | gate_verdict | background_reviewer | reader | verdict=retry -> reader(retry) |
| 12 | `ad5f04b5fac7` | node_end | reader | reader | C_read targets=['hair_length'] family=regression out=[] ids=['id', 'type'] |
| 14 | `515e762d12da` | gate_check | background_reviewer | reader | computed=[S] target leakage: test dataset contains target columns hair_length / f0_status=retry |
| 15 | `6b4062032ced` | gate_verdict | background_reviewer | reader | verdict=retry -> reader(retry) |
| 20 | `5fc499618512` | node_end | reader | reader | C_read targets=['hair_length'] family=regression out=[] ids=['id', 'type'] |
| 22 | `3c7d0d11cc2c` | gate_check | background_reviewer | reader | computed=[S] target leakage: test dataset contains target columns hair_length / f0_status=fatal |
| 23 | `1158a7320958` | gate_verdict | background_reviewer | reader | verdict=fatal -> END(aborted) |
| 26 | `ae5fd01199f3` | run_end | lane_a_runner |  | outcome=completed_graph review=fatal completed=False prediction=False |
