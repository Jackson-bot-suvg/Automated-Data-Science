# Framework Matrix Extended

- Generated: 2026-05-15 09:12:11 UTC
- Reference baseline document: [`framework-matrix-20260505-051038-corrected-with-lightautods.md`](/Users/a99/Downloads/5703-CS33-2/tests/system_reports/framework-matrix-20260505-051038-corrected-with-lightautods.md)
- This report is a fresh rerun with repeatability and token-usage instrumentation.
- Metric fields use 3-run aggregates per framework-task unless explicitly noted otherwise.

## Dataset Winners

| Dataset | Primary Metric | Better | Advantage |
| --- | --- | --- | ---: |
| `spaceship_titanic` | `accuracy` | `aide` | 0.0117 |

## Framework Metrics

| Dataset | Framework | Primary Metric | Score Mean | Reproducibility Std | Runtime Mean (s) | Runtime Std | Valid Output | Input Token Cost | Output Token Cost | Token Total | LLM Call Count |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `spaceship_titanic` | `main` | `accuracy` | 0.7993 | 0.0000 | 158.3656 | 0.0000 | `1/1` | 21704.0000 | 5017.0000 | 26721.0000 | 9.0000 |
| `spaceship_titanic` | `aide` | `accuracy` | 0.8110 | 0.0000 | 130.4996 | 0.0000 | `1/1` | 5781.0000 | 3658.0000 | 9439.0000 | 4.0000 |
| `spaceship_titanic` | `lightautods_lama` | `accuracy` | 0.7964 | 0.0000 | 25.5513 | 0.0000 | `1/1` | 816.0000 | 181.0000 | 997.0000 | 1.0000 |
| `spaceship_titanic` | `autokaggle` | `accuracy` | 0.7792 | 0.0000 | 39.6602 | 0.0000 | `1/1` | 1326.0000 | 202.0000 | 1528.0000 | 1.0000 |

## `spaceship_titanic`

- Description: `single-target binary classification`
- Primary metric: `accuracy`
- Better framework: `aide`
- Advantage: `0.0117`

### main

- Score mean: `0.7993`
- Reproducibility std: `0.0000`
- Runtime mean (s): `158.3656`
- Runtime std: `0.0000`
- Valid output rate: `1/1`
- Success rate: `1/1`
- Input token cost: `21704.0000`
- Output token cost: `5017.0000`
- Token total: `26721.0000`
- LLM call count: `9.0000`
- Representative prediction file: `artifacts/runs/spaceship_titanic_20260515_170803_e241d2b6/predictions/submission.csv`
- Representative run dir: `/Users/a99/Downloads/5703-CS33-2/artifacts/runs/spaceship_titanic_20260515_170803_e241d2b6`

### aide

- Score mean: `0.8110`
- Reproducibility std: `0.0000`
- Runtime mean (s): `130.4996`
- Runtime std: `0.0000`
- Valid output rate: `1/1`
- Success rate: `1/1`
- Input token cost: `5781.0000`
- Output token cost: `3658.0000`
- Token total: `9439.0000`
- LLM call count: `4.0000`
- Representative prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/aide_eval/aideml/workspaces/2-proud-silver-mongrel/working/submission.csv`
- Representative run dir: `/Users/a99/Downloads/5703-CS33-2/_external/aide_eval/aideml/logs/2-proud-silver-mongrel`

### lightautods_lama

- Score mean: `0.7964`
- Reproducibility std: `0.0000`
- Runtime mean (s): `25.5513`
- Runtime std: `0.0000`
- Valid output rate: `1/1`
- Success rate: `1/1`
- Input token cost: `816.0000`
- Output token cost: `181.0000`
- Token total: `997.0000`
- LLM call count: `1.0000`
- Representative prediction file: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/tmp/lads-lama-spaceship_titanic-20260515-091005-443133/models/submission.csv`
- Representative run dir: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/tmp/lads-lama-spaceship_titanic-20260515-091005-443133`

### autokaggle

- Score mean: `0.7792`
- Reproducibility std: `0.0000`
- Runtime mean (s): `39.6602`
- Runtime std: `0.0000`
- Valid output rate: `1/1`
- Success rate: `1/1`
- Input token cost: `1326.0000`
- Output token cost: `202.0000`
- Token total: `1528.0000`
- LLM call count: `1.0000`
- Representative prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/autokaggle_eval/AutoKaggle/multi_agents/competition/autokaggle-spaceship_titanic-20260515-091031-675497/submission.csv`
- Representative run dir: `/Users/a99/Downloads/5703-CS33-2/_external/autokaggle_eval/AutoKaggle/multi_agents/competition/autokaggle-spaceship_titanic-20260515-091031-675497`
