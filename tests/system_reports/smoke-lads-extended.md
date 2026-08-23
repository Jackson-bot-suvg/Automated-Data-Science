# Framework Matrix Extended

- Generated: 2026-05-15 09:06:55 UTC
- Reference baseline document: [`framework-matrix-20260505-051038-corrected-with-lightautods.md`](/Users/a99/Downloads/5703-CS33-2/tests/system_reports/framework-matrix-20260505-051038-corrected-with-lightautods.md)
- This report is a fresh rerun with repeatability and token-usage instrumentation.
- Metric fields use 3-run aggregates per framework-task unless explicitly noted otherwise.

## Dataset Winners

| Dataset | Primary Metric | Better | Advantage |
| --- | --- | --- | ---: |
| `spaceship_titanic` | `accuracy` | `lightautods_lama` | - |

## Framework Metrics

| Dataset | Framework | Primary Metric | Score Mean | Reproducibility Std | Runtime Mean (s) | Runtime Std | Valid Output | Input Token Cost | Output Token Cost | Token Total | LLM Call Count |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `spaceship_titanic` | `main` | `` | - | - | - | - | `0/0` | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `spaceship_titanic` | `aide` | `` | - | - | - | - | `0/0` | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `spaceship_titanic` | `lightautods_lama` | `accuracy` | 0.7964 | 0.0000 | 28.7601 | 0.0000 | `1/1` | 816.0000 | 241.0000 | 1057.0000 | 1.0000 |
| `spaceship_titanic` | `autokaggle` | `` | - | - | - | - | `0/0` | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## `spaceship_titanic`

- Description: `single-target binary classification`
- Primary metric: `accuracy`
- Better framework: `lightautods_lama`
- Advantage: `-`

### main

- Score mean: `-`
- Reproducibility std: `-`
- Runtime mean (s): `-`
- Runtime std: `-`
- Valid output rate: `0/0`
- Success rate: `0/0`
- Input token cost: `0.0000`
- Output token cost: `0.0000`
- Token total: `0.0000`
- LLM call count: `0.0000`
- Representative prediction file: ``
- Representative run dir: ``

### aide

- Score mean: `-`
- Reproducibility std: `-`
- Runtime mean (s): `-`
- Runtime std: `-`
- Valid output rate: `0/0`
- Success rate: `0/0`
- Input token cost: `0.0000`
- Output token cost: `0.0000`
- Token total: `0.0000`
- LLM call count: `0.0000`
- Representative prediction file: ``
- Representative run dir: ``

### lightautods_lama

- Score mean: `0.7964`
- Reproducibility std: `0.0000`
- Runtime mean (s): `28.7601`
- Runtime std: `0.0000`
- Valid output rate: `1/1`
- Success rate: `1/1`
- Input token cost: `816.0000`
- Output token cost: `241.0000`
- Token total: `1057.0000`
- LLM call count: `1.0000`
- Representative prediction file: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/tmp/lads-lama-spaceship_titanic-20260515-090624-312541/models/submission.csv`
- Representative run dir: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/tmp/lads-lama-spaceship_titanic-20260515-090624-312541`

### autokaggle

- Score mean: `-`
- Reproducibility std: `-`
- Runtime mean (s): `-`
- Runtime std: `-`
- Valid output rate: `0/0`
- Success rate: `0/0`
- Input token cost: `0.0000`
- Output token cost: `0.0000`
- Token total: `0.0000`
- LLM call count: `0.0000`
- Representative prediction file: ``
- Representative run dir: ``
