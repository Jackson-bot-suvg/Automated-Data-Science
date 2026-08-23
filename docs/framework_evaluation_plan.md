# Cross-Framework Evaluation Plan

## Goal

Build a unified evaluation system for comparing multiple AutoDS-style frameworks on the same task matrix.

The evaluation design should:

- use metrics that can be collected for every framework
- avoid relying on framework-specific internals
- keep benchmark metric contracts fixed per dataset
- support both per-task comparison and framework-level aggregation

## Current Scope

- Comparison unit: one framework run on one dataset
- Benchmark set: current 8-dataset matrix
- Evaluation style: same dataset, same train/test assets, same locked primary metric per task

## Current Benchmark Implementation

### Dataset Registry and Metric Locking

The current matrix benchmark is driven by the shared dataset registry:

- [app/validation_matrix.py](/Users/a99/Downloads/5703-CS33-2/app/validation_matrix.py:1)

This file defines:

- dataset keys
- task descriptions
- train/test dataset paths
- target columns
- task family / target mode
- locked `metric_hint` for each benchmark dataset

The locked metric is propagated into runtime state through:

- [run_multiagent.py](/Users/a99/Downloads/5703-CS33-2/run_multiagent.py:47)
- [app/intake/task_intake.py](/Users/a99/Downloads/5703-CS33-2/app/intake/task_intake.py:23)

This is the main guardrail that keeps benchmark metric contracts stable across reruns.

### Main Framework

The main framework is executed through:

- [run_multiagent.py](/Users/a99/Downloads/5703-CS33-2/run_multiagent.py:1)

Current benchmark behavior:

- uses the shared dataset registry in `app/validation_matrix.py`
- carries the locked `metric_hint` into the pipeline state
- produces run artifacts under `artifacts/runs/<run_id>/`

Important main-framework artifact paths per run:

- `reports/technical_report.md`
- `reports/run_manifest.json`
- `models/model_results.json`
- `predictions/submission.csv`
- `phase_plans/*.json`
- `phase_reviews/*.json`

### AIDE

The current AIDE matrix benchmark is run through:

- [tests/run_framework_matrix_compare.py](/Users/a99/Downloads/5703-CS33-2/tests/run_framework_matrix_compare.py:1)

This script:

- runs the main framework and AIDE on the same dataset list
- reads the locked primary metric from the main-framework target spec
- converts that metric into an AIDE evaluation prompt
- writes a two-framework comparison report

The AIDE execution helper is:

- [tests/helpers/run_aide_experiment.py](/Users/a99/Downloads/5703-CS33-2/tests/helpers/run_aide_experiment.py:1)

The isolated AIDE repository currently used for benchmarking is:

- `/Users/a99/Downloads/5703-CS33-2/_external/aide_eval/aideml`

AIDE result artifacts are typically read from:

- AIDE log/workspace directories under `_external/aide_eval/aideml/`
- helper-emitted JSON under `tests/system_reports/tmp/`

### LightAutoDS

The current stable LightAutoDS benchmark route is the LightAutoML-only route with external scoring on the main-project metric.

The canonical matrix append script is:

- [tests/run_lightautods_lama_compare.py](/Users/a99/Downloads/5703-CS33-2/tests/run_lightautods_lama_compare.py:1)

This script:

- reuses an existing corrected main/AIDE baseline
- runs only LightAutoDS
- keeps the current LightAutoDS route on `LightAutoML-only`
- uses LLM-selected internal config
- scores validation predictions externally using the benchmark metric
- merges LightAutoDS results back into the three-framework report

The current helper used by this route is:

- [tests/helpers/run_lads_lama_experiment.py](/Users/a99/Downloads/5703-CS33-2/tests/helpers/run_lads_lama_experiment.py:1)

The isolated LightAutoDS repository currently used for benchmarking is:

- `/Users/a99/Downloads/5703-CS33-2/_external/lightautods_eval/LADS`

Notes:

- the older exploratory script [tests/run_framework_matrix_compare_lads.py](/Users/a99/Downloads/5703-CS33-2/tests/run_framework_matrix_compare_lads.py:1) exists, but it is not the current canonical route for the maintained benchmark
- the maintained route is the `run_lightautods_lama_compare.py` flow

### Report Maintenance

The currently maintained three-framework report is:

- [framework-matrix-20260505-051038-corrected-with-lightautods.md](/Users/a99/Downloads/5703-CS33-2/tests/system_reports/framework-matrix-20260505-051038-corrected-with-lightautods.md:1)
- [framework-matrix-20260505-051038-corrected-with-lightautods.json](/Users/a99/Downloads/5703-CS33-2/tests/system_reports/framework-matrix-20260505-051038-corrected-with-lightautods.json:1)

When only the `main` framework is rerun and selectively refreshed into the maintained three-framework report, the update script is:

- [tests/run_refresh_main_threeway_matrix.py](/Users/a99/Downloads/5703-CS33-2/tests/run_refresh_main_threeway_matrix.py:1)

### Current Practical Method

At the moment, the practical cross-framework benchmark flow is:

1. Define benchmark datasets and locked metrics in `app/validation_matrix.py`.
2. Run main + AIDE with `tests/run_framework_matrix_compare.py` to create a corrected two-framework baseline.
3. Append LightAutoDS with `tests/run_lightautods_lama_compare.py`.
4. Maintain the long-lived three-framework report in `tests/system_reports/framework-matrix-20260505-051038-corrected-with-lightautods.md`.
5. When needed, selectively refresh improved `main` rows with `tests/run_refresh_main_threeway_matrix.py`.

## Core Metric Families

### 1. Primary Metric

The benchmark metric locked for each dataset before execution.

Examples:

- `accuracy`
- `f1_macro`
- `roc_auc`
- `roc_auc_macro_ovr`
- `neg_root_mean_squared_error`

Purpose:

- serve as the main quality score for cross-framework comparison
- determine per-task winner

### 2. Total Runtime

Wall-clock runtime from task start to final output generation.

Definition:

- starts when the framework begins processing the dataset
- ends when the framework produces its final prediction artifact or final failure state

Purpose:

- measure efficiency
- compare quality/speed tradeoffs across frameworks

### 3. Valid Output

Whether the framework produced a usable final output artifact.

Suggested checks:

- prediction file exists
- required columns are present
- row count matches expected test rows
- output contains no unexpected nulls or malformed values

Purpose:

- distinguish true task completion from partial or broken runs

### 4. Task Success Rate

Whether the framework successfully completes each task end-to-end.

Definition:

- success = run finishes and yields a valid output plus an evaluable score
- failure = crash, blocked run, invalid output, or missing evaluable result

Framework-level aggregation:

- `success_rate = successful_tasks / total_tasks`

### 5. Metric Contract Match

Whether the framework’s final evaluation metric matches the benchmark metric preassigned to the dataset.

Examples:

- expected `f1_macro`, actual `f1_macro` -> match
- expected `f1_macro`, actual `accuracy` -> mismatch

Purpose:

- prevent invalid comparisons caused by metric drift
- ensure benchmark fairness across frameworks

### 6. Stability

How consistent the framework is when the same task is rerun.

Suggested measures:

- `score_std`: standard deviation of the primary metric across repeated runs
- `runtime_std`: standard deviation of runtime across repeated runs
- optional lightweight fallback: repeated-run success rate

Purpose:

- distinguish stable frameworks from fragile or highly variable ones

### 7. Gap to Best

The difference between a framework’s score and the best score achieved on that task.

Definition:

- for higher-is-better metrics: `best_score - framework_score`
- for lower-is-better metrics after normalization: use a directionally consistent gap representation

Recommended reporting:

- store a directionally normalized `gap_to_best`, where `0` is best and larger is worse

Purpose:

- enable aggregate comparison across tasks with different raw metrics

### 8. Cost / Call Volume

Resource usage attributable to the framework run.

Preferred fields:

- `llm_call_count`
- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `api_cost_usd`

Fallback:

- if token or billing data is unavailable, record only `llm_call_count`

Purpose:

- compare accuracy/runtime gains against actual model usage

## Minimum Cross-Framework Schema

Each task-framework result should support the following fields:

- `dataset`
- `framework`
- `primary_metric`
- `score`
- `runtime_sec`
- `success`
- `valid_output`
- `metric_contract_match`
- `gap_to_best`

Recommended extensions:

- `score_std`
- `runtime_std`
- `llm_call_count`
- `total_tokens`
- `api_cost_usd`

## Framework-Level Summary

Each framework summary should aggregate:

- mean normalized rank or win count
- mean or median primary score by task
- total runtime
- success rate
- valid output rate
- metric contract match rate
- mean gap to best
- stability summary
- total LLM calls / tokens / cost when available

## Practical Rollout Order

### Phase 1

Implement first:

- primary metric
- total runtime
- valid output
- success rate
- metric contract match
- gap to best

### Phase 2

Add:

- stability from repeated runs
- LLM call count

### Phase 3

Add when provider telemetry is available:

- token usage
- API cost

## Notes

- Framework-specific diagnostics such as planner retries, phase retries, or internal fallback usage are useful for debugging, but should not be part of the primary cross-framework scoreboard.
- Those internal diagnostics can be maintained in separate framework-specific appendices.
