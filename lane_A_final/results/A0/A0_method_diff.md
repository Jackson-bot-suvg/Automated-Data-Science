# A0 — Code-semantic audit of F0 and function-level method differences

**Read this first.** Unless a line says otherwise, every statement below comes from reading the source.
- Base: `b9dbebed269f62767e149b93a02f71051efc1027` (public release of the manuscript code). All line numbers refer to this commit.
- Behavioural confirmation comes from the DEV runs. Each such statement points to its run IDs in `results/run_level.csv`.

## 1. The contract: where it is created, who may write it, and whether it is frozen

| Question | Finding | Evidence |
|---|---|---|
| Who creates `target_spec` (C_read)? | The reader, using deterministic resolution: explicit fields > structured evidence (sample submission ∩ train/test difference) > overview regex > natural-language hint > `Survived` fallback. | `app/agents/reader.py:344-413` (`_resolve_target_scope`) |
| Is `task_family` deterministic? | **No.** After the LLM call, `target_spec["task_family"]` is taken from the LLM's `predicted_task_type`, which overrides the value inferred from the target. | `reader.py:133-135` |
| Is the frozen contract write-protected? | **No.** `NODE_CONTRACTS` (the reads/writes table) is documentation only and is never enforced. `GraphState.target_spec` has no reducer, so any node that returns `target_spec` silently replaces it. | `app/schemas/io_contracts.py:44+`; `grep NODE_CONTRACTS` shows it is referenced only in docstrings |
| Can an intake retry change the target? | **No.** `_explicit_target_columns` first reads `state.target_spec.target_columns`, i.e. the reader's own previous output. A retry therefore re-resolves exactly the same target. | `reader.py:415-425` |
| Is there an `Aligned(Y,O,f)` predicate? | **Not in the gates.** `_submission_spec` sets O = submission ∩ Y and turns every other submission column into an id column. An empty O is accepted silently. | `reader.py:669-686` |
| What does model building do with an empty O? | It defaults O to Y and synthesises any id column that is missing from test as `range(n)`. | `app/agents/model_building.py:656-659`; `app/tools/model_tools.py:1725-1745` (`_extract_id_frame`, line 1738) |
| What does the fit consume? | `y = train[target_spec.target_columns]`, read from the **current** state at fit time. The planner plan cannot choose y. It **can**, however, override `scoring.primary_metric`. | `model_building.py:529-690`, `:214`; `_apply_plan_to_config` `:822` (`configure_scoring`) |

## 2. The intake gate (`BackgroundReviewerAgent`) compared with manuscript Algorithm 1

| Algorithm 1 line | Manuscript rule | Code (b9dbebe) |
|---|---|---|
| 2 | Y = ∅ / unresolved → fatal | `target unresolved` is a blocking failure but **retry** (fatal only at attempt 3) — `bg_reviewer.py:163-177, 222-232` |
| 3 | y ∉ D_train → fatal | **retry**, then fatal at attempt 3 (hard fatal only when the train *file* is missing or unreadable, `:112, :117`) |
| 4 | y ∈ D_test → fatal | **retry**, then fatal at attempt 3 (`:180-188`) |
| 6 | O = ∅ or ¬Aligned → retry | **not implemented.** The only submission check compares the sample-submission header with `expected_submission`, which was copied from the same file → it is tautological (`:190-195`) |
| 7 | multilabel: O must cover all of Y | **not implemented.** Y is itself derived from submission ∩ (train − test) (`reader.py:427-478`), so an incomplete sample submission narrows Y, and `Y ⊆ cols(train)` holds by construction |
| 8 | freeze C before model construction | C is stored in shared state, **but nothing enforces immutability** (§1) |

Because an intake retry re-runs a reader that cannot change its answer (§1), every "retry" at intake is effectively a delayed fatal. Each one costs about 3 extra node executions and 2–3 LLM calls.

## 3. The phase gates (`ReviewerAgent` + `app/tools/reviewer_tools.py`)

- **The verdict is deterministic.** `_run_phase_review` → `review_<phase>` → `_status_for` (`reviewer_agent.py:139-170`). The LLM only phrases a comment (`render_text`, `:55`), and the comment is never read by routing.
- **The gates validate against the current contract, not the approved one.** Each review re-derives its contract from the *current* `target_spec` (`reviewer_tools.py:517-560`). A contract rewritten after intake is therefore checked against itself.
- **Checks performed:**
  - artifact existence and paths;
  - readable non-empty CSVs;
  - target present in cleaned/engineered train;
  - target absent from cleaned/engineered test;
  - selected features exclude targets;
  - train/test feature parity;
  - model fields;
  - prediction file has `submission_columns` in order, no nulls, row count == test rows (`:324-374, :608-657`).
- **Not checked anywhere:**
  - the y actually passed to `fit`;
  - the task family or metric used at fit;
  - class order or inverse transform;
  - whether the values in an output column are the model's predictions (rather than synthesised ids or permuted rows);
  - whether the ID → prediction pairing is preserved.
- **A leak is repaired silently.** `data_cleaning` drops target columns that leaked into test (`data_cleaning.py:161-166`), so later gates and the M4 end check can no longer see the leak.
- **Retries vs attempts.** `max_retries_per_phase = 3` is compared with `attempt = retries + 1` and returns fatal when `attempt >= 3` (`reviewer_agent.py:29, 163-170`). That means **3 attempts = at most 2 retries**. The manuscript's "retry budget N_max = 3" therefore counts attempts.
- **Where retry can actually repair.** `data_cleaning` and `indepth_eda` use deterministic planners (`planner.py:321`), so a retry repeats the same plan. Only `feature_engineering` and `model_building` re-plan with an LLM that receives the top-3 blocking failures and the previous plan (`planner.py:900-962`). This is the only diagnostic (targeted-repair) channel in F0.

## 4. Computing a verdict vs executing it (A0 step 3)

**Computing:**
- Phase gates: `ReviewerAgent._run_phase_review` → `reviewer_tools.review_*`, then `ReviewerAgent._status_for`.
- Intake gate: `BackgroundReviewerAgent._check_reader_output` + `_status_for`.

**Executing:**
- `ReviewerAgent.run` writes `review_state.latest_review_status` and `retry_count_by_phase`.
- `app/graph/routes.py:20-29` (`route_after_reviewer`):
  - `approved` → planner, or summarizer when the phase is model_building;
  - `fatal` → END;
  - **anything else (including `None`) → retry**.
- `planner_node` (`app/graph/nodes/planner_node.py`) picks the phase from that same status.

**Consequence: routing depends on state written by the gates.** If the gate nodes are removed but the edges are kept (`M0_NAIVE_GATE_REMOVAL`), `latest_review_status` is never set. The graph then loops reader ↔ background_reviewer until the recursion limit.
- Reproduced in the A0 smoke runs `A0S__*__M0_NAIVE_GATE_REMOVAL__*`.
- This reproduces the manuscript's Table 3 row "No reviewer gates: clean completion 0/4, FP 4/4, 8.40 s" as an **interface breakage, not a semantic effect**.
- A correct M0 (verdicts not enforced, state still written) completes clean tasks. See `A0_smoke.csv`.

## 5. Function-level implementation of every method

All variants wrap the same functions at runtime (`src/adapters/lane_a/methods.py`); the base tree is never edited.

| Method | `_run_phase_review` / `_check_reader_output` result passed on | `_status_for` | Other |
|---|---|---|---|
| F0_ORIGINAL | unchanged (logged) | unchanged | – |
| M0_NO_SEMANTIC_ENFORCEMENT | E failures only (S logged as shadow) | unchanged | – |
| M2_INTAKE_ONLY_STRONG | intake: unchanged; phases: E only | unchanged | – |
| M3_SHADOW | no failures (all logged as shadow) | unchanged | the agent sees an approved record |
| M4_END_ONLY | intermediate gates: none; model_building gate: all F0 rules (intake + every phase) re-applied to the final state | intake failure at the end → fatal | redirects `current_phase` to the earliest failing phase |
| M5_BLOCK_ONLY | unchanged | any failure → fatal | – |
| G_GENERIC_RETRY | unchanged | unchanged | `PlannerAgent._retry_context_for_phase` returns no blocking_failures or warnings |
| M0_NAIVE_GATE_REMOVAL | gate nodes are pass-through `{}` | – | A0 only (reconstruction of the old ablation) |

**Behaviour equivalence.** Under the LLM-off deterministic replay, every variant above except M0_NAIVE produced byte-identical submissions, identical phase histories, contracts and review statuses compared with the *unobserved* F0 run on a clean task. See `results/A0/version_diff.csv`.

## 6. Manuscript items that cannot be verified from code

- **Table 3 ablations.** The `No target-spec`, `Schema-only` and `No semantic align.` variants have no switch in b9dbebe, so they cannot be re-run. The `No reviewer gates` row is consistent with the interface-breakage mechanism in §4. This is a reconstruction, not a replay.
- **Table 5, "Missing target column in train: 2/3 stopped".** Algorithm 1 line 3 says fatal; the code says retry → fatal for a single-target task.
  - A 1/3 leak is consistent with a multilabel package losing one target column. The reader would then resolve the remaining columns as Y, and the gate's Y ⊆ train check would pass by construction (§2, line 7).
  - The 16 packages are not available, so the actual cause is **not verifiable**.
- **Runtime statistics come from different samples.**
  - RQ5 reports a model-building mean of **97.62 s**, which is larger than the Table 2 mean of *total* runtime (**75.8 s**). The RQ5 phase times therefore cannot come from the Table 2 runs.
  - Table 1 (164.60 s), Table 2 (75.8 s) and Table 3 (70.84 s) are also different samples.
  - The manuscript does not document these sources, so their causes are not verifiable.
- **"Deterministic pipeline".** The CV splits are seeded. However, `task_family` (§1) and the feature-engineering and model-building plans are LLM outputs, so repeat stability depends on the LLM. With deepseek-flash, see `A0_old_vs_current.csv`.
