# Lane A analysis plan (pre-registered)

Version laneA-analysis-1.0, registered 2026-10-19 before any LLM-live lane-A run. Status: **DEV pilot only.**
G0 is FAIL: `FREEZE_SIGNOFF.yaml` is unsigned and the CONF and TRANSFER identities are not frozen. Nothing in this package
may be called a CONF confirmation.

## Units, keys and denominators
- Statistical unit: the independent base task (DEV: Spaceship / Ghouls / House; one source group each).
- Paired key: `(task_id, case_family, seed_block, host_id, model_id)`. A2/A3 additionally require the same
  fork snapshot (state_hash, candidate_hash, visible_evidence_hash, remaining_budget, rng hash).
- Denominator: actually started runs (`N_started`). N/A and not_started cells are listed but never counted as 0.
- A2 prefix runs are components of their branches, not separate trajectories.

## A0
- Clean reproduction: 8 DEV tasks × 3 repeats, original packages, manuscript presets (q, H, metric hint),
  technical-only. Compared with manuscript Table 2 on completion, validity, CV score and time. A difference is
  attributed to "config" (LLM backend MiniMax-M2.7 is end-of-life, so deepseek-flash is used) or "code"
  (only where the code diff proves it).
- Smoke: M0, M3 and M0_NAIVE on 2 clean DEV tasks (Titanic, Ghouls).
- Hook equivalence: LLM-off deterministic replay. Unobserved F0 vs observed F0/M0/M2/M3/M4/M5/G must give
  identical node sequence, final contract, review statuses and submission bytes.
- Probes: 4 probes on the manuscript ML representative (Steel). The other families reuse A1's F0 runs.

## A1 (DEV, 3 tasks × 6 probes × {F0, M3} = 36 runs; ML family = 12 N/A cells)
For each run, report C_read vs C_gold, C_used (fit y, family, metric, output provenance) and the terminal S.
Failure sources are separated as:
- initial contract error (C_read ≠ C_gold);
- post-intake stage drift (C_read = C_gold but fit ≠ C_gold);
- consumption decoupling (state fields consistent, delivered values ≠ predict() output);
- output mismatch.

## A2 (DEV pilot: 3 tasks × 4 contexts × seed blocks {1101, 1102} = 24 prefixes; branches M3/M5/G/F0/M4; G N/A for a, b)
Per prefix and branch, record:
- next action after the gate;
- whether a wrong-target fit happened after the fork;
- whether a wrong file was delivered;
- the terminal S;
- calls, wall time and wrong-training seconds.

Primary contrasts (DEV pilot, descriptive):
- F0−M3: effect of verdict enforcement on silent error (S5), unsafe delivery (S5+S6) and correct delivery (S1+S2).
- F0−M5: effect of default retry over immediate block on S1+S2 and on cost.
- F0−M4: early versus end-only checking; compares wrong-training seconds and total cost.
- F1−G: not computed (no F1). G−F0 is reported only where applicable.

Every contrast reports S3, S4, S5, S6 and S1+S2, never only the stop rate.

## A3 (DEV pilot: contexts a and c × 3 tasks × 2 blocks = 12 M2 branches, paired with F0 on the same prefix)
- Assert that M2 and F0 produce identical intake verdict trajectories.
- If the code shows that F0 has no post-intake check able to see the drift, report "F0 and M2 scope overlap" and
  do not manufacture a difference.

## Statistics
- Per task, first average the paired differences over the seed blocks. Then report the task-level mean, the
  per-task direction, and a leave-one-task-out range.
- With 3 DEV tasks, the 95% interval is reported as a cluster bootstrap (task resampling, 10,000 draws, seed 1729).
  It is explicitly unstable and only descriptive.
- Rates use exact Clopper-Pearson 95% intervals. A zero event count is reported with its upper bound and never
  as "risk = 0".
- No significance claim and no non-inferiority claim. The 5 pp / 20% tolerances are unsigned and therefore inactive.
- Claims use the protocol labels: QC_PASS/FAIL, RESULT_USABLE/NOT_USABLE, and
  CLAIM_SUPPORTED_STRONG / DIRECTIONAL / NOT_SUPPORTED / INDETERMINATE. A DEV pilot can be at most DIRECTIONAL
  and is labelled pilot.

## Rerun policy
- Runs are never overwritten.
- An API or infrastructure failure is kept as S8. One retry of the same cell is allowed only under a new run_id
  suffix `__rerun1`. Both records are kept, and the first attempt stays in the denominator.
- No seed is added because of interim results.
- A code fix creates a new implementation SHA and a new batch. It never overwrites a previous batch.
