# lane_A_final — RTS-MA-DS / TAI lane A (real execution and local causality)

**Status: DEV pilot. G0 = FAIL (FREEZE_SIGNOFF.yaml unsigned; CONF/TRANSFER tasks not frozen).**
Nothing in this package is a CONF confirmation. Every A2/A3 number comes from the DEV pilot
(3 tasks × 2 seed blocks).

## Research questions (lane A)
1. Does the manuscript's frozen `target_spec` actually bind what `fit`, `predict` and serialization consume?
2. Do reviewer verdicts change the execution path?
3. From one identical candidate action, what do these do: let it pass (M3), block it (M5), apply F0's default
   retry, apply a generic retry (G), or check only at the end (M4)?
4. Does F0 add anything over a strong intake-only check (M2)?

## Run it
```bash
./reproduce.sh --smoke      # clean env, base clone b9dbebe, task packages, LLM-off equivalence runs, smoke gates (no API key needed)
./reproduce.sh --rebuild    # re-score all runs and rebuild every table and figure from the raw logs
./reproduce.sh --pilot      # re-run the DEV pilot (needs LANE_A_ENV_FILE with LLM settings; never overwrites)
python scripts/run.py --dry-run   # print the full plan matrix, N/A cells and a cost estimate
```

## Layout (protocol Part 3 §1)
| Path | Contents |
|---|---|
| `protocol/` | Shared protocol text and PDF; registries; frozen config; methods, outcomes and analysis plan; deviations |
| `env/` | `environment.lock` (exact interpreter packages), `system_deps.txt` (libomp), `runtime_info.json` |
| `src/adapters/lane_a/` | Runtime adapter: read-only observers, method gate policies, F05/F10 injections, same-prefix fork, runner |
| `src/oracle/` | Offline evaluator (S1–S8) and its unit tests |
| `cases/` | `manifest.csv` (every case with its hashes, information level and expected safe action), mutators and verifiers |
| `scripts/` | `build_tasks`, `run`, `smoke`, `evaluate`, `analyze`, `make_paper_assets`, `make_report`, `package` |
| `logs/` | Plan ledger, run index, raw events, raw LLM I/O, run results, final states, specs, fork snapshots, errors |
| `outputs/` | Artifact index; delivered submissions; predict()/serialize frames |
| `results/` | Run, pair and task level results; resource ledger; quality gates; claims; A0 audit tables |
| `paper/` | Tables, figures, event-level cases, `evidence_report.pdf` |

## Completion
- **Plan:** 1223 plan cells in total.
  - DEV: 287 cells, of which 223 were runnable and 64 were pre-registered N/A.
  - CONF: 936 cells, all `not_started` because G0 is FAIL.
- **DEV runs:** all 223 runnable cells were launched and completed, including 24 A2 prefix snapshots. Each of the
  199 terminal runs has exactly one terminal state: S1 43 / S3 50 / S4 1 / S5 67 / S6 7 / S7 31 / S8 0.
- **LLM usage:** 1186 LLM calls on `deepseek/deepseek-flash` (every response reported `deepseek-flash`), 0 API errors.
- **Quality gates:** G0 FAIL. G1–G6, A0-QC, A1-QC, A2-QC (DEV pilot) and A3-QC (DEV pilot) all PASS.
- **Reproducibility check:** `./reproduce.sh --rebuild` was run from the zip in a fresh environment. It reproduced
  `run_level.csv` cell-for-cell, and `pair_level.csv`, `task_level.csv`, `claims.csv` and the A0 table byte-for-byte.
  `./reproduce.sh --smoke` (no API key) passes.

## Main findings (DEV only)
1. **The F0 gate works at intake, and only at intake.**
   - It stops missing-target and misleading-hint packages before any fit (A1 6/6; A2 contexts a and b 12/12 S3).
   - The non-enforcing shadow (M3) turns the same packages into S5/S6/S7.
2. **After intake nothing is bound.**
   - `target_spec` is not write-protected.
   - The phase gates validate the *current* contract against itself.
   - No rule binds the y used by `fit`, the source of the output values, or the ID→prediction pairing.
   - Post-planner target drift (c), ID/prediction misalignment (d) and legal output aliases (B01) are delivered as
     S5/S6 under F0 exactly as under M3 (21/21 same-terminal pairs).
3. **No increment over intake-only M2.** F0 = M2 in 12/12 A3 pairs ("scope overlap").
4. **No repair.**
   - S1+S2 = 0 in every fault context for every method.
   - The intake retries are deterministic. They cost about 4 extra LLM calls per case compared with block-only (M5).
5. **End-only checking with the same rules (M4)** stops post-planner drift 6/6 where F0 stops 0/6. It fails at intake
   (6/6 wrong fits), because cleaning silently deletes the leaked column.
6. **A0 reproduction.**
   - 24/24 clean runs are valid (S1), and the CV scores are close to the manuscript's Table 2.
   - The scores are **not deterministic** across repeats.
   - Spaceship submissions are probabilities, although the task requires True/False labels.
   - Table 3's "No reviewer gates 0/4" is reproduced as an interface breakage (the routing depends on reviewer state).

Details: `paper/evidence_report.pdf` (page 1 answers the five lane-A questions) and `results/A_claim_ledger.md`.

## Not in this package (by design)
- Kaggle data. Rebuild it from the base commit's `data/` with `scripts/build_tasks.py`.
- Private test labels and C_gold. These are evaluator-only (`<LANE_A_ROOT>/private/`).
- API keys.
- The anonymised manuscript PDF.
