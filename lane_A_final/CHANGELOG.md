# CHANGELOG — lane A

## F0 (base, never edited)
- `b9dbebed269f62767e149b93a02f71051efc1027` — public release of the manuscript code base.
- F0 is run unmodified: technical-only graph and `recursion_limit=80`.

## Read-only observation layer (`src/adapters/lane_a/hooks.py`) — adapter 1.0.0
- Observes every graph node, the LLM service, `fit_and_select_model`, `predict_for_submission`, `build_submission`, `save_model_artifacts` and tool-error log records.
- **Behaviour equivalence.** Under the LLM-off deterministic replay, the unobserved F0 run and every observed variant give byte-identical submissions, phase histories, contracts and review statuses (`results/A0/version_diff.csv`).

## Method variants (`src/adapters/lane_a/methods.py`)
- M0, M0_NAIVE, M2, M3, M4, M5 and G are runtime wrappers over the same F0 check functions; see `protocol/methods.yaml`.
- No F1 batch exists.

## Fault injections and mutators
- **Runtime:** F05 and F10 (`src/adapters/lane_a/injections.py`).
- **File-level:** B01, F01 and F03 (`cases/mutation_specs/mutators.py`, laneA-mutators-1.0), each with an independent verifier.
- **specs 1.0 → 1.1** (before any experimental run): the Spaceship F05 drift column changed from VIP to CryoSleep. VIP still contained NaN after deterministic cleaning, so `fit()` raised an error instead of training on the wrong target (deviation D-09).

## Oracle (evaluator side)
- **1.0 → 1.0.1** (before evaluation): ID columns are cast to `str` before the private-label merge. This bug was found by `src/oracle/test_oracle.py`.
- **1.0.1 → 1.0.2** (before evaluation): added descriptive after-fork fields. The S1–S8 rules are unchanged.
- `scripts/evaluate.py`:
  - in-progress runs are skipped unless `--final` is given;
  - the unobserved hook-equivalence reference is scored S1 only through byte identity with its observed replay.

## Evaluator and analysis (post-run, evaluator side only; agent processes unaffected)
- `scripts/analyze.py`:
  - A1 failure-source categories are derived from events;
  - claims now carry primary and secondary estimands and counter-example pair_ids;
  - Steel holdout scores (`val_<metric>`) are read for A0;
  - pandas-3 string-dtype fixes;
  - a strict label-domain sensitivity table is added (a sensitivity only; the frozen S rules are unchanged).
- `scripts/make_report.py` counts each LLM call once. A2 prefixes are shared by their branches, so per-trajectory token
  fields double-count them.
- `scripts/package.py`:
  - the secret-scan regex now requires a token boundary (`risk-classifier` in a Kaggle URL was a false positive);
  - `model_results.json` is packaged.
- `scripts/restore_from_package.py` was added; `./reproduce.sh --rebuild` restores the run dirs from the packaged logs.

## Batches
- **DEV-pilot-batch-1** (2026-10-19):
  - Agent-side adapter tree frozen at launch (`logs/implementation_sha.json`; per-file hashes).
  - After launch, only evaluator-side files changed (`scripts/evaluate.py`, `src/oracle/*`, `scripts/analyze.py`, `scripts/make_paper_assets.py`). These files do not run inside agent processes.
