# A_claim_ledger — lane A (DEV pilot; G0 = FAIL; nothing here is a CONF confirmation)

Every row below can be traced to `results/claims.csv`: same claim_id prefix, the per-context and pooled estimates,
and the counter-example pair_ids. All numbers are task-level means of same-prefix paired differences over 3 DEV base
tasks × 2 seed blocks. Bootstrap intervals over 3 tasks are unstable and descriptive only.

| Claim | QC | Usable | Support | Evidence | Counter-examples / limits |
|---|---|---|---|---|---|
| **A-C1 (intake)** — F0's verdict execution changes the wrong action at intake (vs M3 shadow) | QC_PASS (DEV) | yes | **CLAIM_DIRECTIONAL (DEV pilot)** | (a): Δunsafe −0.5, ΔS3 +1.0, Δwrong-fit seconds −8.1; (b): ΔS3 +1.0 (S7→S3). A1: F01/F03 6/6 pairs change terminal | Steel F01: correct stop for the wrong reason (fuzzy hint picked a feature column). Spaceship (a) under M3 was S7, not S5, so the per-task Δunsafe is 0. |
| **A-C1 (post-intake)** — the same holds after intake | QC_PASS (DEV) | yes | **CLAIM_NOT_SUPPORTED (no difference)** | (c)(d): F0 = M3 in 12/12 pairs (S5×10, S7×2). A1: F05/F10/B01 9/9 pairs identical | Ghouls (c) S7 comes from one-hot encoding of the drift column (an execution failure), not from detection |
| **A-C2** — post-intake checking adds protection over the identical intake-only M2 | QC_PASS (DEV) | yes | **CLAIM_NOT_SUPPORTED** — "F0 and M2 scope overlap" | A3: 12/12 pairs identical ((a) S3×6; (c) S5×4, S7×2). Code: phase gates validate the *current* target_spec, never C_read | No counter-example in favour of F0 |
| **A-C3a** — F0's default retry preserves more correct deliveries than M5 block-only | QC_PASS (DEV) | yes | **CLAIM_NOT_SUPPORTED** | S1+S2 = 0 in every fault context for every method. F0 costs +4 calls/pair over M5 at intake | Intake retry is deterministic (reader re-reads its own target_spec), so it cannot repair |
| **A-C3b** — G (no diagnostics) differs from F0 | QC_PASS (DEV) | yes | **CLAIM_NOT_SUPPORTED (no difference)** | (c)(d): 12/12 identical. G is pre-registered N/A in (a)(b) | F0's diagnostic retry was exercised only in Ghouls (c), and only for execution failures |
| **A-C4** — early gates beat end-only checking with the same rules (M4) | QC_PASS (DEV) | yes | **mixed**: DIRECTIONAL in (a); **reverse in (c)** | (a): Δunsafe −0.5, Δwrong-fit seconds −8.9 (cleaning erases the leak evidence before the end). (c): M4 stops 6/6, F0 0/6 (Δunsafe +0.67) | (c) shows that F0 never re-validates contract invariants after intake; M4's end check does |
| **A1-binding** — the approved contract constrains the real fit/serialization | QC_PASS (DEV) | yes | **only at intake** | Failure sources (event-derived): initial contract error stopped (F03); stage drift delivered (F05); consumption decoupling (B01: synthesized ids in the output column); output mismatch (F10) | Legal alias B01 is never blocked yet always corrupted (S6/S6/S5). Steel B01 is falsely blocked (S4) |
| **A0 reproduction** — manuscript 24/24 valid clean outputs | QC_PASS | yes | **reproduced (24/24 S1)** with a config change (deepseek-flash) | T-A0a | Not deterministic across repeats (SD 0.0002–0.0013). Spaceship outputs are probabilities, not the required True/False labels (strict-domain sensitivity: 29/29 would be S6) |
| **A0 ablation** — "No reviewer gates 0/4" is interface breakage | QC_PASS | yes | **outcome reproduced, timing not** | Naive gate removal 0/2 (recursion limit); correct M0 2/2 | Old ablation code is missing, so the exact mechanism is unverifiable (8.40 s vs ~172 s) |

## Boundaries that could not be observed reliably
- Anything at CONF level, and the ML family (no genuine DEV multilabel task).
- The manuscript's 16 fault packages and its ablation code (not in the repository).
- The MiniMax-M2.7 backend (end-of-life).
- Content-level target drift under an unchanged column name (y fingerprints are recorded but not adjudicated).
- Wall-clock comparisons (4 concurrent runs; descriptive only).
- In (c)(d) the wrong fit precedes the fork point by construction: F0 has no pre-fit gate.
