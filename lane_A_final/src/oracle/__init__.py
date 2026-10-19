"""Lane-A offline oracle (evaluator-only; reads private gold and private test labels)."""

ORACLE_VERSION = "laneA-oracle-1.0.2"
# 1.0.1 -> 1.0.2 (before any experimental evaluation): descriptive after-fork fields added
# (fork_gate_verdict/next_action, wrong_fits_after_fork); S1-S8 rules unchanged.
# 1.0 -> 1.0.1 (before any experimental evaluation): id columns are cast to str before the
# private-label merge (numeric ids raised a dtype merge error; found by test_oracle.py).
