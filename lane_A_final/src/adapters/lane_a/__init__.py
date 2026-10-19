"""Lane A adapter for RTS-TAI-PARALLEL-v4.0.

Runtime-only adapter around the F0 source tree (base_commit b9dbebe). Nothing in
the base clone is edited: method variants, read-only observers and fault
injections are installed as runtime wrappers selected by configuration, so every
variant can be run independently from the same immutable source files.
"""

ADAPTER_VERSION = "laneA-adapter-1.0.0"
