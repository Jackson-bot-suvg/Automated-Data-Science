"""Location of the lane-A package, base clone, work area and private oracle area.

Defaults assume the layout
    <LANE_ROOT>/lane_A_final   (this package)
    <LANE_ROOT>/base           (git clone of the F0 repository at BASE_COMMIT)
    <LANE_ROOT>/work           (agent-visible task packages + run workspaces)
    <LANE_ROOT>/private        (oracle_gold/, private_test_labels/; never visible to agents)
Every location can be overridden with an environment variable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
LANE_ROOT = Path(os.environ.get("LANE_A_ROOT", PKG.parent)).resolve()
BASE = Path(os.environ.get("LANE_A_BASE", LANE_ROOT / "base")).resolve()
DATA_ROOT = Path(os.environ.get("DATA_ROOT", BASE / "data")).resolve()
WORK = Path(os.environ.get("LANE_A_WORK", LANE_ROOT / "work")).resolve()
PRIVATE = Path(os.environ.get("LANE_A_PRIVATE", LANE_ROOT / "private")).resolve()
ADAPTERS = PKG / "src" / "adapters"
ORACLE_SRC = PKG / "src"
SPECS = PKG / "cases" / "mutation_specs"

BASE_COMMIT = "b9dbebed269f62767e149b93a02f71051efc1027"
BASE_REPO = "https://github.com/Jackson-bot-suvg/Automated-Data-Science.git"
SPLIT_SEED = 1729
DEV_SEED_BLOCKS = (1101, 1102)
CONF_SEED_BLOCKS = (2101, 2102, 2103)
TRANSFER_SEED_BLOCKS = (3101, 3102, 3103)

for p in (str(SPECS), str(ORACLE_SRC), str(ADAPTERS)):
    if p not in sys.path:
        sys.path.insert(0, p)
