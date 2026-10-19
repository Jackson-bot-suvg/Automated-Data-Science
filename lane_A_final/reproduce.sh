#!/usr/bin/env bash
# Lane A reproduction script (RTS-TAI-PARALLEL-v4.0).  Uses only this directory + legal external sources.
#
#   ./reproduce.sh --smoke     clean env -> base clone -> task packages -> LLM-off equivalence runs -> smoke gates
#   ./reproduce.sh --rebuild   re-score every run and rebuild all tables/figures from the raw logs in logs/ + work/
#   ./reproduce.sh --pilot     re-run the DEV pilot (needs an LLM key in $LANE_A_ENV_FILE; never overwrites runs)
#
# Environment (all optional):
#   LANE_A_ROOT     parent directory holding lane_A_final/, base/, work/, private/  (default: parent of this dir)
#   DATA_ROOT       DEV data root (default: <base>/data from the base commit, which ships the 8 DEV packages)
#   LANE_A_ENV_FILE KEY=VALUE file with LLM_PROVIDERS / LLM_DEEPSEEK_* / AGENT_*_MODEL (only for --pilot)
#   PYTHON          python 3.13 interpreter (default: python3.13)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LANE_A_ROOT="${LANE_A_ROOT:-$(dirname "$HERE")}"
BASE="${LANE_A_BASE:-$LANE_A_ROOT/base}"
PYTHON="${PYTHON:-python3.13}"
VENV="$LANE_A_ROOT/.venv_laneA"
BASE_COMMIT="b9dbebed269f62767e149b93a02f71051efc1027"
MODE="${1:---smoke}"

if [ ! -d "$BASE/.git" ]; then
  git clone https://github.com/Jackson-bot-suvg/Automated-Data-Science.git "$BASE"
fi
git -C "$BASE" checkout -q "$BASE_COMMIT"
test "$(git -C "$BASE" rev-parse HEAD)" = "$BASE_COMMIT"

if [ ! -x "$VENV/bin/python" ]; then
  "$PYTHON" -m venv "$VENV"
  "$VENV/bin/python" -m pip install -q --upgrade pip
  # exact interpreter packages used for the lane-A runs (the base commit's uv.lock is stale: deviation D-10)
  grep -v '^-e ' "$HERE/env/environment.lock" > "$LANE_A_ROOT/.laneA_requirements.txt"
  "$VENV/bin/python" -m pip install -q -r "$LANE_A_ROOT/.laneA_requirements.txt"
fi
if [[ "$(uname)" == "Darwin" ]] && ! brew list libomp >/dev/null 2>&1; then
  echo "macOS: xgboost/lightgbm need OpenMP -> brew install libomp" >&2
fi
PY="$VENV/bin/python"
export LANE_A_PYTHON="$PY"

"$PY" -P "$HERE/scripts/build_tasks.py"

case "$MODE" in
  --smoke)
    "$PY" -P "$HERE/scripts/run.py" --pilot --stage A0-EQUIV --workers 4
    "$PY" -P "$HERE/scripts/smoke.py"
    ;;
  --rebuild)
    if [ ! -d "${LANE_A_WORK:-$LANE_A_ROOT/work}/runs" ]; then
      "$PY" -P "$HERE/scripts/restore_from_package.py"   # rebuild run dirs from the packaged raw logs
    fi
    "$PY" -P "$HERE/scripts/evaluate.py" --final
    "$PY" -P "$HERE/scripts/analyze.py"
    "$PY" -P "$HERE/scripts/make_paper_assets.py"
    ;;
  --pilot)
    : "${LANE_A_ENV_FILE:?set LANE_A_ENV_FILE to a KEY=VALUE file with the LLM provider settings}"
    "$PY" -P "$HERE/scripts/run.py" --pilot --stage A0 A1 A2 A3 --workers 4
    "$PY" -P "$HERE/scripts/evaluate.py" --final
    "$PY" -P "$HERE/scripts/analyze.py"
    "$PY" -P "$HERE/scripts/make_paper_assets.py"
    ;;
  *)
    echo "usage: $0 [--smoke|--rebuild|--pilot]" >&2; exit 2;;
esac
