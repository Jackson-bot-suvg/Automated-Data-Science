#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${LANE_D_PYTHON:-python3}"
case "${1:---smoke}" in
  --prepare) "$PYTHON" "$HERE/scripts/prepare.py"; "$PYTHON" "$HERE/scripts/configure.py" ;;
  --smoke) "$PYTHON" "$HERE/scripts/smoke.py" ;;
  --dry-run) "$PYTHON" "$HERE/scripts/run.py" --dry-run ;;
  --pilot) "$PYTHON" "$HERE/scripts/run.py" --pilot ;;
  --resume) "$PYTHON" "$HERE/scripts/run.py" --resume ;;
  --rebuild) "$PYTHON" "$HERE/scripts/evaluate.py"; "$PYTHON" "$HERE/scripts/analyze.py"; "$PYTHON" "$HERE/scripts/make_paper_assets.py" ;;
  *) echo 'Usage: reproduce.sh --prepare|--smoke|--dry-run|--pilot|--resume|--rebuild'; exit 2 ;;
esac
