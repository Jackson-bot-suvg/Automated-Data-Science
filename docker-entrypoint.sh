#!/usr/bin/env sh
set -e
# Allow `docker compose run app python main.py ...` to bypass defaults.
if [ "$#" -gt 0 ]; then
  exec "$@"
fi

TASK="${MAIN_TASK:-predict customer churn}"
DATASET="${MAIN_DATASET:-data/bank_churn}"
ENV_FILE="${MAIN_ENV_FILE:-.env}"

if [ "${MAIN_YES:-1}" = "1" ]; then
  exec python main.py --task "$TASK" --dataset "$DATASET" --env-file "$ENV_FILE" --yes
else
  exec python main.py --task "$TASK" --dataset "$DATASET" --env-file "$ENV_FILE"
fi
