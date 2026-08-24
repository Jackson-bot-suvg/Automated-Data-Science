# 26S1_CS33-2

> For the system architecture, engineering trade-offs, evaluation evidence, and productionization roadmap, see the [Industrial Architecture and Engineering Overview](INDUSTRIAL_OVERVIEW.md).

## Quick Start

This project supports two runtime modes:
- local virtual environment (`.venv`)
- Docker / Docker Compose

Before running, copy `.env.example` to `.env` and fill in valid API keys.

```bash
cp .env.example .env
```

Windows PowerShell:
```powershell
Copy-Item .env.example .env
```

## Run With venv (recommended for development)

### 1) Create and activate virtual environment

macOS / Linux:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

If `python3.13` is not found, check available versions:
```bash
python3 --version
```

Windows PowerShell:
```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows CMD:
```cmd
py -3.13 -m venv .venv
.venv\Scripts\activate.bat
```

### 2) Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3) Start application

```bash
python main.py --task "predict customer churn" --dataset data/bank_churn
```

`main.py` accepts a natural-language task and a dataset directory or CSV file.
It first runs Reader and Background Reviewer, shows the inferred task
understanding, asks for confirmation, then runs the full workflow to the
business report. Useful options:

```bash
python main.py \
  --task "predict customer churn" \
  --dataset data/bank_churn \
  --target "customer churn" \
  --yes
```

- `--target` is optional and may be either a column name or a natural-language hint.
- `--target` may also describe a multi-target task. When you provide exact target
  column names in the hint, the CLI preserves the full target scope instead of
  collapsing it to a single column.
- `--yes` skips the confirmation prompt after the understanding is approved.
- The CLI discovers `train.csv`, optional `test.csv`, optional
  `sample_submission.csv` / `submission.csv`, and optional overview text files
  inside dataset directories.
- For Kaggle-style multi-target datasets, Reader now prefers structured
  evidence from `sample_submission.csv` and the train/test column difference
  before falling back to fuzzy natural-language matching.
- Reader stores the authoritative target contract in `state.target_spec`,
  including resolved target columns, submission schema, metric hints, and
  supporting evidence.
- When multiple target columns are resolved, the confirmation step shows
  `Target columns` and the workflow continues without forcing you to pick only
  one target.
- Legacy `target_column` and `target_columns` fields are still written as
  compatibility mirrors for downstream phases that have not fully migrated yet.
- If no `test.csv` exists, the workflow continues with train-only validation and
  records that limitation in the reports.

The older Titanic technical-only demo remains available:

```bash
python run_multiagent.py titanic
```

## Run With Docker

Compose runs `main.py` with host-mounted `data/` and `artifacts/` (same paths as local runs). Ensure `.env` exists at the repo root (Compose loads it via `env_file`).

Default task/dataset match the README example; override with environment variables:

- `MAIN_TASK` — natural-language `--task` (default: `predict customer churn`)
- `MAIN_DATASET` — `--dataset` path under `/app` (default: `data/bank_churn`)
- `MAIN_ENV_FILE` — passed to `--env-file` (default: `.env`)
- `MAIN_YES` — set to `1` to add `--yes` (default); set to `0` for interactive confirmation (use a TTY, e.g. `docker compose run --rm -it app`)

### 1) Build and run with Compose

```bash
docker compose up --build
```

### 2) Custom run (one-off)

```bash
docker compose run --rm app python main.py --task "your task" --dataset data/titanic_autokaggle --yes
```

### 3) Run in background (optional)

```bash
docker compose up --build -d
```

### 4) Stop services

```bash
docker compose down
```

## Notes

- Python version requirement is `>=3.13` (see `pyproject.toml`).
- `docker-compose.yml` injects env vars from `.env`.
- Do not commit `.env` with real credentials.
- If PowerShell blocks activation scripts, run:
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
