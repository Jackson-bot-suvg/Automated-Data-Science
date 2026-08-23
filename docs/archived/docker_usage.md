# Docker Usage Guide

This document defines the standard Docker workflow for this project.

The source of truth is:

- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`

If there is any conflict, follow these files.

## 1. Prerequisites

- Docker Desktop is installed and running.
- Docker Compose V2 is available (`docker compose version`).
- A valid `.env` file exists in the project root.
- Run all commands from the project root directory.
- Windows teammates can use PowerShell or CMD with the same Docker commands.

## 2. Build Rules

Build image:

```bash
docker build -t cs33-app .
```

Verify image:

```bash
docker images cs33-app
```

## 3. Runtime Commands

Run directly with Docker:

```bash
docker run --rm --env-file .env cs33-app
```

Run with Compose (foreground):

```bash
docker compose up --build
```

Run with Compose (background):

```bash
docker compose up --build -d
```

Stop Compose services:

```bash
docker compose down
```

## 4. Logs and Status

Check container status:

```bash
docker ps -a
```

Follow service logs:

```bash
docker compose logs -f app
```

## 5. Troubleshooting

- `docker: command not found`
  - Install Docker Desktop and restart terminal.
  - On Windows, make sure Docker Desktop is started before opening a new terminal.
- No output for a long time
  - Check logs with `docker compose logs -f app`.
  - Some flows are API-bound and can be slow without frequent console output.
- Build fails due to dependency cache
  - Rebuild without cache:
    ```bash
    docker build --no-cache -t cs33-app .
    ```
- Environment variable issues
  - Ensure `.env` exists and required keys are filled.
  - On Windows, confirm the file name is exactly `.env` (not `.env.txt`).

## 6. Security Rules

- Never commit real secrets in `.env`.
- Do not expose ports unless required by the service design.
- Keep `.dockerignore` updated to exclude local artifacts and sensitive files.

## 7. 2026-04 Docker Adaptation Summary

This section summarizes the recent Docker changes made to ensure `main.py` runs correctly in the current environment.

### 7.1 Dockerfile updates

- Base image remains `python:3.13-slim`.
- Added runtime env:
  - `PYTHONDONTWRITEBYTECODE=1`
  - `PYTHONUNBUFFERED=1`
  - `MPLCONFIGDIR=/tmp/mpl-cache`
- Installed `libgomp1` to support `scikit-learn` / `xgboost` runtime on slim images.
- Changed container startup from fixed `CMD ["python", "main.py"]` to:
  - `ENTRYPOINT ["/app/docker-entrypoint.sh"]`

### 7.2 Entrypoint behavior (`docker-entrypoint.sh`)

- Default behavior (no extra args): runs `main.py` with env-driven CLI args:
  - `--task` from `MAIN_TASK`
  - `--dataset` from `MAIN_DATASET`
  - `--env-file` from `MAIN_ENV_FILE`
  - optional `--yes` controlled by `MAIN_YES`
- Override behavior: if explicit command args are provided, entrypoint executes them directly.
  - Example:
    ```bash
    docker compose run --rm app python main.py --task "..." --dataset data/bank_churn --yes
    ```

### 7.3 Compose updates (`docker-compose.yml`)

- Added env defaults for `main.py`:
  - `MAIN_TASK`, `MAIN_DATASET`, `MAIN_ENV_FILE`, `MAIN_YES`
- Kept `.env` injection via `env_file`.
- Added/confirmed host mounts:
  - `./data:/app/data:ro`
  - `./artifacts:/app/artifacts`
- Enabled interactive mode support:
  - `stdin_open: true`
  - `tty: true`

### 7.4 Artifact path behavior in Docker

- New runs now write into per-run directories under:
  - `artifacts/runs/<run_label>/...`
- `run_label` is human-readable (`dataset + timestamp + short_run_id`).
- Business report may not appear at the legacy fixed path first; check:
  - `artifacts/runs/<run_label>/reports/business_report.md`
- Run metadata index is appended to:
  - `artifacts/run_index.jsonl`

### 7.5 Quick validation checklist

1. Start Docker Desktop and ensure daemon is running.
2. Run:
   ```bash
   docker compose up --build
   ```
3. Verify outputs exist on host under mounted `artifacts/`:
   - `artifacts/runs/<run_label>/reports/technical_report.md`
   - `artifacts/runs/<run_label>/reports/business_report.md`
   - `artifacts/runs/<run_label>/figures/*`
