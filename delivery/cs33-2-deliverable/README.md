# CS33-2 Deliverable

This is the cleaned deliverable package. It keeps the core runtime code, sample datasets, and the local demo page needed to run the project.

## Included Contents

- `app/`: core multi-agent workflow code
- `data/`: built-in sample datasets
- `demo_web/`: local demo frontend and server
- `main.py`: natural-language task entry point
- `run_multiagent.py`: preset dataset runner
- `visualize_json_logs.py`: workflow log visualization script
- `artifacts/`: runtime output directory, pre-created as empty folders

## Removed Contents

- External projects: `AutoKaggle/`, `_external/`
- Development documents and planning materials: `docs/`, `openspec/`, `phase_plans/`, `benchmarks/`
- Test code and test artifacts: `tests/`, `test_*.py`, `.pytest_cache/`
- Historical outputs, reports, logs, caches, and temp files: `outputs/`, `output/`, `logs/`, existing `artifacts/` contents, `workspaces/`, `tmp/`
- Deployment-related files: `Dockerfile`, `docker-compose.yml`, `docker-entrypoint.sh`

## Environment Setup

1. Create a Python 3.13 virtual environment
2. Install dependencies
3. Copy `.env.example` to `.env` and fill in valid API keys

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Running the Project

Main workflow example:

```bash
python main.py \
  --task "predict customer churn" \
  --dataset data/bank_churn \
  --target Exited \
  --yes
```

Preset dataset example:

```bash
python run_multiagent.py bank_churn
```

Local demo page:

```bash
python demo_web/server.py --port 8765
```

Then open `http://127.0.0.1:8765`.

## Output Notes

Intermediate artifacts, predictions, reports, and logs generated during execution are written to `artifacts/`.
