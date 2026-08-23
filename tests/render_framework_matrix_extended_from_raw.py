#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_framework_matrix_extended import (
    DATASETS,
    FRAMEWORKS,
    TMP_DIR,
    aggregate_runs,
    build_output_paths,
    load_jsonl,
    parse_usage_log,
    render_markdown,
    save_json,
    utc_now,
)
from app.validation_matrix import ALL_VALIDATION_DATASETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the extended framework matrix report from an existing raw JSONL log without rerunning tasks."
    )
    parser.add_argument("--output-stem", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--score-mode", choices=("mean", "top"), default="mean")
    parser.add_argument("--raw-jsonl", default="")
    return parser.parse_args()


def recover_missing_row(dataset: str, framework: str, repeat: int) -> dict[str, Any]:
    usage_log = TMP_DIR / "usage_logs" / f"{framework}-{dataset}-r{repeat}.jsonl"
    usage = parse_usage_log(usage_log)
    error = ""
    if usage_log.exists():
        try:
            rows = [json.loads(line) for line in usage_log.read_text(encoding="utf-8").splitlines() if line.strip()]
            if rows:
                last = rows[-1]
                error = str(last.get("error") or "")
        except Exception:
            error = ""
    spec = DATASETS[dataset]
    return {
        "framework": framework,
        "status": "FAIL",
        "metric_name": spec.metric_name,
        "metric_value": None,
        "runtime_sec": None,
        "valid_output": False,
        "prediction_file": "",
        "run_dir": "",
        "stdout_tail": "",
        "stderr_tail": "",
        "usage_log": str(usage_log),
        **usage,
        "error": error or "Result row missing from raw matrix log.",
        "dataset": dataset,
        "repeat": repeat,
        "generated_at": utc_now(),
        "recovered_from_logs": True,
    }


def main() -> int:
    args = parse_args()
    dataset_specs = [DATASETS[spec.key] for spec in ALL_VALIDATION_DATASETS]
    md_path, json_path, raw_jsonl = build_output_paths(args.output_stem)
    if args.raw_jsonl:
        raw_jsonl = Path(args.raw_jsonl).resolve()
    rows = load_jsonl(raw_jsonl)

    expected = {
        (spec.key, framework, repeat)
        for spec in dataset_specs
        for framework in FRAMEWORKS
        for repeat in range(1, args.repeats + 1)
    }
    seen = {
        (str(row.get("dataset") or ""), str(row.get("framework") or ""), int(row.get("repeat") or 0))
        for row in rows
    }
    missing = sorted(expected - seen)
    recovered_rows = [recover_missing_row(dataset, framework, repeat) for dataset, framework, repeat in missing]
    all_rows = rows + recovered_rows

    aggregated: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in dataset_specs:
        aggregated[spec.key] = {}
        for framework in FRAMEWORKS:
            framework_runs = [
                row
                for row in all_rows
                if row.get("dataset") == spec.key and row.get("framework") == framework
            ]
            aggregated[spec.key][framework] = aggregate_runs(
                spec.key,
                framework,
                sorted(framework_runs, key=lambda item: item["repeat"]),
            )

    score_field = "top_score" if args.score_mode == "top" else "score_mean"
    score_label = "Top Score" if args.score_mode == "top" else "Score Mean"
    payload = {
        "generated_at": utc_now(),
        "datasets": [spec.key for spec in dataset_specs],
        "frameworks": list(FRAMEWORKS),
        "repeats": args.repeats,
        "score_mode": args.score_mode,
        "aggregated": aggregated,
        "raw_results_count": len(rows),
        "recovered_rows_count": len(recovered_rows),
        "recovered_rows": recovered_rows,
        "raw_results_path": str(raw_jsonl),
    }
    save_json(json_path, payload)
    md_text = render_markdown(
        "# Framework Matrix Extended",
        dataset_specs,
        aggregated,
        payload["generated_at"],
        score_field=score_field,
        score_label=score_label,
    )
    md_text += "\n## Rendering Notes\n\n"
    md_text += f"- Raw run rows found: `{len(rows)}`\n"
    md_text += f"- Recovered rows appended from existing logs: `{len(recovered_rows)}`\n"
    if recovered_rows:
        for row in recovered_rows:
            md_text += (
                f"- Recovered `{row['framework']}` / `{row['dataset']}` / `r{row['repeat']}` as `FAIL` "
                f"from existing usage logs only.\n"
            )
    md_path.write_text(md_text, encoding="utf-8")
    print(f"[RENDER-EXTENDED] markdown={md_path}")
    print(f"[RENDER-EXTENDED] json={json_path}")
    print(f"[RENDER-EXTENDED] recovered_rows={len(recovered_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
