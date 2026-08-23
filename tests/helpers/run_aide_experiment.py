#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import aide


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one AIDE experiment and emit JSON.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--code-model", required=True)
    parser.add_argument("--feedback-model", required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--num-drafts", type=int, default=1)
    parser.add_argument("--debug-prob", type=float, default=0.0)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _safe_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _count_loc(path: Path) -> int:
    if not path.exists():
        return 0
    return len(path.read_text(encoding="utf-8").splitlines())


def main() -> int:
    args = parse_args()
    output_path = Path(args.output_json).resolve()
    started = time.time()

    try:
        aide_repo_root = Path(aide.__file__).resolve().parents[1]
        os.chdir(aide_repo_root)
        exp = aide.Experiment(
            data_dir=args.data_dir,
            goal=args.goal,
            eval=args.eval,
        )
        exp.cfg.agent.code.model = args.code_model
        exp.cfg.agent.feedback.model = args.feedback_model
        exp.cfg.agent.search.num_drafts = args.num_drafts
        exp.cfg.agent.search.debug_prob = args.debug_prob
        exp.agent.acfg = exp.cfg.agent

        solution = exp.run(steps=args.steps)
        best_node = exp.journal.get_best_node(only_good=False)

        log_dir = Path(exp.cfg.log_dir)
        workspace_dir = Path(exp.cfg.workspace_dir)
        best_solution_path = log_dir / "best_solution.py"
        submission_path = workspace_dir / "working" / "submission.csv"
        tree_plot_path = log_dir / "tree_plot.html"

        payload = {
            "status": "PASS",
            "runtime_sec": round(time.time() - started, 4),
            "metric_value": solution.valid_metric,
            "log_dir": str(log_dir),
            "workspace_dir": str(workspace_dir),
            "best_solution_path": str(best_solution_path),
            "best_solution_loc": _count_loc(best_solution_path),
            "submission_path": str(submission_path),
            "submission_exists": submission_path.exists(),
            "tree_plot_path": str(tree_plot_path),
            "tree_plot_exists": tree_plot_path.exists(),
            "steps_requested": args.steps,
            "nodes_total": len(exp.journal.nodes),
            "good_nodes": len(exp.journal.good_nodes),
            "buggy_nodes": len(exp.journal.buggy_nodes),
            "best_node_analysis": getattr(best_node, "analysis", ""),
            "best_node_buggy": getattr(best_node, "is_buggy", False),
        }
        _safe_write_json(output_path, payload)
        print(f"AIDE_RESULT_JSON::{output_path}")
        return 0
    except Exception as exc:
        payload = {
            "status": "FAIL",
            "runtime_sec": round(time.time() - started, 4),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        _safe_write_json(output_path, payload)
        print(f"AIDE_RESULT_JSON::{output_path}")
        print(payload["error"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
