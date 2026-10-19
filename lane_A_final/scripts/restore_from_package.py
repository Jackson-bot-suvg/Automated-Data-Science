#!/usr/bin/env python3
"""Recreate <WORK>/runs and <WORK>/specs from the packaged raw logs (for ./reproduce.sh --rebuild).

The zip ships, per run: events (+ prefix events), run_result, final_state, stdout, errors,
predict()/serialize frames, the delivered submission and model_results.json.  This script
puts them back into the run-directory layout that scripts/evaluate.py reads, rewriting the
absolute prediction path recorded at run time to the restored file.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import PKG, WORK  # noqa: E402


def main() -> int:
    logs = PKG / "logs"
    runs = WORK / "runs"
    if runs.exists() and any(runs.iterdir()):
        print(f"{runs} already populated; refusing to overwrite")
        return 2
    n = 0
    for res_file in sorted((logs / "run_results").glob("*.json")):
        rid = res_file.stem
        run_dir = runs / rid
        run_dir.mkdir(parents=True)
        for src, name in [(logs / "events" / f"{rid}.jsonl", "events.jsonl"), (logs / "events" / f"{rid}.prefix.jsonl", "prefix_events.jsonl"),
                          (logs / "final_states" / f"{rid}.json", "final_state.json"), (logs / "stdout" / f"{rid}.log", "stdout.log"),
                          (logs / "llm_raw" / f"{rid}.jsonl", "llm_calls.jsonl")]:
            if src.exists():
                shutil.copy2(src, run_dir / name)
        art = PKG / "outputs" / "artifacts" / rid
        res = json.loads(res_file.read_text())
        if art.exists():
            (run_dir / "hooks").mkdir()
            for f in art.glob("hook_*.csv"):
                shutil.copy2(f, run_dir / "hooks" / f.name[len("hook_"):])
            if (art / "model_results.json").exists():
                shutil.copy2(art / "model_results.json", run_dir / "model_results.json")
            if (art / "submission.csv").exists():
                shutil.copy2(art / "submission.csv", run_dir / "submission.csv")
                res["prediction_path"] = str(run_dir / "submission.csv")
        elif res.get("prediction_path"):
            res["prediction_path_original"] = res["prediction_path"]
            res["prediction_path"] = ""
        (run_dir / "run_result.json").write_text(json.dumps(res, indent=2))
        if (logs / "errors" / rid).exists():
            shutil.copytree(logs / "errors" / rid, run_dir / "errors")
        n += 1
    (WORK / "specs").mkdir(parents=True, exist_ok=True)
    for spec in (logs / "specs").glob("*.json"):
        shutil.copy2(spec, WORK / "specs" / spec.name)
    print(f"restored {n} run directories into {runs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
