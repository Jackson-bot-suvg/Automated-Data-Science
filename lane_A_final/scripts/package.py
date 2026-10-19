#!/usr/bin/env python3
"""Collect raw evidence into the package, scan for secrets, write SHA256SUMS, build the zip.

Copied from <WORK> (per run): events.jsonl, prefix_events.jsonl, llm_calls.jsonl, run_result.json,
final_state.json, stdout.log, errors/*, hooks/*.csv, the delivered submission; per prefix:
fork_meta.json and state.json.  Never copied: agent-visible data packages (Kaggle data; rebuilt by
scripts/build_tasks.py), <PRIVATE>/ (oracle gold + private test labels), model pickles, API keys.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import PKG, PRIVATE, WORK  # noqa: E402

# token must start at a boundary (e.g. "risk-classifier" in a Kaggle URL is not a key)
SECRET_RES = [re.compile(r"(?<![A-Za-z0-9\-])sk-[A-Za-z0-9_\-]{20,}"), re.compile(r"(?<![A-Za-z0-9])lsv2_[A-Za-z0-9_]{20,}")]


def copy_raw() -> int:
    logs = PKG / "logs"
    n = 0
    for run_dir in sorted((WORK / "runs").iterdir()):
        if not run_dir.is_dir():
            continue
        rid = run_dir.name
        for name, dest in [("events.jsonl", logs / "events" / f"{rid}.jsonl"), ("prefix_events.jsonl", logs / "events" / f"{rid}.prefix.jsonl"),
                           ("llm_calls.jsonl", logs / "llm_raw" / f"{rid}.jsonl"), ("run_result.json", logs / "run_results" / f"{rid}.json"),
                           ("final_state.json", logs / "final_states" / f"{rid}.json"), ("stdout.log", logs / "stdout" / f"{rid}.log")]:
            src = run_dir / name
            if src.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                n += 1
        if (run_dir / "errors").exists():
            shutil.copytree(run_dir / "errors", logs / "errors" / rid, dirs_exist_ok=True)
        hooks = run_dir / "hooks"
        if hooks.exists():
            for f in hooks.glob("*.csv"):
                d = PKG / "outputs" / "artifacts" / rid
                d.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, d / f"hook_{f.name}")
        for mr in run_dir.rglob("model_results.json"):
            d = PKG / "outputs" / "artifacts" / rid
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mr, d / "model_results.json")
            break
        res = run_dir / "run_result.json"
        if res.exists():
            p = json.loads(res.read_text()).get("prediction_path") or ""
            if p and Path(p).exists():
                d = PKG / "outputs" / "artifacts" / rid
                d.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, d / "submission.csv")
    for spec in (WORK / "specs").glob("*.json"):
        d = logs / "specs"
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(spec, d / spec.name)
    for fork in (WORK / "forks").glob("*"):
        d = logs / "forks" / fork.name
        d.mkdir(parents=True, exist_ok=True)
        for name in ("fork_meta.json", "state.json"):
            if (fork / name).exists():
                shutil.copy2(fork / name, d / name)
    if (WORK / "orchestrator_batch1.log").exists():
        shutil.copy2(WORK / "orchestrator_batch1.log", logs / "orchestrator_batch1.log")
    return n


def scan_secrets() -> list[str]:
    hits = []
    for p in PKG.rglob("*"):
        if p.is_file() and p.suffix in {".json", ".jsonl", ".csv", ".log", ".md", ".txt", ".yaml", ".py", ".sh", ".html"}:
            text = p.read_text(errors="ignore")
            for rx in SECRET_RES:
                if rx.search(text):
                    hits.append(str(p.relative_to(PKG)))
                    break
    # private test labels must not be inside the package
    label_hashes = {hashlib.sha256(f.read_bytes()).hexdigest() for f in (PRIVATE / "private_test_labels").glob("*.csv")}
    for p in PKG.rglob("*.csv"):
        if hashlib.sha256(p.read_bytes()).hexdigest() in label_hashes:
            hits.append("PRIVATE LABEL FILE: " + str(p.relative_to(PKG)))
    return hits


def sha256sums() -> int:
    lines = []
    for p in sorted(PKG.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS" and "__pycache__" not in p.parts and p.suffix != ".zip":
            lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(PKG)}")
    (PKG / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    return len(lines)


def build_zip() -> Path:
    out = PKG.parent / "lane_A_final.zip"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(PKG.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith(".pyc"):
                zf.write(p, Path("lane_A_final") / p.relative_to(PKG))
    return out


def main() -> int:
    n = copy_raw()
    hits = scan_secrets()
    if hits:
        print("REFUSING TO PACKAGE: secrets or private labels found in", hits)
        return 1
    k = sha256sums()
    z = build_zip()
    print(f"raw files copied={n}; SHA256SUMS entries={k}; zip={z} ({z.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
