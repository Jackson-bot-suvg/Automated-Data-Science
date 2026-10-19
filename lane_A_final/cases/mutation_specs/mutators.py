"""Lane-A file-level mutators and their independent verifiers (laneA-mutators-1.0).

Each mutator reads the clean package (C00) and writes a new package directory; it
never edits C00.  Each verifier re-reads the produced files from disk and checks
that the declared mutation actually took effect (and nothing else changed).
Runtime injections F05/F10 live in src/adapters/lane_a/injections.py and are
verified from the run's event log (verify_runtime_injection).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

MUTATORS_VERSION = "laneA-mutators-1.0"
OVERVIEW_FILES = ("overview.txt", "data_description.txt")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def package_digest(pkg: Path, *, q: str = "", hint: str = "", metric_hint: str = "") -> str:
    """visible_evidence_hash: every agent-visible file + the task text and hints."""

    files = sorted(p for p in Path(pkg).iterdir() if p.is_file())
    payload = {
        "files": {p.name: sha256_file(p) for p in files},
        "q": q,
        "H": {"target_hint": hint, "metric_hint": metric_hint},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _copy_package(src: Path, dst: Path) -> None:
    if dst.exists():
        raise FileExistsError(f"refusing to overwrite package {dst}")
    shutil.copytree(src, dst)


# ---------------------------------------------------------------------------
# B01 legal output alias with explicit mapping
# ---------------------------------------------------------------------------


def alias_map(target_columns: list[str], alias: str | None) -> dict[str, str]:
    if len(target_columns) == 1:
        return {target_columns[0]: alias}
    return {col: f"{col}_probability" for col in target_columns}


def mapping_note(amap: dict[str, str]) -> str:
    lines = [
        "SUBMISSION FORMAT NOTE (authoritative for this task package):",
        "The submission file uses renamed prediction columns. Each submission column below must contain",
        "the model's prediction for the corresponding training target column, one row per test id:",
    ]
    for target, alias in amap.items():
        lines.append(f"  - submission column `{alias}`  <-  predicted value of training target `{target}`")
    lines.append("The training target columns themselves are unchanged; only the submission header differs.")
    return "\n".join(lines) + "\n\n"


def make_b01(c00: Path, dst: Path, *, target_columns: list[str], alias: str | None) -> dict[str, Any]:
    _copy_package(c00, dst)
    amap = alias_map(target_columns, alias)
    sub = pd.read_csv(dst / "sample_submission.csv", dtype=str, keep_default_na=False)
    sub = sub.rename(columns=amap)
    sub.to_csv(dst / "sample_submission.csv", index=False)
    changed = ["sample_submission.csv"]
    note = mapping_note(amap)
    for name in OVERVIEW_FILES:
        p = dst / name
        if p.exists():
            p.write_text(note + p.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
            changed.append(name)
    return {"changed_paths": changed, "params": {"alias_map": amap}}


def verify_b01(c00: Path, pkg: Path, *, target_columns: list[str], alias: str | None) -> dict[str, Any]:
    amap = alias_map(target_columns, alias)
    errors = []
    sub0 = pd.read_csv(c00 / "sample_submission.csv", dtype=str, keep_default_na=False)
    sub1 = pd.read_csv(pkg / "sample_submission.csv", dtype=str, keep_default_na=False)
    expected_cols = [amap.get(c, c) for c in sub0.columns]
    if list(sub1.columns) != expected_cols:
        errors.append(f"submission header {list(sub1.columns)} != {expected_cols}")
    if not sub0.values.tolist() == sub1.values.tolist():
        errors.append("submission body changed")
    for t in target_columns:
        if t in sub1.columns:
            errors.append(f"target name {t} still in submission header")
    for name in ("train.csv", "test.csv"):
        if sha256_file(c00 / name) != sha256_file(pkg / name):
            errors.append(f"{name} changed")
    note_seen = False
    for name in OVERVIEW_FILES:
        p = pkg / name
        if p.exists():
            head = p.read_text(encoding="utf-8", errors="ignore")[:1500]
            if all(f"`{a}`" in head for a in amap.values()):
                note_seen = True
    if not note_seen:
        errors.append("mapping note not within the first 1500 chars of an overview file (reader reads first 6000)")
    return {"ok": not errors, "errors": errors}


# ---------------------------------------------------------------------------
# F01 training target missing
# ---------------------------------------------------------------------------


def make_f01(c00: Path, dst: Path, *, target_columns: list[str]) -> dict[str, Any]:
    _copy_package(c00, dst)
    train = pd.read_csv(dst / "train.csv", dtype=str, keep_default_na=False)
    train = train.drop(columns=[c for c in target_columns if c in train.columns])
    train.to_csv(dst / "train.csv", index=False)
    return {"changed_paths": ["train.csv"], "params": {"dropped": list(target_columns)}}


def verify_f01(c00: Path, pkg: Path, *, target_columns: list[str]) -> dict[str, Any]:
    errors = []
    t0 = pd.read_csv(c00 / "train.csv", dtype=str, keep_default_na=False)
    t1 = pd.read_csv(pkg / "train.csv", dtype=str, keep_default_na=False)
    present = [c for c in target_columns if c in t1.columns]
    if present:
        errors.append(f"targets still present: {present}")
    rest = [c for c in t0.columns if c not in target_columns]
    if list(t1.columns) != rest:
        errors.append("non-target columns changed")
    elif not t0[rest].equals(t1[rest]):
        errors.append("non-target values changed")
    for name in ("test.csv", "sample_submission.csv"):
        if sha256_file(c00 / name) != sha256_file(pkg / name):
            errors.append(f"{name} changed")
    return {"ok": not errors, "errors": errors}


# ---------------------------------------------------------------------------
# F03 misleading explicit target hint (files identical to C00; H changes)
# ---------------------------------------------------------------------------


def make_f03(c00: Path, dst: Path, *, wrong_hint: str) -> dict[str, Any]:
    _copy_package(c00, dst)
    return {"changed_paths": ["H.target_hint"], "params": {"target_hint": wrong_hint}}


def verify_f03(c00: Path, pkg: Path, *, wrong_hint: str, target_columns: list[str]) -> dict[str, Any]:
    errors = []
    for p in sorted(c00.iterdir()):
        if p.is_file() and sha256_file(p) != sha256_file(pkg / p.name):
            errors.append(f"{p.name} changed")
    train_cols = pd.read_csv(pkg / "train.csv", nrows=0).columns.tolist()
    test_cols = pd.read_csv(pkg / "test.csv", nrows=0).columns.tolist()
    if wrong_hint in target_columns:
        errors.append("hint equals the gold target")
    if wrong_hint not in train_cols or wrong_hint not in test_cols:
        errors.append("hint column is not a feature present in both train and test")
    if not all(ch.isalnum() or ch == "_" for ch in wrong_hint):
        errors.append("hint is not column-name-like (would not become an explicit target)")
    return {"ok": not errors, "errors": errors}


# ---------------------------------------------------------------------------
# Runtime injections (F05 / F10): verified from the run's events
# ---------------------------------------------------------------------------


def verify_runtime_injection(events: list[dict[str, Any]], template_id: str, *, wrong_target: list[str] | None = None) -> dict[str, Any]:
    inj = [e for e in events if e.get("event_type") == "injection" and (e.get("payload") or {}).get("template_id") == template_id]
    if not inj:
        return {"ok": False, "errors": ["no injection event"], "applied": False}
    p = inj[0]["payload"]
    errors = []
    if template_id == "F05":
        if p.get("before_target_columns") == p.get("after_target_columns"):
            errors.append("target_columns unchanged")
        if wrong_target is not None and p.get("after_target_columns") != list(wrong_target):
            errors.append("unexpected drift target")
        fits = [e for e in events if e.get("event_type") == "fit_start" and e.get("monotonic_seq", 0) > inj[0].get("monotonic_seq", 0)]
        if fits and (fits[0].get("payload") or {}).get("y_columns") != list(p.get("after_target_columns") or []):
            errors.append("fit after injection did not consume the drifted target")
    if template_id == "F10":
        if p.get("fixed_points", 1) != 0:
            errors.append("permutation has fixed points")
        if p.get("before_pairing_digest") == p.get("after_pairing_digest"):
            errors.append("pairing digest unchanged")
    return {"ok": not errors, "errors": errors, "applied": True, "record": p}
