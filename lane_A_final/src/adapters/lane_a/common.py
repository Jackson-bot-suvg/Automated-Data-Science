"""Shared helpers: hashing, canonical JSON, and the per-run event logger."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=str)
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return str(obj)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=_json_default, separators=(",", ":"))


def sha256_json(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def normalize_paths(obj: Any, replacements: dict[str, str]) -> Any:
    """Replace run-specific path prefixes so forked copies hash identically."""

    text = canonical_json(obj)
    for old, new in sorted(replacements.items(), key=lambda kv: -len(kv[0])):
        if old:
            text = text.replace(old, new)
    return json.loads(text)


def series_fingerprint(values: pd.Series | pd.DataFrame) -> dict[str, Any]:
    """Order-insensitive fingerprint of a target vector (robust to row filtering)."""

    frame = values.to_frame() if isinstance(values, pd.Series) else values
    out: dict[str, Any] = {"columns": [str(c) for c in frame.columns], "n": int(len(frame))}
    per_col = {}
    for col in frame.columns:
        s = frame[col]
        info: dict[str, Any] = {"dtype": str(s.dtype), "n_null": int(s.isna().sum())}
        nonnull = s.dropna()
        if pd.api.types.is_numeric_dtype(nonnull) and not pd.api.types.is_bool_dtype(nonnull) and nonnull.nunique() > 20:
            q = np.quantile(nonnull.astype(float), [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]) if len(nonnull) else []
            info["kind"] = "continuous"
            info["quantiles"] = [round(float(x), 6) for x in q]
            info["mean"] = round(float(nonnull.astype(float).mean()), 6) if len(nonnull) else None
        else:
            vc = nonnull.astype(str).value_counts().sort_index()
            info["kind"] = "discrete"
            info["value_counts"] = {str(k): int(v) for k, v in vc.items()}
        per_col[str(col)] = info
    out["per_column"] = per_col
    out["digest"] = sha256_json(per_col)
    return out


def ordered_values_digest(values: pd.Series | pd.DataFrame) -> str:
    frame = values.to_frame() if isinstance(values, pd.Series) else values
    return sha256_text(frame.astype(str).to_csv(index=False))


def pairing_digest(frame: pd.DataFrame, id_columns: list[str], output_columns: list[str]) -> str:
    """Digest of the ID -> prediction pairing, independent of row order."""

    cols = [c for c in id_columns if c in frame.columns] + [c for c in output_columns if c in frame.columns]
    if not cols:
        return ""
    rows = frame[cols].astype(str).agg("\x1f".join, axis=1).sort_values()
    return sha256_text("\n".join(rows.tolist()))


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


# ---------------------------------------------------------------------------
# Event logger (events.jsonl; monotonic sequence; one file per run)
# ---------------------------------------------------------------------------

EVENT_FIELDS = (
    "event_id",
    "run_id",
    "monotonic_seq",
    "timestamp",
    "phase",
    "actor",
    "event_type",
    "state_hash",
    "candidate_hash",
    "contract_digest",
    "source_fields",
    "verdict",
    "next_action",
    "actual_fit_target_hash",
    "artifact_mapping_digest",
    "error_subtype",
)


class EventLogger:
    def __init__(self, path: Path, run_id: str, *, start_seq: int = 0) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self._counter = itertools.count(start_seq + 1)
        self._lock = threading.Lock()
        self._t0 = time.monotonic()

    def emit(self, event_type: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            seq = next(self._counter)
            record: dict[str, Any] = {key: "" for key in EVENT_FIELDS}
            record.update(
                {
                    "event_id": uuid.uuid4().hex,
                    "run_id": self.run_id,
                    "monotonic_seq": seq,
                    "timestamp": utc_now(),
                    "elapsed_s": round(time.monotonic() - self._t0, 4),
                    "event_type": event_type,
                }
            )
            payload = fields.pop("payload", None)
            for key, value in fields.items():
                record[key] = value
            if payload is not None:
                record["payload"] = payload
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(canonical_json(record) + "\n")
            return record


def read_events(path: str | Path) -> list[dict[str, Any]]:
    events = []
    p = Path(path)
    if not p.exists():
        return events
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default, sort_keys=True), encoding="utf-8")
    os.replace(tmp, p)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
