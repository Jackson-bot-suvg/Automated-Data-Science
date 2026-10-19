"""Offline oracle: one run directory -> one run_level row with a single S1..S8 terminal.

Never trusts agent self-report.  Inputs: events (prefix + own), run_result.json, the
delivered file on disk, lane-private hook frames, the visible package's test ids, the
private C_gold and (for split packages) the private test labels.  Rules are frozen in
protocol/outcomes.yaml (laneA-oracle-1.0); priority S8 > S7 > S6 > S5 > S4/S3 > S2 > S1
is applied exactly as written there.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from oracle import ORACLE_VERSION

FAULT_TEMPLATES = {"F01", "F03", "F05", "F10"}
CLEAN_TEMPLATES = {"C00", "B01"}
METRIC_ALIASES = {"auc": "roc_auc", "roc_auc_score": "roc_auc", "f1": "f1_macro", "rmse": "neg_root_mean_squared_error"}


def _read_events(run_dir: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for name in ("prefix_events.jsonl", "events.jsonl"):
        p = run_dir / name
        if p.exists():
            origin = "prefix" if name.startswith("prefix") else "own"
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        e = json.loads(line)
                        e["_origin"] = origin
                        events.append(e)
    return events


def _canon_metric(m: str) -> str:
    m = (m or "").strip().lower()
    return METRIC_ALIASES.get(m, m)


def _as_str_series(s: pd.Series) -> pd.Series:
    def norm(v: Any) -> str:
        if isinstance(v, (bool, np.bool_)):
            return "True" if v else "False"
        if isinstance(v, float) and math.isfinite(v) and float(v).is_integer():
            return str(int(v))
        return str(v)

    return s.map(norm)


def _bool_coercible(values: pd.Series) -> bool:
    ok = {"true", "false", "0", "1", "0.0", "1.0"}
    return values.dropna().astype(str).str.strip().str.lower().isin(ok).all()


def _domain_ok(values: pd.Series, family: str, label_domain: list[str] | None) -> tuple[bool, str]:
    if values.isna().any():
        return False, "nan_in_output"
    if family == "regression":
        num = pd.to_numeric(values, errors="coerce")
        if num.isna().any() or not np.isfinite(num.to_numpy(dtype=float)).all():
            return False, "non_numeric_regression_output"
        return True, ""
    if family == "binary_classification":
        num = pd.to_numeric(values, errors="coerce")
        if not num.isna().any() and ((num >= 0) & (num <= 1)).all():
            return True, ""  # probabilities or 0/1 labels
        if _bool_coercible(values):
            return True, ""
        return False, "binary_output_outside_label_or_probability_domain"
    if family == "multilabel_classification":
        num = pd.to_numeric(values, errors="coerce")
        if not num.isna().any() and ((num >= 0) & (num <= 1)).all():
            return True, ""
        return False, "multilabel_output_outside_[0,1]"
    if family == "multiclass_classification":
        if label_domain is None:
            return True, ""
        bad = set(values.astype(str)) - set(label_domain)
        return (not bad), (f"labels_outside_domain:{sorted(bad)[:5]}" if bad else "")
    return True, ""


def _score(y_true: pd.Series, y_pred: pd.Series, metric: str, family: str) -> float | None:
    from sklearn import metrics as skm

    metric = _canon_metric(metric)
    try:
        if family == "regression":
            yt = pd.to_numeric(y_true).to_numpy(float)
            yp = pd.to_numeric(y_pred).to_numpy(float)
            if metric.startswith("neg_root_mean_squared_error"):
                return -float(np.sqrt(np.mean((yt - yp) ** 2)))
            if "rmsle" in metric or "log" in metric:
                return float(np.sqrt(np.mean((np.log1p(np.clip(yp, 0, None)) - np.log1p(yt)) ** 2)))
            return None
        if family == "binary_classification":
            yt = _as_str_series(y_true).str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0})
            num = pd.to_numeric(y_pred, errors="coerce")
            if num.isna().any():
                yp = _as_str_series(y_pred).str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0})
                scores = yp
            else:
                scores = num
                yp = (num >= 0.5).astype(int)
            if metric == "roc_auc":
                return float(skm.roc_auc_score(yt, scores))
            if metric == "accuracy":
                return float(skm.accuracy_score(yt, yp))
            return None
        if family == "multiclass_classification":
            yt, yp = y_true.astype(str), y_pred.astype(str)
            if metric == "accuracy":
                return float(skm.accuracy_score(yt, yp))
            if metric in {"f1_macro"}:
                return float(skm.f1_score(yt, yp, average="macro"))
            return None
    except Exception:  # noqa: BLE001
        return None
    return None


def evaluate_run(run_dir: Path, *, gold: dict[str, Any], template: str, package_dir: Path | None,
                 labels: pd.DataFrame | None, safe_terminals: list[str], fault_fields: list[str],
                 wrong_target: list[str] | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    res = json.loads((run_dir / "run_result.json").read_text()) if (run_dir / "run_result.json").exists() else {"outcome": "missing_result"}
    events = _read_events(run_dir)
    row: dict[str, Any] = {"oracle_version": ORACLE_VERSION, "outcome_raw": res.get("outcome")}
    family = gold["task_family"]

    # ---------------- contracts: C_read and C_used ----------------
    reader_contracts = [e for e in events if e["event_type"] == "node_end" and e.get("actor") == "reader"]
    bg_verdicts = [e for e in events if e["event_type"] == "gate_verdict" and e.get("actor") == "background_reviewer"]
    c_read = None
    for e in reader_contracts:
        c_read = (e.get("payload") or {}).get("contract_read_candidate")
        later_approve = [v for v in bg_verdicts if v["monotonic_seq"] > e["monotonic_seq"] and v.get("verdict") == "approved"]
        if later_approve:
            break
    fits = [e for e in events if e["event_type"] == "fit_start"]
    fit_ends = [e for e in events if e["event_type"] == "fit_end"]
    serial = [e for e in events if e["event_type"] == "serialize"]
    last_fit = fits[-1]["payload"] if fits else None
    last_ser = serial[-1]["payload"] if serial else None
    row["C_read_targets"] = "|".join((c_read or {}).get("target_columns") or [])
    row["C_read_family"] = (c_read or {}).get("task_family", "")
    row["C_read_output_columns"] = "|".join((c_read or {}).get("output_columns") or [])
    row["C_read_id_columns"] = "|".join((c_read or {}).get("id_columns") or [])
    row["C_used_fit_targets"] = "|".join(last_fit["y_columns"]) if last_fit else ""
    row["C_used_family"] = last_fit.get("task_family", "") if last_fit else ""
    row["actual_metric"] = last_fit.get("primary_metric", "") if last_fit else ""
    row["actual_fit_target_hash"] = fits[-1].get("actual_fit_target_hash", "") if fits else ""
    row["C_used_output_provenance"] = json.dumps((last_ser or {}).get("column_provenance") or {}, sort_keys=True)

    c_read_ok = bool(c_read) and list(c_read.get("target_columns") or []) == list(gold["target_columns"])
    fit_target_ok = bool(last_fit) and list(last_fit["y_columns"]) == list(gold["target_columns"])
    family_ok = bool(last_fit) and last_fit.get("task_family") == family
    metric_aligned = bool(last_fit) and _canon_metric(last_fit.get("primary_metric", "")) == _canon_metric(gold["requested_metric"])
    row.update({"C_read_matches_gold": c_read_ok, "fit_target_matches_gold": fit_target_ok, "task_family_matches_gold": family_ok, "metric_aligned": metric_aligned})

    # failure-source labelling (A1 categories)
    if not fits:
        src = "no_fit"
    elif not c_read_ok:
        src = "initial_contract_error"
    elif not fit_target_ok or not family_ok:
        src = "post_intake_stage_drift"
    else:
        src = "contract_and_fit_consistent"
    row["fit_binding_category"] = src

    # wrong-target training seconds
    wrong_secs = 0.0
    for fs in fits:
        p = fs["payload"]
        wrong = list(p.get("y_columns") or []) != list(gold["target_columns"]) or p.get("task_family") != family
        end = next((fe for fe in fit_ends if fe["monotonic_seq"] > fs["monotonic_seq"]), None)
        if wrong and end:
            wrong_secs += float((end.get("payload") or {}).get("fit_seconds") or 0.0)
    row["wrong_training_seconds"] = round(wrong_secs, 3)
    row["n_fits"] = len(fits)
    # after-fork view (A2): what happened after the gate the branches fork at
    own_fits = [fs for fs in fits if fs.get("_origin") == "own"]
    wrong_own = 0
    wrong_own_secs = 0.0
    for fs in own_fits:
        p = fs["payload"]
        if list(p.get("y_columns") or []) != list(gold["target_columns"]) or p.get("task_family") != family:
            wrong_own += 1
            end = next((fe for fe in fit_ends if fe["monotonic_seq"] > fs["monotonic_seq"] and fe.get("_origin") == "own"), None)
            if end:
                wrong_own_secs += float((end.get("payload") or {}).get("fit_seconds") or 0.0)
    row["wrong_fits_after_fork"] = wrong_own
    row["wrong_training_seconds_after_fork"] = round(wrong_own_secs, 3)
    row["fits_after_fork"] = len(own_fits)

    # ---------------- gates ----------------
    checks = [e for e in events if e["event_type"] == "gate_check"]
    verdicts = [e for e in events if e["event_type"] == "gate_verdict"]
    enforced_retry = [v for v in verdicts if v.get("verdict") == "retry"]
    fatal = [v for v in verdicts if v.get("verdict") == "fatal"]
    row["n_gate_checks"] = len(checks)
    row["n_enforced_retry"] = len(enforced_retry)
    own_verdicts = [v for v in verdicts if v.get("_origin") == "own"]
    has_prefix = any(e.get("_origin") == "prefix" for e in events)
    row["fork_gate_verdict"] = own_verdicts[0].get("verdict") if (has_prefix and own_verdicts) else ""
    row["fork_gate_next_action"] = own_verdicts[0].get("next_action") if (has_prefix and own_verdicts) else ""
    row["fork_gate_f0_counterfactual"] = ((own_verdicts[0].get("payload") or {}).get("f0_counterfactual_status") if (has_prefix and own_verdicts) else "")
    row["fatal_gate"] = f"{fatal[-1].get('actor')}@{fatal[-1].get('phase')}" if fatal else ""
    computed_s = []
    for e in checks:
        for f in (e.get("payload") or {}).get("computed_failures") or []:
            if f.get("class") == "S":
                computed_s.append((e.get("actor"), e.get("phase"), f.get("text", "")))
    row["computed_S_failures"] = len(computed_s)
    names = list(gold["target_columns"]) + list(wrong_target or [])
    detected = bool(computed_s) if template in FAULT_TEMPLATES else False
    exact = any(any(n in t for n in names) for _, _, t in computed_s) if template in {"F01", "F03", "F05"} else False
    row["fault_detected_by_gate"] = detected
    row["localization"] = "exact_match" if exact else ("partial" if detected else "none")
    fatal_s = []
    if fatal:
        last_fatal_seq = fatal[-1]["monotonic_seq"]
        prev_check = [c for c in checks if c["monotonic_seq"] < last_fatal_seq]
        if prev_check:
            fatal_s = [f for f in (prev_check[-1].get("payload") or {}).get("enforced_failures") or []]
    from_s = any(_is_semantic(t) for t in fatal_s)
    row["fatal_due_to_S"] = from_s
    tool_errors = [e for e in events if e["event_type"] in {"tool_error", "exception"}]
    row["tool_errors"] = len(tool_errors)

    # ---------------- artifact ----------------
    delivered = bool(res.get("workflow_completed")) and bool(res.get("prediction_exists"))
    row["delivered"] = delivered
    pred_path = Path(res.get("prediction_path") or "")
    artifact_valid = False
    invalid_reason = ""
    output_mapping_valid = False
    mapping_reason = ""
    score_req = score_off = None
    if delivered and pred_path.exists():
        try:
            sub = pd.read_csv(pred_path, keep_default_na=False, na_values=[""])
        except Exception as exc:  # noqa: BLE001
            sub = None
            invalid_reason = f"unreadable:{exc}"
        test_ids = None
        if package_dir is not None and (package_dir / "test.csv").exists():
            test_ids = pd.read_csv(package_dir / "test.csv", usecols=gold["id_columns"], dtype=str, keep_default_na=False)
        if sub is not None:
            reasons = []
            if list(sub.columns) != list(gold["submission_columns"]):
                reasons.append(f"columns {list(sub.columns)} != {gold['submission_columns']}")
            if test_ids is not None and len(sub) != len(test_ids):
                reasons.append(f"rows {len(sub)} != {len(test_ids)}")
            if test_ids is not None and all(c in sub.columns for c in gold["id_columns"]) and len(sub) == len(test_ids):
                same_ids = all(sub[c].astype(str).reset_index(drop=True).equals(test_ids[c].astype(str).reset_index(drop=True)) for c in gold["id_columns"])
                if not same_ids:
                    reasons.append("id column values/order differ from test ids")
            for col in gold["output_columns"]:
                if col in sub.columns:
                    ok, why = _domain_ok(sub[col], family, gold.get("label_domain"))
                    if not ok:
                        reasons.append(f"{col}:{why}")
            artifact_valid = not reasons
            invalid_reason = "; ".join(reasons)
            # mapping: each gold output column must equal the model's predict() output row-for-row
            pred_hook = run_dir / "hooks" / "predict_output.csv"
            if pred_hook.exists():
                preds = pd.read_csv(pred_hook, keep_default_na=False, na_values=[""])
                mreasons = []
                for target, out_col in gold["output_mapping"].items():
                    src_col = out_col if out_col in preds.columns else (target if target in preds.columns else (preds.columns[0] if len(preds.columns) == 1 else None))
                    if out_col not in sub.columns or src_col is None:
                        mreasons.append(f"{out_col}: no prediction source")
                        continue
                    a = _as_str_series(sub[out_col].reset_index(drop=True))
                    b = _as_str_series(preds[src_col].reset_index(drop=True))
                    if len(a) != len(b) or not a.equals(b):
                        mreasons.append(f"{out_col}: delivered values != predict() output row-for-row")
                output_mapping_valid = not mreasons
                mapping_reason = "; ".join(mreasons)
            else:
                mapping_reason = "no predict() hook frame"
        if labels is not None and sub is not None and artifact_valid:
            sub_k = sub.copy()
            lab_k = labels.copy()
            for c in gold["id_columns"]:
                sub_k[c] = _as_str_series(sub_k[c])
                lab_k[c] = _as_str_series(lab_k[c])
            merged = lab_k.merge(sub_k, on=gold["id_columns"], how="left", suffixes=("_true", ""))
            t0, o0 = gold["target_columns"][0], gold["output_columns"][0]
            true_col = f"{t0}_true" if f"{t0}_true" in merged.columns else t0
            if o0 in merged.columns and len(gold["target_columns"]) == 1:
                score_req = _score(merged[true_col], merged[o0], gold["requested_metric"], family)
                off = gold["official_metric"].split(" ")[0]
                score_off = _score(merged[true_col], merged[o0], "rmsle" if "rmse_of_log" in gold["official_metric"] else off, family)
    row.update({"artifact_valid": artifact_valid, "invalid_reason": invalid_reason, "output_mapping_valid": output_mapping_valid, "mapping_reason": mapping_reason})
    semantic_correct = bool(delivered and fit_target_ok and family_ok and output_mapping_valid)
    row["semantic_correct"] = semantic_correct

    # ---------------- terminal ----------------
    outcome = res.get("outcome")
    sub_reason = ""
    if outcome in {"budget_llm_calls", "budget_wall_clock"}:
        terminal, sub_reason = "S8", "budget:" + outcome
    elif outcome in {"infra_process_failure", "missing_result"}:
        terminal, sub_reason = "S8", "infrastructure:" + str(outcome)
    elif outcome == "forked":
        terminal, sub_reason = "PREFIX", "fork snapshot (not a terminal run)"
    elif outcome in {"exception", "recursion_limit"}:
        terminal, sub_reason = "S7", outcome
    elif res.get("workflow_aborted"):
        if template in CLEAN_TEMPLATES:
            terminal = "S4" if from_s else "S7"
            sub_reason = "semantic gate blocked a solvable/legal task" if from_s else "aborted on execution/structural failures"
        else:
            justified = from_s  # a stop counts as S3 only if the fatal verdict enforced a contract (S) failure
            terminal = "S3" if justified else "S7"
            sub_reason = "stopped on contract violation" if justified else "aborted on unrelated execution failures"
    elif res.get("workflow_completed") and not res.get("prediction_exists"):
        terminal = "S7" if tool_errors else "S6"
        sub_reason = "workflow completed without a prediction artifact"
    elif delivered:
        if not artifact_valid:
            terminal, sub_reason = "S6", invalid_reason[:300]
        elif not semantic_correct:
            terminal = "S5"
            why = []
            if not fit_target_ok:
                why.append("fit target != gold")
            if not family_ok:
                why.append("task family != gold")
            if not output_mapping_valid:
                why.append("output mapping invalid: " + mapping_reason[:200])
            sub_reason = "; ".join(why)
        else:
            terminal = "S2" if enforced_retry else "S1"
    else:
        terminal, sub_reason = "S7", f"unclassified outcome {outcome}"
    row["terminal_S"] = terminal
    row["terminal_subreason"] = sub_reason
    row["safe_terminal"] = terminal in safe_terminals
    row["rejected_valid_case"] = terminal == "S4"
    row["score_requested_metric"] = score_req
    row["score_official_metric"] = score_off

    # ---------------- resources ----------------
    llm = [e for e in events if e["event_type"] == "llm_call" and e.get("error_subtype") != "llm_disabled_replay"]
    own_llm = [e for e in llm if e["_origin"] == "own"]
    tin = [((e.get("payload") or {}).get("usage") or {}).get("input_tokens") for e in llm]
    tout = [((e.get("payload") or {}).get("usage") or {}).get("output_tokens") for e in llm]
    row["api_calls_total"] = len(llm)
    row["api_calls_own"] = len(own_llm)
    row["api_errors"] = sum(1 for e in llm if e.get("error_subtype"))
    row["tokens_in_if_observed"] = int(sum(x for x in tin if isinstance(x, (int, float)))) if any(isinstance(x, (int, float)) for x in tin) else None
    row["tokens_out_if_observed"] = int(sum(x for x in tout if isinstance(x, (int, float)))) if any(isinstance(x, (int, float)) for x in tout) else None
    models = sorted({(e.get("payload") or {}).get("returned_model") or "" for e in llm} - {""})
    row["returned_model_ids"] = "|".join(models)
    row["wall_seconds_own"] = res.get("wall_seconds")
    starts = [e for e in events if e["event_type"] == "run_start"]
    row["prefix_wall_seconds"] = None
    if any(e["_origin"] == "prefix" for e in events):
        pre = [e for e in events if e["_origin"] == "prefix"]
        row["prefix_wall_seconds"] = max(float(e.get("elapsed_s") or 0) for e in pre)
    inj = [e for e in events if e["event_type"] == "injection"]
    row["injection_applied"] = bool(inj) if template in {"F05", "F10"} else None
    row["n_events"] = len(events)
    row["n_run_start"] = len(starts)
    return row


def _is_semantic(text: str) -> bool:
    import re

    t = re.sub(r"\S*[/\\]\S*", " ", text or "")
    return bool(re.search(r"target|submission", t, re.I))
