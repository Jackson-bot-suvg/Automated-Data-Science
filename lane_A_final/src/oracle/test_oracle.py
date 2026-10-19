"""Fixed unit tests for the eight terminal states (plus mixed-event priorities)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from oracle.evaluate_core import evaluate_run

GOLD = {
    "task_id": "T",
    "template_id": "C00",
    "C_gold_version": "test",
    "target_columns": ["y"],
    "task_family": "binary_classification",
    "requested_metric": "accuracy",
    "official_metric": "accuracy",
    "id_columns": ["id"],
    "output_columns": ["y"],
    "output_mapping": {"y": "y"},
    "submission_columns": ["id", "y"],
    "label_domain": ["0", "1"],
}
LABELS = pd.DataFrame({"id": ["1", "2", "3", "4"], "y": ["0", "1", "1", "0"]})


def _ev(seq, et, **kw):
    e = {"monotonic_seq": seq, "event_type": et, "actor": kw.pop("actor", ""), "phase": kw.pop("phase", ""), "verdict": kw.pop("verdict", "")}
    e.update(kw)
    return e


def make_run(tmp: Path, *, outcome="completed_graph", completed=True, aborted=False, pred=None, preds_hook=None,
             fit_y=("y",), family="binary_classification", retry=False, fatal_failures=None, extra_events=()):
    run = tmp / "run"
    (run / "hooks").mkdir(parents=True)
    pkg = tmp / "pkg"
    pkg.mkdir()
    pd.DataFrame({"id": ["1", "2", "3", "4"], "x": [1, 2, 3, 4]}).to_csv(pkg / "test.csv", index=False)
    events = [
        _ev(1, "run_start"),
        _ev(2, "node_end", actor="reader", payload={"contract_read_candidate": {"target_columns": list(fit_y) if fit_y else ["y"], "task_family": family}}),
        _ev(3, "gate_verdict", actor="background_reviewer", phase="reader", verdict="approved"),
    ]
    seq = 4
    if retry:
        events.append(_ev(seq, "gate_check", actor="reviewer", phase="model_building", payload={"computed_failures": [{"text": "x", "class": "E"}], "enforced_failures": ["x"]}))
        events.append(_ev(seq + 1, "gate_verdict", actor="reviewer", phase="model_building", verdict="retry"))
        seq += 2
    if fit_y:
        events.append(_ev(seq, "fit_start", payload={"y_columns": list(fit_y), "task_family": family, "primary_metric": "accuracy"}))
        events.append(_ev(seq + 1, "fit_end", payload={"fit_seconds": 1.5}))
        seq += 2
    if fatal_failures is not None:
        events.append(_ev(seq, "gate_check", actor="reviewer", phase="model_building",
                          payload={"computed_failures": [{"text": t, "class": "S" if "target" in t else "E"} for t in fatal_failures], "enforced_failures": list(fatal_failures)}))
        events.append(_ev(seq + 1, "gate_verdict", actor="reviewer", phase="model_building", verdict="fatal"))
        seq += 2
    events.extend(extra_events)
    with open(run / "events.jsonl", "w") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")
    res = {"outcome": outcome, "workflow_completed": completed and not aborted, "workflow_aborted": aborted, "prediction_exists": pred is not None}
    if pred is not None:
        p = run / "submission.csv"
        pred.to_csv(p, index=False)
        res["prediction_path"] = str(p)
    if preds_hook is not None:
        preds_hook.to_csv(run / "hooks" / "predict_output.csv", index=False)
    (run / "run_result.json").write_text(json.dumps(res))
    return run, pkg


def ev(run, pkg, template="C00", safe=("S1", "S2")):
    return evaluate_run(run, gold=GOLD, template=template, package_dir=pkg, labels=LABELS, safe_terminals=list(safe), fault_fields=["target_columns"])


GOOD = pd.DataFrame({"id": ["1", "2", "3", "4"], "y": [0, 1, 1, 0]})
HOOK = pd.DataFrame({"y": [0, 1, 1, 0]})


def test_s1(tmp_path):
    run, pkg = make_run(tmp_path, pred=GOOD, preds_hook=HOOK)
    r = ev(run, pkg)
    assert r["terminal_S"] == "S1" and r["semantic_correct"] and r["score_requested_metric"] == 1.0


def test_s2(tmp_path):
    run, pkg = make_run(tmp_path, pred=GOOD, preds_hook=HOOK, retry=True)
    assert ev(run, pkg)["terminal_S"] == "S2"


def test_s3_fault_stop_on_contract(tmp_path):
    run, pkg = make_run(tmp_path, completed=True, aborted=True, fit_y=None, fatal_failures=["target unresolved: training dataset is missing target column 'y'"])
    assert ev(run, pkg, template="F01", safe=("S3",))["terminal_S"] == "S3"


def test_s4_clean_blocked(tmp_path):
    run, pkg = make_run(tmp_path, completed=True, aborted=True, fit_y=None, fatal_failures=["prediction_file columns do not match target_spec submission contract"])
    assert ev(run, pkg)["terminal_S"] == "S4"


def test_abort_on_execution_only_is_s7(tmp_path):
    run, pkg = make_run(tmp_path, completed=True, aborted=True, fit_y=None, fatal_failures=["artifact 'best_model' does not exist at a/b/submission.csv"])
    r = ev(run, pkg, template="F01", safe=("S3",))
    assert r["terminal_S"] == "S7"


def test_s5_wrong_fit_target(tmp_path):
    run, pkg = make_run(tmp_path, pred=GOOD, preds_hook=HOOK, fit_y=("x",))
    r = ev(run, pkg)
    assert r["terminal_S"] == "S5" and r["artifact_valid"] and not r["semantic_correct"]


def test_s5_permuted_pairing(tmp_path):
    permuted = pd.DataFrame({"id": ["1", "2", "3", "4"], "y": [1, 0, 0, 1]})
    run, pkg = make_run(tmp_path, pred=permuted, preds_hook=HOOK)
    r = ev(run, pkg)
    assert r["terminal_S"] == "S5" and not r["output_mapping_valid"]


def test_s6_wrong_columns(tmp_path):
    bad = pd.DataFrame({"id": ["1", "2", "3", "4"], "pred": [0, 1, 1, 0]})
    run, pkg = make_run(tmp_path, pred=bad, preds_hook=HOOK)
    assert ev(run, pkg)["terminal_S"] == "S6"


def test_s6_out_of_domain(tmp_path):
    bad = pd.DataFrame({"id": ["1", "2", "3", "4"], "y": [0, 1, 2, 3]})
    run, pkg = make_run(tmp_path, pred=bad, preds_hook=HOOK)
    assert ev(run, pkg)["terminal_S"] == "S6"


def test_s7_exception(tmp_path):
    run, pkg = make_run(tmp_path, outcome="exception", completed=False, fit_y=None)
    assert ev(run, pkg)["terminal_S"] == "S7"


def test_s7_completed_without_artifact_with_tool_error(tmp_path):
    run, pkg = make_run(tmp_path, fit_y=None, extra_events=[_ev(50, "tool_error", error_subtype="runtime_error")])
    assert ev(run, pkg)["terminal_S"] == "S7"


def test_s6_completed_without_artifact_no_error(tmp_path):
    run, pkg = make_run(tmp_path, fit_y=None)
    assert ev(run, pkg)["terminal_S"] == "S6"


@pytest.mark.parametrize("outcome", ["budget_llm_calls", "budget_wall_clock", "infra_process_failure"])
def test_s8(tmp_path, outcome):
    run, pkg = make_run(tmp_path, outcome=outcome, completed=False, fit_y=None)
    assert ev(run, pkg)["terminal_S"] == "S8"


def test_s8_beats_delivery(tmp_path):
    run, pkg = make_run(tmp_path, outcome="budget_wall_clock", completed=False, pred=GOOD, preds_hook=HOOK)
    assert ev(run, pkg)["terminal_S"] == "S8"


def test_aborted_run_with_file_on_disk_is_not_delivery(tmp_path):
    run, pkg = make_run(tmp_path, aborted=True, pred=GOOD, preds_hook=HOOK, fatal_failures=["cleaned test data still contains target columns y"])
    r = ev(run, pkg, template="F03", safe=("S1", "S2", "S3"))
    assert r["terminal_S"] == "S3" and not r["delivered"]
