#!/usr/bin/env python3
"""Build paper/evidence_report.html (+ PDF via headless Chrome) from results/ and paper/ assets.

Every number in the report is read from the result files at build time; the narrative text
lives in paper/report_text.json (written by the lane owner after reading the results) and may
only reference numbers through {placeholders} filled from the computed `facts` dict.
"""

from __future__ import annotations

import base64
import html
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import PKG  # noqa: E402

RES = PKG / "results"
PAPER = PKG / "paper"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def rd(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=object, keep_default_na=False) if path.exists() else pd.DataFrame()


def img(name: str) -> str:
    p = PAPER / "figures" / f"{name}.png"
    if not p.exists():
        return f"<p class='missing'>[figure {name} not generated]</p>"
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f"<img src='data:image/png;base64,{b64}' alt='{name}'/>"


def table_html(df: pd.DataFrame, cols: list[str] | None = None, rename: dict | None = None, max_rows: int = 60) -> str:
    if df.empty:
        return "<p class='missing'>[no rows]</p>"
    d = df[cols] if cols else df
    if rename:
        d = d.rename(columns=rename)
    d = d.head(max_rows)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in d.columns)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>" for row in d.itertuples(index=False))
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def facts() -> dict:
    rl = rd(RES / "run_level.csv")
    f: dict = {}
    started = rl[rl["terminal_S"] != "PREFIX_FORKED"]
    f["n_runs_launched"] = len(rl)
    f["n_terminal_runs"] = len(started)
    f["n_prefix_forked"] = int((rl["terminal_S"] == "PREFIX_FORKED").sum())
    f["n_llm_live"] = int((rl["model_id"] == "deepseek/deepseek-flash").sum())
    # count every LLM call exactly once (own events only; A2 prefixes are shared by their branches)
    calls = tin = tout = errs = 0
    for ev_file in (PKG / "logs" / "events").glob("*.jsonl"):
        if ev_file.name.endswith(".prefix.jsonl"):
            continue
        for line in ev_file.read_text().splitlines():
            e = json.loads(line)
            if e.get("event_type") != "llm_call" or e.get("error_subtype") == "llm_disabled_replay":
                continue
            calls += 1
            u = (e.get("payload") or {}).get("usage") or {}
            tin += u.get("input_tokens") or 0
            tout += u.get("output_tokens") or 0
            errs += 1 if e.get("error_subtype") else 0
    f["calls_total"], f["tokens_in"], f["tokens_out"], f["api_errors"] = calls, tin, tout, errs
    for stage in ["A0-CLEAN", "A0-SMOKE", "A0-PROBE", "A0-EQUIV", "A1", "A2-PILOT", "A3-PILOT", "A2-PREFIX"]:
        sub = rl[rl["stage"] == stage]
        f[f"n_{stage}"] = len(sub)
        f[f"terminals_{stage}"] = ", ".join(f"{k}:{v}" for k, v in sorted(sub["terminal_S"].value_counts().to_dict().items()))
    return f


def main() -> int:
    text_path = PAPER / "report_text.json"
    text = json.loads(text_path.read_text(encoding="utf-8")) if text_path.exists() else {}
    F = facts()
    T = {k: (v.format(**F) if isinstance(v, str) else v) for k, v in text.items()}

    qg = rd(RES / "quality_gates.csv")
    a0 = rd(RES / "A0" / "A0_old_vs_current.csv")
    ab = rd(RES / "A0" / "A0_old_vs_corrected_ablation.csv")
    probes = rd(RES / "A0" / "A0_probe_outcomes.csv")
    a1 = rd(RES / "A1_run_binding.csv")
    ta2 = rd(PAPER / "tables" / "T-A2_context_x_decision.csv")
    tl = rd(RES / "task_level.csv")
    ta3 = rd(PAPER / "tables" / "T-A3_intake_only_vs_full.csv")
    claims = rd(RES / "claims.csv")

    css = """
    @page { size: A4; margin: 14mm 13mm 14mm 13mm; }
    body { font-family: 'PingFang SC','Hiragino Sans GB','Noto Sans CJK SC','Helvetica Neue',Arial,sans-serif; font-size: 9.6pt; line-height: 1.45; color: #111; }
    h1 { font-size: 15pt; margin: 0 0 4px 0; } h2 { font-size: 12pt; margin: 14px 0 4px 0; border-bottom: 1px solid #333; padding-bottom: 2px; }
    h3 { font-size: 10.5pt; margin: 10px 0 3px 0; } p { margin: 4px 0; } .meta { color: #444; font-size: 8.6pt; }
    table { border-collapse: collapse; width: 100%; margin: 5px 0 6px 0; font-size: 6.9pt; } th, td { border: 1px solid #999; padding: 2px 4px; vertical-align: top; }
    th { background: #eee; } img { max-width: 100%; max-height: 88mm; display: block; margin: 4px auto; object-fit: contain; } .box { border: 1.2px solid #111; padding: 6px 9px; margin: 6px 0; }
    .qa p { margin: 3px 0; } .qa b { display: inline-block; min-width: 0; } .missing { color: #a00; } .small { font-size: 8pt; color: #333; }
    .pb { page-break-before: always; } code { font-size: 8pt; }
    """
    parts = [f"<html><head><meta charset='utf-8'><title>lane A evidence report</title><style>{css}</style></head><body>"]
    parts.append(f"<h1>{T.get('title', 'RTS-MA-DS / TAI — 负责人 A 证据报告')}</h1>")
    parts.append(f"<p class='meta'>{T.get('meta', '')}</p>")
    parts.append(f"<div class='box'><b>状态声明</b>：{T.get('status', '')}</div>")
    parts.append("<h2>结论首页：五个问题</h2><div class='qa'>")
    for q in ["q1", "q2", "q3", "q4", "q5"]:
        parts.append(f"<p>{T.get(q, '')}</p>")
    parts.append("</div>")
    parts.append(f"<h3>运行账本</h3><p>{T.get('ledger', '')}</p>")
    parts.append("<h3>质量门控（G0–G6 与本 lane 实验门控）</h3>")
    qg2 = qg.copy()
    if not qg2.empty:
        qg2["reason"] = qg2["reason"].str.slice(0, 115)
    parts.append(table_html(qg2, ["experiment_id", "gate_id", "status", "checked_count", "eligible_count", "reason"]))

    parts.append("<h2 class='pb'>A0 旧结果核对、无门控可运行性与代码语义审计</h2>")
    parts.append(f"<p>{T.get('a0_text', '')}</p>")
    parts.append(table_html(a0, ["task_id", "manuscript_metric", "manuscript_cv_score", "current_cv_score_mean", "current_cv_score_sd", "score_delta_vs_manuscript",
                                 "manuscript_valid", "current_valid", "manuscript_time_s_mean", "current_wall_s_mean", "current_terminals"],
                            {"manuscript_cv_score": "原稿分数", "current_cv_score_mean": "当前均值", "current_cv_score_sd": "当前SD", "score_delta_vs_manuscript": "Δ",
                             "manuscript_valid": "原稿有效", "current_valid": "当前有效", "manuscript_time_s_mean": "原稿秒", "current_wall_s_mean": "当前秒", "current_terminals": "终态"}))
    parts.append(f"<p>{T.get('a0_ablation_text', '')}</p>")
    parts.append(table_html(ab, ["source", "variant", "clean_completion", "clean_valid", "stop", "fp", "time_s"]))
    parts.append(f"<p>{T.get('a0_code_text', '')}</p>")

    parts.append("<h2>A1 C_gold、C_read 与真实 fit/output 的绑定</h2>")
    parts.append(f"<p>{T.get('a1_text', '')}</p>")
    if not a1.empty:
        rows = []
        for (task, tmpl), g in a1.groupby(["task_id", "template_id"], sort=True):
            f0 = g[g["method_id"] == "F0_ORIGINAL"]
            m3 = g[g["method_id"] == "M3_SHADOW"]
            r0 = f0.iloc[0] if not f0.empty else None
            rows.append({"任务": task.replace("DEV-", ""), "探针": tmpl,
                         "F0 终态": r0["terminal_S"] if r0 is not None else "", "M3 终态": m3.iloc[0]["terminal_S"] if not m3.empty else "(A0 only)",
                         "F0 C_read": r0["C_read_targets"][:28] if r0 is not None else "", "F0 fit y": r0["C_used_fit_targets"][:28] if r0 is not None else "",
                         "F0 失败来源（事件判定）": r0["a1_failure_source"] if r0 is not None else "", "定位": r0["localization"] if r0 is not None else ""})
        parts.append(table_html(pd.DataFrame(rows)))
    parts.append(img("F-A1_action_chain"))
    parts.append(f"<p class='small'>{T.get('a1_cases', '')}</p>")

    parts.append("<h2 class='pb'>A2 同一候选动作下的放行—阻断—恢复配对干预（DEV pilot）</h2>")
    parts.append(f"<p>{T.get('a2_text', '')}</p>")
    parts.append(table_html(ta2, ["context", "method_id", "n", "next_action_after_gate", "wrong_fit_after_fork", "wrong_file_delivered", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "mean_api_calls_after_fork", "strict_common_prefix"],
                            {"next_action_after_gate": "gate 后下一动作", "wrong_fit_after_fork": "分叉后错误fit", "wrong_file_delivered": "错误交付", "mean_api_calls_after_fork": "分叉后调用", "strict_common_prefix": "严格同前缀"}))
    sel = tl[(tl["context"] == "ALL") & tl["metric"].isin(["d_unsafe_delivery_S5S6", "d_justified_stop_S3", "d_correct_delivery_S1S2", "d_wrong_training_seconds_after_fork", "d_api_calls_own"])] if not tl.empty else tl
    parts.append(table_html(sel, ["contrast", "metric", "n_tasks", "n_pairs", "task_level_mean", "boot95_low", "boot95_high", "per_task", "direction"]))
    parts.append(img("F-A2_forest"))
    parts.append(img("F-A3_same_prefix_branches"))

    parts.append("<h2>A3 强一次性语义检查是否已经足够？（DEV pilot）</h2>")
    parts.append(f"<p>{T.get('a3_text', '')}</p>")
    parts.append(table_html(ta3, ["context", "task_id", "seed_block", "F0_terminal", "M2_terminal", "d_unsafe_delivery_S5S6", "d_correct_delivery_S1S2", "strict_common_prefix"]))
    parts.append(img("F-A4_location_x_increment"))

    parts.append("<h2>主张账本、不可观察边界与下一步</h2>")
    parts.append(f"<p>{T.get('limits', '')}</p>")
    prim = claims[claims["estimand_role"] == "primary"] if "estimand_role" in claims.columns else claims
    parts.append(table_html(prim, ["claim_id", "primary_contrast", "estimate", "ci_low", "ci_high", "per_task_direction", "support_level"], max_rows=40))
    parts.append("<p class='small'>完整主张账本（含次要估计量、反例 pair_id 与允许的论文措辞）：results/claims.csv；文字版：results/A_claim_ledger.md。</p>")
    parts.append(f"<p>{T.get('next', '')}</p>")
    parts.append(f"<h3>证据链矩阵</h3>{T.get('evidence_matrix', '')}")
    parts.append("</body></html>")
    out_html = PAPER / "evidence_report.html"
    out_html.write_text("\n".join(parts), encoding="utf-8")
    out_pdf = PAPER / "evidence_report.pdf"
    if Path(CHROME).exists():
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={out_pdf}", out_html.as_uri()],
                       check=False, capture_output=True, timeout=180)
    print("report:", out_html, out_pdf if out_pdf.exists() else "(pdf not generated)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
