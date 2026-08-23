

## 13. Recent Validation Snapshots

The following end-to-end runs were executed after the latest BAT updates:

1. `titanic_autokaggle` (classification)
   - Result: report generated successfully with required chapter structure.
   - BAT call pattern: single `business_translator` LLM call (`llm_calls = 1`).
   - BAT latency: `53.6s` (within `<=60s` target).
   - Industry positioning quality: specific domain label produced (premium passenger transport / maritime tourism context).

2. `ghouls_goblins_and_ghosts_boo` (multiclass classification)
   - Result: report generated successfully with required chapter structure and action-card formatting.
   - BAT call pattern: single `business_translator` LLM call (`llm_calls = 1`).
   - BAT latency: `41.7s` and `44.3s` in repeat runs (within `<=60s` target).
   - Industry positioning quality: specific domain label produced (supernatural entity risk/containment context), with concrete operating rules.



- # Business Action Translator (BAT) v3.0

> **Before running**: verify your `.env` contains one of `DASHSCOPE_API_KEY`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY`. A missing key aborts the business report at the very end of the workflow (the technical pipeline still runs).

## Purpose

BAT v3 converts the upstream technical state — Reader's `target_spec`, EDA findings, feature importance, model summary, technical report — into a stakeholder-facing English business report rendered as a **self-contained single HTML file** with **inlined Chart.js v4** for interactive visualizations.

Output file: `<workspace_root>/artifacts/reports/business_report.html`.

The report has **five fixed sections**, anchored on the analyst voice and structure of the sample reports in `示例商业报告/`:

1. **Preface** — opens with a specific industry role voice (e.g. "Chief Retention Officer of a mid-sized European retail bank"), names the dataset scale and a headline business figure, sets the three progressive business questions the report answers.
2. **Business Context Modeling** — feature-to-business-variable mapping table, situational context, P&L and risk lens defining gold / high-risk / distressed segments.
3. **Core Insight Discovery** — 3–5 insights ranked by P&L kill-power (not statistical significance), each with quantified title, evidence with numbers embedded in prose, and a business-interpretation callout.
4. **Strategic Action Recommendations** — 3–5 action cards with target segment, problem diagnosis, time-bounded action levers, and a quantified KPI target.
5. **Executive Briefing** — 4–6 board-line judgments, a 90-day priority action table, a one-line tone-setter on a dark hero callout.

## Public API

`app.agents.business_translator` exposes:

- `run_business_translation(state: GraphState) -> dict` — the agent entrypoint called by `business_translator_node.py`. **Never raises.** Returns either a success delta or an abort delta (see below).
- `preflight_business_translator() -> tuple[bool, str | None]` — public helper for callers that want to fail-fast on missing env keys before the node runs.

## State contract

### Reads

- `state.task_config.raw_task`, `task_type`, `dataset_paths.train`/`.test`
- `state.target_spec.target_columns`, `task_family`
- `state.problem_understanding.predicted_task_type`, `data_overview`
- `state.data_state.eda_findings`, `indepth_eda_summary`
- `state.feature_state.feature_importance_summary`
- `state.model_state.best_model_name`, `validation_metrics`, `baseline_metrics`
- `state.report_state.technical_report_path`, `limitations`
- `state.runtime_state.workspace_root`

### Writes — success delta

```python
{
  "report_state": {
    "business_report_path": "<abs path to .html>",
    "executive_summary": "<one-line tone-setter>",
    "recommended_actions": ["Action 1 title", "Action 2 title", ...],
    "limitations": [<existing limitations + leak notes + chart-drop notes>],
  },
  "artifacts": {
    "business_report_html": "<abs path to .html>",
    "business_report": "<abs path to .html>",  # legacy key repurposed
    "business_translator": {
      "status": "completed",
      "error": None,
      "error_class": None,
      "charts_count": <int>,
      "industry_label": "<specific sub-industry>",
      "format": "html_v3",
    },
  },
}
```

### Writes — abort delta (any failure)

```python
{
  "report_state": {
    "business_report_path": "",
    "limitations": ["Business report aborted: <reason>"],
  },
  "artifacts": {
    "business_report_html": None,
    "business_translator": {
      "status": "aborted",
      "error": "<reason>",
      "error_class": "<exception class name>",
      "charts_count": 0,
      "industry_label": None,
      "format": "html_v3",
    },
  },
}
```

## LLM contract

Single call via `app.llm.helpers.complete_json` (agent_name = `business_translator`), routed by `app.llm.agent_config.get_agent_config`. The LLM returns a fenced JSON block matching this top-level shape:

```json
{
  "industry_label": "<specific sub-industry>",
  "industry_role_voice": "<role title>",
  "preface_html": "<sanitized html>",
  "business_context_html": "<sanitized html>",
  "insights":  [{"id": "insight-1", "title": "...", "evidence_html": "...", "interpretation_html": "..."}, ...],
  "actions":   [{"id": "action-1",  "title": "...", "target_segment_html": "...", "diagnosis_html": "...", "levers_html": "...", "kpi_html": "..."}, ...],
  "executive_briefing": {
    "judgments_html": "<ol>...</ol>",
    "ninety_day_table": [{"action": "...", "owner": "...", "metric": "...", "expected_benefit": "..."}, ...],
    "tone_line": "..."
  },
  "charts": [{
    "chart_id": "c1",
    "section": "business-context | key-insights | action-recommendations",
    "type": "bar | horizontal_bar | pie | line | scatter | heatmap",
    "title": "<conclusion title>",
    "caption": "<one or two sentences with at least one number>",
    "insight_ref": "insight-1 | action-1",
    "x_label": "...", "y_label": "...",
    "data": {"labels": [...], "series": [{"name": "...", "values": [...]}]}
  }, ...]
}
```

Hard floors enforced by the orchestrator:
- ≥ 3 insights, ≥ 3 actions (Pydantic schema)
- Total kept charts ≥ 2 AND charts in `key-insights` section ≥ 1 (business validator). Below this, the run aborts and no HTML is written.

## Chart rules

- 3–5 charts elastic.
- Each chart must anchor to an existing `insight-N` or `action-N` id (orphan charts are dropped).
- Title must state a conclusion. Descriptive titles ("Distribution of X") are rejected.
- Caption must contain at least one number.
- Series and labels lengths must match.
- Pie with > 5 slices is coerced to bar.

## HTML rendering

- Self-contained single file at `artifacts/reports/business_report.html`.
- Inline CSS in `<style>` (no external stylesheets).
- Inline Chart.js v4.4.7 UMD min (~200 KB) read once from `app/agents/vendor/chart.umd.min.js`, cached via `lru_cache`.
- Each chart emits a `<figure class="chart">` plus a `<script type="application/json" id="data-cN">` data island. A bootstrap script at the bottom of the document iterates the islands and instantiates Chart.js charts using a unified theme.
- Clicking a chart `<figcaption>` smooth-scrolls to the anchored insight/action.

## Error protocol

| Trigger | When | Result |
|---|---|---|
| `.env` missing any of `DASHSCOPE_API_KEY`/`OPENAI_API_KEY`/`ANTHROPIC_API_KEY` | First line of BAT | Abort delta with clear CLI message; zero LLM calls |
| LLM call timeout (default 55 s), 4xx/5xx, network failure | Runtime | Abort delta |
| Fenced JSON parse failure | Post-processing | Abort delta |
| Pydantic schema mismatch | Post-processing | Abort delta with field info |
| Charts below the floor (total < 2 OR `key-insights` < 1) | Post-processing | Abort delta |

**Invariant**: `run_business_translation` never raises. Every failure path runs through the top-level `try/except` and is converted into an abort delta. No half-baked HTML is ever written.

## Generalisation across task types

Prompt-conditioned framing (no Python branching):

- `target_spec.task_family == "regression"` → pricing / valuation / forecast accuracy vocabulary; "at-risk" / "exposure" vocabulary is suppressed.
- `task_family == "multiclass"` (or `multi_class` / `multi-class`) → tier distribution / category-level interventions.
- Anything else (default, including `binary`) → at-risk segments, exposure, intervention ROI.

## Tests

Unit suite at `tests/unit/test_business_translator.py`. After v3:

- ≥ 50 tests covering every private helper and the full orchestrator.
- The contract test `test_v3_run_business_translation_never_raises` parametrises over mocked failure sources (LLM raises, JSON fails, schema fails, etc.) and confirms the public function always returns a dict.
- The contract test `test_v3_run_business_translation_missing_key_aborts_without_calling_llm` confirms zero LLM calls when env keys are missing.

## File layout

- `app/agents/business_translator.py` — single file, regions ordered: V3 errors → V3 abort delta → public `run_business_translation` shim → (v2.1 dead helpers retained as-is for safety) → `_run_coro_sync` (shared) → V3 schemas / sanitizer / scanner / extractor / validator / evidence pack / prompts / LLM bridge / renderer / preflight / orchestrator.
- `app/agents/vendor/chart.umd.min.js` — pinned Chart.js v4.4.7, sha256 in `app/agents/vendor/CHART_JS_VERSION`.

## What was deprecated in v3

- Markdown report renderer
- PDF report renderer
- Bilingual fallback
- State keys `artifacts.business_report_pdf` (no longer written)

The legacy artifact key `artifacts.business_report` is retained but now points at the `.html` file for backward-compatible consumers.

---

## Code implementation walkthrough

This section indexes the actual implementation of BAT v3 after the prompt refactor that moved prompt content out to its own module. Two source files matter:

- [app/agents/business_translator.py](../app/agents/business_translator.py) — ~3.7k lines, single-file agent with clearly marked `# region` blocks. Contains the orchestrator, schemas, sanitizer, validator, evidence pack, LLM bridge, HTML renderer, preflight, and error/delta builders.
- [app/prompts/business_prompt.py](../app/prompts/business_prompt.py) — owns the v3 system prompt, the JSON schema hint, and the user-prompt builder. Imported at the top of `business_translator.py`.

The LangGraph adapter is the thin wrapper [app/graph/nodes/business_translator_node.py](../app/graph/nodes/business_translator_node.py).

### File layout — `app/agents/business_translator.py` (top to bottom)

| Region | Lines | Purpose |
|---|---|---|
| Module imports | 1–32 | stdlib + `app.llm.helpers.complete_json`, observability, `GraphState`, tools, plus `BUSINESS_TRANSLATOR_SYSTEM_PROMPT` / `build_business_translation_prompt` from `app.prompts.business_prompt` |
| V3 Errors | 34–41 | [`_V3BusinessTranslatorError`](../app/agents/business_translator.py#L39) — internal exception type, never escapes the public entrypoint |
| V3 Abort Delta | 43–70 | [`_v3_build_abort_delta`](../app/agents/business_translator.py#L48) — single source of truth for the failure-mode state delta |
| Public entrypoint | 75–87 | [`run_business_translation`](../app/agents/business_translator.py#L75) — wraps `_v3_run_business_translation_impl` in `try/except` so it never raises |
| Legacy v2.1 helpers | ~90–2110 | Markdown/PDF helpers, deterministic insight/action briefs, industry profiler, hypothesis/critic plumbing. **Dead code under v3** — retained because some unit tests still import them; no v3 path calls into this block. The four v2.1 prompt names they reference (`BUSINESS_TRANSLATOR_CRITIC_SYSTEM_PROMPT`, etc.) are no longer imported, so calling these helpers would `NameError` at runtime — fine since they're never called. |
| `_run_coro_sync` | [2113](../app/agents/business_translator.py#L2113)–2137 | Shared sync→async bridge. Preserves `contextvars` across the thread hop so LangSmith tracing stays attached |
| V3 Schemas | 2305–2480 | Pydantic models (see below) |
| V3 HTML Sanitizer | 2483–2618 | `_V3HtmlSanitizer` (HTMLParser-based) + `_v3_inject_preface_question_list` + `_v3_sanitize_html_fragment` |
| V3 Metric-Token Scanner | 2620–2656 | `_v3_scan_metric_token_leaks` — detects banned tokens (AUC/F1/RMSE/…) in rendered HTML; **non-blocking**, appended to limitations |
| V3 Fenced JSON Extractor | 2658–2681 | `_v3_extract_fenced_json` — fallback parser for ```json …``` fences |
| V3 Chart Validator | 2683–2766 | `_v3_validate_charts` — business rules layered on top of Pydantic validation |
| V3 Evidence Pack | 2768–2913 | Reads the technical report and `artifacts/`, builds the evidence string passed to the LLM |
| V3 Prompts (delegated) | 2915–2928 | Region body is only two alias lines: `_V3_SYSTEM_PROMPT = BUSINESS_TRANSLATOR_SYSTEM_PROMPT` and `_v3_build_user_prompt = build_business_translation_prompt`. Authoritative source is in [`app/prompts/business_prompt.py`](../app/prompts/business_prompt.py). |
| V3 LLM Bridge | 2929–2977 | [`_v3_call_llm_sync`](../app/agents/business_translator.py#L2934) — uses `app.llm.agent_config.get_agent_config("business_translator")`, fresh `LLMService` per call |
| V3 HTML Renderer | 2979–3534 | [`_v3_render_html_document`](../app/agents/business_translator.py#L3431) and partials — embeds inlined CSS, Chart.js UMD, JSON data islands |
| V3 Preflight + Success Delta | 3536–3589 | [`preflight_business_translator`](../app/agents/business_translator.py#L3542), `_v3_build_success_delta` |
| V3 Public Orchestrator | 3591–end | [`_v3_run_business_translation_impl`](../app/agents/business_translator.py#L3595) — the 8-step pipeline |

### File layout — `app/prompts/business_prompt.py`

A short module (~180 lines) with three public symbols:

| Symbol | Purpose |
|---|---|
| `BUSINESS_TRANSLATOR_SYSTEM_PROMPT` | Long fixed string. Covers five-section voice rules, allowed HTML tags, the exact preface skeleton, three `<h3>` sub-blocks of Business Context, banned metric tokens, industry-label rules, and chart caps. |
| `BUSINESS_TRANSLATOR_OUTPUT_SCHEMA_HINT` | The JSON-shape contract appended verbatim to every user prompt. |
| `build_business_translation_prompt(*, evidence_pack, task_family)` | Branches on `task_family` (`regression` / `multiclass` / default-binary) → returns `f"{framing}\n\n{schema_hint}\n\n## Evidence\n\n{evidence_pack}\n"`. **The only place where task-family branching enters the v3 pipeline.** |

The v2.1 prompts (`BUSINESS_TRANSLATOR_CRITIC_SYSTEM_PROMPT`, `BUSINESS_INSIGHT_HYPOTHESIS_SCHEMA_HINT`, `build_business_hypothesis_prompt`, `build_business_critic_prompt`, `build_industry_identification_prompt`, etc.) have been removed — they were only ever called by the dead v2.1 code in `business_translator.py`.

### Public orchestrator pipeline

[`_v3_run_business_translation_impl`](../app/agents/business_translator.py#L3595) is the canonical entry path. Eight numbered steps:

1. **Preflight** — [`preflight_business_translator()`](../app/agents/business_translator.py#L3542) scans `os.environ` for any var whose name contains `API_KEY` with a non-empty value. Raises `_V3BusinessTranslatorError` if none found. Zero LLM calls in this path.
2. **Build evidence pack** — [`_v3_read_technical_report`](../app/agents/business_translator.py#L2788) loads the technical markdown, [`_v3_collect_artifact_evidence`](../app/agents/business_translator.py#L2856) summarises referenced CSV/JSON artifacts (via `_summarize_csv`/`_summarize_json`), [`_v3_build_evidence_pack`](../app/agents/business_translator.py#L2891) concatenates them. The technical report is the **sole** data source — no direct reads from `state.data_state`, `state.model_state`, etc.
3. **Call LLM** — [`_v3_call_llm_sync`](../app/agents/business_translator.py#L2934) runs `LLMService.chat(...)` via `_run_coro_sync`, then `parse_json_response`. Fallback to `_v3_extract_fenced_json` if the parsed dict lacks `industry_label` but carries a `raw_text` field.
4. **Pydantic validation** — `_V3BATResponse.model_validate(raw_dict)`. Any `ValidationError` is wrapped into `_V3BusinessTranslatorError("Output schema mismatch: …")`.
5. **HTML sanitization** — `_v3_sanitize_html_fragment` is applied in place to every `*_html` field. `_v3_inject_preface_question_list` post-processes the preface to ensure the three operational questions become a sibling `<ol>` (regex-based, idempotent if `<ol>` already present).
6. **Metric-token leak scan** — `_v3_scan_metric_token_leaks` returns a list of leak strings; these are appended to `limitations` (**non-blocking** — known quality gap, see below).
7. **Chart validation** — `_v3_validate_charts` enforces the rules below and returns `(kept, dropped_reasons, abort_reason)`. A non-None `abort_reason` raises `_V3BusinessTranslatorError`.
8. **Render + write** — `_v3_render_html_document` produces the full HTML string; the file is written to `Path(workspace_root) / "artifacts/reports" / _v3_title_to_filename(f"{dataset_name} business recommendation")`. Then `_v3_build_success_delta` assembles the state delta.

### Pydantic schemas (region: V3 Schemas, L2305–2480)

All schemas use `model_config = ConfigDict(extra="forbid")`.

- [`_V3ChartSeries`](../app/agents/business_translator.py#L2317) — `name: str`, `values: list[float]` (1–12). Pre-validator coerces `[x, y]` scatter pairs to `y` only.
- [`_V3ChartData`](../app/agents/business_translator.py#L2337) — `labels: list[str]` (1–12), `series: list[_V3ChartSeries]` (1–3). Pre-validator wraps a flat numeric list into a single anonymous series; stringifies labels; truncates series to 3.
- [`_V3ChartSpec`](../app/agents/business_translator.py#L2362) — `chart_id`, `section` (Literal of 3), `type` (Literal of 6), `title` (1–200), `caption` (1–400), `insight_ref`, `x_label`, `y_label`, `data`. Pre-validator normalises common section typos (`key_insights` → `key-insights`, etc.) and coerces null label/ref fields to `""`.
- [`_V3InsightBlock`](../app/agents/business_translator.py#L2399) — `id`, `title`, `evidence_html`, `interpretation_html`. Pre-validator accepts `title_html`/`evidence`/`interpretation` aliases.
- [`_V3ActionBlock`](../app/agents/business_translator.py#L2425) — `id`, `title`, `target_segment_html`, `diagnosis_html`, `levers_html`, `kpi_html`. Same alias coercion.
- [`_V3NinetyDayAction`](../app/agents/business_translator.py#L2455) — `action`, `owner`, `metric`, `expected_benefit`.
- [`_V3ExecutiveBriefing`](../app/agents/business_translator.py#L2463) — `judgments_html`, `ninety_day_table` (1–8 rows), `tone_line`.
- [`_V3BATResponse`](../app/agents/business_translator.py#L2470) — top level: `industry_label`, `industry_role_voice`, `preface_html`, `business_context_html`, `insights` (3–6), `actions` (3–6), `executive_briefing`, `charts` (0–5).

### HTML sanitization (L2483–2618)

[`_V3_ALLOWED_TAGS`](../app/agents/business_translator.py#L2488) is the strict allowlist: `p h3 h4 table thead tbody tr td th ol ul li blockquote strong em br`. All attributes are stripped. `script style iframe object embed` are dropped tag **and** content. Unknown tags drop the tag but keep the text. A regex post-pass normalises the `Report perspective:` paragraph so only the label is bold, never the body.

### Chart validator rules ([`_v3_validate_charts`](../app/agents/business_translator.py#L2694))

Per-chart filters, applied in order:
1. `insight_ref` must resolve to a known `insight-N` or `action-N` id (else drop as "orphan").
2. Title is rejected if it starts with a descriptive prefix: `distribution of`, `histogram of`, `frequency of`, `breakdown of`.
3. Caption must contain at least one digit.
4. Every series' `values` length must equal `labels` length.
5. `type == "pie"` with > 5 labels is coerced to `bar` (kept, with a drop-reason note).

Per-bucket caps:
- At most **1** chart in `business-context`.
- At most **1** chart per `insight_ref` (no two cards under the same insight/action).

Global floor (abort if violated): `total ≥ 2 AND key-insights count ≥ 1`. On failure, the orchestrator raises and the whole run produces an abort delta.

### LLM call ([`_v3_call_llm_sync`](../app/agents/business_translator.py#L2934))

- Uses `app.llm.agent_config.get_agent_config("business_translator")` to resolve model/provider/temperature/max_tokens.
- Instantiates a fresh `LLMService()` (not the module singleton) so it works under `_run_coro_sync`, which spawns a private event loop in a worker thread.
- `module_name="business_translator"` for token-usage attribution.
- `timeout_seconds` parameter is **kept for signature compatibility only**; no code-level timeout is enforced (per project requirement). Real timeouts come from the provider HTTP layer.
- Output goes through `app.llm.helpers.parse_json_response(resp.content, fallback={})`. Empty / non-dict response → `_V3BusinessTranslatorError("LLM returned empty or non-dict response")`.

### Prompts ([`app/prompts/business_prompt.py`](../app/prompts/business_prompt.py))

The orchestrator builds its prompt from exactly two symbols imported at the top of `business_translator.py`:

```python
from app.prompts.business_prompt import (
    BUSINESS_TRANSLATOR_SYSTEM_PROMPT,
    build_business_translation_prompt,
)
```

- `BUSINESS_TRANSLATOR_SYSTEM_PROMPT` — covers five-section voice rules, allowed HTML tags, the **exact required preface skeleton** (paragraph + sibling `<ol>` of three questions), the three `<h3>` sub-blocks of Business Context, banned metric tokens (AUC, F1, ROC, RMSE, MAE, MAPE, R^2, precision, recall, p-value, feature importance), industry-label rules (no "general/various/miscellaneous"), and chart caps.
- `BUSINESS_TRANSLATOR_OUTPUT_SCHEMA_HINT` — JSON schema hint appended verbatim by the user-prompt builder.
- `build_business_translation_prompt(*, evidence_pack, task_family)` — produces `framing + schema_hint + "## Evidence\n\n" + evidence_pack`. Branching:
  - `regression` → "Frame everything as pricing / valuation accuracy / pricing bands. Avoid at-risk or exposure vocabulary."
  - `multiclass` / `multi_class` / `multi-class` → "Frame everything as tier distribution and category-level interventions."
  - anything else (including `binary`) → "Frame everything as at-risk segments, exposure, and intervention ROI."

Inside `business_translator.py` the private aliases `_V3_SYSTEM_PROMPT` and `_v3_build_user_prompt` (defined at the V3 Prompts region) re-export the two symbols so any caller / test that imported the v3-internal names keeps working.

### HTML renderer (region: V3 HTML Renderer, L2979–3534)

- [`_v3_read_vendored_chartjs`](../app/agents/business_translator.py#L2987) is `@lru_cache(maxsize=1)`'d — Chart.js v4.4.7 UMD min (~200 KB) is read from `app/agents/vendor/chart.umd.min.js` once per process.
- [`_v3_chart_figure_html`](../app/agents/business_translator.py#L3004) emits the `<figure class="chart">` markup plus a sibling `<script type="application/json" id="data-cN">` JSON data island. The bootstrap JS inside [`_v3_render_html_document`](../app/agents/business_translator.py#L3431) iterates these islands, hydrates each one into a Chart.js chart with a unified theme, and wires a `click` handler on each `<figcaption>` that smooth-scrolls to `#insight-N` / `#action-N`.
- [`_v3_title_to_filename`](../app/agents/business_translator.py#L3424) slugifies the report title for the final HTML filename.

### LangGraph adapter

[`business_translator_node`](../app/graph/nodes/business_translator_node.py) is 5 lines: it wraps `run_business_translation` in a `_runner` and hands it to `timed_node`, which provides per-node latency telemetry. There is no async work in BAT itself — the LLM call is bridged back to sync inside the agent.

### Preflight contract

[`preflight_business_translator`](../app/agents/business_translator.py#L3542) is intentionally permissive: it accepts **any** non-empty env var whose name contains `API_KEY`. The legacy whitelist `_V3_API_KEY_ENV_VARS` (`DASHSCOPE_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `NEWAPI_API_KEY`, `LLM_NEWAPI_API_KEY`) is retained for documentation only. The function returns `(True, None)` on success or `(False, reason)` on failure.

### Known quality gap: metric-token leaks

The metric-token scanner at step 6 is **non-blocking by design** — it records leaks in `report_state.limitations` and lets the run finish. In practice, LLMs frequently slip technical terms (RMSE / MAE / precision) into evidence and KPI prose, especially on regression tasks. Example: a successful house-prices test produced a complete report but flagged 12 leaks (RMSE × 1, MAE × 10, precision × 1) plus 1 orphan chart drop.

Two ways to tighten this in the future without breaking the orchestrator's invariants:
- Add stricter formatting examples to `BUSINESS_TRANSLATOR_SYSTEM_PROMPT` (e.g. "write 'about 1 in 8' not 'MAPE 12 %'").
- Promote the scanner to blocking — raise `_V3BusinessTranslatorError` when the leak count exceeds some threshold. This requires adding the threshold knob and is a behaviour change, not a refactor.

### Invariants worth not breaking

- `run_business_translation` is the **only** public surface and must never raise. Adding a new failure mode means adding a new `_V3BusinessTranslatorError` raise *inside* `_v3_run_business_translation_impl`, not propagating a different exception class out of it.
- The HTML file is written only at the very end of the orchestrator. If any step 1–7 fails, no partial file is written.
- Task-family branching belongs in `build_business_translation_prompt` only (i.e. inside `app/prompts/business_prompt.py`). Do not introduce `if task_family == "regression":` branches in the validator, renderer, or evidence pack.
- The evidence pack is the only thing the LLM sees about the run — anything BAT should "know" must flow through `_v3_build_evidence_pack`, not through new top-level `state.*` reads.
- Prompt content lives in `app/prompts/business_prompt.py`. Do not move it back into `business_translator.py`, and do not inline new prompt strings at the call site.

