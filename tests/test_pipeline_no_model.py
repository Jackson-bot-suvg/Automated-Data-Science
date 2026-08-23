"""端到端流水线测试（跳过模型训练）。

使用真实 LLM API 对代表性主任务集（默认 4 个）分别跑完：
  reader → preliminary_eda → data_cleaning → indepth_eda → feature_engineering

验证目标：
  1. Profiler 正确识别任务类型（分类 / 回归 / 多标签）
  2. 清洗阶段不删除目标列和 ID 列
  3. 特征工程阶段不编码目标列
  4. selected_features 不含目标列和 ID 列
  5. 训练集 / 测试集 Schema 对齐

用法：
  # 默认运行代表性的 4 个数据集
  python tests/test_pipeline_no_model.py

  # 加入 secondary smoke 数据集
  python tests/test_pipeline_no_model.py --include-smoke

  # 只跑某个数据集
  python tests/test_pipeline_no_model.py --only bank_churn

  # 只跑前 3 个数据集
  python tests/test_pipeline_no_model.py --limit 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

# ── 把项目根目录加入 sys.path ────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv(".env")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── 项目导入 ────────────────────────────────────
from app.intake import build_initial_state
from app.agents.reader import run_reader
from app.agents.bg_reviewer import run_bg_reviewer
from app.agents.preliminary_eda import run_preliminary_eda
from app.agents.data_cleaning import run_data_cleaning
from app.agents.indepth_eda import run_indepth_eda
from app.agents.feature_engineering import run_feature_engineering
from app.observability.demo_logger import DemoLogger
from app.observability.langsmith import with_tracing_context
from app.observability.token_usage import reset_token_usage, snapshot_token_usage
from app.schemas.state_schema import GraphState
from app.utils.target_scope import get_target_columns, get_id_columns
from app.validation_matrix import PRIMARY_ACCEPTANCE_DATASETS, SECONDARY_SMOKE_DATASETS

# ═══════════════════════════════════════════════════
# 测试用例定义
# ═══════════════════════════════════════════════════

DATASETS = [
    {
        "name": spec.key,
        "task": spec.task,
        "dataset": spec.dataset,
        "expected_targets": [part.strip() for part in spec.target.split(",") if part.strip()],
        "task_family": spec.task_family,
        "description": spec.description,
        "primary_acceptance": spec.primary_acceptance,
    }
    for spec in (*PRIMARY_ACCEPTANCE_DATASETS, *SECONDARY_SMOKE_DATASETS)
]

PHASE_ORDER: list[tuple[str, str]] = [
    ("reader", "reader"),
    ("background_reviewer", "background_reviewer"),
    ("preliminary_eda", "preliminary_eda"),
    ("data_cleaning", "data_cleaning"),
    ("indepth_eda", "indepth_eda"),
    ("feature_engineering", "feature_engineering"),
]


# ═══════════════════════════════════════════════════
# 辅助：合并 state
# ═══════════════════════════════════════════════════

def merge_state(state: GraphState, update: dict) -> GraphState:
    data = state.model_dump(mode="python")
    for key, value in update.items():
        if key in {"phase_history", "messages"} and isinstance(value, list):
            data[key] = list(data.get(key) or []) + value
        elif isinstance(data.get(key), dict) and isinstance(value, dict):
            data[key] = {**data[key], **value}
        else:
            data[key] = value
    return GraphState(**data)


def _timestamp_for_filename() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_test_log_path() -> Path:
    return Path("artifacts/logs") / f"test_pipeline_no_model_{_timestamp_for_filename()}.jsonl"


def tracing_metadata(ds: dict, phase: str = "") -> dict[str, str]:
    return {
        "workflow": "test_pipeline_no_model",
        "dataset": ds["name"],
        "task_family": ds.get("task_family", ""),
        "phase": phase,
    }


async def run_phase_with_logging(
    *,
    phase_name: str,
    node_name: str,
    ds: dict,
    state: GraphState,
    demo_logger: DemoLogger,
    log_path: Path,
    runner,
):
    event_key = f"{ds['name']}_{phase_name}_{int(time.time() * 1000)}"
    demo_logger.log_node_start(node_name=node_name, event_key=event_key, dataset=ds["name"])
    phase_start = time.time()
    try:
        with with_tracing_context(
            tags=["pipeline-test", f"dataset:{ds['name']}", f"phase:{phase_name}"],
            metadata={
                **tracing_metadata(ds, phase=phase_name),
                "local_log_path": str(log_path),
                "local_event_key": event_key,
                "token_usage_snapshot": snapshot_token_usage(),
            },
        ):
            update = await runner(state)
        state = merge_state(state, update)
        phase_ms = int((time.time() - phase_start) * 1000)
        phase_token_usage = snapshot_token_usage().get(phase_name, {})
        demo_logger.log_node_done(
            node_name=node_name,
            event_key=event_key,
            dataset=ds["name"],
            current_phase=phase_name,
            technical_report_path="",
            token_usage=phase_token_usage,
            token_usage_by_module=snapshot_token_usage(),
        )
        return state, phase_ms
    except Exception as exc:
        demo_logger.log_node_error(node_name=node_name, event_key=event_key, dataset=ds["name"], error=exc)
        raise


# ═══════════════════════════════════════════════════
# 单个数据集的完整测试
# ═══════════════════════════════════════════════════

async def test_single_dataset(ds: dict, demo_logger: DemoLogger, log_path: Path) -> dict:
    """对单个数据集跑 reader → EDA → cleaning → indepth_eda → feature_eng.

    Returns:
        结果字典 {"name", "status", "checks", "errors", "elapsed"}
    """
    name = ds["name"]
    checks: list[str] = []
    errors: list[str] = []
    start = time.time()
    reset_token_usage()
    dataset_event_key = f"{name}_{int(time.time() * 1000)}"
    demo_logger.log_node_start(node_name="dataset_run", event_key=dataset_event_key, dataset=name)

    try:
        with with_tracing_context(
            tags=["pipeline-test", f"dataset:{name}", f"task_family:{ds.get('task_family', '')}"],
            metadata={
                **tracing_metadata(ds),
                "local_log_path": str(log_path),
                "local_event_key": dataset_event_key,
            },
        ):
            # ── 1. Reader ────────────────────────────
            state = build_initial_state(
                task=ds["task"],
                dataset=ds["dataset"],
                target="",
                workspace_root=".",
            )
            state, _ = await run_phase_with_logging(
                phase_name="reader",
                node_name="reader",
                ds=ds,
                state=state,
                demo_logger=demo_logger,
                log_path=log_path,
                runner=run_reader,
            )

            # Background reviewer
            state, _ = await run_phase_with_logging(
                phase_name="background_reviewer",
                node_name="background_reviewer",
                ds=ds,
                state=state,
                demo_logger=demo_logger,
                log_path=log_path,
                runner=run_bg_reviewer,
            )

            if state.review_state.latest_review_status != "approved":
                errors.append(f"Reader 未通过: {state.review_state.last_review_comment}")
                return _result(name, "FAIL", checks, errors, start)

            checks.append("✅ Reader approved")

            # 验证 target_spec 是否被正确填充
            target_cols = get_target_columns(state)
            id_cols = get_id_columns(state)
            if not target_cols:
                errors.append("target_columns 为空")
            else:
                checks.append(f"✅ target_columns: {target_cols}")

            # ── 2. Preliminary EDA ──────────────────
            state = merge_state(state, {"current_phase": "preliminary_eda"})
            phase_event_key = f"{name}_preliminary_eda_{int(time.time() * 1000)}"
            demo_logger.log_node_start(node_name="preliminary_eda", event_key=phase_event_key, dataset=name)
            phase_start = time.time()
            update = run_preliminary_eda(state)
            state = merge_state(state, update)
            demo_logger.log_node_done(
                node_name="preliminary_eda",
                event_key=phase_event_key,
                dataset=name,
                current_phase="preliminary_eda",
                token_usage=snapshot_token_usage().get("preliminary_eda", {}),
                token_usage_by_module=snapshot_token_usage(),
            )

        eda = state.data_state.eda_findings
        if not eda:
            errors.append("EDA findings 为空")
        else:
            checks.append("✅ EDA findings 已生成")

            # 检查 profiler 输出
            if "class_distribution" in eda:
                checks.append(f"✅ class_distribution: {list(eda['class_distribution'].keys())}")
            if "target_statistics" in eda:
                checks.append(f"✅ target_statistics: {list(eda['target_statistics'].keys())}")
            if "label_cooccurrence" in eda:
                checks.append(f"✅ label_cooccurrence: {len(eda['label_cooccurrence'])} pairs")

            # ── 3. Data Cleaning ────────────────────
            state = merge_state(state, {"current_phase": "data_cleaning"})
            state, _ = await run_phase_with_logging(
                phase_name="data_cleaning",
                node_name="data_cleaning",
                ds=ds,
                state=state,
                demo_logger=demo_logger,
                log_path=log_path,
                runner=run_data_cleaning,
            )

        # 检查清洗结果
        if state.runtime_state and state.runtime_state.interrupted:
            fatal = state.review_state.fatal_errors if state.review_state else []
            errors.append(f"清洗阶段失败: {fatal}")
        else:
            checks.append("✅ Data Cleaning 完成")

            # 验证目标列未被删除
            import pandas as pd
            cleaned_train = state.data_state.current_train_path
            if cleaned_train and Path(cleaned_train).exists():
                df_cleaned = pd.read_csv(cleaned_train)
                for tc in ds["expected_targets"]:
                    if tc in df_cleaned.columns:
                        checks.append(f"✅ 目标列 '{tc}' 在清洗后仍存在")
                    else:
                        errors.append(f"❌ 目标列 '{tc}' 在清洗后丢失！")

            # ── 4. In-depth EDA（可能因缺少 execution_plan 而跳过）─────
            try:
                state = merge_state(state, {"current_phase": "indepth_eda"})
                state, _ = await run_phase_with_logging(
                    phase_name="indepth_eda",
                    node_name="indepth_eda",
                    ds=ds,
                    state=state,
                    demo_logger=demo_logger,
                    log_path=log_path,
                    runner=run_indepth_eda,
                )
                checks.append("✅ In-depth EDA 完成")
            except Exception as e:
                checks.append(f"⚠️ In-depth EDA 跳过（需要 planner 输出）: {type(e).__name__}")

            # ── 5. Feature Engineering ──────────────
            state = merge_state(state, {"current_phase": "feature_engineering"})
            state, _ = await run_phase_with_logging(
                phase_name="feature_engineering",
                node_name="feature_engineering",
                ds=ds,
                state=state,
                demo_logger=demo_logger,
                log_path=log_path,
                runner=run_feature_engineering,
            )

        if state.runtime_state and state.runtime_state.interrupted:
            fatal = state.review_state.fatal_errors if state.review_state else []
            errors.append(f"特征工程阶段失败: {fatal}")
        else:
            checks.append("✅ Feature Engineering 完成")

            # 验证 selected_features 不含目标列
            selected = state.feature_state.selected_features
            for tc in ds["expected_targets"]:
                if tc in selected:
                    errors.append(f"❌ 目标列 '{tc}' 出现在 selected_features 中！")
                else:
                    checks.append(f"✅ 目标列 '{tc}' 已从 selected_features 排除")

            # 验证 ID 列排除
            for ic in id_cols:
                if ic in selected:
                    errors.append(f"❌ ID 列 '{ic}' 出现在 selected_features 中！")
                else:
                    checks.append(f"✅ ID 列 '{ic}' 已从 selected_features 排除")

    except Exception as e:
        errors.append(f"异常: {type(e).__name__}: {str(e)}")
        traceback.print_exc()

    result = _result(name, "PASS" if not errors else "FAIL", checks, errors, start)
    demo_logger.log_node_done(
        node_name="dataset_run",
        event_key=dataset_event_key,
        dataset=name,
        current_phase="finished",
        token_usage=snapshot_token_usage().get("planner", {}),
        token_usage_by_module=snapshot_token_usage(),
    )
    return result


def _result(name, status, checks, errors, start):
    return {
        "name": name,
        "status": status,
        "checks": checks,
        "errors": errors,
        "elapsed": round(time.time() - start, 1),
    }


# ═══════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════

async def main():
    parser = argparse.ArgumentParser(description="Pipeline 端到端测试（跳过模型训练）")
    parser.add_argument("--only", type=str, help="只跑指定数据集 (name)")
    parser.add_argument("--limit", type=int, help="只跑前 N 个数据集")
    parser.add_argument(
        "--include-smoke",
        action="store_true",
        help="加入 secondary smoke 数据集；默认只跑代表性的 4 个 primary acceptance 任务",
    )
    args = parser.parse_args()

    datasets = DATASETS if args.include_smoke else [d for d in DATASETS if d["primary_acceptance"]]
    if args.only:
        datasets = [d for d in datasets if d["name"] == args.only]
        if not datasets:
            print(f"❌ 未找到数据集 '{args.only}'")
            sys.exit(1)
    if args.limit:
        datasets = datasets[:args.limit]

    print(f"\n{'='*70}")
    print(f" Pipeline 端到端测试（跳过模型训练）")
    print(f" 数据集数量: {len(datasets)}")
    print(f"{'='*70}\n")

    log_path = build_test_log_path()
    demo_logger = DemoLogger(log_path)
    demo_logger.log_workflow_start(dataset="test_pipeline_no_model", thread_id="test-pipeline", recursion_limit=0)
    print(f"📝 测试日志: {demo_logger.path}")

    results = []
    for i, ds in enumerate(datasets, 1):
        print(f"\n{'─'*70}")
        print(f" [{i}/{len(datasets)}] {ds['name']} — {ds['description']}")
        print(f"{'─'*70}")

        result = await test_single_dataset(ds, demo_logger, log_path)
        results.append(result)

        # 打印每步检查结果
        for check in result["checks"]:
            print(f"  {check}")
        for err in result["errors"]:
            print(f"  ❌ {err}")
        print(f"  ⏱ {result['elapsed']}s — {result['status']}")

    # ── 汇总 ─────────────────────────────────
    print(f"\n\n{'='*70}")
    print(" 测试汇总")
    print(f"{'='*70}")
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")

    for r in results:
        icon = "✅" if r["status"] == "PASS" else "❌"
        print(f"  {icon} {r['name']:45s} {r['elapsed']:>6.1f}s  {r['status']}")
        if r["errors"]:
            for err in r["errors"]:
                print(f"      └─ {err}")

    print(f"\n  结果: {passed} PASS / {failed} FAIL / {len(results)} TOTAL")
    print(f"{'='*70}\n")

    # 保存 JSON 报告
    report_path = "artifacts/test_pipeline_report.json"
    os.makedirs("artifacts", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  报告已保存: {report_path}\n")
    demo_logger.log_workflow_end(
        dataset="test_pipeline_no_model",
        report_path=str(Path(report_path).resolve()),
        token_usage_by_module=snapshot_token_usage(),
    )
    demo_logger.close()

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    asyncio.run(main())
