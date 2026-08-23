"""初步探索性数据分析（Preliminary EDA）节点。

读取原始 CSV，调用底层的 profiler 工具生成精确的上下文 JSON，
从而为 LLM Planner 准备结构化的画像数据。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from app.schemas.state_schema import GraphState
from app.tools.profiler import profile_dataset
from app.tools import file_tools
from app.utils.target_scope import get_target_columns

# ──────────────────────────────────────────────
# 保留工件路径
# ──────────────────────────────────────────────
_EDA_REPORT_KEY = "preliminary_eda_report"


def run_preliminary_eda(state: GraphState) -> dict:
    """执行 Preliminary EDA 阶段，返回局部 State 更新字典。

    核心逻辑已变更为使用 profile_dataset 提取高密度特征画像（JSON），
    存入 data_state.eda_findings，供下一节点的 LLM Planner 消费。
    """
    train_path = state.data_state.current_train_path
    test_path = state.data_state.current_test_path

    # ── 1. 路径校验 ──────────────────────────────
    if not train_path or not Path(train_path).exists():
        return _error_envelope(
            phase="preliminary_eda",
            error_type="artifact_missing",
            message=f"训练集文件不存在或路径为空: '{train_path}'",
        )

    # ── 2. 读取数据 ──────────────────────────────
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path) if test_path and Path(test_path).exists() else None

    # ── 3. 从 target_spec 获取目标列（不再硬编码）────
    target_cols = get_target_columns(state)

    # 提取高度浓缩的统计画像
    eda_findings_json = profile_dataset(df_train, target_cols=target_cols)

    # ── 4. Schema 摘要为了向下兼容暂只含简易信息 ─────
    schema_summary = {"columns": list(df_train.columns)}

    # ── 5. 确保目录存在 ───────────────────────────
    report_path = str(state.artifacts.get(_EDA_REPORT_KEY) or "artifacts/eda/preliminary_eda.md")
    os.makedirs(Path(report_path).parent, exist_ok=True)

    # ── 6. 生成基础分布图表（为了人类可视化）────────
    new_figures: list[str] = []
    # 缺失值
    missing = df_train.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=missing.index.tolist(), y=missing.values.tolist(), ax=ax, palette="viridis")
        ax.set_title("Missing Values Training Set")
        plt.tight_layout()
        path = file_tools.figure_path(
            workspace_root=state.runtime_state.workspace_root,
            artifact_root=state.runtime_state.artifact_root,
            filename="eda_missingness.png",
        )
        fig.savefig(path, dpi=100)
        plt.close(fig)
        new_figures.append(path)

    # ── 7. 将结构化 profile 写入 Markdown (让人看) ──
    _write_eda_report(df_train, eda_findings_json, new_figures, report_path)

    existing_figures: list[str] = list(state.artifacts.get("figures", []))

    # ── 8. 将提取出的画像推到下游流程给 Planner ──
    return {
        "data_state": {
            **state.data_state.model_dump(),
            "schema_summary": schema_summary,
            "eda_findings": eda_findings_json,  # 将变成 Dict/JSON 发给 LLM Planner
        },
        "artifacts": {
            **state.artifacts,
            "preliminary_eda_report": report_path,
            "figures": existing_figures + new_figures,
        },
        "current_phase": "preliminary_eda",
        "phase_history": ["preliminary_eda"],  
    }


# ──────────────────────────────────────────────
# 内部辅助函数
# ──────────────────────────────────────────────

def _write_eda_report(
    df: pd.DataFrame,
    profile_json: dict,
    figures: list[str],
    report_path: str,
) -> None:
    """将 EDA 提取出的 JSON Profile 简单铺设成 Markdown。"""
    lines: list[str] = [
        "# Preliminary EDA 特征画像报告\n\n",
        f"该报告由底层 Profiler 生成，它将被完整提交给 LLM Planner 进行工具调用。\n\n",
        "## 数据集概览\n",
        f"- 行数：{profile_json['dataset_shape']['rows']}\n",
        f"- 列数：{profile_json['dataset_shape']['columns']}\n\n",
        "## 核心特征侧写 (Context 给 LLM)\n\n"
        "```json\n" + json.dumps(profile_json, ensure_ascii=False, indent=2) + "\n```\n\n"
    ]

    if figures:
        lines.append("\n## 生成图表\n\n")
        for fig_path in figures:
            lines.append(f"- `{fig_path}`\n")

    with open(report_path, "w", encoding="utf-8") as fh:
        fh.writelines(lines)


def _error_envelope(phase: str, error_type: str, message: str, details: dict | None = None) -> dict:
    error_str = f"[{phase}] {error_type}: {message}"
    return {
        "runtime_state": {
            "interrupted": True,
        },
        "review_state": {
            "fatal_errors": [error_str]
        },
    }
