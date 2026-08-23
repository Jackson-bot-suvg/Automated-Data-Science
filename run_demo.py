import asyncio
import os
import sys
from pprint import pprint
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from app.schemas.state_schema import GraphState, TaskConfig, DatasetPaths, DataState
from app.agents.preliminary_eda import run_preliminary_eda
from app.agents.data_cleaning import run_data_cleaning
from app.agents.feature_engineering import run_feature_engineering


def _check_errors(res: dict, phase_name: str) -> bool:
    """检查节点返回值是否包含 fatal_errors，如果有则打印并返回 True。"""
    review = res.get("review_state", {})
    errors = review.get("fatal_errors", [])
    if errors:
        print(f"\n❌ {phase_name} 出现错误!")
        for e in errors:
            print(f"   → {e}")
        return True
    return False


async def main():
    state = GraphState(
        task_config=TaskConfig(
            task_type="classification",
            target_column="Survived",
            dataset_paths=DatasetPaths(
                train="data/titanic/train.csv", test="data/titanic/test.csv"
            ),
        ),
        data_state=DataState(
            current_train_path="data/titanic/train.csv",
            current_test_path="data/titanic/test.csv",
        ),
    )

    # ── Phase 1: Preliminary EDA ──
    print("🚀 [1/3] 启动 Preliminary EDA 节点...")
    res = run_preliminary_eda(state)
    if _check_errors(res, "Preliminary EDA"):
        return
    state = GraphState(**{**state.model_dump(), **res})
    print("   ✅ EDA 画像生成完毕\n")

    # ── Phase 2: Data Cleaning (调用 GLM-5) ──
    print("🚀 [2/3] 启动 Data Cleaning 节点 (调用 GLM-5 via DashScope)...")
    res = await run_data_cleaning(state)
    if _check_errors(res, "Data Cleaning"):
        return
    state = GraphState(**{**state.model_dump(), **res})
    print("   ✅ 数据清洗完毕\n")

    # ── Phase 3: Feature Engineering (调用 GLM-5) ──
    print("🚀 [3/3] 启动 Feature Engineering 节点 (调用 GLM-5 via DashScope)...")
    res = await run_feature_engineering(state)
    if _check_errors(res, "Feature Engineering"):
        return
    state = GraphState(**{**state.model_dump(), **res})
    print("   ✅ 特征工程完毕\n")

    # ── 结果汇总 ──
    print("=" * 60)
    print("✅ 全管道执行完毕！最终生成的工件：")
    print("=" * 60)
    for key, path in state.artifacts.items():
        if isinstance(path, str) and os.path.isfile(path):
            print(f"  📄 {key}: {path}")

    print(f"\n🔍 数据集变化：")
    raw = pd.read_csv("data/titanic/train.csv")
    final = pd.read_csv(state.feature_state.engineered_train_path)
    print(f"  原始训练集: {raw.shape}")
    print(f"  最终特征集: {final.shape}")
    print(f"\n✨ 最终保留的特征列:")
    print(f"  {', '.join(state.feature_state.selected_features)}")


if __name__ == "__main__":
    asyncio.run(main())
