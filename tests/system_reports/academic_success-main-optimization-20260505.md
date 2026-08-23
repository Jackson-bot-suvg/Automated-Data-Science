# Academic Success Main Optimization

- Generated: `2026-05-05 10:34:00 UTC`
- Baseline matrix row source: [framework-matrix-20260505-051038-corrected-with-lightautods.md](/Users/a99/Downloads/5703-CS33-2/tests/system_reports/framework-matrix-20260505-051038-corrected-with-lightautods.md)
- Old main run: [run_timing.json](/Users/a99/Downloads/5703-CS33-2/artifacts/runs/classification_with_an_academic_success_dataset_20260505_151630_c8277cc2/reports/run_timing.json), [model_results.json](/Users/a99/Downloads/5703-CS33-2/artifacts/runs/classification_with_an_academic_success_dataset_20260505_151630_c8277cc2/models/model_results.json)
- Refreshed main run, matrix-comparable mode: [run_timing.json](/Users/a99/Downloads/5703-CS33-2/artifacts/runs/classification_with_an_academic_success_dataset_20260505_222908_f0fd4fb4/reports/run_timing.json), [model_results.json](/Users/a99/Downloads/5703-CS33-2/artifacts/runs/classification_with_an_academic_success_dataset_20260505_222908_f0fd4fb4/models/model_results.json), [technical_report.md](/Users/a99/Downloads/5703-CS33-2/artifacts/runs/classification_with_an_academic_success_dataset_20260505_222908_f0fd4fb4/reports/technical_report.md)

## Root Cause

旧 run 的高耗时不是清洗或特征工程执行慢，而是两块叠加：

- `model_building` 极重：旧 run 的 `model_building=358.5561s`，见 [旧 run_timing.json](/Users/a99/Downloads/5703-CS33-2/artifacts/runs/classification_with_an_academic_success_dataset_20260505_151630_c8277cc2/reports/run_timing.json)。
- `planner` 也很重：旧 run 的 `planner=123.0479s`，同样见 [旧 run_timing.json](/Users/a99/Downloads/5703-CS33-2/artifacts/runs/classification_with_an_academic_success_dataset_20260505_151630_c8277cc2/reports/run_timing.json)。

其中真正的工程瓶颈是 `model_building`。旧 run 的候选模型是 `LogisticRegression`、`RandomForest`、`GradientBoosting`，最佳模型也是 `GradientBoosting`，见 [旧 model_results.json](/Users/a99/Downloads/5703-CS33-2/artifacts/runs/classification_with_an_academic_success_dataset_20260505_151630_c8277cc2/models/model_results.json)。这个数据集有 `76,518` 行、是单目标三分类，传统 `GradientBoostingClassifier` 在这种规模上明显不合适。

我做了同口径小基准，使用这次任务的 engineered 训练数据、相同 `3-fold f1_macro`：

- `GradientBoosting`: `66.019s`, `f1_macro=0.7881`
- `HistGradientBoosting`: `5.495s`, `f1_macro=0.7918`

这说明慢点主要就在旧的 boosting 候选本身，而不是预处理或 I/O。

## Optimization

这次优化没有改评测口径，只改了 multiclass 建模候选：

- [app/tools/model_tools.py](/Users/a99/Downloads/5703-CS33-2/app/tools/model_tools.py:365)
  - 把分类候选里的 `GradientBoosting` 替换成 `HistGradientBoosting`
  - 保留 `balanced` 类权重
  - 保持同样的 `f1_macro` 选模流程
- [app/agents/model_building.py](/Users/a99/Downloads/5703-CS33-2/app/agents/model_building.py:891)
  - 默认分类候选改成 `LogisticRegression`、`RandomForest`、`HistGradientBoosting`
- [app/agents/planner.py](/Users/a99/Downloads/5703-CS33-2/app/agents/planner.py:679)
  - 允许 `HistGradientBoosting`
  - 把 `gradient_boosting` 这类旧名字别名归一到 `HistGradientBoosting`
- [app/prompts/planner_prompt.py](/Users/a99/Downloads/5703-CS33-2/app/prompts/planner_prompt.py:113)
  - 提示词里的分类 baseline 偏好同步更新为 `HistGradientBoosting`

验证：

- 相关单测已通过：`111 passed`

## Before/After

| Item | Old Main Run | New Main Run |
| --- | ---: | ---: |
| Runtime mode | technical-summary stop | technical-summary stop |
| Total runtime | `507.8830s` | `221.0487s` |
| Planner time | `123.0479s` | `150.7528s` |
| Model building time | `358.5561s` | `43.9175s` |
| Best model | `GradientBoosting` | `HistGradientBoosting` |
| Main metric | `0.7881` | `0.7920` |

结论：

- `model_building` 降了约 `314.64s`，约 `87.8%`
- 总 runtime 从 `507.9s` 降到 `221.0s`，降了约 `56.5%`
- `f1_macro` 还从 `0.7881` 提升到了 `0.7920`

## Remaining Bottleneck

现在 `academic_success` 的主耗时已经不是建模，而是 `planner`。

在新的 matrix-comparable run 里：

- `planner=150.7528s`
- `model_building=43.9175s`

也就是说，工程建模瓶颈已经修掉了，但当前总耗时仍然受 LLM 规划延迟影响，而且这一段有明显波动。这个更像是“规划链路偏重/响应抖动”问题，不是 sklearn 训练问题。

## Refreshed Row

如果只刷新 `academic_success` 这一行，三框架对比会变成：

| Dataset | Metric | Main | AIDE | LightAutoDS | Better | Advantage | Main Time (s) | AIDE Time (s) | LightAutoDS Time (s) |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `academic_success` | `f1_macro` | 0.7920 | 0.7921 | 0.7929 | `lightautods_lama` | 0.0008 | 221.0487 | 147.5690 | 109.4121 |

这表示：

- `main` 已经从原来的明显落后，提升到几乎追平
- 现在 `main` 比 `AIDE` 只低约 `0.0001`
- 距 `LightAutoDS` 只低约 `0.0008`

如果你要，我下一步可以继续只盯 `planner` 做第二轮提速，把 academic_success 的总耗时再往下压。 
