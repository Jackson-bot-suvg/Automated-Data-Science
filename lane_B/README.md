# lane_B — 契约门控机制语义故障注入评估（基础验证）

> 对应 `RTS-TAI-PARALLEL-v4.0` 中**负责人 B** 专项的基础验证版。确定性、离线运行，无需 LLM API key。

## 研究问题

对同一个语义故障，分别用「**完整门控 / 普通结构检查 / 仅入口检查**」三种版本判定，
我们的契约门控机制到底比它们多解决了什么问题？

## 运行方式

在 `cs33-2-deliverable` 目录下：

```bash
python -m lane_B.runner --dry-run   # 打印计划矩阵（不运行）
python -m lane_B.runner             # 生成案例 → 运行 M1/M2/F0 → 落盘 judgments/results
python -m lane_B.analyze            # 汇总生成 summary.md
```

产物均写入 `lane_B/outputs/`。

## 目录结构

```
lane_B/
├── registry.py        # C_gold 真值契约（validation_matrix + sample_submission）
├── cases.py           # 7 类模板 + 案例清单
├── mutators.py        # 在数据拷贝上制造故障（不污染原始 data/）
├── checkers/
│   ├── base.py        # Judgment / 可见包加载 / 契约解析 / 语义检查
│   ├── m1_structural.py  # M1 普通结构检查
│   ├── m2_intake.py      # M2 仅入口检查
│   └── f0_full.py        # F0 完整门控
├── oracle.py          # 离线判 TP/FN/FP/TN + 根因定位
├── runner.py          # 主运行器
├── analyze.py         # 生成 summary.md
└── outputs/           # 运行产物（cases/ manifest/ judgments/ results/ summary.md）
```

## 模板（7 个）与三方法

- **模板**：`C00` 干净对照、`B02` 合法列序变化；`F01` 目标缺失、`F02` 目标泄漏、
  `F03` 目标提示误导、`F04` 输出列不一致、`F05` 入口后漂移。
- **方法**：`M1` 普通结构检查、`M2` 仅入口检查、`F0` 完整门控（入口 + 后阶段持续核查）。
- **数据**：4 个 DEV 数据集覆盖四类任务（二分类 / 多分类 / 回归 / 多标签）。

## 主要发现

| 方法 | TP | FN | FP | TN | 故障检出率 | 误拦率 |
|---|---|---|---|---|---|---|
| M1 普通结构检查 | 0 | 20 | 0 | 8 | 0% | 0% |
| M2 仅入口检查 | 16 | 4 | 0 | 8 | 80% | 0% |
| F0 完整门控 | **20** | **0** | **0** | 8 | **100%** | 0% |

- **F0 vs M1**：多检出 20 个「结构合法但语义错误」的故障（M1 完全看不到）。
- **F0 vs M2**：多检出 4 个 **F05 入口后漂移**（`post_intake`），体现「多阶段持续核查」的增量。
- **零误拦**：三种方法对 C00/B02 均不误拦（FP=0）。
- 结果在 4 个数据集上完全一致（每数据集 F0 均 `5 TP + 2 TN`）。

## 完成率

- ✅ 已完成：本地部署、C_gold 注册表、7 类变异器、3 方法判定器、离线 oracle、
  runner、analyze、`summary.md`、README。
- ⬜ 后续扩展：F06/F07/F08/F09/F10 模板、12 个 CONF 与 4 个 TRANSFER 数据集、
  B1/B3 完整矩阵、带 LLM 与真实训练的端到端验证。

## 说明与限制

- 本版本**确定性离线**：复用项目 `ReaderAgent` 确定性契约解析与 `reviewer_tools` 后阶段检查，
  后阶段故障用合成预测文件模拟（不实际训练模型、不调用 LLM）。
- 原始 `data/` 只读；所有变异写入 `outputs/cases/`，可追溯、可复现。
- 部分目标列故障在 academic_success / plate_defect 上定位为 `partial`（解析器回退到错误目标，
  同时触发 target_columns 与 output_columns），详见 `outputs/summary.md` 第 7 节。
