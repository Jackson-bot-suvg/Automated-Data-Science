## 背景与原因 (Why)

当前的 Titanic 流水线完全依赖硬编码的数据预处理规则，这些规则无法推广到其他的表格数据集上。为了演变为一个真正的类 AutoKaggle 框架，系统架构必须进行重构。原始的 AutoKaggle 依赖 LLM 动态生成不可预测的 Python 脚本，并通过子进程执行和无限报错循环来 Debug，这导致了运行时错误多、调试繁琐且测试性极差。本次更改将引入一个结构化、确定性的工具调用 (Tool Calling) 架构。LLM 将仅作为“规划师” (Planner)，输出结构化的 JSON 调用计划，然后在一个健壮的底层工具库上确定性地执行，从而保证了系统的高可靠性和高可测试性。

## 改动内容 (What Changes)

- 实现一个强大的纯基础数据科学操作底层函数库，覆盖特征侧写、缺失值清洗和特征工程工具。
- 开发一个 LLM Planner 模块，能够读取高度浓缩的数据画像 (Dataset Profiles)，并返回结构化的 JSON 数据处理计划。
- 重构流水线现有的 Agent 节点 (`preliminary_eda.py`, `data_cleaning.py`, `feature_engineering.py`)，从手动规则执行切换为“LLM 生成工具清单 -> 节点执行原生函数”的全新模式。
- 利用 Pydantic 确立严格的数据模型 (Schemas)，约束所有工具调用的参数输入。

## 能力演进 (Capabilities)

### 新增能力 (New Capabilities)
- `llm-planner`: 数据智能规划能力。负责分析统计学视角的数据画像 (JSON Context)，由 LLM 推理后输出结构化的处理任务计划。
- `ds-toolset`: 确定性表格变异能力。这是一个强类型、纯 Python 的函数工具基座，通过底层的原语支持（如删除列、中位数填充等）执行变异。

### 修改的能力 (Modified Capabilities)

## 影响范围 (Impact)

- `app/agents/*.py` 下的所有节点将抛弃自定的业务规则，彻底重构为接入新 Planner 引擎与纯底层工具调用的中继转发站。
- 需要针对新引入的 Pydantic 模型和底层无依赖 Tool 函数重写或扩展现存的单元测试与集成测试框架（全面解耦 LLM 和业务执行逻辑的捆绑）。
