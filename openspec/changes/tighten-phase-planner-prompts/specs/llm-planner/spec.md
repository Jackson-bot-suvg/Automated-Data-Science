## MODIFIED Requirements

### Requirement: 输出受控结构调令 (Output Structured Tool Calls)
LLM Planner 在被赋予数据集画像配置约束时，必须且只能吐出格式化的离散 `ToolCall` 调用清单，严禁输出原生包裹格式的 Python 执行脚本。结构化输出 SHALL 通过 `app/llm/` 服务层的 `ChatParams.response_schema` 参数实现，而非直接使用 LangChain。对于执行 phase，Planner SHALL 只基于当前 phase 的工具 allowlist 生成计划，并且 SHALL 在提示词中显式声明禁止使用其它 phase 的工具。

#### Scenario: Planner 使用 LLM 服务层生成结构化计划
- **WHEN** Planner 接收到数据集画像 JSON 并需要生成工具调用计划
- **THEN** Planner SHALL 通过 `LLMService.chat(model, messages, response_schema=ToolCallPlan, stream=False)` 调用 LLM，并从 `ChatResponse.parsed` 获取 `ToolCallPlan` 实例

#### Scenario: Planner 不再依赖 LangChain
- **WHEN** `BasePlanner` 初始化或执行计划生成
- **THEN** 系统 SHALL NOT 导入或使用 `langchain_openai`、`langchain_core` 的任何模块，所有 LLM 调用统一走 `app/llm/` 服务层

#### Scenario: Planner 保持异步接口
- **WHEN** 调用 `BasePlanner.plan_data_cleaning()` 或 `plan_feature_engineering()`
- **THEN** 这些方法 SHALL 为 `async` 方法，返回 `ToolCallPlan`，调用者需使用 `await`

#### Scenario: Execution phase planner sees only phase-scoped tools
- **WHEN** Planner 为 `data_cleaning`、`indepth_eda`、`feature_engineering` 或 `model_building` 生成计划
- **THEN** 提示词 SHALL 只暴露当前 phase 的工具 allowlist、当前 phase 的参数 schema，以及当前 phase 的约束说明，而 SHALL NOT 要求模型参考其他 phase 的工具集合

### Requirement: Planner prompt guardrails for executable phase plans
对于可执行 phase，Planner prompt SHALL 提供足够强的护栏来提高工具选择和参数生成的正确率。护栏 MUST 包括 phase-specific hard rules、常见反模式约束、面向输出前的内部自检要求，以及在 phase 明显可行动时优先返回安全最小计划而不是空步骤列表。

#### Scenario: Feature engineering prompt forbids protected-column transformations
- **WHEN** Planner 为 `feature_engineering` 生成计划
- **THEN** 提示词 SHALL 明确禁止对 `target_spec.target_columns` 和 `submission.id_columns` 规划特征变换步骤

#### Scenario: Prompt discourages schema-incompatible tool choices
- **WHEN** Planner 为任意执行 phase 生成计划
- **THEN** 提示词 SHALL 明确要求工具选择与已观测 schema 一致，例如 regex 工具仅用于可解析字符串列，算术交互仅用于数值列，类别归组仅用于分类列

#### Scenario: Prompt prefers safe minimal executable plans
- **WHEN** 当前 phase 存在明确可执行动作且 profile evidence 足以支持保守计划
- **THEN** Planner SHALL 优先返回一个安全最小计划，而 SHALL NOT 仅因证据不完美就返回空 `steps`

#### Scenario: Prompt requires internal plan self-check
- **WHEN** Planner 生成 `ToolCallPlan`
- **THEN** 提示词 SHALL 要求模型在输出前内部检查工具是否在 allowlist 中、参数名是否匹配 schema、必填字段是否存在、列名是否来自观测 schema、以及是否违反 protected-column 约束

### Requirement: Task-aware model-building planning guidance
当 Planner 为 `model_building` 生成计划时，提示词 SHALL 根据 `target_spec.task_family` 与 `target_spec.target_mode` 提供任务类型感知的候选模型、交叉验证和评分指导，以减少语义正确但任务不匹配的配置。

#### Scenario: Multilabel model-building prompt avoids binary-only scoring assumptions
- **WHEN** `target_spec` 表示 `multilabel_classification` 或 `multi_label`
- **THEN** 提示词 SHALL 明确避免仅适用于二分类的评分假设，并要求 scoring 配置与 multi-label 目标语义一致

#### Scenario: Binary or multiclass model-building prompt prefers task-compatible baselines
- **WHEN** `target_spec` 表示二分类或多分类任务
- **THEN** 提示词 SHALL 提供与该任务族兼容的候选模型、CV 策略和主要指标指导，而不是只依赖通用默认描述

#### Scenario: Regression model-building prompt uses regression-compatible guidance
- **WHEN** `target_spec` 表示回归任务
- **THEN** 提示词 SHALL 引导 Planner 选择回归兼容的候选模型和评分配置，而 SHALL NOT 复用分类导向的建议
