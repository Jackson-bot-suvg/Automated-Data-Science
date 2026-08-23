## MODIFIED Requirements

### Requirement: 输出受控结构调令 (Output Structured Tool Calls)
LLM Planner 在被赋予数据集画像配置约束时，必须且只能吐出格式化的离散 `ToolCall` 调用清单，严禁输出原生包裹格式的 Python 执行脚本。结构化输出 SHALL 通过 `app/llm/` 服务层的 `ChatParams.response_schema` 参数实现，而非直接使用 LangChain。

Planner 的数据清洗和特征工程 Prompt MUST 注入 `target_spec` 摘要信息，使 LLM 感知当前任务的目标列、ID 列和任务类型，从而生成目标安全的工具调用计划。

#### Scenario: Planner 使用 LLM 服务层生成结构化计划
- **WHEN** Planner 接收到数据集画像 JSON 并需要生成工具调用计划
- **THEN** Planner SHALL 通过 `LLMService.chat(model, messages, response_schema=ToolCallPlan, stream=False)` 调用 LLM，并从 `ChatResponse.parsed` 获取 `ToolCallPlan` 实例

#### Scenario: Planner 不再依赖 LangChain
- **WHEN** `BasePlanner` 初始化或执行计划生成
- **THEN** 系统 SHALL NOT 导入或使用 `langchain_openai`、`langchain_core` 的任何模块，所有 LLM 调用统一走 `app/llm/` 服务层

#### Scenario: Planner 保持异步接口
- **WHEN** 调用 `BasePlanner.plan_data_cleaning()` 或 `plan_feature_engineering()`
- **THEN** 这些方法 SHALL 为 `async` 方法，返回 `ToolCallPlan`，调用者需使用 `await`

#### Scenario: 数据清洗 Prompt 包含目标上下文
- **WHEN** 调用 `build_data_cleaning_prompt(profile_str, target_spec_summary)`
- **THEN** 生成的 Prompt 中 MUST 包含 `【目标协议 (TargetSpec Summary)】` 段落，列出目标列名、任务类型和 ID 列

#### Scenario: 特征工程 Prompt 包含目标上下文
- **WHEN** 调用 `build_feature_engineering_prompt(profile_str, target_spec_summary)`
- **THEN** 生成的 Prompt 中 MUST 包含 `【目标协议 (TargetSpec Summary)】` 段落，列出目标列名、任务类型和 ID 列

#### Scenario: BasePlanner 方法签名更新
- **WHEN** 调用 `plan_data_cleaning()` 或 `plan_feature_engineering()`
- **THEN** 方法签名 MUST 接受 `target_spec_summary: dict` 参数，并传递给对应的 Prompt 构建函数
