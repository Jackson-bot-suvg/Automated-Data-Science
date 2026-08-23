## ADDED Requirements

### Requirement: Agent LLM 配置数据模型
系统 SHALL 提供 `AgentLLMConfig` Pydantic 模型，包含以下字段：
- `model`（str）：模型标识符，格式为 `<provider>/[<prefix>/]<model>`，如 `dashscope/glm-5`
- `temperature`（float，默认 `0.0`）
- `max_tokens`（int，默认 `65536`）
- `stream`（bool，默认 `False`）

#### Scenario: 创建默认配置
- **WHEN** 未在 .env 中配置任何 `AGENT_<NAME>_` 变量
- **THEN** `AgentLLMConfig` SHALL 使用默认值：`model="dashscope/glm-5"`, `temperature=0.0`, `max_tokens=65536`, `stream=False`

#### Scenario: 从 .env 读取完整配置
- **WHEN** .env 中设置了 `AGENT_PLANNER_MODEL=dashscope/glm-5`, `AGENT_PLANNER_TEMPERATURE=1.0`, `AGENT_PLANNER_MAX_TOKENS=32768`, `AGENT_PLANNER_STREAM=false`
- **THEN** `get_agent_config("planner")` SHALL 返回 `AgentLLMConfig(model="dashscope/glm-5", temperature=1.0, max_tokens=32768, stream=False)`

### Requirement: get_agent_config 配置读取函数
系统 SHALL 提供 `get_agent_config(agent_name: str) -> AgentLLMConfig` 函数，根据 agent 名称从环境变量读取其 LLM 配置。

环境变量命名规则：`AGENT_<UPPER_NAME>_<FIELD>`，其中 `<UPPER_NAME>` 为 agent 名称的大写形式。

#### Scenario: 读取 agent 配置
- **WHEN** 调用 `get_agent_config("reader")`
- **THEN** 系统 SHALL 查找环境变量 `AGENT_READER_MODEL`、`AGENT_READER_TEMPERATURE`、`AGENT_READER_MAX_TOKENS`、`AGENT_READER_STREAM`

#### Scenario: 部分配置覆盖
- **WHEN** .env 中仅设置了 `AGENT_READER_MODEL=dashscope/qwen-plus`，未设置其他变量
- **THEN** 系统 SHALL 使用 `model="dashscope/qwen-plus"`，其余字段使用默认值

#### Scenario: 无效 temperature 值
- **WHEN** `AGENT_READER_TEMPERATURE` 的值无法转换为 float
- **THEN** 系统 SHALL 使用默认值 `0.0` 并输出警告日志

#### Scenario: 无效 stream 值
- **WHEN** `AGENT_READER_STREAM` 的值不是合法的布尔值
- **THEN** 系统 SHALL 使用默认值 `False`

### Requirement: .env 中支持全部 agent 的配置
系统 SHALL 在 `.env.example` 中为以下 agent 提供配置模板，每个 agent 使用 `AGENT_<NAME>_` 前缀：
- `reader`：`AGENT_READER_MODEL`、`AGENT_READER_TEMPERATURE`、`AGENT_READER_MAX_TOKENS`、`AGENT_READER_STREAM`
- `planner`：`AGENT_PLANNER_MODEL`、`AGENT_PLANNER_TEMPERATURE`、`AGENT_PLANNER_MAX_TOKENS`、`AGENT_PLANNER_STREAM`
- `summarizer`：`AGENT_SUMMARIZER_MODEL`、`AGENT_SUMMARIZER_TEMPERATURE`、`AGENT_SUMMARIZER_MAX_TOKENS`、`AGENT_SUMMARIZER_STREAM`
- `bg_reviewer`：`AGENT_BG_REVIEWER_MODEL`、`AGENT_BG_REVIEWER_TEMPERATURE`、`AGENT_BG_REVIEWER_MAX_TOKENS`、`AGENT_BG_REVIEWER_STREAM`
- `reviewer`：`AGENT_REVIEWER_MODEL`、`AGENT_REVIEWER_TEMPERATURE`、`AGENT_REVIEWER_MAX_TOKENS`、`AGENT_REVIEWER_STREAM`
- `data_cleaning`：`AGENT_DATA_CLEANING_MODEL`、`AGENT_DATA_CLEANING_TEMPERATURE`、`AGENT_DATA_CLEANING_MAX_TOKENS`、`AGENT_DATA_CLEANING_STREAM`
- `feature_engineering`：`AGENT_FEATURE_ENGINEERING_MODEL`、`AGENT_FEATURE_ENGINEERING_TEMPERATURE`、`AGENT_FEATURE_ENGINEERING_MAX_TOKENS`、`AGENT_FEATURE_ENGINEERING_STREAM`

#### Scenario: .env.example 包含所有 agent 配置段
- **WHEN** 开发者查看 `.env.example` 文件
- **THEN** 文件 SHALL 包含上述所有 7 个 agent 的 MODEL、TEMPERATURE、MAX_TOKENS、STREAM 配置项，带有注释说明
