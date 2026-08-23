## MODIFIED Requirements

### Requirement: 可选参数支持
系统 SHALL 支持以下可选参数，传递给对应的 LLM API：
- `temperature`（float，控制随机性）
- `top_p`（float，核采样参数）
- `stream`（bool，默认 `True`）
- `thinking_budget`（int，思考预算 token 数，仅部分模型支持）

此外，当调用方通过 `get_agent_config()` 获取的 `AgentLLMConfig` 传入时，系统 SHALL 将其 `temperature`、`max_tokens`、`stream` 字段作为默认参数传递给 `LLMService.chat()`，调用方显式传入的同名参数 SHALL 覆盖 agent 配置中的默认值。

#### Scenario: 传递 temperature 和 top_p
- **WHEN** 调用 `chat("openai/GPT-5.4", messages, temperature=0.7, top_p=0.9)`
- **THEN** 系统 SHALL 将 `temperature=0.7` 和 `top_p=0.9` 传递给底层 OpenAI API 调用

#### Scenario: Agent 配置提供默认参数
- **WHEN** `AgentLLMConfig` 中 `temperature=1.0`, `max_tokens=32768`, `stream=False`，且调用方未显式传入这些参数
- **THEN** 系统 SHALL 使用 agent 配置中的值作为参数传递给 LLM API

#### Scenario: 显式参数覆盖 agent 配置
- **WHEN** `AgentLLMConfig` 中 `temperature=0.0`，但调用方显式传入 `temperature=0.7`
- **THEN** 系统 SHALL 使用 `temperature=0.7`
