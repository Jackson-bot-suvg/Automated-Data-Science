## ADDED Requirements

### Requirement: complete_json 异步辅助函数
系统 SHALL 在 `app/llm/helpers.py` 中提供 `async complete_json()` 函数，通过 `LLMService.chat()` 调用 LLM 并解析 JSON 响应，失败时返回 fallback 值。

函数签名：`async complete_json(*, agent_name: str, system_prompt: str, user_prompt: str, fallback: Mapping[str, Any]) -> dict[str, Any]`

#### Scenario: LLM 返回合法 JSON
- **WHEN** 调用 `await complete_json(agent_name="reader", system_prompt="...", user_prompt="...", fallback={"key": "default"})` 且 LLM 返回可解析的 JSON
- **THEN** 函数 SHALL 通过 `get_agent_config("reader")` 获取模型配置，调用 `LLMService.chat()` 的非流式模式，解析 JSON 并返回解析后的 dict

#### Scenario: LLM 返回 fenced JSON
- **WHEN** LLM 返回包含 ```` ```json ... ``` ```` 围栏的文本
- **THEN** 函数 SHALL 提取围栏内的 JSON 并解析

#### Scenario: LLM 调用失败
- **WHEN** LLM 调用抛出异常（网络错误、API 错误等）
- **THEN** 函数 SHALL 返回 `fallback` 的副本，不抛出异常

#### Scenario: JSON 解析失败
- **WHEN** LLM 返回的文本无法解析为 JSON dict
- **THEN** 函数 SHALL 返回 `fallback` 的副本

### Requirement: render_text 异步辅助函数
系统 SHALL 在 `app/llm/helpers.py` 中提供 `async render_text()` 函数，通过 `LLMService.chat()` 调用 LLM 并返回文本响应，失败时返回 fallback 文本。

函数签名：`async render_text(*, agent_name: str, system_prompt: str, user_prompt: str, fallback: str) -> str`

#### Scenario: LLM 返回文本
- **WHEN** 调用 `await render_text(agent_name="reviewer", system_prompt="...", user_prompt="...", fallback="default comment")` 且 LLM 调用成功
- **THEN** 函数 SHALL 返回 LLM 响应的 `content` 文本

#### Scenario: LLM 调用失败
- **WHEN** LLM 调用抛出异常
- **THEN** 函数 SHALL 返回 `fallback` 字符串，不抛出异常

#### Scenario: LLM 返回空内容
- **WHEN** LLM 返回空字符串或仅空白字符
- **THEN** 函数 SHALL 返回 `fallback` 字符串

### Requirement: helpers 模块导出
`app/llm/__init__.py` SHALL 导出 `complete_json` 和 `render_text` 函数。

#### Scenario: 从 app.llm 导入
- **WHEN** 使用 `from app.llm import complete_json, render_text`
- **THEN** 导入 SHALL 成功且函数可用

### Requirement: parse_json_response 保留为独立工具函数
`parse_json_response()` 函数 SHALL 迁移到 `app/llm/helpers.py` 并保持原有行为：解析纯文本或围栏 JSON 对象文本。

#### Scenario: 解析围栏 JSON
- **WHEN** 传入 ```` ```json {"key": "value"} ``` ```` 文本
- **THEN** 函数 SHALL 返回 `{"key": "value"}`

#### Scenario: 解析纯 JSON
- **WHEN** 传入 `{"key": "value"}` 文本
- **THEN** 函数 SHALL 返回 `{"key": "value"}`

#### Scenario: 解析失败
- **WHEN** 传入无法解析的文本
- **THEN** 函数 SHALL 返回 `fallback` 的副本
