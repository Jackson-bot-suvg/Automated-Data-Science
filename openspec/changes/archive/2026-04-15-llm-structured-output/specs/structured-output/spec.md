## ADDED Requirements

### Requirement: ChatParams 支持 response_schema 参数
`ChatParams` 模型 SHALL 包含一个可选字段 `response_schema: type[BaseModel] | None`，默认值为 `None`。当该字段不为 `None` 时，表示调用者期望 LLM 输出符合指定 Pydantic 模型的结构化 JSON。

#### Scenario: 传入 response_schema 进行结构化调用
- **WHEN** 调用者通过 `ChatParams(response_schema=ToolCallPlan)` 传入 Pydantic 模型
- **THEN** provider SHALL 将该 schema 传递给底层 LLM API 的结构化输出功能，并返回解析后的 Pydantic 对象

#### Scenario: 不传 response_schema 保持原有行为
- **WHEN** 调用者不传 `response_schema`（默认 `None`）
- **THEN** provider SHALL 按照原有纯文本模式工作，`ChatResponse.parsed` 为 `None`

### Requirement: ChatParams 支持 max_tokens 参数
`ChatParams` 模型 SHALL 包含一个可选字段 `max_tokens: int | None`，默认值为 `None`。当该字段不为 `None` 时，provider SHALL 将其作为生成 token 上限传递给底层 API。

#### Scenario: 设置 max_tokens 限制输出长度
- **WHEN** 调用者通过 `ChatParams(max_tokens=65536)` 设置 token 上限
- **THEN** provider SHALL 将该参数传递给底层 LLM API 的 `max_tokens` 参数

#### Scenario: 不传 max_tokens 使用 API 默认值
- **WHEN** 调用者不传 `max_tokens`（默认 `None`）
- **THEN** provider SHALL 不设置 `max_tokens` 参数，使用 API 服务端的默认值

### Requirement: ChatResponse 支持 parsed 字段
`ChatResponse` 模型 SHALL 包含一个可选字段 `parsed: BaseModel | None`，默认值为 `None`。当 `ChatParams.response_schema` 不为 `None` 且解析成功时，`parsed` SHALL 包含解析后的 Pydantic 模型实例。

#### Scenario: 结构化输出解析成功
- **WHEN** LLM 返回符合 `response_schema` 的 JSON 且 Pydantic 解析成功
- **THEN** `ChatResponse.parsed` SHALL 为对应 Pydantic 模型的实例，`ChatResponse.content` SHALL 保留原始文本内容

#### Scenario: 结构化输出解析失败
- **WHEN** LLM 返回的 JSON 不符合 `response_schema` 或解析过程出错
- **THEN** 系统 SHALL 抛出 `LLMError`，包含明确的解析失败信息

### Requirement: OpenAI provider 支持结构化输出
`OpenAICompatProvider` SHALL 在 `ChatParams.response_schema` 不为 `None` 时，使用 OpenAI SDK 的 `beta.chat.completions.parse()` API 传入 `response_format` 参数实现结构化输出。

#### Scenario: 使用 parse API 进行结构化调用
- **WHEN** 调用 `OpenAICompatProvider.chat()` 时 `params.response_schema` 不为 `None`
- **THEN** provider SHALL 调用 `client.beta.chat.completions.parse(response_format=schema)` 并从响应中提取解析后的对象到 `ChatResponse.parsed`

#### Scenario: parse API 不被服务端支持时的降级
- **WHEN** `parse()` API 因服务端不支持而返回错误
- **THEN** provider SHALL 降级为 `chat.completions.create(response_format={"type": "json_object"})` + 在 messages 中注入 schema + 手动 `model_validate_json()` 解析

### Requirement: Anthropic provider 支持结构化输出
`AnthropicProvider` SHALL 在 `ChatParams.response_schema` 不为 `None` 时，通过在 system prompt 中注入 JSON schema 约束 + JSON mode + Pydantic `model_validate_json()` 实现结构化输出。

#### Scenario: 使用 prompt 约束进行结构化调用
- **WHEN** 调用 `AnthropicProvider.chat()` 时 `params.response_schema` 不为 `None`
- **THEN** provider SHALL 在 system prompt 末尾追加 JSON schema 描述，启用 JSON mode，并在收到响应后使用 `response_schema.model_validate_json()` 解析内容到 `ChatResponse.parsed`

#### Scenario: Anthropic 结构化输出仅支持非流式
- **WHEN** 调用 `AnthropicProvider.chat()` 时 `params.response_schema` 不为 `None` 且 `params.stream=True`
- **THEN** 系统 SHALL 抛出 `ValueError`，提示结构化输出仅支持非流式模式

### Requirement: 结构化输出仅支持非流式模式
当 `ChatParams.response_schema` 不为 `None` 时，系统 SHALL 强制使用非流式模式（`stream=False`），即使调用者传入了 `stream=True`。

#### Scenario: 传入 response_schema 且 stream=True
- **WHEN** 调用者传入 `ChatParams(response_schema=SomeModel, stream=True)`
- **THEN** 系统 SHALL 将 `stream` 覆盖为 `False` 并执行非流式调用，或在 provider 层抛出明确错误
