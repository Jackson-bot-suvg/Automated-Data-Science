## ADDED Requirements

### Requirement: 统一聊天接口
系统 SHALL 提供统一的 `chat()` 方法，接受模型标识、消息列表和可选参数，返回 LLM 响应。方法签名 SHALL 为：`chat(model: str, messages: list[ChatMessage], **params) -> ChatResponse | AsyncGenerator[str, None]`。

#### Scenario: 非流式调用
- **WHEN** 调用 `chat("openai/GPT-5.4", messages, stream=False)`
- **THEN** 系统 SHALL 返回 `ChatResponse` 对象，包含完整的响应文本、模型名称、使用量信息

#### Scenario: 流式调用（默认）
- **WHEN** 调用 `chat("openai/GPT-5.4", messages)` （未指定 stream 或 stream=True）
- **THEN** 系统 SHALL 返回 `AsyncGenerator[str, None]`，逐步 yield 响应文本的 token

#### Scenario: Anthropic 流式调用
- **WHEN** 调用 `chat("anthropic/claude-opus-4.6", messages, stream=True)`
- **THEN** 系统 SHALL 通过 Anthropic Messages API 流式接口返回 token 生成器

### Requirement: 消息格式
系统 SHALL 定义统一的 `ChatMessage` 数据结构，包含 `role`（system/user/assistant）和 `content`（文本内容）。内部自动转换为各供应商所需的 message 格式。

#### Scenario: 基本消息转换
- **WHEN** 传入 `ChatMessage(role="user", content="Hello")`
- **THEN** OpenAI provider SHALL 将其转换为 `{"role": "user", "content": "Hello"}`
- **AND** Anthropic provider SHALL 将其转换为相同格式的 dict（Anthropic Messages API 使用兼容格式）

#### Scenario: System 消息
- **WHEN** 传入 `ChatMessage(role="system", content="You are a helpful assistant")`
- **THEN** OpenAI provider SHALL 将其放入 messages 列表
- **AND** Anthropic provider SHALL 将其提取为独立的 `system` 参数（Anthropic API 要求 system 消息与 messages 分离）

### Requirement: 可选参数支持
系统 SHALL 支持以下可选参数，传递给对应的 LLM API：
- `temperature`（float，控制随机性）
- `top_p`（float，核采样参数）
- `stream`（bool，默认 `True`）
- `thinking_budget`（int，思考预算 token 数，仅部分模型支持）

#### Scenario: 传递 temperature 和 top_p
- **WHEN** 调用 `chat("openai/GPT-5.4", messages, temperature=0.7, top_p=0.9)`
- **THEN** 系统 SHALL 将 `temperature=0.7` 和 `top_p=0.9` 传递给底层 OpenAI API 调用

#### Scenario: 传递 thinking_budget
- **WHEN** 调用 `chat("anthropic/claude-opus-4.6", messages, thinking_budget=10000)`
- **THEN** 系统 SHALL 将思考预算配置传递给 Anthropic API 的 `thinking` 参数

#### Scenario: thinking_budget 对不支持模型的处理
- **WHEN** 调用 `chat("openai/GPT-5.4", messages, thinking_budget=10000)` 且 OpenAI 不原生支持思考预算参数
- **THEN** 系统 SHALL 忽略该参数，不将其传递给 API，不抛出错误

### Requirement: 流式响应默认开启
当未指定 `stream` 参数时，系统 SHALL 默认使用流式模式（`stream=True`）。

#### Scenario: 未指定 stream 参数
- **WHEN** 调用 `chat("openai/GPT-5.4", messages)` 且未提供 stream 参数
- **THEN** 系统 SHALL 以流式模式调用 LLM API 并返回 AsyncGenerator

### Requirement: 响应数据结构
系统 SHALL 定义 `ChatResponse` 数据结构，包含：
- `content`（str，完整响应文本）
- `model`（str，实际使用的模型名称）
- `usage`（UsageInfo | None，包含 input_tokens 和 output_tokens）

#### Scenario: 非流式响应结构
- **WHEN** 以 `stream=False` 调用并成功获取响应
- **THEN** 返回的 `ChatResponse` SHALL 包含 `content`（完整文本）、`model`（模型名）、`usage`（token 使用量）

### Requirement: 错误处理
系统 SHALL 捕获底层 SDK 异常并转换为统一的 `LLMError` 异常，包含原始错误信息和 provider 标识。

#### Scenario: API Key 无效
- **WHEN** 调用 LLM 时 API Key 无效导致底层 SDK 抛出认证错误
- **THEN** 系统 SHALL 抛出 `LLMError`，包含 provider 名称、模型名称和原始错误信息

#### Scenario: 网络超时
- **WHEN** 调用 LLM 时发生网络超时
- **THEN** 系统 SHALL 抛出 `LLMError`，包含 provider 名称和超时错误信息

#### Scenario: 模型不存在
- **WHEN** 调用不存在的模型名称
- **THEN** 系统 SHALL 抛出 `LLMError`，包含 provider 名称、模型名称和原始 404 错误信息
