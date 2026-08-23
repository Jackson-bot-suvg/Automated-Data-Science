## Context

当前 `app/llm/` 服务层已具备完整的 provider 抽象架构（`BaseProvider` → `OpenAICompatProvider` / `AnthropicProvider`），支持流式/非流式 chat 调用。但该层尚不支持结构化输出（Structured Output）——即约束 LLM 返回符合指定 Pydantic 模型的 JSON。

`BasePlanner` 是唯一需要结构化输出的消费者，当前通过 `langchain-openai` 的 `ChatOpenAI.with_structured_output(ToolCallPlan, method="json_schema")` 实现，引入了 `langchain-openai` 和 `langchain-core` 两个额外依赖。

相关源码位置：
- `app/llm/schemas.py`: `ChatParams`（4 个可选字段）、`ChatResponse`、`ChatMessage`
- `app/llm/providers/base.py`: `BaseProvider.chat()` 返回 `ChatResult = ChatResponse | AsyncGenerator`
- `app/llm/providers/openai_compat.py`: `_complete()` 使用 `chat.completions.create()`
- `app/llm/providers/anthropic.py`: `_complete()` 使用 `messages.create()`
- `app/planners/base.py`: `BasePlanner` 使用 LangChain 的结构化输出

## Goals / Non-Goals

**Goals:**
- 在 `app/llm/` 服务层增加结构化输出能力，通过 `ChatParams.response_schema` 传入 Pydantic 模型
- OpenAI provider 使用 `client.beta.chat.completions.parse()` 原生结构化输出
- Anthropic provider 使用 `client.messages.create()` + `response_format` 参数（或 JSON mode + 手动解析）
- `BasePlanner` 完全迁移到 `app/llm/` 服务层，移除 LangChain 依赖
- 保持向后兼容：`ChatParams.response_schema=None` 时行为不变

**Non-Goals:**
- 流式结构化输出（当前无此需求，结构化输出仅用于非流式场景）
- 支持 OpenAI function calling / tool_use 模式（仅使用 json_schema 模式）
- 为其他模块（非 BasePlanner）提供结构化输出能力（未来可扩展，但不在本次范围内）
- 更换 LLM 服务层的整体架构或 API 设计

## Decisions

### Decision 1: `ChatParams` 增加 `response_schema` 字段

**选择**: 在 `ChatParams` 中添加 `response_schema: type[BaseModel] | None = None`

**理由**: 与现有参数风格一致，可选参数不破坏现有调用。当 `response_schema` 非 None 时，provider 负责将结构化输出解析后的 Pydantic 对象放入 `ChatResponse.parsed`。

**备选方案**:
- 独立 `structured_chat()` 方法 → 增加接口复杂度，且 chat/structured 逻辑高度重叠
- 新增 `StructuredChatParams` 子类 → 过度设计，仅多一个字段

### Decision 2: `ChatResponse` 增加 `parsed` 字段

**选择**: 在 `ChatResponse` 中添加 `parsed: BaseModel | None = None`

**理由**: 保持 `content`（原始文本）和 `parsed`（结构化对象）并存，方便调试和日志。当 `response_schema` 非 None 时，`parsed` 包含解析后的 Pydantic 实例。

### Decision 3: OpenAI provider 使用 `client.beta.chat.completions.parse()`

**选择**: 使用 OpenAI SDK 的 `parse()` 方法，传入 `response_format=schema`

**理由**: `parse()` 是 OpenAI SDK 提供的一等结构化输出 API，自动处理 json_schema 约束和解析。比手动 JSON mode + `json.loads()` + `schema.model_validate()` 更可靠。

**备选方案**:
- `chat.completions.create(response_format={"type": "json_object"})` + 手动解析 → 可行但需额外处理 schema 约束注入 prompt
- `instructor` 库 → 引入新依赖，与移除 LangChain 的目标矛盾

### Decision 4: Anthropic provider 使用 JSON mode + 手动解析

**选择**: Anthropic SDK 暂不支持原生 `parse()` API 的 json_schema 模式。采用方案：在 system prompt 中注入 schema 约束 + 启用 JSON mode + `response_schema.model_validate_json()` 手动解析。

**理由**: Anthropic 的 `beta.messages.parse()` 目前对 json_schema 的支持有限。使用 prompt engineering + JSON mode + Pydantic 验证是更稳妥的方案，且 `ToolCallPlan` 结构简单，prompt 约束足够可靠。

**备选方案**:
- 等待 Anthropic SDK 原生支持 → 阻塞进度
- 使用 `instructor` 库 → 引入新依赖

### Decision 5: `BasePlanner` 迁移为 `LLMService` 的使用者

**选择**: `BasePlanner` 使用 `LLMService.chat()` + `response_schema=ToolCallPlan`，移除所有 LangChain 导入

**理由**: 统一调用路径，LangChain 的唯一用途就是 `with_structured_output()`，可以在服务层替代。

**保留**: LangSmith tracing 逻辑保持不变（通过 `with_tracing_context` 上下文管理器）。

### Decision 6: max_tokens 通过 ChatParams 传递

**选择**: 在 `ChatParams` 中新增 `max_tokens: int | None = None`

**理由**: `BasePlanner` 当前使用 `max_tokens=65536`，迁移后需要通过服务层传递此参数。这也是一个通用需求，其他调用者可能也需要控制 token 上限。

## Risks / Trade-offs

- **[OpenAI parse() API 兼容性]** → parse() 是 beta API，部分 OpenAI 兼容服务（如 ZhipuAI/DashScope）可能不支持 → 检测失败时 fallback 到 JSON mode + 手动解析
- **[Anthropic 结构化输出可靠性]** → prompt 约束 + JSON mode 不如原生 json_schema 严格 → 在 `ToolCallPlan` 级别添加 `model_validate` 校验，解析失败时抛出明确错误
- **[BasePlanner 同步/异步]** → `BasePlanner` 当前是同步调用（`runnable.invoke()`），而 `LLMService.chat()` 是 async → 需要将 `BasePlanner` 的 plan 方法改为 async，或在内部使用 `asyncio.run()` / `loop.run_until_complete()`
- **[LangSmith tracing 适配]** → 移除 LangChain 后，LangSmith 的自动 tracing 不再生效 → 需要确认 `with_tracing_context` 是否仍能正常工作，可能需要改用 LangSmith SDK 的手动 tracing API
