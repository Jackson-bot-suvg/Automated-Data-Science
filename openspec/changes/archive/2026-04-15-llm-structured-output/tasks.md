## 1. 数据模型扩展

- [x] 1.1 在 `ChatParams` 中新增 `response_schema: type[BaseModel] | None = None` 字段
- [x] 1.2 在 `ChatParams` 中新增 `max_tokens: int | None = None` 字段
- [x] 1.3 在 `ChatResponse` 中新增 `parsed: BaseModel | None = None` 字段
- [x] 1.4 更新 `BaseProvider.chat()` 的 docstring，说明 `response_schema` 和 `max_tokens` 的处理约定

## 2. OpenAI Provider 结构化输出

- [x] 2.1 在 `OpenAICompatProvider.chat()` 中处理 `params.max_tokens`，传入 `max_tokens` 参数
- [x] 2.2 在 `OpenAICompatProvider.chat()` 中检测 `params.response_schema`，当非 None 时走结构化输出路径
- [x] 2.3 实现 `_structured_complete()` 方法：调用 `client.beta.chat.completions.parse(response_format=schema)` 并将解析结果填入 `ChatResponse.parsed`
- [x] 2.4 实现降级逻辑：当 `parse()` API 抛出不支持错误时，fallback 到 `create(response_format={"type": "json_object"})` + schema 注入 prompt + `model_validate_json()` 手动解析
- [x] 2.5 当 `response_schema` 非 None 时强制 `stream=False`

## 3. Anthropic Provider 结构化输出

- [x] 3.1 在 `AnthropicProvider.chat()` 中处理 `params.max_tokens`，传入 `max_tokens` 参数
- [x] 3.2 在 `AnthropicProvider.chat()` 中检测 `params.response_schema`，当非 None 时走结构化输出路径
- [x] 3.3 实现 `_structured_complete()` 方法：在 system prompt 末尾注入 JSON schema 约束，启用 JSON mode，响应后用 `model_validate_json()` 解析
- [x] 3.4 当 `response_schema` 非 None 且 `stream=True` 时抛出 `ValueError`
- [x] 3.5 解析失败时抛出 `LLMError` 并包含解析失败的原始内容和异常信息

## 4. LLM Service 层适配

- [x] 4.1 更新 `LLMService.chat()` 的 docstring，说明 `response_schema` 和 `max_tokens` 参数
- [x] 4.2 确认 `ChatParams(**kwargs)` 能正确透传 `response_schema` 和 `max_tokens`
- [x] 4.3 在 `app/llm/__init__.py` 中确认导出完整（无需新增导出）

## 5. BasePlanner 迁移

- [x] 5.1 移除 `_build_structured_llm()` 方法中的 `langchain_openai.ChatOpenAI` 导入和使用
- [x] 5.2 移除 `_build_messages()` 方法中的 `langchain_core.messages` 导入和使用，改用 `ChatMessage`
- [x] 5.3 将 `BasePlanner.__init__()` 改为使用 `LLMService` 替代手动构建 LangChain LLM
- [x] 5.4 将 `plan_data_cleaning()` 和 `plan_feature_engineering()` 改为 `async` 方法
- [x] 5.5 在 `_invoke_structured_llm()` 中使用 `await self._service.chat(...)` 替代 `runnable.invoke()`
- [x] 5.6 保留 LangSmith tracing 逻辑（`with_tracing_context`），确保追踪上下文不受影响

## 6. 调用者适配

- [x] 6.1 搜索所有 `BasePlanner` 的调用点，将同步调用改为 `await`
- [x] 6.2 确保调用链上的所有函数都是 `async` 的（或在同步入口点使用 `asyncio.run()`）

## 7. 依赖清理与测试

- [x] 7.1 从项目依赖中移除 `langchain-openai` 和 `langchain-core`
- [x] 7.2 编写 `OpenAICompatProvider` 结构化输出的单元测试（mock `parse()` API）
- [x] 7.3 编写 `AnthropicProvider` 结构化输出的单元测试（mock `messages.create()`）
- [x] 7.4 编写 `BasePlanner` 迁移后的集成测试，验证 `ToolCallPlan` 结构化输出正确
- [x] 7.5 运行全部测试，确认无回归
