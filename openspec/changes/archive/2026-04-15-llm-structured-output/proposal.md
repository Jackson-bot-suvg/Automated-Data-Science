## Why

`BasePlanner` 目前通过 `langchain-openai` 的 `ChatOpenAI.with_structured_output()` 实现结构化输出，是项目中唯一引入 LangChain 的模块。而 `app/llm/` 服务层已经具备完整的 provider 抽象（OpenAI / Anthropic），但缺少结构化输出能力。将结构化输出下沉到 LLM 服务层，可以彻底移除 LangChain 依赖、统一 LLM 调用路径，并为未来其他模块的结构化输出需求提供基础设施。

## What Changes

- 在 `ChatParams` 中新增 `response_schema: type[BaseModel] | None` 参数，支持传入 Pydantic 模型以约束 LLM 输出格式
- OpenAI 兼容 provider 使用 `chat.completions.parse(response_format=...)` 原生结构化输出 API
- Anthropic provider 使用 `beta.messages.parse(output_format=...)` 结构化输出 API
- `LLMService.chat()` 返回结果中增加解析后的结构化对象
- **BREAKING**: `BasePlanner` 移除所有 LangChain 依赖，改用 `app/llm/` 服务层的结构化输出能力
- 移除 `langchain-openai` 和 `langchain-core` 依赖

## Capabilities

### New Capabilities
- `structured-output`: 在 LLM 服务层提供基于 Pydantic 模型的结构化输出能力，支持 OpenAI 和 Anthropic 两个 provider

### Modified Capabilities
- `llm-planner`: BasePlanner 改用 `app/llm/` 服务层的结构化输出接口，移除 LangChain 依赖

## Impact

- **代码**: `app/llm/schemas.py`（ChatParams 扩展）、`app/llm/providers/openai_compat.py`、`app/llm/providers/anthropic.py`、`app/llm/service.py`、`app/planners/base.py`
- **API**: `LLMService.chat()` 方法签名变更（增加可选参数，向后兼容）
- **依赖**: 移除 `langchain-openai`、`langchain-core`；可能需要升级 `openai` SDK 版本以支持 `parse()` API
- **风险**: 结构化输出的解析行为在 OpenAI 和 Anthropic 两家实现上可能有细微差异，需要针对 `ToolCallPlan` 模型进行验证
