## Why

当前项目通过 `app/planners/base.py` 中的 `ChatOpenAI` 直接调用 GLM-5，LLM 调用逻辑散落在各 agent/planner 中，没有统一的服务层来管理多供应商、多模型的调用。随着系统扩展，需要支持 OpenAI 兼容格式（包括第三方 OpenAI 兼容服务如 zai/glm-5.1）和 Anthropic 格式的模型调用，并且需要一种统一的模型标识方式来路由请求。现在建立统一的 LLM 服务层，可以解耦模型调用细节、支持多供应商切换、便于测试和扩展。

## What Changes

- 新增 LLM 服务层模块（`app/llm/`），提供统一的模型调用接口
- 支持 OpenAI 兼容格式（Chat Completion），可配置自定义 `base_url` 和 `api_key`
- 支持 Anthropic 格式（Messages API）
- 引入统一模型标识格式：`<provider>/<model>` 或 `<provider>/<prefix>/<model>`，例如 `openai/GPT-5.4`、`openai/zai/glm-5.1`、`anthropic/claude-opus-4.6`
- 支持可选参数：`temperature`、`top_p`、`stream`（默认 `true`）、思考预算（`thinking_budget`）
- 重构现有 `BasePlanner` 等调用方，迁移至新服务层

## Capabilities

### New Capabilities
- `llm-router`: 模型路由与客户端工厂 — 解析模型标识、路由到正确的 LLM 供应商客户端（OpenAI 兼容 / Anthropic），管理 API 配置
- `llm-chat`: 统一聊天接口 — 提供统一的 `chat()` 方法，支持流式/非流式响应，封装 message 格式转换、可选参数传递

### Modified Capabilities
<!-- 无现有规格需要修改 -->

## Impact

- **新增代码**: `app/llm/` 模块（router.py、service.py、config.py、schemas.py、providers/）
- **修改代码**: `app/planners/base.py` 及各 agent 中的 LLM 调用方式，迁移至新服务层
- **依赖**: 新增 `openai` 和 `anthropic` 原生 SDK 依赖
- **配置**: 需要新增 LLM 供应商配置结构（base_url、api_key 的环境变量或配置文件）
- **测试**: 需要为路由解析、供应商切换、参数传递编写单元测试
- **文档更新**:
  - `docs/architecture.md`: 在 High-Level Layers 中加入 `app/llm/` 层描述
  - `docs/engineering-specification.md`: 在 Repository Structure 中加入 `app/llm/` 目录树，更新环境变量说明
  - `docs/observability.md`: 加入新 LLM 服务层相关的环境变量（`OPENAI_API_KEY`、`OPENAI_BASE_URL`、`ANTHROPIC_API_KEY` 等）
  - `docs/skills/create-module/SKILL.md`: 在 Module Selection 中加入 `app/llm/` 包的职责说明和所有权规则
