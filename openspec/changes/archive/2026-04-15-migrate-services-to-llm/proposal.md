## Why

`app/services/` 中的 `llm_service.py` 和 `reviewer_service.py` 使用 LangChain `ChatOpenAI` 直接调用 LLM，绕过了已建成的 `app/llm/` 统一服务层。这导致项目中存在两套 LLM 调用架构：5 个 agent（reader、planner、summarizer、bg_reviewer、reviewer_agent）走旧的 LangChain 路径，而 2 个 agent（data_cleaning、feature_engineering）走新的 `app/llm/` 路径。同时各 agent 的模型配置（提供商、模型名、温度、max_tokens、流式传输）散落在环境变量前缀和硬编码常量中，缺乏统一的 .env 配置方案。

## What Changes

- **删除 `app/services/` 包**，将 `complete_json()` 和 `render_review_comment()` 的功能迁移到 `app/llm/` 服务层，不保留过渡兼容层
- **将所有 agent 的 LLM 调用统一迁移到 `app/llm/` 服务层**，消除 LangChain `ChatOpenAI` 依赖
- **在 `.env` 中为每个 agent 提供统一配置**：提供商/模型标识、温度、max_tokens、是否流式传输
- **更新文档**：`docs/llm-usage_cn.md`、`docs/architecture.md`、`docs/skills/create-module/SKILL.md`
- **更新 skills**：`.opencode/skills/` 中的相关技能说明

## Capabilities

### New Capabilities
- `agent-llm-config`: 每个 agent 的 LLM 配置（模型、提供商、参数、流式传输）统一在 .env 中管理，通过 `app/llm/` 服务层读取和路由
- `llm-helpers`: 在 `app/llm/` 中提供 `complete_json()` 和 `render_text()` 等高层辅助函数，替代原 `app/services/` 中的同名功能

### Modified Capabilities
- `llm-chat`: `ChatParams` 新增从 .env 读取 agent 级别配置的能力，支持按 agent 名称解析模型标识和参数

## Impact

- **代码变更**：删除 `app/services/`、`app/agents/`（5 个 agent 的 import 和 LLM 调用路径）、`app/llm/`（新增辅助模块）、`app/planners/base.py`（模型配置来源改为 .env 且支持按调用方 agent 名读取）
- **依赖影响**：移除 `langchain-openai` 在 agent 层的依赖（`BasePlanner` 已移除）
- **配置变更**：`.env` 新增 7 个 agent 的 LLM 配置段（`AGENT_*_MODEL`、`AGENT_*_TEMPERATURE`、`AGENT_*_MAX_TOKENS`、`AGENT_*_STREAM`）
- **文档变更**：`docs/llm-usage_cn.md`、`docs/architecture.md`、`docs/skills/create-module/SKILL.md` 需同步更新
