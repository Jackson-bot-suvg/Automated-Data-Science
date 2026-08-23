## Context

当前项目存在两套并行的 LLM 调用架构：

1. **旧架构（`app/services/`）**：5 个 agent（reader、planner、summarizer、bg_reviewer、reviewer_agent）使用 `langchain_openai.ChatOpenAI` 直接调用 LLM，配置通过分散的 `env_prefix` 环境变量前缀管理（如 `READER_LLM_MODEL`、`PLANNER_LLM_MODEL`）。
2. **新架构（`app/llm/`）**：2 个 agent（data_cleaning、feature_engineering）通过 `BasePlanner` → `LLMService.chat()` 统一调用，但模型参数硬编码在 `BasePlanner.__init__` 中。

这导致了配置分散、依赖重复（LangChain + OpenAI SDK 并存）、无法统一管理各 agent 的模型选择和参数。

## Goals / Non-Goals

**Goals:**
- 消除 `app/services/` 中的 LangChain LLM 调用，将所有 agent 的 LLM 调用统一到 `app/llm/` 服务层
- 在 `.env` 中提供每个 agent 独立的 LLM 配置（提供商/模型、温度、max_tokens、流式传输）
- 在 `app/llm/` 中提供 `complete_json()` 和 `render_text()` 等高层辅助函数，保持现有 agent 调用接口的兼容性
- 更新文档和 skills，反映新架构

**Non-Goals:**
- 不改变各 agent 的业务逻辑和 prompt 内容
- 不重构 `app/llm/` 的 provider 实现或 router
- 不引入新的 LLM 提供商
- 不改变 `BasePlanner` 的结构化输出机制

## Decisions

### Decision 1: Agent 配置使用 `AGENT_<NAME>_` 环境变量前缀

**选择**：使用 `AGENT_<NAME>_MODEL`、`AGENT_<NAME>_TEMPERATURE`、`AGENT_<NAME>_MAX_TOKENS`、`AGENT_<NAME>_STREAM` 格式。

**替代方案**：
- JSON 配置文件 → 过重，当前 7 个 agent 不需要
- 复用旧的 `<AGENT>_LLM_*` 前缀 → 语义不清晰，与 provider 配置混淆

**理由**：与现有 `LLM_<PROVIDER>_` 前缀风格一致，`AGENT_` 前缀清楚表明是 agent 级配置。model 字段使用 `provider/model` 格式（如 `dashscope/glm-5`），复用 router 的路由能力。

### Decision 2: 在 `app/llm/` 中新增 `helpers.py` 模块

**选择**：新增 `app/llm/helpers.py`，包含 `complete_json()` 和 `render_text()` 两个异步辅助函数。

**理由**：
- `complete_json()` 被多个 agent 使用，是通用的"调用 LLM → 解析 JSON"模式
- `render_text()` 替代 `render_review_comment()` 的通用"调用 LLM → 返回文本"模式
- 放在 `app/llm/` 包内，保持内聚性，与 `LLMService` 紧密协作

### Decision 3: 删除 `app/services/` 包

**选择**：删除 `app/services/llm_service.py`、`app/services/reviewer_service.py`、`app/services/__init__.py`，不保留过渡兼容层。

**理由**：项目内 LLM 调用已统一迁移到 `app/llm/`，继续保留壳层会增加认知负担并制造“旧入口仍可用”的误导。

### Decision 4: 每个 agent 读取自己的 .env 配置段

**选择**：新增 `app/llm/agent_config.py` 模块，提供 `get_agent_config(agent_name: str) -> AgentLLMConfig` 函数，统一从 .env 读取并解析 agent 的 LLM 配置。

**理由**：集中管理 agent 配置读取逻辑，避免每个 agent 重复解析环境变量。返回强类型的 `AgentLLMConfig` Pydantic 模型。

### Decision 5: `BasePlanner` 的模型配置改为按调用方 agent 从 .env 读取

**选择**：`BasePlanner` 不再硬编码 `model_name`、`temperature`、`max_tokens`，改为通过可注入的 `agent_name` 调用 `get_agent_config(agent_name)` 读取。

**理由**：确保 `planner`、`data_cleaning`、`feature_engineering` 都能使用各自的 `AGENT_<NAME>_*` 配置，避免“配置已声明但不生效”。

## Risks / Trade-offs

- **[Breaking Change] agent 的 LLM 调用从同步变为异步**：`complete_json()` 当前是同步函数，迁移到 `app/llm/` 后变为异步。所有调用方需使用 `await`。→ 缓解：graph nodes 已经是异步的，影响可控。
- **[配置迁移] 旧的 `<AGENT>_LLM_*` 环境变量不再使用**：用户需要迁移到 `AGENT_<NAME>_` 格式。→ 缓解：在 `.env.example` 中提供清晰的配置模板和注释。
- **[测试影响] 现有使用 `complete_json` 的测试需要更新**：→ 缓解：helpers 提供相同的签名和 fallback 语义。
- **[LangChain 依赖]**：移除 `langchain-openai` 在 agent 层的依赖后，如果项目其他地方不再使用，可以从 requirements 中移除。→ 需要全局检查。
