## Context

当前项目是一个基于 LangGraph 的多代理数据科学自动化系统。LLM 调用散落在 `app/planners/base.py` 中，通过 `ChatOpenAI` 直接调用 GLM-5（DashScope API）。系统需要扩展支持多种 LLM 供应商（OpenAI 兼容服务、Anthropic），并提供统一的调用接口。

当前状态：
- 仅支持 OpenAI 兼容格式（通过 `langchain-openai` 的 `ChatOpenAI`）
- API 配置（base_url、api_key）硬编码或通过构造参数传递
- 无模型标识体系，无法灵活路由到不同供应商
- 各 agent/planner 直接持有 LLM 客户端实例，无统一抽象

## Goals / Non-Goals

**Goals:**
- 提供统一的 LLM 调用接口，屏蔽不同供应商的 API 差异
- 通过模型标识字符串（如 `openai/GPT-5.4`、`openai/zai/glm-5.1`、`anthropic/claude-opus-4.6`）自动路由到正确的供应商客户端
- 支持流式响应（默认）和非流式响应
- 支持可选参数：temperature、top_p、thinking_budget
- 便于现有代码迁移，接口设计简洁

**Non-Goals:**
- 不构建 HTTP API 服务器（本层是 Python 内部服务层，不是 REST API）
- 不实现模型负载均衡、故障转移、重试策略（可后续扩展）
- 不实现 Token 计费、用量统计
- 不支持 embedding、image generation 等非聊天类 API
- 不实现自定义 provider 插件机制（仅支持 OpenAI 兼容和 Anthropic 两种）

## Decisions

### Decision 1: 模型标识格式 — `provider/[prefix/]model`

**选择**: 使用 `/` 分隔的字符串格式 `<provider>/[<prefix>/]<model>`

**格式规则**:
- `openai/GPT-5.4` → OpenAI 兼容，模型名为 `GPT-5.4`
- `openai/zai/glm-5.1` → OpenAI 兼容，prefix 为 `zai`，实际模型名为 `glm-5.1`（某些 OpenAI 兼容服务需要路径前缀）
- `anthropic/claude-opus-4.6` → Anthropic 格式，模型名为 `claude-opus-4.6`

**备选方案**:
- 使用 `provider:model` 格式（冒号分隔）→ 模型名本身可能包含冒号，不够可靠
- 使用结构化对象 → 调用繁琐，不符合"填入模型"的简洁需求

**理由**: `/` 分隔直观且与 URL 路径风格一致；provider 固定为第一段，解析简单可靠。

### Decision 2: 使用原生 SDK 而非 LangChain

**选择**: 使用 `openai` 和 `anthropic` 原生 SDK，不依赖 LangChain 封装。

**理由**:
- LangChain 的抽象层在多供应商场景下增加了不必要的复杂度
- 原生 SDK 对流式响应、思考预算等新特性支持更及时
- 减少 LangChain 版本耦合风险
- 当前项目 LangChain 主要用于 LangGraph 工作流，LLM 调用可以独立

**备选方案**: 继续使用 `langchain-openai` + `langchain-anthropic` → 抽象层过厚，新增特性需要等 LangChain 更新

### Decision 3: 配置管理 — Pydantic + .env 文件

**选择**: 使用 Pydantic 模型定义各供应商的配置（base_url、api_key），从 `.env` 文件加载环境变量后读取。

项目已依赖 `python-dotenv`（`main.py`、`run_demo.py` 中已有 `load_dotenv()`）。LLM 服务层 SHALL 在初始化时调用 `load_dotenv()` 确保 `.env` 中的变量已加载到 `os.environ`，然后通过 Pydantic Settings 或手动读取 `os.environ` 获取配置。

```python
class ProviderConfig(BaseModel):
    providers: dict[str, OpenAICompatConfig | AnthropicConfig]
```

**配置来源优先级**: 系统环境变量 > `.env` 文件 > 代码默认值

**理由**: Pydantic 已在项目中广泛使用；`python-dotenv` 已是项目依赖；`.env` 文件便于本地开发配置且不会被 git 追踪（已在 `.gitignore` 中）。

### Decision 4: 模块结构

```
app/llm/
├── __init__.py          # 导出 LLMService, chat()
├── config.py            # ProviderConfig, 全局配置
├── router.py            # 模型标识解析 + 路由
├── service.py           # LLMService 主类（统一入口）
├── providers/
│   ├── __init__.py
│   ├── base.py          # BaseProvider 抽象基类
│   ├── openai_compat.py # OpenAI 兼容实现
│   └── anthropic.py     # Anthropic 实现
└── schemas.py           # ChatMessage, ChatResponse, ChatParams 等
```

### Decision 5: 流式响应返回 AsyncGenerator

**选择**: 流式响应返回 `AsyncGenerator[str, None]`（逐 token 文本），非流式返回完整的 `ChatResponse`。

**理由**: LangGraph 的 agent 节点需要逐步处理 LLM 输出；返回纯文本生成器最通用。

### Decision 6: 文档与技能更新

**选择**: 同步更新以下文档，确保 `app/llm/` 层被纳入项目规范体系：

| 文档 | 变更内容 |
|------|----------|
| `docs/architecture.md` | High-Level Layers 新增 `app/llm/` 条目，Ownership Rules 新增 LLM 层所有权规则 |
| `docs/engineering-specification.md` | Repository Structure 目录树新增 `app/llm/` 子目录及文件列表 |
| `docs/observability.md` | 环境变量列表新增 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`ANTHROPIC_API_KEY` 等 |
| `docs/skills/create-module/SKILL.md` | Module Selection 新增 `app/llm/` 包及其职责说明 |

**理由**: 项目采用文档驱动开发，所有模块必须反映在架构文档和工程规范中。`create-module` skill 是创建新模块时的操作指南，必须包含 `app/llm/` 以指导后续开发。

## Risks / Trade-offs

- **[风险] 模型标识中 `/` 的歧义** → 通过规则限定：provider 固定为第一段，最后一个 `/` 前为 prefix，最后一段为 model name。解析逻辑明确。
- **[风险] 双 SDK 依赖增加包大小** → `openai` 和 `anthropic` 都是轻量 SDK，影响可忽略。
- **[权衡] 不使用 LangChain 抽象** → 灵活性更高但需要自行维护统一的 message 格式转换。message 格式本身不复杂（OpenAI 和 Anthropic 的 message 结构相似）。
- **[风险] 迁移现有代码** → 需要修改 `BasePlanner` 及各 agent 中的 LLM 调用。通过保持接口简洁（`service.chat(model, messages, **params)`），迁移成本可控。
