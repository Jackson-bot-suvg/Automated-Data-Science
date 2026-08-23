# LLM 服务层与应用集成指南

## 1. 概述

`app/llm/` 模块为项目中所有 LLM 交互提供了**统一、与提供商无关的接口**。业务节点（agent、planner）从不直接持有 SDK 客户端——它们通过服务层进行调用，由服务层统一处理提供商路由、配置管理、流式传输、结构化输出和错误归一化。

### 架构总览

```
业务节点 (app/agents/, app/planners/)
        │
        ▼
  LLMService (app/llm/service.py)
        │
        ├── 模型路由器 (app/llm/router.py)
        │       解析 "provider/prefix/model" 格式的标识符
        │
        ├── 提供商注册表
        │       ├── OpenAICompatProvider  →  任何 OpenAI 兼容 API
        │       └── AnthropicProvider     →  Anthropic Messages API
        │
        └── 配置 (app/llm/config.py)
                从环境变量读取凭据
```

---

## 2. 快速开始

### 2.1 配置环境

将 `.env.example` 复制为 `.env` 并填入你的凭据。

#### 推荐方式：统一多 Provider 配置

通过 `LLM_PROVIDERS` 环境变量声明你要使用的 provider 列表（逗号分隔）。每个 provider 通过 `LLM_<NAME>_TYPE`、`LLM_<NAME>_API_KEY`、`LLM_<NAME>_BASE_URL` 配置：

```bash
# 声明启用的 provider（逗号分隔）
LLM_PROVIDERS=dashscope,zhipuai

# DashScope（OpenAI 兼容）
LLM_DASHSCOPE_TYPE=openai_compat
LLM_DASHSCOPE_API_KEY=sk-xxxxx
LLM_DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# ZhipuAI（OpenAI 兼容）
LLM_ZHIPUAI_TYPE=openai_compat
LLM_ZHIPUAI_API_KEY=xxxxx
LLM_ZHIPUAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4

# Anthropic（Claude 模型）— 按需添加
# LLM_PROVIDERS=dashscope,zhipuai,anthropic
# LLM_ANTHROPIC_TYPE=anthropic
# LLM_ANTHROPIC_API_KEY=sk-ant-xxxxx
```

**支持的 provider 类型：**
- `openai_compat`（默认）：适用于任何 OpenAI 兼容 API（DashScope、ZhipuAI、vLLM、DeepSeek 等）
- `anthropic`：适用于 Anthropic Messages API（Claude 模型）

**环境变量名称映射：** `LLM_PROVIDERS` 中的名称会转为大写查找对应环境变量。例如 `dashscope` → `LLM_DASHSCOPE_API_KEY`。

**Provider 名称规则：** 仅允许字母、数字、`-`、`_`。

#### 旧格式（向后兼容）

如果 `LLM_PROVIDERS` 未设置或为空，系统自动回退到旧格式：

```bash
# OpenAI 兼容提供商（DashScope、ZhipuAI、vLLM 等）必填
OPENAI_API_KEY=sk-xxxxx
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# Anthropic 提供商必填
ANTHROPIC_API_KEY=sk-ant-xxxxx
# ANTHROPIC_BASE_URL=https://api.anthropic.com   # 可选覆盖
```

现有部署无需任何修改即可继续工作。

### 2.2 基本用法

```python
from app.llm import get_service, ChatMessage

service = get_service()

# 非流式调用
response = await service.chat(
    model="openai/glm-5",
    messages=[
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="Hello!"),
    ],
    stream=False,
)
print(response.content)        # 回复文本
print(response.usage)          # UsageInfo(input_tokens=..., output_tokens=...)

# 流式调用（默认）
async for token in await service.chat(
    model="openai/glm-5",
    messages=[ChatMessage(role="user", content="Tell me a story.")],
):
    print(token, end="", flush=True)
```

### 2.3 结构化输出

传入一个 Pydantic 模型类作为 `response_schema`，即可获得类型安全的、经过验证的响应。**此参数会强制使用非流式模式**，无论 `stream` 参数如何设置。

```python
from pydantic import BaseModel
from app.llm import get_service, ChatMessage

class SentimentResult(BaseModel):
    label: str          # "positive" | "negative" | "neutral"
    confidence: float   # 0.0 – 1.0

service = get_service()
response = await service.chat(
    model="openai/glm-5",
    messages=[ChatMessage(role="user", content="I love this product!")],
    response_schema=SentimentResult,
    stream=False,
)

result: SentimentResult = response.parsed  # 已验证的对象
print(result.label)       # "positive"
print(result.confidence)  # 0.95
```

---

## 3. 模块参考

### 3.1 `app/llm/__init__.py` — 公开 API

包通过 `get_service()` 暴露一个便捷的单例，并导出辅助函数和配置工具：

```python
from app.llm import (
    get_service, ChatMessage, ChatResponse, ChatParams, LLMError, UsageInfo,
    complete_json, render_text, parse_json_response,
    get_agent_config, AgentLLMConfig,
)
```

| 导出项 | 说明 |
|---|---|
| `get_service()` | 返回模块级单例 `LLMService` |
| `ChatMessage` | `(role: str, content: str)` — 单条消息 |
| `ChatResponse` | `(content, model, usage?, parsed?)` — 完整响应 |
| `ChatParams` | 参数容器（详见 §3.3） |
| `LLMError` | 统一异常，包含 `provider` 和 `model` 上下文 |
| `UsageInfo` | `(input_tokens, output_tokens)` |
| `complete_json` | 异步辅助：调用 LLM → 解析 JSON → 失败返回 fallback |
| `render_text` | 异步辅助：调用 LLM → 返回文本 → 失败返回 fallback |
| `parse_json_response` | 同步工具：从文本中提取 JSON 对象 |
| `get_agent_config` | 根据 agent 名称读取 LLM 配置 |
| `AgentLLMConfig` | agent LLM 配置 Pydantic 模型 |

### 3.2 `LLMService`（`app/llm/service.py`）

核心调度器。一个 `chat()` 方法处理所有请求。

```python
class LLMService:
    def __init__(self, providers: dict[str, BaseProvider] | None = None): ...
    async def chat(self, model: str, messages: list[ChatMessage], **kwargs) -> ChatResult: ...
```

**构造函数** — 当 `providers` 为 `None`（默认值）时，会通过 `load_provider_config()` 自动从环境变量构建提供商。可传入自定义字典用于测试或高级配置。

**`chat()` 参数：**

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | `str` | 必填 | 标识符：`<provider>/[<prefix>/]<model>` |
| `messages` | `list[ChatMessage]` | 必填 | 对话历史 |
| `temperature` | `float` | `None` | 采样随机性 |
| `top_p` | `float` | `None` | 核采样阈值 |
| `stream` | `bool` | `True` | 流式返回 token 还是返回完整响应 |
| `max_tokens` | `int` | `None` | 生成 token 的上限 |
| `thinking_budget` | `int` | `None` | 扩展思考 token 数（仅 Anthropic） |
| `response_schema` | `type[BaseModel]` | `None` | 用于结构化输出的 Pydantic 类（强制非流式） |

**返回值：** 当 `stream=False`（或设置了 `response_schema`）时返回 `ChatResponse`，否则返回 `AsyncGenerator[str, None]`。

### 3.3 `ChatParams`（`app/llm/schemas.py`）

所有可选的聊天参数在内部被归一化为这个 Pydantic 模型。通常不需要直接构造——只需将关键字参数传给 `chat()` 即可。

### 3.4 `ChatMessage`（`app/llm/schemas.py`）

```python
class ChatMessage(BaseModel):
    role: str      # "system" | "user" | "assistant"
    content: str
```

### 3.5 `ChatResponse`（`app/llm/schemas.py`）

```python
class ChatResponse(BaseModel):
    content: str                     # LLM 返回的原始文本
    model: str                       # API 返回的实际模型名
    usage: UsageInfo | None = None   # Token 使用统计
    parsed: Any | None = None        # 验证后的 Pydantic 对象（结构化输出）
```

---

## 4. 模型标识符格式

模型标识符遵循格式：`<provider>/[<prefix>/]<model>`

路由器（`app/llm/router.py`）将其解析为 `ModelId(provider, prefix, model)` 数据类。Provider 名称由配置层动态决定。

| 标识符 | 提供商 | 前缀 | 模型 |
|---|---|---|---|
| `openai/glm-5` | `openai` | `None` | `glm-5` |
| `openai/zai/glm-5.1` | `openai` | `zai` | `glm-5.1` |
| `anthropic/claude-opus-4.6` | `anthropic` | `None` | `claude-opus-4.6` |
| `dashscope/qwen-plus` | `dashscope` | `None` | `qwen-plus` |

**注意：** provider 名称不再限定为 `openai` 和 `anthropic`——任何在 `LLM_PROVIDERS` 中声明（或旧格式中注册）的 provider 名称均可使用。

无效标识符或未注册的提供商会抛出 `ValueError`。

---

## 5. 提供商

### 5.1 `OpenAICompatProvider`（`app/llm/providers/openai_compat.py`）

适用于**任何暴露 OpenAI 兼容 Chat Completions 端点的服务**（DashScope、ZhipuAI、本地 vLLM、Ollama 等）。

**核心行为：**
- 使用 `openai.AsyncOpenAI`，支持配置 `base_url`
- 模型标识符中的 `prefix` 段会被前置拼接：`prefix/model`（如 `zai/glm-5.1`）
- `thinking_budget` 会被静默忽略（OpenAI API 不支持）

**结构化输出策略（两级回退）：**
1. **首选方案：** 使用 `beta.chat.completions.parse()` 并传入 `response_format=schema`——利用原生结构化输出支持
2. **回退方案：** 若 `parse()` 不支持，则切换到 JSON 模式（`response_format={"type": "json_object"}`），将 JSON Schema 注入用户提示词中，然后用 `schema.model_validate_json()` 验证响应

### 5.2 `AnthropicProvider`（`app/llm/providers/anthropic.py`）

适用于 **Anthropic Messages API**（Claude 模型）。

**核心行为：**
- 使用 `anthropic.AsyncAnthropic` 客户端
- 系统消息会被提取并通过独立的 `system` 参数传递（Anthropic 要求）
- 模型标识符中的 `prefix` 段会被忽略（不适用）
- 支持**扩展思考**：`thinking_budget` 映射为 `thinking={"type": "enabled", "budget_tokens": N}`
- 需要设置 `max_tokens`；结构化输出时默认为 `8192`

**结构化输出策略：**
- JSON Schema 被注入系统提示词，附带严格的格式要求
- 响应通过 `schema.model_validate_json()` 验证

---

## 6. 配置（`app/llm/config.py`）

配置通过 `python-dotenv` 从**环境变量**加载。优先级：系统环境变量 > `.env` 文件 > 代码默认值。

### 6.1 统一格式（推荐）

当设置了 `LLM_PROVIDERS` 时，使用统一格式加载。每个 provider 通过 `LLM_<NAME>_` 前缀的环境变量配置。

| 变量 | 说明 | 默认值 |
|---|---|---|
| `LLM_PROVIDERS` | 逗号分隔的 provider 名称列表 | — |
| `LLM_<NAME>_TYPE` | Provider 类型：`openai_compat` 或 `anthropic` | `openai_compat` |
| `LLM_<NAME>_API_KEY` | API 密钥 | — |
| `LLM_<NAME>_BASE_URL` | API 端点 URL（`openai_compat` 默认 `https://api.openai.com/v1`，`anthropic` 默认 `https://api.anthropic.com`） | 见左 |

**示例：**
```bash
LLM_PROVIDERS=dashscope,zhipuai,anthropic
LLM_DASHSCOPE_TYPE=openai_compat
LLM_DASHSCOPE_API_KEY=sk-xxx
LLM_DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_ZHIPUAI_TYPE=openai_compat
LLM_ZHIPUAI_API_KEY=xxx
LLM_ZHIPUAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4
LLM_ANTHROPIC_TYPE=anthropic
LLM_ANTHROPIC_API_KEY=sk-ant-xxx
```

### 6.2 旧格式（向后兼容）

当 `LLM_PROVIDERS` 未设置或为空时，自动回退到旧格式。

| 变量 | 提供商 | 说明 | 默认值 |
|---|---|---|---|
| `OPENAI_API_KEY` | OpenAI-compat | API 密钥 | *（必填）* |
| `OPENAI_BASE_URL` | OpenAI-compat | API 端点 URL | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `ANTHROPIC_API_KEY` | Anthropic | API 密钥 | *（必填）* |
| `ANTHROPIC_BASE_URL` | Anthropic | API 端点 URL | `https://api.anthropic.com` |

旧格式下 `openai` 和 `anthropic` 两个 provider 始终会被注册。`api_key` 为空的 provider 在调用时会抛出 `ValueError`。

### 6.3 配置验证

- Provider 名称仅允许字母、数字、`-`、`_`，否则启动时抛出 `ValueError`
- 不支持的 provider 类型（非 `openai_compat` / `anthropic`）启动时抛出 `ValueError`
- 缺少 API key 的 provider 会输出警告日志，但不会阻止启动（实际调用时抛出错误）

---

## 7. 错误处理

所有提供商特定的 SDK 异常都会被捕获并重新抛出为 `LLMError`：

```python
class LLMError(Exception):
    provider: str   # 例如 "openai"
    model: str      # 例如 "glm-5"
    # 消息格式："[openai/glm-5] <原始 SDK 错误>"
```

这为调用方提供了统一的错误类型，无需关心底层提供商。

---

## 8. 应用集成模式

业务节点通过三种模式与 LLM 服务层集成：

### 8.1 高层辅助函数（推荐，适用于大多数 Agent）

`app/llm/helpers.py` 提供了两个异步辅助函数，封装了"调用 LLM → 处理响应 → 失败返回 fallback"的通用模式：

- **`complete_json(agent_name, system_prompt, user_prompt, fallback)`** — 调用 LLM 并解析 JSON 响应，失败时返回 fallback 副本
- **`render_text(agent_name, system_prompt, user_prompt, fallback)`** — 调用 LLM 并返回文本响应，失败时返回 fallback 字符串

两个函数都会自动通过 `get_agent_config(agent_name)` 读取该 agent 的模型配置，调用方无需关心模型选择和参数。

```python
from app.llm import complete_json, render_text

# JSON 模式 — 用于需要结构化输出的 agent
payload = await complete_json(
    agent_name="reader",
    system_prompt=READER_SYSTEM_PROMPT,
    user_prompt=build_reader_prompt(...),
    fallback={"key": "default_value"},
)

# 文本模式 — 用于需要自由文本的 agent
comment = await render_text(
    agent_name="reviewer",
    system_prompt=REVIEWER_COMMENT_SYSTEM_PROMPT,
    user_prompt=build_reviewer_comment_prompt(...),
    fallback="default comment",
)
```

### 8.2 规划器模式（结构化工具调用）

本项目的主要模式是**规划器模式（Planner Pattern）**——业务节点将 LLM 推理委托给规划器类，规划器利用结构化输出生成可执行的工具调用计划。

```
Agent 节点（如 data_cleaning.py）
    │
    ├─ 1. 收集上下文（数据集画像、EDA 发现）
    │
    ├─ 2. BasePlanner(agent_name="data_cleaning").plan_data_cleaning(profile_json)
    │       │
    │       ├─ 构建 ChatMessage 列表（system + user 提示词）
    │       ├─ 通过 get_agent_config(agent_name) 获取模型配置
    │       ├─ LLMService.chat(model=cfg.model,
    │       │       response_schema=ToolCallPlan, ...)
    │       └─ 返回强类型的 ToolCallPlan
    │
    └─ 3. 执行 plan.steps → 动态工具分派
```

### 8.3 直接服务调用（高级场景）

直接使用 `LLMService.chat()` 进行精细控制：

```python
from app.llm import get_service, ChatMessage

service = get_service()
response = await service.chat("openai/glm-5", messages, stream=False)
```

**当前 LLM 消费者：**

| 组件 | 集成方式 | 模型配置 | 结构化输出 | 用途 |
|---|---|---|---|---|
| `ReaderAgent` | `complete_json(agent_name="reader")` | `AGENT_READER_*` | JSON 解析 | 任务理解与数据集分析 |
| `PlannerAgent` | `complete_json(agent_name="planner")` | `AGENT_PLANNER_*` | JSON 解析 | 阶段选择与执行计划 |
| `TechnicalSummarizerAgent` | `complete_json(agent_name="summarizer")` | `AGENT_SUMMARIZER_*` | JSON 解析 | 技术报告生成 |
| `BackgroundReviewerAgent` | `complete_json(agent_name="bg_reviewer")` | `AGENT_BG_REVIEWER_*` | JSON 解析 | Reader 输出审查 |
| `ReviewerAgent` | `render_text(agent_name="reviewer")` | `AGENT_REVIEWER_*` | — | 阶段审查评语 |
| `BasePlanner(agent_name="data_cleaning").plan_data_cleaning()` | 直接调用 `LLMService.chat()` | `AGENT_DATA_CLEANING_*` | `ToolCallPlan` | 数据清洗工具调用 |
| `BasePlanner(agent_name="feature_engineering").plan_feature_engineering()` | 直接调用 `LLMService.chat()` | `AGENT_FEATURE_ENGINEERING_*` | `ToolCallPlan` | 特征工程工具调用 |

**不使用 LLM 的节点：**
- `preliminary_eda` — 纯数据分析和可视化
- `model_building` — 传统机器学习（scikit-learn）
- `business_translator` — 确定性的指标到业务语言转换

### 8.4 Agent 配置

每个 agent 的 LLM 配置通过 `app/llm/agent_config.py` 中的 `get_agent_config(agent_name)` 从环境变量读取：

```python
from app.llm import get_agent_config

cfg = get_agent_config("reader")
# AgentLLMConfig(model="dashscope/glm-5", temperature=0.0, max_tokens=65536, stream=False)
```

环境变量命名规则：`AGENT_<UPPER_NAME>_{MODEL,TEMPERATURE,MAX_TOKENS,STREAM}`

详见 `.env.example` 中的 "Per-Agent LLM Configuration" 段。

### 8.5 依赖注入

`LLMService` 和 `BasePlanner` 均支持依赖注入，便于测试。`BasePlanner` 同时支持通过 `agent_name` 选择配置段：

```python
# 向 LLMService 注入 mock 提供商
mock_service = LLMService(providers={"openai": mock_provider})

# 向 BasePlanner 注入 mock 服务，并指定使用哪个 AGENT_<NAME>_* 配置
planner = BasePlanner(agent_name="data_cleaning", service=mock_service)
```

---

## 9. 可观测性集成

通过 `BasePlanner` 发起的 LLM 调用会自动包装在 **LangSmith 追踪**上下文中：

```python
# 在 BasePlanner._invoke_structured_llm() 中：
metadata = planner_trace_metadata(
    planner_type="data_cleaning",
    profile_json=profile_json,
    model_name=self.model_name,
    ...
)
tags = planner_trace_tags(planner_type="data_cleaning")

with with_tracing_context(tags=tags, metadata=metadata):
    result = await self._service.chat(...)
```

这对服务层是透明的——追踪发生在规划器层面。直接的 `service.chat()` 调用不会被追踪，除非调用方自行包装。

---

## 10. 文件索引

```
app/llm/
├── __init__.py              # 公开 API 与单例 (get_service)
├── agent_config.py          # Agent LLM 配置 (AgentLLMConfig, get_agent_config)
├── helpers.py               # 高层辅助函数 (complete_json, render_text, parse_json_response)
├── service.py               # LLMService — 统一调度
├── router.py                # ModelId 解析与提供商路由
├── config.py                # 基于环境变量的提供商配置
├── schemas.py               # ChatMessage、ChatResponse、ChatParams、LLMError
└── providers/
    ├── base.py              # BaseProvider 抽象基类与 ChatResult 类型别名
    ├── openai_compat.py     # OpenAI 兼容提供商实现
    └── anthropic.py         # Anthropic Messages API 提供商
```

---

## 11. 添加新提供商

### 11.1 通过配置添加（推荐，适用于 OpenAI 兼容 / Anthropic 类型）

只需在 `.env` 中添加新的 provider 条目即可，无需修改任何代码：

```bash
# 在 LLM_PROVIDERS 中添加新 provider 名称
LLM_PROVIDERS=dashscope,deepseek

# 配置新 provider
LLM_DEEPSEEK_TYPE=openai_compat
LLM_DEEPSEEK_API_KEY=sk-xxx
LLM_DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
```

然后即可通过 `deepseek/deepseek-chat` 使用。

### 11.2 添加新的 Provider 类型（需要修改代码）

如果需要支持非 OpenAI 兼容、非 Anthropic 的新类型：

1. 在 `app/llm/providers/<name>.py` 中创建 `BaseProvider` 的实现：

```python
from app.llm.providers.base import BaseProvider, ChatResult
from app.llm.schemas import ChatMessage, ChatParams, ChatResponse, LLMError

class MyProvider(BaseProvider):
    async def chat(
        self,
        messages: list[ChatMessage],
        params: ChatParams,
        model_name: str,
        prefix: str | None = None,
    ) -> ChatResult:
        # 实现提供商特定的 API 调用
        ...
```

2. 在 `config.py` 中添加新的 Config 模型，并在 `_load_from_unified()` 中支持新类型。

3. 在 `LLMService._build_providers()` 中添加新 provider 的构建逻辑。

---

## 12. 设计原则

1. **单一入口：** 所有 LLM 调用都通过 `LLMService.chat()` 进行，业务代码不持有 SDK 客户端。
2. **提供商无关：** 模型标识符（`openai/glm-5`、`anthropic/claude-opus-4.6`）决定路由，调用方无需感知提供商。
3. **结构化输出优先：** `response_schema` 参数确保 LLM 输出的类型安全，提供商内部处理验证和回退。
4. **默认流式：** 流式传输（常见场景）为默认行为，非流式为主动选择。
5. **环境变量配置：** 无硬编码凭据，`python-dotenv` 将 `.env` 文件桥接到 `os.environ`。
6. **可测试性：** 服务层和规划器层面均支持依赖注入，便于 mock。
