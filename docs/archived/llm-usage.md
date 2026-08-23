# LLM Service Layer & App Integration Guide

## 1. Overview

The `app/llm/` module provides a **unified, provider-agnostic interface** for all LLM interactions in the project. Business nodes (agents, planners) never hold SDK clients directly — they call through the service layer, which handles provider routing, configuration, streaming, structured output, and error normalization.

### Architecture at a Glance

```
Business Nodes (app/agents/, app/planners/)
        │
        ▼
  LLMService (app/llm/service.py)
        │
        ├── Model Router (app/llm/router.py)
        │       parses "provider/prefix/model" identifiers
        │
        ├── Provider Registry
        │       ├── OpenAICompatProvider  →  any OpenAI-compatible API
        │       └── AnthropicProvider     →  Anthropic Messages API
        │
        └── Config (app/llm/config.py)
                reads credentials from environment variables
```

---

## 2. Quick Start

### 2.1 Configure Environment

Copy `.env.example` to `.env` and fill in your credentials.

#### Recommended: Unified Multi-Provider Configuration

Declare the providers you want to use via the `LLM_PROVIDERS` environment variable (comma-separated). Each provider is configured through `LLM_<NAME>_TYPE`, `LLM_<NAME>_API_KEY`, and `LLM_<NAME>_BASE_URL`:

```bash
# Declare enabled providers (comma-separated)
LLM_PROVIDERS=dashscope,zhipuai

# DashScope (OpenAI-compatible)
LLM_DASHSCOPE_TYPE=openai_compat
LLM_DASHSCOPE_API_KEY=sk-xxxxx
LLM_DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# ZhipuAI (OpenAI-compatible)
LLM_ZHIPUAI_TYPE=openai_compat
LLM_ZHIPUAI_API_KEY=xxxxx
LLM_ZHIPUAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4

# Anthropic (Claude models) — add as needed
# LLM_PROVIDERS=dashscope,zhipuai,anthropic
# LLM_ANTHROPIC_TYPE=anthropic
# LLM_ANTHROPIC_API_KEY=sk-ant-xxxxx
```

**Supported provider types:**
- `openai_compat` (default): Works with any OpenAI-compatible API (DashScope, ZhipuAI, vLLM, DeepSeek, etc.)
- `anthropic`: Works with the Anthropic Messages API (Claude models)

**Environment variable name mapping:** The name in `LLM_PROVIDERS` is uppercased to look up corresponding env vars. For example, `dashscope` → `LLM_DASHSCOPE_API_KEY`.

**Provider name rules:** Only letters, digits, `-`, and `_` are allowed.

#### Legacy Format (Backward Compatible)

If `LLM_PROVIDERS` is not set or is empty, the system automatically falls back to the legacy format:

```bash
# Required for OpenAI-compatible provider (DashScope, ZhipuAI, vLLM, etc.)
OPENAI_API_KEY=sk-xxxxx
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# Required for Anthropic provider
ANTHROPIC_API_KEY=sk-ant-xxxxx
# ANTHROPIC_BASE_URL=https://api.anthropic.com   # optional override
```

Existing deployments continue to work without any changes.

### 2.2 Basic Usage

```python
from app.llm import get_service, ChatMessage

service = get_service()

# Non-streaming call
response = await service.chat(
    model="openai/glm-5",
    messages=[
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="Hello!"),
    ],
    stream=False,
)
print(response.content)        # The reply text
print(response.usage)          # UsageInfo(input_tokens=..., output_tokens=...)

# Streaming call (default)
async for token in await service.chat(
    model="openai/glm-5",
    messages=[ChatMessage(role="user", content="Tell me a story.")],
):
    print(token, end="", flush=True)
```

### 2.3 Structured Output

Pass a Pydantic model class as `response_schema` to get a typed, validated response. **This forces non-streaming mode** regardless of the `stream` parameter.

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

result: SentimentResult = response.parsed  # Already validated
print(result.label)       # "positive"
print(result.confidence)  # 0.95
```

---

## 3. Module Reference

### 3.1 `app/llm/__init__.py` — Public API

The package exposes a convenience singleton via `get_service()`:

```python
from app.llm import get_service, ChatMessage, ChatResponse, ChatParams, LLMError, UsageInfo
```

| Export | Description |
|---|---|
| `get_service()` | Returns a module-level singleton `LLMService` |
| `ChatMessage` | `(role: str, content: str)` — a single message |
| `ChatResponse` | `(content, model, usage?, parsed?)` — complete response |
| `ChatParams` | Parameter holder (see §3.3) |
| `LLMError` | Unified exception with `provider` and `model` context |
| `UsageInfo` | `(input_tokens, output_tokens)` |

### 3.2 `LLMService` (`app/llm/service.py`)

The central dispatcher. One `chat()` method handles everything.

```python
class LLMService:
    def __init__(self, providers: dict[str, BaseProvider] | None = None): ...
    async def chat(self, model: str, messages: list[ChatMessage], **kwargs) -> ChatResult: ...
```

**Constructor** — When `providers` is `None` (the default), providers are built automatically from environment variables via `load_provider_config()`. Pass a custom dict for testing or advanced setups.

**`chat()` parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | required | Identifier: `<provider>/[<prefix>/]<model>` |
| `messages` | `list[ChatMessage]` | required | Conversation history |
| `temperature` | `float` | `None` | Sampling randomness |
| `top_p` | `float` | `None` | Nucleus sampling threshold |
| `stream` | `bool` | `True` | Stream tokens or return full response |
| `max_tokens` | `int` | `None` | Hard cap on generated tokens |
| `thinking_budget` | `int` | `None` | Extended-thinking tokens (Anthropic only) |
| `response_schema` | `type[BaseModel]` | `None` | Pydantic class for structured output (forces non-streaming) |

**Returns:** `ChatResponse` when `stream=False` (or when `response_schema` is set), otherwise `AsyncGenerator[str, None]`.

### 3.3 `ChatParams` (`app/llm/schemas.py`)

All optional chat parameters are normalized into this Pydantic model internally. You rarely construct it directly — just pass keyword arguments to `chat()`.

### 3.4 `ChatMessage` (`app/llm/schemas.py`)

```python
class ChatMessage(BaseModel):
    role: str      # "system" | "user" | "assistant"
    content: str
```

### 3.5 `ChatResponse` (`app/llm/schemas.py`)

```python
class ChatResponse(BaseModel):
    content: str                     # Raw text from the LLM
    model: str                       # Actual model name returned by the API
    usage: UsageInfo | None = None   # Token usage stats
    parsed: Any | None = None        # Validated Pydantic object (structured output)
```

---

## 4. Model Identifier Format

Model identifiers follow the format: `<provider>/[<prefix>/]<model>`

The router (`app/llm/router.py`) parses these into a `ModelId(provider, prefix, model)` dataclass. Provider names are determined dynamically by the configuration layer.

| Identifier | Provider | Prefix | Model |
|---|---|---|---|
| `openai/glm-5` | `openai` | `None` | `glm-5` |
| `openai/zai/glm-5.1` | `openai` | `zai` | `glm-5.1` |
| `anthropic/claude-opus-4.6` | `anthropic` | `None` | `claude-opus-4.6` |
| `dashscope/qwen-plus` | `dashscope` | `None` | `qwen-plus` |

**Note:** Provider names are no longer limited to `openai` and `anthropic` — any provider name declared in `LLM_PROVIDERS` (or registered via the legacy format) can be used.

Invalid identifiers or unregistered providers raise `ValueError`.

---

## 5. Providers

### 5.1 `OpenAICompatProvider` (`app/llm/providers/openai_compat.py`)

Works with **any service that exposes an OpenAI-compatible Chat Completions endpoint** (DashScope, ZhipuAI, local vLLM, Ollama, etc.).

**Key behaviors:**
- Uses `openai.AsyncOpenAI` with a configurable `base_url`
- The `prefix` segment of the model identifier is prepended: `prefix/model` (e.g., `zai/glm-5.1`)
- `thinking_budget` is silently ignored (not supported by OpenAI APIs)

**Structured output strategy (two-tier fallback):**
1. **Primary:** `beta.chat.completions.parse()` with `response_format=schema` — leverages native structured output support
2. **Fallback:** If `parse()` is not supported, switches to JSON mode (`response_format={"type": "json_object"}`) with the JSON schema injected into the user prompt, then validates the response with `schema.model_validate_json()`

### 5.2 `AnthropicProvider` (`app/llm/providers/anthropic.py`)

Works with the **Anthropic Messages API** (Claude models).

**Key behaviors:**
- Uses `anthropic.AsyncAnthropic` client
- System messages are extracted and passed via the separate `system` parameter (required by Anthropic)
- The `prefix` segment of model identifiers is ignored (not applicable)
- Supports **extended thinking** via `thinking_budget` → maps to `thinking={"type": "enabled", "budget_tokens": N}`
- Requires `max_tokens` to be set; defaults to `8192` for structured output if not specified

**Structured output strategy:**
- JSON schema is injected into the system prompt with strict formatting instructions
- Response is validated with `schema.model_validate_json()`

---

## 6. Configuration (`app/llm/config.py`)

Configuration is loaded from **environment variables** using `python-dotenv`. Priority: system env vars > `.env` file > code defaults.

### 6.1 Unified Format (Recommended)

When `LLM_PROVIDERS` is set, the unified format is used. Each provider is configured through the `LLM_<NAME>_` prefixed environment variables.

| Variable | Description | Default |
|---|---|---|
| `LLM_PROVIDERS` | Comma-separated list of provider names | — |
| `LLM_<NAME>_TYPE` | Provider type: `openai_compat` or `anthropic` | `openai_compat` |
| `LLM_<NAME>_API_KEY` | API key | — |
| `LLM_<NAME>_BASE_URL` | API endpoint URL (`openai_compat` defaults to `https://api.openai.com/v1`, `anthropic` defaults to `https://api.anthropic.com`) | See left |

**Example:**
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

### 6.2 Legacy Format (Backward Compatible)

When `LLM_PROVIDERS` is not set or is empty, the system automatically falls back to the legacy format.

| Variable | Provider | Description | Default |
|---|---|---|---|
| `OPENAI_API_KEY` | OpenAI-compat | API key | *(required)* |
| `OPENAI_BASE_URL` | OpenAI-compat | API endpoint URL | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `ANTHROPIC_API_KEY` | Anthropic | API key | *(required)* |
| `ANTHROPIC_BASE_URL` | Anthropic | API endpoint URL | `https://api.anthropic.com` |

Under the legacy format, both `openai` and `anthropic` providers are always registered. A provider with an empty `api_key` will raise `ValueError` at call time.

### 6.3 Configuration Validation

- Provider names must contain only letters, digits, `-`, and `_`; otherwise a `ValueError` is raised at startup
- Unsupported provider types (not `openai_compat` / `anthropic`) raise `ValueError` at startup
- Providers with missing API keys emit a warning log but do not block startup (errors occur at call time)

---

## 7. Error Handling

All provider-specific SDK exceptions are caught and re-raised as `LLMError`:

```python
class LLMError(Exception):
    provider: str   # e.g., "openai"
    model: str      # e.g., "glm-5"
    # Message format: "[openai/glm-5] <original SDK error>"
```

This gives callers a uniform error type regardless of the underlying provider.

---

## 8. App Integration Patterns

Business nodes integrate with the LLM service layer through two patterns:

### 8.1 Direct Service Usage (simple cases)

```python
from app.llm import get_service, ChatMessage

service = get_service()
response = await service.chat("openai/glm-5", messages, stream=False)
```

### 8.2 Planner Pattern (production usage)

The project's primary pattern is the **Planner Pattern** — business nodes delegate LLM reasoning to planner classes, which use structured output to generate executable tool-call plans.

```
Agent Node (e.g., data_cleaning.py)
    │
    ├─ 1. Gather context (dataset profile, EDA findings)
    │
    ├─ 2. BasePlanner(agent_name="data_cleaning").plan_data_cleaning(profile_json)
    │       │
    │       ├─ Build ChatMessage list (system + user prompts)
    │       ├─ LLMService.chat(model="openai/glm-5",
    │       │       response_schema=ToolCallPlan, ...)
    │       └─ Return typed ToolCallPlan
    │
    └─ 3. Execute plan.steps → dynamic tool dispatch
```

**Current LLM consumers:**

| Component | Model | Structured Output | Purpose |
|---|---|---|---|
| `BasePlanner(agent_name="data_cleaning").plan_data_cleaning()` | `AGENT_DATA_CLEANING_MODEL` (e.g. `dashscope/glm-5`) | `ToolCallPlan` | Generate data cleaning tool calls |
| `BasePlanner(agent_name="feature_engineering").plan_feature_engineering()` | `AGENT_FEATURE_ENGINEERING_MODEL` (e.g. `dashscope/glm-5`) | `ToolCallPlan` | Generate feature engineering tool calls |

**Nodes that do NOT use LLM:**
- `preliminary_eda` — pure data profiling and visualization
- `model_building` — traditional ML (scikit-learn)
- `business_translator` — deterministic metric-to-business translation

### 8.3 Dependency Injection

Both `LLMService` and `BasePlanner` support dependency injection for testing. `BasePlanner` also accepts `agent_name` to select the corresponding `AGENT_<NAME>_*` config segment:

```python
# Inject mock providers into LLMService
mock_service = LLMService(providers={"openai": mock_provider})

# Inject mock service into BasePlanner and bind agent config segment
planner = BasePlanner(agent_name="data_cleaning", service=mock_service)
```

---

## 9. Observability Integration

LLM calls made through `BasePlanner` are automatically wrapped in **LangSmith tracing** contexts:

```python
# In BasePlanner._invoke_structured_llm():
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

This is transparent to the service layer — tracing happens at the planner level. Direct `service.chat()` calls are not traced unless wrapped by the caller.

---

## 10. File Index

```
app/llm/
├── __init__.py              # Public API & singleton (get_service)
├── service.py               # LLMService — unified dispatch
├── router.py                # ModelId parsing & provider routing
├── config.py                # Environment-based provider configuration
├── schemas.py               # ChatMessage, ChatResponse, ChatParams, LLMError
└── providers/
    ├── base.py              # BaseProvider ABC & ChatResult type alias
    ├── openai_compat.py     # OpenAI-compatible provider implementation
    └── anthropic.py         # Anthropic Messages API provider
```

---

## 11. Adding a New Provider

### 11.1 Via Configuration (Recommended, for OpenAI-compatible / Anthropic types)

Simply add a new provider entry in `.env` — no code changes needed:

```bash
# Add the new provider name to LLM_PROVIDERS
LLM_PROVIDERS=dashscope,deepseek

# Configure the new provider
LLM_DEEPSEEK_TYPE=openai_compat
LLM_DEEPSEEK_API_KEY=sk-xxx
LLM_DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
```

Then use it via `deepseek/deepseek-chat`.

### 11.2 Adding a New Provider Type (requires code changes)

If you need to support a provider type that is neither OpenAI-compatible nor Anthropic:

1. Create `app/llm/providers/<name>.py` implementing `BaseProvider`:

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
        # Implement provider-specific API call
        ...
```

2. Add a new Config model in `config.py` and support the new type in `_load_from_unified()`.

3. Add the provider construction logic in `LLMService._build_providers()`.

---

## 12. Design Principles

1. **Single entry point:** All LLM calls go through `LLMService.chat()`. No business code holds SDK clients.
2. **Provider-agnostic:** Model identifiers (`openai/glm-5`, `anthropic/claude-opus-4.6`) determine routing. Callers don't know about providers.
3. **Structured output first:** The `response_schema` parameter ensures type-safe LLM outputs. Providers handle validation and fallback internally.
4. **Streaming by default:** The common case (streaming) is the default. Non-streaming is opt-in.
5. **Configuration via environment:** No hardcoded credentials. `python-dotenv` bridges `.env` files to `os.environ`.
6. **Testable:** Dependency injection at both service and planner levels enables easy mocking.
