## Context

当前 `app/llm/config.py` 中 `load_provider_config()` 硬编码了两个 provider：
- `openai` → `OpenAICompatConfig`，读取 `OPENAI_API_KEY` / `OPENAI_BASE_URL`
- `anthropic` → `AnthropicConfig`，读取 `ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL`

`LLMService._build_providers()` 根据 config 类型分发到 `OpenAICompatProvider` 或 `AnthropicProvider`。

用户常需要同时接入多个 OpenAI 兼容 API（DashScope、ZhipuAI、DeepSeek、vLLM 等），当前无法通过配置实现。

## Goals / Non-Goals

**Goals:**
- 用户在 `.env` 中声明多个 provider，无需修改代码
- 支持任意数量的 OpenAI 兼容 provider 和 Anthropic provider
- 向后兼容：现有 `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` 配置继续工作
- 通过模型标识符 `<provider>/model` 路由到对应 provider（已有机制不变）

**Non-Goals:**
- 不实现 Anthropic 以外的非 OpenAI 兼容 provider 类型（当前只需 `openai_compat` 和 `anthropic` 两种类型）
- 不实现运行时动态添加/移除 provider（重启加载即可）
- 不改变 `BaseProvider` 接口、`ChatMessage`/`ChatResponse` 等核心 schema
- 不实现 provider 级别的负载均衡或故障转移

## Decisions

### Decision 1: 环境变量命名方案

**选择**：统一前缀 `LLM_<NAME>_` 方案，配合 `LLM_PROVIDERS` 列表。

```env
# 声明启用的 provider（逗号分隔）
LLM_PROVIDERS=dashscope,zhipuai,anthropic

# 每个 provider 独立配置
LLM_DASHSCOPE_TYPE=openai_compat
LLM_DASHSCOPE_API_KEY=sk-xxx
LLM_DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

LLM_ZHIPUAI_TYPE=openai_compat
LLM_ZHIPUAI_API_KEY=xxx
LLM_ZHIPUAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4

LLM_ANTHROPIC_TYPE=anthropic
LLM_ANTHROPIC_API_KEY=sk-ant-xxx
```

**备选方案 A**：保持旧变量 + 仅扩展额外 provider（`OPENAI_EXTRA_PROVIDERS` + `OPENAI_<NAME>_*`）
- ❌ 放弃原因：新旧两套命名规则共存增加认知负担，且"openai"这个 provider 名称特指某个具体服务，不够通用

**备选方案 B**：JSON/YAML 配置文件
- ❌ 放弃原因：项目已全面使用 `.env`，引入新配置格式增加复杂度

**选择理由**：统一 `LLM_<NAME>_*` 命名空间简洁清晰，`LLM_PROVIDERS` 显式列出所有 provider，方便用户一目了然。

### Decision 2: 向后兼容回退策略

**选择**：`LLM_PROVIDERS` 未设置时，回退到旧行为。

```
if LLM_PROVIDERS is set:
    使用新格式解析
else:
    使用旧 OPENAI_API_KEY/ANTHROPIC_API_KEY 格式（当前行为）
```

**理由**：零破坏性迁移。现有用户什么都不用改，新用户用新格式。

### Decision 3: config.py 重构策略

**选择**：在 `config.py` 中新增 `load_provider_config_v2()` 函数，原 `load_provider_config()` 保持不变作为回退。`load_provider_config()` 内部根据 `LLM_PROVIDERS` 是否存在来选择路径。

```python
def load_provider_config() -> ProviderConfig:
    load_dotenv()
    providers_env = os.environ.get("LLM_PROVIDERS", "").strip()
    if providers_env:
        return _load_from_unified(providers_env)
    return _load_from_legacy()
```

**备选方案**：直接修改 `load_provider_config()`
- 选择当前方案的理由：最小化改动范围，新旧逻辑隔离清晰

### Decision 4: ProviderConfig 模型保持统一

**选择**：`ProviderConfig` 的 `providers` 字典值类型不变，仍然是 `OpenAICompatConfig | AnthropicConfig`。新增 `ProviderType` 字段到各个 Config 中（或通过 isinstance 区分）。

**理由**：`service.py` 中的 `_build_providers()` 已经通过 `isinstance` 分发，无需修改模型结构。

## Risks / Trade-offs

- **[Risk] 新旧配置共存时用户混淆** → 在 `.env.example` 中清晰标注两种格式，文档中推荐新格式
- **[Risk] provider 名称大小写不一致** → 规范：`LLM_PROVIDERS` 中写小写，env var 中名称自动转大写查找。解析时统一 `upper()` 匹配环境变量
- **[Trade-off] 不支持 provider 级别的额外参数（如自定义 headers）** → 当前足够，未来可扩展 `LLM_<NAME>_EXTRA_*`
- **[Trade-off] `LLM_PROVIDERS` 不存在时的静默回退可能让用户以为新配置生效** → 在日志中打印当前使用的配置模式
