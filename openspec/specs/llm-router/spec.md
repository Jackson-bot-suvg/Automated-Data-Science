## ADDED Requirements

### Requirement: 模型标识解析
系统 SHALL 解析模型标识字符串，提取 provider、可选 prefix 和 model name。解析规则：以 `/` 分隔，第一段为 provider，最后一段为 model name，中间部分为 prefix（可为空）。

#### Scenario: 标准两段式标识
- **WHEN** 输入模型标识 `openai/GPT-5.4`
- **THEN** 解析结果为 provider=`openai`，prefix=`None`，model=`GPT-5.4`

#### Scenario: 三段式标识（含 prefix）
- **WHEN** 输入模型标识 `openai/zai/glm-5.1`
- **THEN** 解析结果为 provider=`openai`，prefix=`zai`，model=`glm-5.1`

#### Scenario: Anthropic 标识
- **WHEN** 输入模型标识 `anthropic/claude-opus-4.6`
- **THEN** 解析结果为 provider=`anthropic`，prefix=`None`，model=`claude-opus-4.6`

#### Scenario: 无效标识 — 缺少分隔符
- **WHEN** 输入模型标识 `GPT-5.4`（无 `/`）
- **THEN** 系统 SHALL 抛出 `ValueError`，提示模型标识格式不正确

#### Scenario: 无效标识 — 未知 provider
- **WHEN** 输入模型标识 `google/gemini-pro`（provider 未注册）
- **THEN** 系统 SHALL 抛出 `ValueError`，提示 provider 不受支持

### Requirement: Provider 注册与路由
系统 SHALL 维护动态的 provider 列表（由 `LLM_PROVIDERS` 环境变量或旧格式环境变量决定），并根据解析出的 provider 将请求路由到对应的 Provider 实现。不再限定为固定的 `openai` 和 `anthropic` 两个名称。

#### Scenario: 路由到任意已注册的 OpenAI 兼容 provider
- **WHEN** 模型标识 provider 为 `dashscope`（已在 `LLM_PROVIDERS` 中声明）
- **THEN** 系统 SHALL 将请求路由到名为 `dashscope` 的 OpenAICompatProvider 实例

#### Scenario: 路由到 Anthropic provider（自定义名称）
- **WHEN** 模型标识 provider 为 `claude`（在 `LLM_PROVIDERS` 中声明且 TYPE 为 `anthropic`）
- **THEN** 系统 SHALL 将请求路由到名为 `claude` 的 AnthropicProvider 实例

#### Scenario: 路由到未注册的 provider
- **WHEN** 模型标识 provider 为 `unknown-provider`（未在任何配置中声明）
- **THEN** 系统 SHALL 抛出 `ValueError`，提示 provider 不受支持并列出当前已注册的 provider 列表

#### Scenario: 向后兼容路由到 openai provider
- **WHEN** 未设置 `LLM_PROVIDERS`，使用旧格式 `OPENAI_API_KEY`，模型标识为 `openai/glm-5`
- **THEN** 系统 SHALL 将请求路由到名为 `openai` 的 OpenAICompatProvider（与当前行为一致）

#### Scenario: 向后兼容路由到 anthropic provider
- **WHEN** 未设置 `LLM_PROVIDERS`，使用旧格式 `ANTHROPIC_API_KEY`，模型标识为 `anthropic/claude-opus-4.6`
- **THEN** 系统 SHALL 将请求路由到名为 `anthropic` 的 AnthropicProvider（与当前行为一致）

### Requirement: Provider 配置管理
系统 SHALL 支持两种配置格式：统一格式（`LLM_PROVIDERS` + `LLM_<NAME>_*`）和旧格式（`OPENAI_API_KEY`/`OPENAI_BASE_URL` + `ANTHROPIC_API_KEY`/`ANTHROPIC_BASE_URL`）。当 `LLM_PROVIDERS` 存在时使用统一格式，否则回退到旧格式。配置来源优先级：系统环境变量 > `.env` 文件 > 代码默认值。

#### Scenario: 从 .env 文件加载统一格式配置
- **WHEN** `.env` 中设置 `LLM_PROVIDERS=dashscope,zhipuai` 及各自 `LLM_DASHSCOPE_API_KEY`、`LLM_ZHIPUAI_API_KEY`
- **THEN** 系统 SHALL 通过 `load_dotenv()` 加载后，为每个 provider 使用对应配置创建 Provider 实例

#### Scenario: 旧格式配置加载（未设置 LLM_PROVIDERS）
- **WHEN** 未设置 `LLM_PROVIDERS`，`.env` 中存在 `OPENAI_API_KEY=sk-xxx` 和 `ANTHROPIC_API_KEY=sk-ant-xxx`
- **THEN** 系统 SHALL 以旧行为加载 `openai` 和 `anthropic` 两个 provider

#### Scenario: 配置缺失
- **WHEN** 某个 provider 的 `api_key` 在 `.env` 文件和环境变量中均未设置
- **THEN** 在使用该 provider 时 SHALL 抛出明确的配置错误，而非在 API 调用时失败
