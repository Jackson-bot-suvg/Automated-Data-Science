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
系统 SHALL 维护已注册的 provider 列表（至少包含 `openai` 和 `anthropic`），并根据解析出的 provider 将请求路由到对应的 Provider 实现。

#### Scenario: 路由到 OpenAI 兼容 provider
- **WHEN** 模型标识 provider 为 `openai`
- **THEN** 系统 SHALL 将请求路由到 OpenAI 兼容 provider 实例

#### Scenario: 路由到 Anthropic provider
- **WHEN** 模型标识 provider 为 `anthropic`
- **THEN** 系统 SHALL 将请求路由到 Anthropic provider 实例

### Requirement: Provider 配置管理
系统 SHALL 支持为每个 provider 配置 `base_url` 和 `api_key`。配置 SHALL 优先从 `.env` 文件加载（使用 `python-dotenv` 的 `load_dotenv()`），然后从 `os.environ` 中读取，格式为 `<PROVIDER>_BASE_URL` 和 `<PROVIDER>_API_KEY`（provider 名大写）。配置来源优先级：系统环境变量 > `.env` 文件 > 代码默认值。

#### Scenario: 从 .env 文件加载 OpenAI 配置
- **WHEN** `.env` 文件中存在 `OPENAI_BASE_URL=https://api.openai.com/v1` 和 `OPENAI_API_KEY=sk-xxx`
- **THEN** 系统 SHALL 通过 `load_dotenv()` 加载后，OpenAI provider 使用该 base_url 和 api_key 创建客户端

#### Scenario: 从系统环境变量加载配置（覆盖 .env）
- **WHEN** 系统环境变量中设置了 `OPENAI_API_KEY=sk-system-env`，同时 `.env` 文件中也有 `OPENAI_API_KEY=sk-dotenv`
- **THEN** 系统 SHALL 使用系统环境变量中的值 `sk-system-env`（`load_dotenv` 默认不覆盖已存在的环境变量）

#### Scenario: 从环境变量加载 Anthropic 配置
- **WHEN** `.env` 文件或环境变量中存在 `ANTHROPIC_API_KEY=sk-ant-xxx`
- **THEN** Anthropic provider SHALL 使用该 api_key 创建客户端，base_url 默认为 Anthropic 官方 API 地址

#### Scenario: 配置缺失
- **WHEN** 某个 provider 的 `api_key` 在 `.env` 文件和环境变量中均未设置
- **THEN** 在使用该 provider 时 SHALL 抛出明确的配置错误，而非在 API 调用时失败
