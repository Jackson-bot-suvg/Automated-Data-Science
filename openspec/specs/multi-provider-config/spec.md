## ADDED Requirements

### Requirement: 统一 Provider 声明式配置
系统 SHALL 支持通过 `LLM_PROVIDERS` 环境变量声明启用的 provider 列表（逗号分隔，如 `dashscope,zhipuai,anthropic`）。列表中每个名称对应一组 `LLM_<NAME>_TYPE`、`LLM_<NAME>_API_KEY`、`LLM_<NAME>_BASE_URL` 环境变量。

#### Scenario: 配置多个 OpenAI 兼容 provider
- **WHEN** `.env` 中设置 `LLM_PROVIDERS=dashscope,zhipuai`，且配置了 `LLM_DASHSCOPE_TYPE=openai_compat`、`LLM_DASHSCOPE_API_KEY=sk-a`、`LLM_DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`、`LLM_ZHIPUAI_TYPE=openai_compat`、`LLM_ZHIPUAI_API_KEY=sk-b`、`LLM_ZHIPUAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4`
- **THEN** 系统 SHALL 注册两个 provider：`dashscope`（OpenAICompatProvider）和 `zhipuai`（OpenAICompatProvider），各自使用对应的 base_url 和 api_key

#### Scenario: 混合配置 OpenAI 兼容和 Anthropic provider
- **WHEN** `.env` 中设置 `LLM_PROVIDERS=dashscope,anthropic`，且 `LLM_DASHSCOPE_TYPE=openai_compat`、`LLM_ANTHROPIC_TYPE=anthropic`
- **THEN** 系统 SHALL 注册 `dashscope`（OpenAICompatProvider）和 `anthropic`（AnthropicProvider）

#### Scenario: LLM_PROVIDERS 为空或未设置时回退
- **WHEN** `.env` 中未设置 `LLM_PROVIDERS` 或其值为空字符串
- **THEN** 系统 SHALL 回退到旧格式，从 `OPENAI_API_KEY`/`OPENAI_BASE_URL` 和 `ANTHROPIC_API_KEY`/`ANTHROPIC_BASE_URL` 加载配置（与当前行为一致）

### Requirement: Provider 类型识别
系统 SHALL 通过 `LLM_<NAME>_TYPE` 环境变量识别 provider 类型。支持的类型为 `openai_compat` 和 `anthropic`。

#### Scenario: TYPE 为 openai_compat
- **WHEN** `LLM_MYPROVIDER_TYPE=openai_compat`
- **THEN** 系统 SHALL 为该 provider 创建 `OpenAICompatProvider` 实例

#### Scenario: TYPE 为 anthropic
- **WHEN** `LLM_MYPROVIDER_TYPE=anthropic`
- **THEN** 系统 SHALL 为该 provider 创建 `AnthropicProvider` 实例

#### Scenario: TYPE 缺失时的默认值
- **WHEN** 声明了某个 provider 但未设置 `LLM_<NAME>_TYPE`
- **THEN** 系统 SHALL 默认为 `openai_compat` 类型

#### Scenario: TYPE 值不支持
- **WHEN** `LLM_MYPROVIDER_TYPE=gemini`（不支持的类型）
- **THEN** 系统 SHALL 在启动时抛出 `ValueError`，明确提示不支持的 provider 类型

### Requirement: 环境变量名称映射
系统 SHALL 将 `LLM_PROVIDERS` 中的 provider 名称转换为大写后查找对应环境变量。例如 `dashscope` → 查找 `LLM_DASHSCOPE_API_KEY`。

#### Scenario: 大小写转换
- **WHEN** `LLM_PROVIDERS=MyProvider`
- **THEN** 系统 SHALL 查找 `LLM_MYPROVIDER_API_KEY`、`LLM_MYPROVIDER_BASE_URL`、`LLM_MYPROVIDER_TYPE`

#### Scenario: provider 名称包含特殊字符
- **WHEN** `LLM_PROVIDERS` 中的名称包含非字母数字字符（除 `-` 和 `_`）
- **THEN** 系统 SHALL 在启动时抛出 `ValueError`，提示 provider 名称格式无效

### Requirement: 配置验证
系统 SHALL 在加载配置时验证每个声明 provider 的必要字段。`openai_compat` 类型需要 `api_key` 和 `base_url`；`anthropic` 类型需要 `api_key`，`base_url` 默认为 `https://api.anthropic.com`。

#### Scenario: 缺少 API Key
- **WHEN** `LLM_PROVIDERS` 中声明了 `myprovider` 但未设置 `LLM_MYPROVIDER_API_KEY`
- **THEN** 系统 SHALL 在加载时发出警告日志，该 provider 的 `api_key` 设为空字符串（与旧行为一致），在实际调用时抛出 `ValueError`

#### Scenario: OpenAI 兼容 provider 缺少 BASE_URL
- **WHEN** 声明了 `openai_compat` 类型但未设置 `LLM_MYPROVIDER_BASE_URL`
- **THEN** 系统 SHALL 默认 base_url 为 `https://api.openai.com/v1`
