## Why

当前 `app/llm/config.py` 硬编码了恰好两个 provider（`openai` 和 `anthropic`），每个 provider 有固定的环境变量名（`OPENAI_API_KEY`、`ANTHROPIC_API_KEY` 等）。如果用户需要同时使用多个 OpenAI 兼容供应商（例如 DashScope、ZhipuAI、vLLM、DeepSeek），无法通过 `.env` 配置，必须修改代码。

需要将 provider 配置从硬编码改为声明式——用户在 `.env` 中列出要使用的 provider 名称及各自的凭据即可，无需改动代码。

## What Changes

- **新增统一 provider 注册机制**：引入 `LLM_PROVIDERS` 环境变量声明已启用的 provider 列表，每个 provider 通过 `LLM_<NAME>_TYPE`、`LLM_<NAME>_API_KEY`、`LLM_<NAME>_BASE_URL` 配置
- **向后兼容**：当 `LLM_PROVIDERS` 未设置时，回退到现有 `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` 行为，无破坏性变更
- **支持多个 OpenAI 兼容 provider**：同一 `.env` 中可声明 `dashscope`、`zhipuai`、`deepseek`、`vllm` 等任意数量的 OpenAI 兼容供应商，各自独立配置 base_url 和 api_key
- **更新 `config.py`**：将硬编码的 `load_provider_config()` 改为动态扫描 `LLM_<NAME>_*` 环境变量
- **更新 `.env.example`**：补充新格式的配置示例和说明
- **更新文档**：同步更新 `docs/llm-usage.md` 和 `docs/llm-usage_cn.md` 中的配置章节
- **更新 skills**：补充 OpenSpec skills 中涉及 LLM 配置的内容

## Capabilities

### New Capabilities
- `multi-provider-config`: 在 `.env` 中声明式配置多个 OpenAI 兼容 provider，包括统一环境变量格式、动态 provider 发现和注册、向后兼容回退逻辑

### Modified Capabilities
- `llm-router`: Provider 注册与路由规则需要适配动态 provider 列表（不再仅限 openai/anthropic 两个固定名称）

## Impact

- **代码**：`app/llm/config.py`（主要修改）、`app/llm/service.py`（`_build_providers` 适配）、`.env.example`
- **文档**：`docs/llm-usage.md`、`docs/llm-usage_cn.md`
- **配置**：`.env` 文件格式扩展（新增 `LLM_PROVIDERS` 和 `LLM_<NAME>_*` 变量族）
- **向后兼容**：现有 `.env` 配置无需修改即可继续工作
