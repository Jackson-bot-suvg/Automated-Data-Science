## 1. 配置层重构

- [x] 1.1 在 `app/llm/config.py` 中实现 `_load_from_legacy()` 函数，提取现有 `load_provider_config()` 的逻辑（读取 `OPENAI_API_KEY`/`OPENAI_BASE_URL` 和 `ANTHROPIC_API_KEY`/`ANTHROPIC_BASE_URL`）
- [x] 1.2 在 `app/llm/config.py` 中实现 `_load_from_unified(providers_env: str)` 函数，解析 `LLM_PROVIDERS` 逗号分隔列表，为每个 provider 读取 `LLM_<NAME>_TYPE`、`LLM_<NAME>_API_KEY`、`LLM_<NAME>_BASE_URL`，支持 provider 名称大写映射、类型默认为 `openai_compat`、不支持的类型抛出 `ValueError`
- [x] 1.3 重构 `load_provider_config()`：根据 `LLM_PROVIDERS` 环境变量是否存在，分派到 `_load_from_unified()` 或 `_load_from_legacy()`
- [x] 1.4 添加 provider 名称格式校验（仅允许字母、数字、`-`、`_`），无效名称抛出 `ValueError`
- [x] 1.5 为缺少 `api_key` 的 provider 添加 `logging.warning` 日志输出

## 2. 服务层适配

- [x] 2.1 验证 `app/llm/service.py` 的 `_build_providers()` 方法无需修改（应已通过 `isinstance` 分发兼容动态 provider 列表）
- [x] 2.2 如需修改 `_build_providers()`，确保 `openai_compat` 类型创建 `OpenAICompatProvider`，`anthropic` 类型创建 `AnthropicProvider`

## 3. 测试

- [x] 3.1 为 `_load_from_unified()` 编写单元测试：多 OpenAI 兼容 provider、混合类型、默认类型、不支持的类型、无效名称
- [x] 3.2 为向后兼容回退编写单元测试：`LLM_PROVIDERS` 未设置时回退到旧格式
- [x] 3.3 为 `LLMService` 编写集成测试：使用新格式配置时，`chat()` 能正确路由到动态注册的 provider
- [x] 3.4 运行全量测试确保无回归

## 4. 配置文件更新

- [x] 4.1 更新 `.env.example`：添加 `LLM_PROVIDERS` 和 `LLM_<NAME>_*` 格式示例，保留旧格式注释，标注推荐使用新格式

## 5. 文档更新

- [x] 5.1 更新 `docs/llm-usage_cn.md`：在配置章节添加多 provider 配置说明、环境变量格式说明、向后兼容说明、使用示例
- [x] 5.2 更新 `docs/llm-usage.md`：同步英文版配置章节，内容与中文版一致
- [x] 5.3 检查并更新 `.opencode/skills/` 中涉及 LLM 配置的 skill 文件（如有）

## 6. 编译验证

- [x] 6.1 运行项目编译/类型检查，确保无错误
- [x] 6.2 运行测试套件，确保所有测试通过
