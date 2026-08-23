## 1. 项目结构搭建

- [x] 1.1 创建 `app/llm/` 模块目录结构（`__init__.py`、`providers/` 子目录）
- [x] 1.2 在 `pyproject.toml` 中确认/添加 `openai` 和 `anthropic` SDK 依赖

## 2. 数据结构与配置

- [x] 2.1 创建 `app/llm/schemas.py`：定义 `ChatMessage`（role, content）、`ChatResponse`（content, model, usage）、`UsageInfo`（input_tokens, output_tokens）、`ChatParams`（temperature, top_p, stream, thinking_budget）、`LLMError` 异常类
- [x] 2.2 创建 `app/llm/config.py`：定义 `ProviderConfig`（base_url, api_key）和全局配置加载逻辑，使用 `python-dotenv` 的 `load_dotenv()` 加载 `.env` 文件，然后从 `os.environ` 读取 `<PROVIDER>_BASE_URL` 和 `<PROVIDER>_API_KEY`，使用 Pydantic 模型
- [x] 2.3 创建 `.env.example` 模板文件：列出所有 LLM 服务层需要的环境变量（`OPENAI_API_KEY`、`OPENAI_BASE_URL`、`ANTHROPIC_API_KEY` 等），含注释说明，不包含实际密钥

## 3. 模型路由

- [x] 3.1 创建 `app/llm/router.py`：实现模型标识解析函数 `parse_model_id(model: str) -> ModelId(provider, prefix, model)`，支持两段式和三段式格式
- [x] 3.2 实现无效标识校验：缺少 `/` 时抛出 ValueError，未知 provider 时抛出 ValueError

## 4. Provider 实现

- [x] 4.1 创建 `app/llm/providers/base.py`：定义 `BaseProvider` 抽象基类，声明 `chat()` 异步方法签名和 `chat_stream()` 异步方法签名
- [x] 4.2 创建 `app/llm/providers/openai_compat.py`：实现 OpenAI 兼容 provider，使用 `openai` SDK 的 `AsyncOpenAI` 客户端，支持自定义 `base_url`、stream/non-stream 模式，处理 prefix（用于某些兼容服务的路径前缀）
- [x] 4.3 创建 `app/llm/providers/anthropic.py`：实现 Anthropic provider，使用 `anthropic` SDK 的 `AsyncAnthropic` 客户端，处理 system 消息分离、thinking_budget 参数传递
- [x] 4.4 在各 provider 中实现错误捕获，将底层 SDK 异常转换为统一的 `LLMError`

## 5. 服务层主入口

- [x] 5.1 创建 `app/llm/service.py`：实现 `LLMService` 类，维护 provider 注册表（openai → OpenAICompatProvider, anthropic → AnthropicProvider），根据路由结果分发请求
- [x] 5.2 实现 `LLMService.chat()` 方法：解析模型标识 → 路由到 provider → 传递参数 → 返回 `ChatResponse` 或 `AsyncGenerator`
- [x] 5.3 实现 `LLMService` 的初始化逻辑：从配置加载各 provider 实例，校验必需配置（api_key）
- [x] 5.4 创建 `app/llm/__init__.py`：导出 `LLMService`、`ChatMessage`、`ChatResponse`、`ChatParams`、`LLMError`，提供便捷的模块级 `get_service()` 函数

## 6. 测试

- [x] 6.1 编写 `tests/unit/llm/test_router.py`：测试模型标识解析（标准两段式、三段式、Anthropic 格式）和无效输入处理
- [x] 6.2 编写 `tests/unit/llm/test_config.py`：测试环境变量配置加载（`.env` 文件加载、系统环境变量覆盖 `.env`）、配置缺失错误
- [x] 6.3 编写 `tests/unit/llm/test_service.py`：测试 provider 路由、参数传递、stream 默认值、thinking_budget 忽略逻辑（mock provider）
- [x] 6.4 编写 `tests/unit/llm/test_providers.py`：测试 OpenAI 兼容 provider 和 Anthropic provider 的消息格式转换、流式/非流式响应、错误处理（mock SDK 调用）

## 7. 文档与技能更新

- [x] 7.1 更新 `docs/architecture.md`：在 High-Level Layers 新增 `app/llm/` 层条目（统一 LLM 调用接口、多供应商路由），在 Ownership Rules 新增 LLM 服务层所有权规则（模型路由、配置管理、provider 实现归属 `app/llm/`，业务节点通过服务层调用而非直接持有客户端）
- [x] 7.2 更新 `docs/engineering-specification.md`：在 Repository Structure 目录树中新增 `app/llm/` 子目录及文件列表（`__init__.py`、`config.py`、`router.py`、`service.py`、`schemas.py`、`providers/`）
- [x] 7.3 更新 `docs/observability.md`：在 Environment Variables 中新增 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`ANTHROPIC_API_KEY` 等新服务层使用的环境变量说明
- [x] 7.4 更新 `docs/skills/create-module/SKILL.md`：在 Module Selection 中新增 `app/llm/` 包条目（统一 LLM 调用服务层、多供应商路由与配置），明确所有权边界

## 8. 现有代码迁移

- [x] 8.1 修改 `app/planners/base.py`：将配置来源迁移至 `app/llm/config.py`，API 配置统一从 LLM 服务层加载（保留 LangChain 结构化输出功能）
- [x] 8.2 检查其他 agent 代码中是否有直接 LLM 调用，如有则迁移至新服务层
- [x] 8.3 运行现有测试确保迁移后功能正常
