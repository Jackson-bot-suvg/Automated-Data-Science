## 1. Agent LLM 配置模块

- [x] 1.1 创建 `app/llm/agent_config.py`：定义 `AgentLLMConfig` Pydantic 模型（model, temperature, max_tokens, stream 字段）和 `get_agent_config(agent_name: str) -> AgentLLMConfig` 函数，从环境变量读取配置
- [x] 1.2 在 `app/llm/__init__.py` 中导出 `AgentLLMConfig` 和 `get_agent_config`

## 2. LLM 辅助函数模块

- [x] 2.1 创建 `app/llm/helpers.py`：实现 `parse_json_response()` 函数（从旧 `llm_service.py` 迁移，保持行为一致）
- [x] 2.2 在 `app/llm/helpers.py` 中实现 `async complete_json()` 函数：通过 `get_agent_config()` 获取 agent 配置 → 调用 `LLMService.chat()` 非流式模式 → 解析 JSON → 失败返回 fallback
- [x] 2.3 在 `app/llm/helpers.py` 中实现 `async render_text()` 函数：通过 `get_agent_config()` 获取 agent 配置 → 调用 `LLMService.chat()` → 返回文本 → 失败返回 fallback
- [x] 2.4 在 `app/llm/__init__.py` 中导出 `complete_json`、`render_text`、`parse_json_response`

## 3. 迁移 Agent LLM 调用

- [x] 3.1 迁移 `app/agents/reader.py`：将 `complete_json(env_prefix="READER_LLM")` 改为 `await complete_json(agent_name="reader")`，更新 import 路径
- [x] 3.2 迁移 `app/agents/planner.py`：将 `complete_json(env_prefix="PLANNER_LLM")` 改为 `await complete_json(agent_name="planner")`，更新 import 路径
- [x] 3.3 迁移 `app/agents/summarizer.py`：将 `complete_json(env_prefix="SUMMARIZER_LLM")` 改为 `await complete_json(agent_name="summarizer")`，更新 import 路径
- [x] 3.4 迁移 `app/agents/bg_reviewer.py`：将 `complete_json(env_prefix="BG_REVIEWER_LLM")` 改为 `await complete_json(agent_name="bg_reviewer")`，更新 import 路径
- [x] 3.5 迁移 `app/agents/reviewer_agent.py`：将 `render_review_comment()` 改为 `await render_text(agent_name="reviewer")`，更新 import 路径和调用方式
- [x] 3.6 迁移 `app/planners/base.py`：将硬编码的 `model_name`、`temperature`、`max_tokens` 改为从 `get_agent_config("planner")` 读取，模型标识符使用配置中的 `model` 字段

## 4. 清除 app/services

- [x] 4.1 删除 `app/services/llm_service.py`
- [x] 4.2 删除 `app/services/reviewer_service.py`
- [x] 4.3 删除 `app/services/__init__.py`（不保留过渡兼容层）

## 5. 环境配置

- [x] 5.1 更新 `.env.example`：添加全部 7 个 agent 的 `AGENT_<NAME>_MODEL`、`AGENT_<NAME>_TEMPERATURE`、`AGENT_<NAME>_MAX_TOKENS`、`AGENT_<NAME>_STREAM` 配置段，带注释说明

## 6. 文档和 Skills 更新

- [x] 6.1 更新 `docs/llm-usage_cn.md`：反映新架构——所有 agent 通过 `app/llm/` 调用，添加 agent 配置说明，更新集成模式章节，移除对 `app/services/` LLM 调用的引用
- [x] 6.2 更新 `docs/architecture.md`：更新 Ownership Rules 和 Implementation Status，反映 `app/services/` 已不再包含 LLM 调用代码
- [x] 6.3 更新 `docs/skills/create-module/SKILL.md`：更新模块选择指南和模板，确保新模块使用 `app/llm/` 服务层和 `get_agent_config()`

## 7. 验证

- [x] 7.1 运行项目编译检查（`python -m py_compile` 或 IDE LSP 检查），确保无 import 错误和类型错误
- [x] 7.2 验证所有 agent 的 LLM 调用路径正确（import 路径、async/await 使用）
- [x] 7.3 验证 `data_cleaning` 和 `feature_engineering` 分别读取 `AGENT_DATA_CLEANING_*` 与 `AGENT_FEATURE_ENGINEERING_*` 配置
