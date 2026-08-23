"""BasePlanner — 智能规划中枢。

通过 ``app/llm/`` 服务层的结构化输出能力调用 GLM-5 大模型，
接收数据集画像 JSON，返回强类型 ToolCallPlan 执行计划。

已移除 LangChain 依赖，统一使用 ``LLMService.chat()`` 的
``response_schema`` 参数实现结构化输出。
"""

from __future__ import annotations

import json
from typing import Any

from app.llm import get_service
from app.llm.agent_config import get_agent_config
from app.llm.schemas import ChatMessage, ChatResponse
from app.llm.service import LLMService
from app.observability.langsmith import (
    get_langsmith_tracing_extra,
    planner_trace_metadata,
    planner_trace_tags,
)
from app.schemas.tools import ToolCallPlan
from app.planners.prompts import (
    SYSTEM_PLANNER_PROMPT,
    build_data_cleaning_prompt,
    build_feature_engineering_prompt,
    build_model_building_prompt,
)


class BasePlanner:
    """LLM 驱动的数据科学执行计划生成器。

    通过 ``app/llm/`` 服务层的结构化输出能力调用 GLM-5，
    以 ``response_schema=ToolCallPlan`` 约束输出格式。
    """

    def __init__(
        self, *, agent_name: str = "planner", service: LLMService | None = None
    ) -> None:
        # Allow each caller to bind its own AGENT_<NAME>_* config segment.
        cfg = get_agent_config(agent_name)
        self.agent_name = agent_name
        self.model_name = cfg.model
        self.temperature = cfg.temperature
        self.max_tokens = cfg.max_tokens
        self._service = service or get_service()

    async def plan_data_cleaning(
        self, profile_json: dict, target_spec_summary: dict | None = None,
    ) -> ToolCallPlan:
        """基于数据集画像，为清洗阶段生成工具调用计划。"""
        profile_str = json.dumps(profile_json, ensure_ascii=False, indent=2)
        target_spec_str = json.dumps(
            target_spec_summary or {}, ensure_ascii=False, indent=2, default=str
        )
        human_prompt = build_data_cleaning_prompt(profile_str, target_spec_str)
        messages = self._build_messages(human_prompt)
        return await self._invoke_structured_llm(
            messages=messages,
            planner_type="data_cleaning",
            profile_json=profile_json,
        )

    async def plan_feature_engineering(
        self, profile_json: dict, target_spec_summary: dict | None = None,
    ) -> ToolCallPlan:
        """基于数据集画像，为特征工程阶段生成工具调用计划。"""
        profile_str = json.dumps(profile_json, ensure_ascii=False, indent=2)
        target_spec_str = json.dumps(
            target_spec_summary or {}, ensure_ascii=False, indent=2, default=str
        )
        human_prompt = build_feature_engineering_prompt(profile_str, target_spec_str)
        messages = self._build_messages(human_prompt)
        return await self._invoke_structured_llm(
            messages=messages,
            planner_type="feature_engineering",
            profile_json=profile_json,
        )

    async def plan_model_building(
        self,
        profile_json: dict,
        target_spec_summary: dict,
    ) -> ToolCallPlan:
        """基于特征工程后画像 + target_spec 摘要，为建模阶段生成配置工具调用计划。

        返回的 ``ToolCallPlan`` 步骤调用 ``configure_*`` 配置工具；实际建模逻辑
        由 ``app/agents/model_building.py`` 根据计划组装参数后执行。
        LLM 失败时由调用方捕获并走 task_family 感知的默认 plan。
        """
        profile_str = json.dumps(profile_json, ensure_ascii=False, indent=2)
        target_spec_str = json.dumps(
            target_spec_summary, ensure_ascii=False, indent=2, default=str
        )
        human_prompt = build_model_building_prompt(profile_str, target_spec_str)
        messages = self._build_messages(human_prompt)
        return await self._invoke_structured_llm(
            messages=messages,
            planner_type="model_building",
            profile_json={
                **profile_json,
                "_target_spec_summary": target_spec_summary,
            },
        )

    @staticmethod
    def _build_messages(human_prompt: str) -> list[ChatMessage]:
        """构建统一格式的 ChatMessage 列表。"""
        return [
            ChatMessage(role="system", content=SYSTEM_PLANNER_PROMPT),
            ChatMessage(role="user", content=human_prompt),
        ]

    async def _invoke_structured_llm(
        self,
        *,
        messages: list[ChatMessage],
        planner_type: str,
        profile_json: dict[str, Any],
    ) -> ToolCallPlan:
        metadata = planner_trace_metadata(
            planner_type=planner_type,
            profile_json=profile_json,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            target_column=str(profile_json.get("target_variable") or ""),
        )
        tags = planner_trace_tags(planner_type=planner_type)

        trace_extra = get_langsmith_tracing_extra(
            run_name=f"planner:{planner_type}",
            tags=tags,
            metadata=metadata,
        )
        result = await self._service.chat(
            model=self.model_name,
            messages=messages,
            module_name="planner",
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=False,
            response_schema=ToolCallPlan,
            trace_extra=trace_extra,
        )

        # stream=False guarantees a ChatResponse; narrow the type.
        assert isinstance(result, ChatResponse)
        response = result

        parsed = response.parsed
        if parsed is None:
            raise RuntimeError(
                f"LLM did not return a structured ToolCallPlan. "
                f"Raw content: {response.content[:500]}"
            )
        if not isinstance(parsed, ToolCallPlan):
            raise RuntimeError(
                f"LLM returned unexpected type {type(parsed).__name__}, "
                f"expected ToolCallPlan."
            )
        return parsed
