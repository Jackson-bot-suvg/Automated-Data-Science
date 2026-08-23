"""BasePlanner - the central planning hub.

Uses the structured output capabilities of the ``app/llm/`` service layer
to call GLM-5, consume dataset profile JSON, and return a strongly typed
``ToolCallPlan`` execution plan.

LangChain dependencies have been removed. Structured output is implemented
through the ``response_schema`` parameter of ``LLMService.chat()``.
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
    """LLM-driven generator for data science execution plans.

    Calls GLM-5 through the structured output support provided by the
    ``app/llm/`` service layer and constrains the output format with
    ``response_schema=ToolCallPlan``.
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
        """Generate a tool-call plan for the data cleaning phase from the dataset profile."""
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
        """Generate a tool-call plan for the feature engineering phase from the dataset profile."""
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
        """Generate a configuration-style tool-call plan for model building.

        The returned ``ToolCallPlan`` invokes ``configure_*`` helper tools.
        The actual modeling logic is executed later by
        ``app/agents/model_building.py`` after assembling parameters from the plan.
        If the LLM call fails, the caller can fall back to a task-family-aware
        default plan.
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
        """Build a standard list of ``ChatMessage`` objects."""
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
