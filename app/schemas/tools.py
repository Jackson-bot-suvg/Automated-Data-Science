from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ToolCall(BaseModel):
    """
    具体的工具调用对象。
    tool: 需要调用的底层 Python 函数名。
    parameters: 传递给该函数的参数字典（无需传入 df 本身，节点会自动注入 df）。
    """
    tool: str = Field(..., description="底层 Python 函数库中的可用函数名（例如：fill_missing, drop_columns, encode_categorical）")
    parameters: Dict[str, Any] = Field(..., description="调用该工具时传入的参数 k-v 对")
    reasoning: Optional[str] = Field(None, description="规划这一步的逻辑原因，比如：'Age列缺失率20%，使用中位数填补'")

class ToolCallPlan(BaseModel):
    """
    LLM Planner 回传的完整数据处理调用计划。
    按数组排列的顺序决定了工具的执行顺序。
    """
    plan_name: str = Field(..., description="该处理计划的短名称")
    steps: List[ToolCall] = Field(..., description="有序排布的工具调用步骤数组")
