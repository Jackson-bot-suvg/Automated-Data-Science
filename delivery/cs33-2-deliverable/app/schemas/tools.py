from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ToolCall(BaseModel):
    """
    A concrete tool invocation object.
    tool: the name of the underlying Python function to call.
    parameters: the parameter dictionary for that function. The dataframe
    itself does not need to be passed in because the node injects it automatically.
    """
    tool: str = Field(..., description="Available function name from the underlying Python tool library, for example: fill_missing, drop_columns, encode_categorical")
    parameters: Dict[str, Any] = Field(..., description="Key-value arguments passed when invoking this tool")
    reasoning: Optional[str] = Field(None, description="Why this step was planned, for example: 'Age has 20% missing values, so use median imputation'")

class ToolCallPlan(BaseModel):
    """
    Complete data processing plan returned by the LLM planner.
    The order of the array determines the execution order of tools.
    """
    plan_name: str = Field(..., description="Short name of this processing plan")
    steps: List[ToolCall] = Field(..., description="Ordered array of tool invocation steps")
