from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field
class AgentAssignment(BaseModel):
    agent_id: str
    agent_type: Optional[str] = None
    execution_order: Optional[Dict[str, Any]] = None

class OrchestrationAssignRequest(BaseModel):
    feature_names: List[str] = Field(..., description="제어 대상이 되는 Manipulated Variable (조작 변수) 이름")
    target_col: str = Field(..., description="제어 타겟이 되는 Controlled Variable (제어 변수) 이름")
    control_setpoint: float = Field(..., description="제어 목표 값(setpoint) 정의")
    control_horizon_minutes: int = Field(..., description="제어 구간(horizon) 정의")
    constraints: Optional[Any] = Field(None, description="제어 제약 조건")

class UpdatedAssignment(BaseModel):
    agent_id: str
    status: str 

class OrchestrationAssignResponse(BaseModel):
    task_id: str
    updated_assignments: List[UpdatedAssignment]
    response: Optional[Dict[str, Any]] = None