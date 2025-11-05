from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field
class AgentAssignment(BaseModel):
    agent_id: str
    agent_type: Optional[str] = None
    execution_order: Optional[Dict[str, Any]] = None

class OrchestrationAssignRequest(BaseModel):
    taskId: str = Field(..., description="task id")
    query: str = Field(..., description="사용자 요청사항")
    feature_names: Optional[List[str]] = Field(None, description="제어 대상이 되는 Manipulated Variable (조작 변수) 이름")
    target_col: Optional[str] = Field(None, description="제어 타겟이 되는 Controlled Variable (제어 변수) 이름")
    control_setpoint: Optional[float] = Field(None, description="제어 목표 값(setpoint) 정의")
    control_horizon_minutes: Optional[int] = Field(None, description="제어 구간(horizon) 정의")
    constraints: Optional[Any] = Field(None, description="제어 제약 조건")
    optimization_objective: Optional[str] = Field(None, description="목적함수")
    safety_mode: Optional[bool] = Field(True, description="안전 모드 활성화")
    simulation_before_apply: Optional[bool] = Field(True, description="적용 전 시뮬레이션 수행")
    timeseries_info: Optional[Any] = Field(None, description="제어 요청 시간")

class UpdatedAssignment(BaseModel):
    agent_id: str
    status: str 

class OrchestrationAssignResponse(BaseModel):
    task_id: str
    updated_assignments: List[UpdatedAssignment]
    response: Optional[Dict[str, Any]] = None