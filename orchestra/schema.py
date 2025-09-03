from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field
class AgentExecutionOrder(BaseModel):
    nl_query: str
    surrogate_model: str = "linear"
    surrogate_params: Optional[str] = None
    opt_method: str = "coordinate"
    maxiter: Optional[int] = None
    agent_base_url: Optional[str] = None
    agent_task_id: Optional[str] = None
    agent_inline: bool = True
    weight_local_path: Optional[str] = None
class AgentAssignment(BaseModel):
    agent_id: str
    agent_type: Optional[str] = None
    execution_order: AgentExecutionOrder 

class OrchestrationAssignRequest(BaseModel):
    task_id: str
    agent_assignments: List[AgentAssignment]

class UpdatedAssignment(BaseModel):
    agent_id: str
    status: str 

class OrchestrationAssignResponse(BaseModel):
    task_id: str
    updated_assignments: List[UpdatedAssignment]
    response: Optional[Dict[str, Any]] = None