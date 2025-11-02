from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field
class AgentAssignment(BaseModel):
    agent_id: str
    agent_type: Optional[str] = None
    execution_order: Optional[Dict[str, Any]] = None

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