from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    thread_id: Optional[str] = None
    model: Optional[str] = Field(
        None,
        description="Model to use (e.g., 'gpt-4o', 'claude-3-5-sonnet-20241022', 'gemini-2.5-flash'). If not specified, uses DEFAULT_MODEL from environment or 'gemini-2.5-flash'"
    )
    temperature: Optional[float] = Field(
        0,
        ge=0,
        le=1,
        description="Temperature for model sampling (0-1). Lower is more deterministic."
    )

class ChatResponse(BaseModel):
    response: str
    thread_id: str
    model_used: Optional[str] = Field(None, description="The model that was actually used for this response")

class Agent(BaseModel):
    id: str
    name: str
    description: str
    agent_type: str

class AgentData(BaseModel):
    userAgentId: str
    agent: Agent
    summary: Dict[str, Any]

class AgentListResponse(BaseModel):
    agents: List[AgentData]
