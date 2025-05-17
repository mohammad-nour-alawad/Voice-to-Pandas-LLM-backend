# schemas.py

from typing import List, Dict, Any, Optional, TypedDict
from pydantic import BaseModel, Field

# TypedDict for internal agent state
class AgentState(TypedDict):
    user_input: str
    conversation_history: List[dict]
    metadata: Dict[str, Any]
    generated_code: Optional[str]
    response_message: Optional[str]
    response_audio: Optional[str]
    decision: Optional[Dict[str, Any]]

    timing_info: Dict[str, float] = Field(default_factory=dict)

    class Config:
        extra = "allow"

# TypedDict for decision result
class Decision(TypedDict):
    action: str

# Pydantic model for conversation requests
class ConversationRequest(BaseModel):
    user_input: str
    metadata: Dict[str, Any]
    conversation_history: List[dict] = []
