from typing import Literal, Optional
from pydantic import BaseModel, Field


class ChatState_aiweb(BaseModel):
    mode: Literal["idle", "sale"] = "idle"
    topic: str = "unknown"
    target_group: str = "unknown"
    goal: str = "unknown"
    pain_point: str = "unknown"
    last_question: str = "none"
    last_question_type: str = "none"


class ChatRequest_aiweb(BaseModel):
    chat_id: str = Field(..., min_length=1, max_length=36)
    user_message: str = Field(..., min_length=1)
    state: Optional[ChatState_aiweb] = None


class ChatResponse_aiweb(BaseModel):
    reply: str
    state: Optional[ChatState_aiweb] = None
    source: Optional[str] = None
    chat_id: Optional[str] = None


class ResetRequest_aiweb(BaseModel):
    chat_id: str = Field(..., min_length=1, max_length=36)