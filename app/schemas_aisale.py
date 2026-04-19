from typing import Literal, Optional
from pydantic import BaseModel, Field


class ChatState_aisale(BaseModel):
    mode: Literal["idle", "sale"] = "idle"
    topic: str = "unknown"
    target_group: str = "unknown"
    goal: str = "unknown"
    pain_point: str = "unknown"

    last_question: str = "none"
    last_question_type: str = "none"


class ChatRequest_aisale(BaseModel):
    user_message: str = Field(..., min_length=1)
    web_no: Optional[int] = None
    member_no: Optional[int] = None
    state: Optional[ChatState_aisale] = None


class ChatIntentOutput_aisale(BaseModel):
    intent: Literal["greeting", "general", "sale"]
    topic: str = "unknown"
    target_group: str = "unknown"
    goal: str = "unknown"
    pain_point: str = "unknown"


class SaleProgressOutput_aisale(BaseModel):
    topic: str = "unknown"
    target_group: str = "unknown"
    goal: str = "unknown"
    pain_point: str = "unknown"
    last_question: str = "none"
    next_action: Literal["ask_topic", "ask_target_group", "ask_goal", "ask_pain_point", "ready", "fallback"] = "fallback"


class ChatResponse_aisale(BaseModel):
    reply: str
    state: Optional[ChatState_aisale] = None
    source: Optional[str] = None


class IntentResult_aisale(BaseModel):
    intent: str


class SaleProgress_aisale(BaseModel):
    topic: str
    target_group: str
    goal: str
    pain_point: str
    last_question: str
    next_action: str
    raw: str


class ResetRequest_aisale(BaseModel):
    web_no: int
    member_no: int