from typing import Literal, Optional
from pydantic import BaseModel, Field


class ChatState(BaseModel):
    mode: Literal["idle", "learning"] = "idle"
    topic: str = "unknown"
    goal: str = "unknown"
    event: str = "unknown"
    last_question: str = "none"
    next_action: str = "none"
    last_question_type: str = "none"


class ChatRequest(BaseModel):
    user_message: str = Field(..., min_length=1)
    web_no: Optional[int] = None
    member_no: Optional[int] = None
    state: Optional[ChatState] = None


class ChatIntentOutput(BaseModel):
    intent: Literal["greeting", "general", "learning"]
    topic: str = "unknown"
    learning_need: str = "unknown"


class LearningProgressOutput(BaseModel):
    topic: str = "unknown"
    learning_need: str = "unknown"
    last_question: str = "none"
    next_action: Literal["ask_topic", "ask_learning_need", "ready", "fallback"] = "fallback"


class ChatResponse(BaseModel):
    reply: str
    state: Optional[ChatState] = None
    source: Optional[str] = None

class IntentResult(BaseModel):
    intent: str

class LearningProgress(BaseModel):
    topic: str
    goal: str
    event: str
    last_question: str
    next_action: str
    raw: str

class ResetRequest(BaseModel):
    web_no: int
    member_no: int