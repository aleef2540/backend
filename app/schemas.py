from typing import Literal, Optional
from pydantic import BaseModel, Field


class ChatState(BaseModel):
    mode: Literal["idle", "learning"] = "idle"
    topic: str = "unknown"
    learning_need: str = "unknown"
    last_question: str = "none"
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
    state: ChatState
    intent: Optional[str] = None
    topic: Optional[str] = None
    learning_need: Optional[str] = None
    source: Optional[str] = None