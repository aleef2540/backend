from typing import Literal, Optional
from pydantic import BaseModel, Field


class ChatState(BaseModel):
    mode: Literal["idle", "feedback"] = "idle"
    applied_action: str = "unknown"   # ทำอะไรไปบ้าง
    result: str = "unknown"           # ผลลัพธ์เป็นยังไง
    strength: str = "unknown"         # สิ่งที่ทำได้ดี
    improvement: str = "unknown"      # อยากพัฒนาอะไรต่อ
    last_question: str = "none"
    next_action: str = "none"
    last_question_type: str = "none"


class ChatRequest_model2(BaseModel):
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


class ChatResponse_model2(BaseModel):
    reply: str
    state: Optional[ChatState] = None
    source: Optional[str] = None

class IntentResult(BaseModel):
    intent: str

class ResetRequest_model2(BaseModel):
    web_no: int
    member_no: int

class FeedbackProgress(BaseModel):
    applied_action: str = "unknown"
    result: str = "unknown"
    strength: str = "unknown"
    improvement: str = "unknown"
    last_question: str = "none"
    next_action: str = "ask_applied_action"
    raw: str = ""