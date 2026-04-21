from pydantic import BaseModel
from typing import Optional, List

class ChatState_aicustom(BaseModel):
    web_no: Optional[int] = None
    member_no: Optional[int] = None
    course_use: List[str] = []
    mode: str = "idle"
    intent: str = "unknown"
    topic: str = "unknown"

    last_user_message: Optional[str] = None
    last_answer: Optional[str] = None

class ChatRequest_aicustom(BaseModel):
    user_message: str
    web_no: Optional[int] = None
    member_no: Optional[int] = None
    course_use: List[str] = []
    state: Optional[ChatState_aicustom] = None

class ChatResponse_aicustom(BaseModel):
    reply: str
    state: Optional[ChatState_aicustom] = None
    source: Optional[str] = None

class ResetRequest_aicustom(BaseModel):
    web_no: Optional[int] = None
    member_no: Optional[int] = None