from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class ChatState_aicustom(BaseModel):
    #ใครกำลังคุย + ใช้ข้อมูลชุดไหน
    web_no: Optional[int] = None
    member_no: Optional[int] = None
    course_use: List[str] = []

    # ตอนนี้กำลังอยู่ phase ไหน
    mode: str = "idle"

    # คำถามนี้ user ต้องการอะไร
    intent: str = "unknown"

    # กำลังเรียนเรื่องอะไรอยู่
    topic: str = "unknown"
    active_course_no: Optional[int] = None

    # ข้อมูลเก่าจากคำถามที่แล้ว
    last_intent: str = "unknown"
    last_answer_type: Optional[str] = None
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
    active_video: Optional[dict] = None

class ResetRequest_aicustom(BaseModel):
    web_no: Optional[int] = None
    member_no: Optional[int] = None