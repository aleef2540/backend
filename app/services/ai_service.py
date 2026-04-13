from app.schemas import ChatIntentOutput


async def detect_intent(
    user_message: str,
) -> ChatIntentOutput:
    """
    TODO:
    ย้าย logic จริงจาก detect_intent() ใน PHP มาใส่ตรงนี้
    ตอนนี้ทำเป็นตัวอย่างง่าย ๆ ไปก่อน
    """
    text = user_message.strip().lower()

    greetings = ["สวัสดี", "hello", "hi", "หวัดดี"]
    if any(g in text for g in greetings):
        return ChatIntentOutput(intent="greeting")

    learning_keywords = [
        "พัฒนา", "เรียน", "อยากเก่ง", "skill", "ทักษะ",
        "communication", "leadership", "teamwork", "coaching"
    ]
    if any(k in text for k in learning_keywords):
        return ChatIntentOutput(intent="learning", topic="unknown", learning_need="unknown")

    return ChatIntentOutput(intent="general", topic="unknown", learning_need="unknown")


async def reply_greeting(user_message: str) -> str:
    """
    TODO:
    ย้าย logic จริงจาก reply_greeting()
    """
    return "สวัสดีครับ 😊 วันนี้อยากพัฒนาเรื่องไหนเป็นพิเศษครับ"


async def reply_general(user_message: str) -> str:
    """
    TODO:
    ย้าย logic จริงจาก reply_general()
    """
    return "เข้าใจครับ ช่วยเล่าเพิ่มเติมได้ไหมครับว่าต้องการคำแนะนำในเรื่องไหน"