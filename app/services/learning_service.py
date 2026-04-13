from app.schemas import ChatState, LearningProgressOutput


async def analyze_learning_progress(
    user_message: str,
    state: ChatState,
) -> LearningProgressOutput:
    """
    TODO:
    ย้าย logic จริงจาก analyze_learning_progress() มาใส่ตรงนี้
    """
    topic = state.topic
    learning_need = state.learning_need

    text = user_message.strip()

    if topic == "unknown" and learning_need == "unknown":
        return LearningProgressOutput(
            topic="unknown",
            learning_need="unknown",
            last_question="ask_topic",
            next_action="ask_topic",
        )

    if topic != "unknown" and learning_need == "unknown":
        return LearningProgressOutput(
            topic=topic,
            learning_need="unknown",
            last_question="ask_learning_need",
            next_action="ask_learning_need",
        )

    if topic != "unknown" and learning_need != "unknown":
        return LearningProgressOutput(
            topic=topic,
            learning_need=learning_need,
            last_question="answering",
            next_action="ready",
        )

    return LearningProgressOutput(
        topic=topic,
        learning_need=learning_need,
        last_question="none",
        next_action="fallback",
    )


async def generate_learning_question(
    user_message: str,
    state: ChatState,
    question_type: int,
) -> str:
    """
    question_type:
    1 = ask topic
    2 = ask learning need
    3 = ask topic when learning_need exists
    """
    if question_type == 1:
        return "ตอนนี้คุณอยากพัฒนาด้านไหนเป็นพิเศษครับ เช่น การสื่อสาร การทำงานเป็นทีม หรือภาวะผู้นำ"
    if question_type == 2:
        topic = state.topic if state.topic != "unknown" else "เรื่องนี้"
        return f"คุณอยากพัฒนาเรื่อง '{topic}' ในมุมไหนเป็นพิเศษครับ เช่น อยากรู้วิธีทำ อยากแก้ปัญหา หรืออยากเข้าใจหลักการ"
    if question_type == 3:
        return "เข้าใจครับ แล้วหัวข้อหลักที่คุณอยากพัฒนาคือเรื่องอะไรเป็นพิเศษครับ"
    return "ช่วยเล่าเพิ่มเติมได้นิดนึงไหมครับ"


async def answer_when_learning_data_complete(
    user_message: str,
    topic: str,
    learning_need: str,
) -> dict:
    """
    TODO:
    ย้าย logic จริงจาก answer_when_learning_data_complete()
    ปัจจุบัน return ให้เหมือน PHP ที่คืน dict มี key 'reply'
    """
    return {
        "reply": f"เข้าใจครับ ตอนนี้หัวข้อคือ '{topic}' และสิ่งที่ต้องการคือ '{learning_need}' เดี๋ยวผมช่วยตอบต่อในมุมนี้ให้ครับ"
    }