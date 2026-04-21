from app.services.course_service_aicustom import get_course_data_by_nos
from app.services.ai_service_aicustom import (
    detect_intent,
    reply_greeting,
    reply_general,
    reply_learning,
    reply_with_topic,
)
from app.schemas_aicustom import ChatState_aicustom


def build_course_name_context(course_data) -> str:
    names = []
    seen = set()

    for row in course_data:
        course_name = str(row[1] or "").strip()

        if not course_name:
            continue

        if course_name in seen:
            continue

        seen.add(course_name)
        names.append(course_name)

    return ", ".join(names)


def find_script_by_topic(course_data, topic: str) -> str:
    topic_clean = str(topic or "").strip().lower()

    if not topic_clean or topic_clean == "unknown":
        return ""

    for row in course_data:
        course_name = str(row[1] or "").strip()
        script = str(row[2] or "").strip()

        if course_name.lower() == topic_clean:
            return script

    for row in course_data:
        course_name = str(row[1] or "").strip().lower()
        script = str(row[2] or "").strip()

        if topic_clean in course_name or course_name in topic_clean:
            return script

    return ""


async def process_chat_aicustom(req, state, conn):
    if state is None:
        state = ChatState_aicustom()

    user_message = (req.user_message or "").strip()

    state.web_no = str(req.web_no) if req.web_no is not None else None
    state.member_no = str(req.member_no) if req.member_no is not None else None

    if req.course_use:
        state.course_use = [str(x).strip() for x in req.course_use if str(x).strip()]

    course_use = state.course_use or []

    if not course_use:
        reply = "ขออภัยครับ ยังไม่พบรายการหลักสูตรที่อนุญาตให้ใช้งาน"
        state.last_user_message = user_message
        state.last_answer = reply

        return type("Obj", (), {
            "reply": reply,
            "status": "no_course",
            "reason": "empty_course_use",
            "state": state,
            "source": "ai_custom_no_course",
        })()

    course_data = get_course_data_by_nos(conn, course_use)
    course_context = build_course_name_context(course_data)

    state.last_user_message = user_message

    # ==================================================
    # 1) ถ้ามี topic ค้างอยู่แล้ว ใช้ topic เดิมก่อนเลย
    # ==================================================
    current_topic = str(getattr(state, "topic", "") or "").strip()

    if current_topic and current_topic != "unknown":

        # 🔥 เช็คก่อนว่าผู้ใช้เปลี่ยน topic ไหม
        intent_data = await detect_intent(user_message, course_context)
        new_topic = intent_data.get("topic", "unknown")

        # ถ้ามี topic ใหม่ และไม่เหมือนเดิม -> เปลี่ยน topic
        if new_topic and new_topic != "unknown" and new_topic != current_topic:
            state.topic = new_topic
            current_topic = new_topic

        script = find_script_by_topic(course_data, current_topic)

        if script:
            reply = await reply_with_topic(user_message, current_topic, script)
        else:
            reply = await reply_learning(user_message, course_context)

        state.intent = "learning"
        state.mode = "learning"
        state.last_answer = reply

        return type("Obj", (), {
            "reply": reply,
            "status": "learning",
            "reason": "use_existing_topic",
            "state": state,
            "source": "ai_custom_topic_continue",
        })()

    # ==================================================
    # 2) ถ้าอยู่ใน learning อยู่แล้ว แต่ยังไม่มี topic
    #    ก็ไม่ต้อง detect ใหม่
    # ==================================================
    if getattr(state, "intent", "") == "learning" or getattr(state, "mode", "") == "learning":
        intent_data = await detect_intent(user_message, course_context)
        topic = intent_data.get("topic", "unknown")

        state.intent = "learning"
        state.topic = topic

        if topic and topic != "unknown":
            script = find_script_by_topic(course_data, topic)

            if script:
                reply = await reply_with_topic(user_message, topic, script)
            else:
                reply = await reply_learning(user_message, course_context)
        else:
            reply = await reply_learning(user_message, course_context)

        state.mode = "learning"
        state.last_answer = reply

        return type("Obj", (), {
            "reply": reply,
            "status": "learning",
            "reason": "continue_learning_mode_detect_topic",
            "state": state,
            "source": "ai_custom_learning_continue",
        })()

    # ==================================================
    # 3) ค่อย detect intent เฉพาะตอนยังไม่มี context เดิม
    # ==================================================
    intent_data = await detect_intent(user_message, course_context)
    intent = intent_data["intent"]
    topic = intent_data["topic"]

    state.intent = intent
    state.topic = topic

    if intent == "greeting":
        reply = await reply_greeting(user_message, course_context)
        state.mode = "idle"
        state.last_answer = reply

        return type("Obj", (), {
            "reply": reply,
            "status": "greeting",
            "reason": "intent_greeting",
            "state": state,
            "source": "ai_custom_greeting",
        })()

    # if intent == "general":
    #     reply = await reply_general(user_message, course_context)
    #     state.mode = "idle"
    #     state.last_answer = reply

    #     return type("Obj", (), {
    #         "reply": reply,
    #         "status": "general",
    #         "reason": "intent_general",
    #         "state": state,
    #         "source": "ai_custom_general",
    #     })()

    # ==================================================
    # 4) learning
    # ==================================================
    if topic and topic != "unknown":
        script = find_script_by_topic(course_data, topic)

        if script:
            reply = await reply_with_topic(user_message, topic, script)
        else:
            reply = await reply_learning(user_message, course_context)
    else:
        reply = await reply_learning(user_message, course_context)

    state.mode = "learning"
    state.last_answer = reply

    return type("Obj", (), {
        "reply": reply,
        "status": "learning",
        "reason": "intent_learning",
        "state": state,
        "source": "ai_custom_learning",
    })()