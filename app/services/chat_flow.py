from app.schemas import ChatRequest, ChatResponse, ChatState
from app.services.ai_service import detect_intent, reply_greeting, reply_general
from app.services.learning_service import (
    analyze_learning_progress,
    generate_learning_question,
    answer_when_learning_data_complete,
)


async def process_chat(req: ChatRequest, state: ChatState) -> ChatResponse:
    user_message = req.user_message.strip()

    if not user_message:
        return ChatResponse(
            reply="กรุณาพิมพ์สิ่งที่ต้องการพัฒนา / ปัญหาที่อยากแก้",
            state=state,
            source="empty_message",
        )

    # ===== CASE 1: อยู่ใน learning mode แล้ว =====
    if state.mode == "learning":
        progress = await analyze_learning_progress(user_message, state)

        topic = progress.topic
        learning_need = progress.learning_need
        last_question = progress.last_question
        next_action = progress.next_action

        new_state = ChatState(
            mode="learning",
            topic=topic,
            learning_need=learning_need,
            last_question=last_question,
            last_question_type="none",
        )

        if next_action == "ask_topic":
            reply = await generate_learning_question(user_message, new_state, 1)
            new_state.last_question = reply
            new_state.last_question_type = "ask_topic"

        elif next_action == "ask_learning_need":
            reply = await generate_learning_question(user_message, new_state, 2)
            new_state.last_question = reply
            new_state.last_question_type = "ask_learning_need"

        elif next_action == "ready":
            ans = await answer_when_learning_data_complete(
                user_message=user_message,
                topic=topic,
                learning_need=learning_need,
            )
            reply = ans["reply"]
            new_state.last_question = reply
            new_state.last_question_type = "answered"

        else:
            reply = "ช่วยเล่าเพิ่มเติมได้นิดนึงไหมครับ ผมอยากเข้าใจคุณให้ชัดขึ้น 😊"

        return ChatResponse(
            reply=reply,
            state=new_state,
            intent="learning",
            topic=topic,
            learning_need=learning_need,
            source="learning_mode",
        )

    # ===== CASE 2: ยังไม่อยู่ใน learning mode =====
    output = await detect_intent(user_message)

    if output.intent == "greeting":
        reply = await reply_greeting(user_message)
        return ChatResponse(
            reply=reply,
            state=state,
            intent="greeting",
            source="greeting",
        )

    if output.intent == "general":
        reply = await reply_general(user_message)
        return ChatResponse(
            reply=reply,
            state=state,
            intent="general",
            source="general",
        )

    # ===== CASE 3: intent = learning =====
    new_state = ChatState(
        mode="learning",
        topic=output.topic,
        learning_need=output.learning_need,
        last_question="none",
        last_question_type="none",
    )

    if output.topic == "unknown" and output.learning_need == "unknown":
        reply = await generate_learning_question(user_message, new_state, 1)
        new_state.topic = "unknown"
        new_state.learning_need = "unknown"
        new_state.last_question = reply
        new_state.last_question_type = "ask_topic"

    elif output.topic != "unknown" and output.learning_need == "unknown":
        reply = await generate_learning_question(user_message, new_state, 2)
        new_state.last_question = reply
        new_state.last_question_type = "ask_learning_need"

    elif output.topic == "unknown" and output.learning_need != "unknown":
        reply = await generate_learning_question(user_message, new_state, 3)
        new_state.last_question = reply
        new_state.last_question_type = "ask_topic"

    else:
        topic = output.topic.strip()
        learning_need = output.learning_need.strip()

        ans = await answer_when_learning_data_complete(
            user_message=user_message,
            topic=topic,
            learning_need=learning_need,
        )
        reply = ans["reply"]
        new_state.topic = topic
        new_state.learning_need = learning_need
        new_state.last_question = reply
        new_state.last_question_type = "answered"

    return ChatResponse(
        reply=reply,
        state=new_state,
        intent="learning",
        topic=new_state.topic,
        learning_need=new_state.learning_need,
        source="learning_entry",
    )