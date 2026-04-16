from app.schemas import ChatRequest, ChatResponse, ChatState
from app.services.ai_service import detect_intent, reply_greeting, reply_general
from app.services.learning_service import (
    analyze_learning_progress,
    generate_learning_question,
    answer_when_learning_data_complete,
)


async def process_chat(req: ChatRequest, state: ChatState, conn) -> ChatResponse:
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
        goal = progress.goal
        event = progress.event
        last_question = progress.last_question
        next_action = progress.next_action

        new_state = ChatState(
            mode="learning",
            topic=topic,
            goal=goal,
            event=event,
            next_action = next_action,
            last_question=last_question,
            last_question_type="none",
        )

        if next_action == "ask_topic":
            # reply = "ask_topic"
            reply = await generate_learning_question(user_message, new_state, 1)
            new_state.last_question = reply
            new_state.last_question_type = "ask_topic"

        elif next_action == "ask_goal":
            # reply = "ask_goal"
            reply = await generate_learning_question(user_message, new_state, 2)
            new_state.last_question = reply
            new_state.last_question_type = "ask_goal"

        elif next_action == "ask_event":
            # reply = "ask_event"
            reply = await generate_learning_question(user_message, new_state, 3)
            new_state.last_question = reply
            new_state.last_question_type = "ask_event"

        elif next_action == "ready":
            # reply = "ready"
            ans = answer_when_learning_data_complete(
                conn=conn,
                user_message=user_message,
                topic=topic,
                goal=goal,
                event=event,
            )
            reply = ans["reply"]
            new_state.last_question = "none"
            new_state.last_question_type = "answered"

        else:
            reply = "ช่วยเล่าเพิ่มเติมได้นิดนึงไหมครับ ผมอยากเข้าใจคุณให้ชัดขึ้น 😊"

        return ChatResponse(
            reply=reply,
            state=new_state,
            source="learning_mode",
        )

    # ===== CASE 2: ยังไม่อยู่ใน learning mode =====
    output = await detect_intent(user_message)

    #pass
    if output.intent == "greeting":
        reply = await reply_greeting(user_message)
        return ChatResponse(
            reply=reply,
            state=state,
            source="greeting",
        )
    
    #pass
    if output.intent == "general":
        reply = await reply_general(user_message)
        return ChatResponse(
            reply=reply,
            state=state,
            source="general",
        )

    # ===== CASE 3: intent = learning =====
    progress = await analyze_learning_progress(user_message, state)

    topic = progress.topic
    goal = progress.goal
    event = progress.event
    last_question = progress.last_question
    next_action = progress.next_action

    new_state = ChatState(
        mode="learning",
        topic=topic,
        goal=goal,
        event=event,
        last_question=last_question,
        next_action = next_action,
        last_question_type="none",
    )

    if next_action == "ask_topic":
        # reply = "ask_topic"
        reply = await generate_learning_question(user_message, new_state, 1)
        new_state.last_question = reply
        new_state.last_question_type = "ask_topic"

    elif next_action == "ask_goal":
        # reply = "ask_goal"
        reply = await generate_learning_question(user_message, new_state, 2)
        new_state.last_question = reply
        new_state.last_question_type = "ask_goal"

    elif next_action == "ask_event":
        # reply = "ask_event"
        reply = await generate_learning_question(user_message, new_state, 3)
        new_state.last_question = reply
        new_state.last_question_type = "ask_event"

    elif next_action == "ready":
        # reply = "ready"
        ans = answer_when_learning_data_complete(
            conn=conn,
            user_message=user_message,
            topic=topic,
            goal=goal,
            event=event,
        )
        reply = ans["reply"]
        new_state.last_question = "none"
        new_state.last_question_type = "answered"

    else:
        reply = "ช่วยเล่าเพิ่มเติมได้นิดนึงไหมครับ ผมอยากเข้าใจคุณให้ชัดขึ้น 😊"
        new_state.last_question = reply
        new_state.last_question_type = "ask_topic"

    return ChatResponse(
        reply=reply,
        state=new_state,
        source="learning_entry",
    )