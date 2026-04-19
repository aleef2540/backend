from app.schemas_aisale import ChatRequest_aisale, ChatResponse_aisale, ChatState_aisale
from app.services.sale_service import (
    detect_sale_intent,
    reply_sale_greeting,
    reply_sale_general,
    analyze_sale_progress,
    generate_sale_question,
    answer_when_sale_data_complete,
)


async def process_chat_aisale(req: ChatRequest_aisale, state: ChatState_aisale, conn) -> ChatResponse_aisale:
    user_message = req.user_message.strip()

    if not user_message:
        return ChatResponse_aisale(
            reply="กรุณาพิมพ์สิ่งที่ต้องการค้นหาหลักสูตร หรือปัญหาที่อยากพัฒนาก่อนครับ",
            state=state,
            source="empty_message",
        )

    # ===== CASE 1: อยู่ใน sale mode แล้ว =====
    if state.mode == "sale":
        progress = await analyze_sale_progress(user_message, state)

        topic = progress.topic
        target_group = progress.target_group
        goal = progress.goal
        pain_point = progress.pain_point
        last_question = progress.last_question
        next_action = progress.next_action

        new_state = ChatState_aisale(
            mode="sale",
            topic=topic,
            target_group=target_group,
            goal=goal,
            pain_point=pain_point,
            last_question=last_question,
            last_question_type="none",
        )

        if next_action == "ask_topic":
            reply = await generate_sale_question(user_message, new_state, 1)
            new_state.last_question = reply
            new_state.last_question_type = "ask_topic"

        elif next_action == "ask_target_group":
            reply = await generate_sale_question(user_message, new_state, 2)
            new_state.last_question = reply
            new_state.last_question_type = "ask_target_group"

        elif next_action == "ask_goal":
            reply = await generate_sale_question(user_message, new_state, 3)
            new_state.last_question = reply
            new_state.last_question_type = "ask_goal"

        elif next_action == "ask_pain_point":
            reply = await generate_sale_question(user_message, new_state, 4)
            new_state.last_question = reply
            new_state.last_question_type = "ask_pain_point"

        elif next_action == "ready":
            ans = await answer_when_sale_data_complete(
                conn=conn,
                user_message=user_message,
                topic=topic,
                target_group=target_group,
                goal=goal,
                pain_point=pain_point,
            )
            reply = ans["reply"]
            new_state.last_question = reply
            new_state.last_question_type = "answered"

        else:
            reply = "ช่วยเล่าเพิ่มเติมได้นิดนึงไหมครับ ผมจะได้ช่วยแนะนำหลักสูตรได้ตรงขึ้น"

        return ChatResponse_aisale(
            reply=reply,
            state=new_state,
            source="sale_mode",
        )

    # ===== CASE 2: ยังไม่อยู่ใน sale mode =====
    output = await detect_sale_intent(user_message)

    if output.intent == "greeting":
        reply = await reply_sale_greeting(user_message)
        return ChatResponse_aisale(
            reply=reply,
            state=state,
            source="greeting",
        )

    if output.intent == "general":
        reply = await reply_sale_general(user_message)
        return ChatResponse_aisale(
            reply=reply,
            state=state,
            source="general",
        )

    # ===== CASE 3: intent = sale =====
    progress = await analyze_sale_progress(user_message, state)

    topic = progress.topic
    target_group = progress.target_group
    goal = progress.goal
    pain_point = progress.pain_point
    last_question = progress.last_question
    next_action = progress.next_action

    new_state = ChatState_aisale(
        mode="sale",
        topic=topic,
        target_group=target_group,
        goal=goal,
        pain_point=pain_point,
        last_question=last_question,
        last_question_type="none",
    )

    if next_action == "ask_topic":
        reply = await generate_sale_question(user_message, new_state, 1)
        new_state.last_question = reply
        new_state.last_question_type = "ask_topic"

    elif next_action == "ask_target_group":
        reply = await generate_sale_question(user_message, new_state, 2)
        new_state.last_question = reply
        new_state.last_question_type = "ask_target_group"

    elif next_action == "ask_goal":
        reply = await generate_sale_question(user_message, new_state, 3)
        new_state.last_question = reply
        new_state.last_question_type = "ask_goal"

    elif next_action == "ask_pain_point":
        reply = await generate_sale_question(user_message, new_state, 4)
        new_state.last_question = reply
        new_state.last_question_type = "ask_pain_point"

    elif next_action == "ready":
        ans = await answer_when_sale_data_complete(
            conn=conn,
            user_message=user_message,
            topic=topic,
            target_group=target_group,
            goal=goal,
            pain_point=pain_point,
        )
        reply = ans["reply"]
        new_state.last_question = "none"
        new_state.last_question_type = "answered"

    else:
        reply = "ช่วยเล่าเพิ่มเติมได้นิดนึงไหมครับ ผมจะได้ช่วยแนะนำหลักสูตรได้ตรงขึ้น"
        new_state.last_question = reply
        new_state.last_question_type = "ask_topic"

    return ChatResponse_aisale(
        reply=reply,
        state=new_state,
        source="sale_entry",
    )