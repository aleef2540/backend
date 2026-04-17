from app.schemas_model2 import ChatRequest_model2, ChatResponse_model2, ChatState
from app.services.learning_service_model2 import (
    analyze_feedback_progress,
    generate_feedback_question,
    answer_when_feedback_data_complete,
    reply_greeting,
    reply_general,
    detect_intent,
)


async def process_chat_model2(req: ChatRequest_model2, state: ChatState, conn) -> ChatResponse_model2:
    user_message = req.user_message.strip()

    if not user_message:
        return ChatResponse_model2(
            reply="กรุณาพิมพ์สิ่งที่ต้องการพัฒนา / ปัญหาที่อยากแก้",
            state=state,
            source="empty_message",
        )

    # ===== CASE 1: อยู่ใน feedback mode แล้ว =====
    if state.mode == "feedback":
        progress = await analyze_feedback_progress(user_message, state)

        applied_action = progress.applied_action
        result = progress.result
        strength = progress.strength
        improvement = progress.improvement
        last_question = progress.last_question
        next_action = progress.next_action

        new_state = ChatState(
            mode="feedback",
            applied_action=applied_action,
            result=result,
            strength=strength,
            improvement=improvement,
            last_question=last_question,
            next_action=next_action,
            last_question_type="none",
        )

        if next_action == "ask_applied_action":
            # reply = "ask_applied_action"
            reply = await generate_feedback_question(user_message, new_state, 1)
            new_state.last_question = reply
            new_state.last_question_type = "ask_applied_action"

        elif next_action == "ask_result":
            # reply = "ask_result"
            reply = await generate_feedback_question(user_message, new_state, 2)
            new_state.last_question = reply
            new_state.last_question_type = "ask_result"

        elif next_action == "ask_strength":
            # reply = "ask_strength"
            reply = await generate_feedback_question(user_message, new_state, 3)
            new_state.last_question = reply
            new_state.last_question_type = "ask_strength"
        
        elif next_action == "ask_improvement":
            # reply = "ask_improvement"
            reply = await generate_feedback_question(user_message, new_state, 4)
            new_state.last_question = reply
            new_state.last_question_type = "ask_improvement"

        elif next_action == "ready":
            # reply = "ready"
            ans = answer_when_feedback_data_complete(
                conn=conn,
                user_message=user_message,
                applied_action=applied_action,
                result=result,
                strength=strength,
                improvement=improvement,
            )

            reply = ans["reply"]
            new_state.last_question = "none"
            new_state.last_question_type = "answered"
            new_state.next_action = "ready"

        else:
            reply = "ช่วยเล่าเพิ่มเติมได้นิดนึงไหมครับ ผมอยากเข้าใจคุณให้ชัดขึ้น 😊"

        return ChatResponse_model2(
            reply=reply,
            state=new_state,
            source="learning_mode",
        )

    # ===== CASE 2: ยังไม่อยู่ใน learning mode =====
    output = await detect_intent(user_message)

    #pass
    if output.intent == "greeting":
        reply = await reply_greeting(user_message)
        return ChatResponse_model2(
            reply=reply,
            state=state,
            source="greeting",
        )
    
    #pass
    if output.intent == "general":
        reply = await reply_general(user_message)
        return ChatResponse_model2(
            reply=reply,
            state=state,
            source="general",
        )

    # ===== CASE 3: intent = learning =====
    progress = await analyze_feedback_progress(user_message, state)

    applied_action = progress.applied_action
    result = progress.result
    strength = progress.strength
    improvement = progress.improvement
    last_question = progress.last_question
    next_action = progress.next_action

    new_state = ChatState(
        mode="feedback",
        applied_action=applied_action,
        result=result,
        strength=strength,
        improvement=improvement,
        last_question=last_question,
        next_action=next_action,
        last_question_type="none",
    )

    if next_action == "ask_applied_action":
        # reply = "ask_applied_action"
        reply = await generate_feedback_question(user_message, new_state, 1)
        new_state.last_question = reply
        new_state.last_question_type = "ask_applied_action"

    elif next_action == "ask_result":
        # reply = "ask_result"
        reply = await generate_feedback_question(user_message, new_state, 2)
        new_state.last_question = reply
        new_state.last_question_type = "ask_result"

    elif next_action == "ask_strength":
        # reply = "ask_strength"
        reply = await generate_feedback_question(user_message, new_state, 3)
        new_state.last_question = reply
        new_state.last_question_type = "ask_strength"
        
    elif next_action == "ask_improvement":
        # reply = "ask_improvement"
        reply = await generate_feedback_question(user_message, new_state, 4)
        new_state.last_question = reply
        new_state.last_question_type = "ask_improvement"

    elif next_action == "ready":
        # reply = "ready"
        ans = answer_when_feedback_data_complete(
            conn=conn,
            user_message=user_message,
            applied_action=applied_action,
            result=result,
            strength=strength,
            improvement=improvement,
        )

        reply = ans["reply"]
        new_state.last_question = "none"
        new_state.last_question_type = "answered"
        new_state.next_action = "ready"

    else:
        reply = "ช่วยเล่าเพิ่มเติมได้นิดนึงไหมครับ ผมอยากเข้าใจคุณให้ชัดขึ้น 😊"

    return ChatResponse_model2(
        reply=reply,
        state=new_state,
        source="learning_entry",
    )

