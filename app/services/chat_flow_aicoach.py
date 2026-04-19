from app.schemas_aicoach import ChatRequest_aicoach, ChatResponse_aicoach, ChatState
from app.services.ai_service import detect_intent, reply_greeting, reply_general
from app.services.learning_service_aicoach import (
   generate_opening_ai_coach_question,
   evaluate_user_answer,
   generate_retry_same_step_question,
   generate_probe_same_step_question,
   generate_next_step_question,
)
from app.constants.coach_questions import FIXED_QUESTIONS


async def process_chat_aicoach(req: ChatRequest_aicoach, state: ChatState) -> ChatResponse_aicoach:
    user_message = req.user_message.strip()

    if not user_message:
        return ChatResponse_aicoach(
            reply="กรุณาพิมพ์สิ่งที่ต้องการพัฒนา / ปัญหาที่อยากแก้",
            state=state,
            source="empty_message",
        )

    if state.step == 0:
        # 1) ดึง fixed question
        fixed_q = state.fixed_question

        # 2) ให้ AI rewrite ให้เป็นโค้ช
        q1 = await generate_opening_ai_coach_question(
            fixed_question=fixed_q
        )

        # 3) update state
        new_state = ChatState(
            step=1,
            fixed_question=fixed_q,
            last_question=q1,
            answers_by_step=state.answers_by_step,
            history=state.history,
        )

        # เก็บ history: AI ถามเปิดข้อแรก
        new_state.history.append({
            "step": 1,
            "event": "ai_question",
            "fixed_question": fixed_q,
            "asked_question": q1,
        })

        reply = q1

    else:
        fixed_q = state.fixed_question
        current_step = state.step

        # ตั้งต้นไว้ก่อนกัน UnboundLocalError
        new_state = state

        # เก็บ history: user answer
        state.history.append({
            "step": current_step,
            "event": "user_answer",
            "fixed_question": fixed_q,
            "asked_question": state.last_question,
            "user_answer": user_message,
        })

        check = await evaluate_user_answer(
            question=fixed_q,
            user_answer=user_message,
        )

        status = check.get("status", "off_topic")
        reason = check.get("reason", "")
        confidence = check.get("confidence", 0.0)

        # เก็บ history: evaluation result
        state.history.append({
            "step": current_step,
            "event": "evaluation",
            "fixed_question": fixed_q,
            "asked_question": state.last_question,
            "user_answer": user_message,
            "status": status,
            "reason": reason,
            "confidence": confidence,
            "raw": check.get("raw", ""),
        })

        # สร้างที่เก็บของข้อปัจจุบันถ้ายังไม่มี
        if current_step not in state.answers_by_step:
            state.answers_by_step[current_step] = {
                "fixed_question": fixed_q,
                "asked_question": state.last_question,
                "user_answer": "",
                "status": "",
                "reason": "",
                "confidence": 0.0,
                "is_completed": False,
            }

        # update ข้อมูลของข้อปัจจุบัน
        state.answers_by_step[current_step]["fixed_question"] = fixed_q
        state.answers_by_step[current_step]["asked_question"] = state.last_question
        state.answers_by_step[current_step]["user_answer"] = user_message
        state.answers_by_step[current_step]["status"] = status
        state.answers_by_step[current_step]["reason"] = reason
        state.answers_by_step[current_step]["confidence"] = confidence

        new_state = state

        if status in {"off_topic", "too_short"}:
            reply = await generate_retry_same_step_question(
                fixed_question=fixed_q,
                user_answer=user_message,
                status=status,
            )
            new_state.last_question = reply

            # เก็บ history: AI retry same step
            state.history.append({
                "step": current_step,
                "event": "ai_retry_question",
                "fixed_question": fixed_q,
                "asked_question": reply,
                "based_on_status": status,
            })

        elif status in {"partial", "reflecting", "clear_but_needs_guidance"}:
            # ถ้าข้อนี้เริ่มมีคำตอบที่ใช้ได้แล้ว ค่อย mark completed
            if status == "clear_but_needs_guidance":
                state.answers_by_step[current_step]["is_completed"] = True

            reply = await generate_probe_same_step_question(
                fixed_question=fixed_q,
                user_answer=user_message,
                status=status,
            )
            new_state.last_question = reply

            # เก็บ history: AI probe same step
            state.history.append({
                "step": current_step,
                "event": "ai_probe_question",
                "fixed_question": fixed_q,
                "asked_question": reply,
                "based_on_status": status,
            })

        elif status == "clear_complete":
            state.answers_by_step[current_step]["is_completed"] = True

            next_step = current_step + 1
            next_fixed_q = FIXED_QUESTIONS.get(next_step)

            if next_fixed_q:
                next_question = await generate_next_step_question(
                    fixed_question=next_fixed_q,
                    previous_answer=user_message,
                )
                new_state.step = next_step
                new_state.fixed_question = next_fixed_q
                new_state.last_question = next_question
                reply = next_question

                # เก็บ history: move to next step
                state.history.append({
                    "step": next_step,
                    "event": "ai_next_question",
                    "previous_step": current_step,
                    "fixed_question": next_fixed_q,
                    "asked_question": next_question,
                    "previous_answer": user_message,
                })
            else:
                all_answers = state.answers_by_step

                reply_lines = [
                    "ขอบคุณมากครับ ตอนนี้เราได้สำรวจประเด็นสำคัญครบแล้ว",
                    "",
                    "สรุปคำตอบของคุณ:"
                ]

                for step_no, item in sorted(all_answers.items()):
                    fixed_question_item = item.get("fixed_question", "")
                    user_answer_item = item.get("user_answer", "")
                    status_item = item.get("status", "")

                    reply_lines.append(
                        f"\nข้อ {step_no}\n"
                        f"คำถาม: {fixed_question_item}\n"
                        f"คำตอบ: {user_answer_item}\n"
                        f"สถานะ: {status_item}"
                    )

                reply = "\n".join(reply_lines)

                # เก็บ history: conversation completed
                state.history.append({
                    "step": current_step,
                    "event": "conversation_completed",
                    "summary_count": len(all_answers),
                    "final_reply": reply,
                })

        else:
            reply = "ขอชวนเล่าเพิ่มเติมอีกนิดนะครับ"

            # เก็บ history: fallback
            state.history.append({
                "step": current_step,
                "event": "fallback_reply",
                "fixed_question": fixed_q,
                "asked_question": state.last_question,
                "user_answer": user_message,
            })

    return ChatResponse_aicoach(
        reply=reply,
        state=new_state,
        source="learning_entry",
    )