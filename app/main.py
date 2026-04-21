from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mysql.connector
import sqlite3


def get_db_connection():
    return sqlite3.connect("ai_idp_script.db")


from app.schemas_all import ChatRequest, ChatResponse, ResetRequest
from app.state_store_all import chat_state_store_all
from app.services.chat_flow_all import process_chat

from app.schemas_model1 import ChatRequest_model1, ChatResponse_model1, ResetRequest_model1
from app.state_store_model1 import chat_state_store_model1
from app.services.chat_flow_model1 import process_chat_model1

from app.schemas_model2 import ChatRequest_model2, ChatResponse_model2, ResetRequest_model2
from app.state_store_model2 import chat_state_store_model2
from app.services.chat_flow_model2 import process_chat_model2

from app.schemas_aisale import ChatRequest_aisale, ChatResponse_aisale, ResetRequest_aisale
from app.state_store_aisale import chat_state_store_aisale
from app.services.chat_flow_aisale import process_chat_aisale

from app.schemas_aicoach import ChatResponse_aicoach, ChatRequest_aicoach, ResetRequest_aicaoch, ChatState
from app.state_store_aicoach import chat_state_store_aicoach
from app.services.chat_flow_aicoach import process_chat_aicoach
from app.constants.coach_questions import FIXED_QUESTIONS


from app.utils.debug_state import print_state, print_debug

from fastapi import Request
from app.schemas_aiweb import ChatRequest_aiweb, ChatResponse_aiweb, ResetRequest_aiweb
from app.services.chat_flow_aiweb import process_chat_aiweb
from app.services.chat_db_aiweb import (
    get_mysql_connection,
    ensure_chat_session,
    load_chat_state,
    save_chat_state,
    insert_chat_message,
    insert_request_log,
    reset_chat_state,
)

from app.schemas_aiselflearning import ChatRequest_aiselflearning, ChatResponse_aiselflearning
from app.state_store_aiselflearning import chat_state_store_aiselflearning
from app.services.chat_flow_aiselflearning import process_chat_aiselflearning
from app.services.chat_history_aiselflearning import insert_chat_history_aiselflearning

from app.schemas_aicustom import (
    ChatRequest_aicustom,
    ChatResponse_aicustom,
    ResetRequest_aicustom,
)
from app.state_store_aicustom import chat_state_store_aicustom
from app.services.chat_flow_aicustom import process_chat_aicustom

#fortest api
from app.services.ai_service import detect_intent
from app.services.learning_service_all import analyze_learning_progress

app = FastAPI(title="Entraining Chat API")
origins = [
    "https://www.enmarks.com",
    "https://www.entraining.net",
    "https://entraining.net",
    "http://localhost",
    "http://127.0.0.1",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    print("🔥 CHAT ENDPOINT HIT 🔥", flush=True)
    return {"status": "ok"}

@app.post("/detect-intent")
async def detect_intent_route(req: ChatRequest):
    if not req.user_message or not req.user_message.strip():
        raise HTTPException(status_code=400, detail="user_message is required")

    user_message = req.user_message.strip()
    result = await detect_intent(user_message)
    return result

@app.post("/analyze")
async def analyze_route(req: ChatRequest):
    if not req.user_message or not req.user_message.strip():
        raise HTTPException(status_code=400, detail="user_message is required")

    user_message = req.user_message.strip()

    state = {
        "topic": req.state.topic if req.state else "unknown",
        "learning_need": req.state.learning_need if req.state else "unknown",
        "last_question": req.state.last_question if req.state else "none",
    }

    result = await analyze_learning_progress(
        user_message=user_message,
        state=state,
        model="gpt-4.1-mini",
    )
    return result

@app.post("/chat/all", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.user_message or not req.user_message.strip():
        raise HTTPException(status_code=400, detail="user_message is required")

    user_message = req.user_message.strip()
    req.user_message = user_message
    conn = get_db_connection()

    if req.state:
        state = req.state
    else:
        state = chat_state_store_all.get_state(req.web_no, req.member_no)

    print_debug("req.user_message", user_message)
    # print_debug("before state", state)
    print_state("BEFORE STATE", state)

    try:
        result = await process_chat(req, state, conn)
        print_state("AFTER STATE", result.state)
        # print_debug("result", result)   

        chat_state_store_all.set_state(req.web_no, req.member_no, result.state)

        return ChatResponse(
            reply=result.reply,
            state=result.state,
            source=result.source or "debug_chat",
        )

    except Exception as e:
        print("DEBUG ERROR =", repr(e))
        return ChatResponse(
            reply=f"DEBUG ERROR: {str(e)}",
            state=state,
            source="debug_error",
        )

    
@app.post("/chat/reset/all")
async def reset_chat(payload: ResetRequest):
    state = chat_state_store_all.reset_state(payload.web_no, payload.member_no)
    return {"status": "ok", "state": state.model_dump()}
    
@app.post("/chat/model1", response_model=ChatResponse_model1)
async def chat(req: ChatRequest_model1):
    if not req.user_message or not req.user_message.strip():
        raise HTTPException(status_code=400, detail="user_message is required")

    user_message = req.user_message.strip()
    req.user_message = user_message
    conn = get_db_connection()

    if req.state:
        state = req.state
    else:
        state = chat_state_store_model1.get_state(req.web_no, req.member_no)

    print_debug("req.user_message", user_message)
    print_debug("before state", state)
    print_state("BEFORE STATE", state)

    try:
        result = await process_chat_model1(req, state, conn)
        print_state("AFTER STATE", result.state)
        print_debug("result", result)   

        chat_state_store_model1.set_state(req.web_no, req.member_no, result.state)

        return ChatResponse_model1(
            reply=result.reply,
            state=result.state,
            source=result.source or "debug_chat",
        )

    except Exception as e:
        print("DEBUG ERROR =", repr(e))
        return ChatResponse_model1(
            reply=f"DEBUG ERROR: {str(e)}",
            state=state,
            source="debug_error",
        )
    

    # result = await process_chat(req, state)
    # chat_state_store.set_state(req.web_no, req.member_no, result.state)

    # return result

@app.post("/chat/reset/model1")
async def reset_chat(payload: ResetRequest_model1):
    state = chat_state_store_model1.reset_state(payload.web_no, payload.member_no)
    return {"status": "ok", "state": state.model_dump()}

@app.post("/chat/model2", response_model=ChatResponse_model2)
async def chat(req: ChatRequest_model2):
    if not req.user_message or not req.user_message.strip():
        raise HTTPException(status_code=400, detail="user_message is required")

    user_message = req.user_message.strip()
    req.user_message = user_message
    conn = get_db_connection()

    if req.state:
        state = req.state
    else:
        state = chat_state_store_model2.get_state(req.web_no, req.member_no)

    print_debug("req.user_message", user_message)
    print_debug("before state", state)
    print_state("BEFORE STATE", state)

    try:
        result = await process_chat_model2(req, state, conn)
        print_state("AFTER STATE", result.state)
        print_debug("result", result)

        chat_state_store_model2.set_state(req.web_no, req.member_no, result.state)

        return ChatResponse_model2(
            reply=result.reply,
            state=result.state,
            source=result.source or "debug_chat",
        )

    except Exception as e:
        print("DEBUG ERROR =", repr(e))
        return ChatResponse_model2(
            reply=f"DEBUG ERROR: {str(e)}",
            state=state,
            source="debug_error",
        )
    
@app.post("/chat/reset/model2")
async def reset_chat(payload: ResetRequest_model2):
    state = chat_state_store_model2.reset_state(payload.web_no, payload.member_no)
    return {"status": "ok", "state": state.model_dump()}

@app.post("/start/ai-coach")
async def chat(req: ChatRequest_aicoach):
    if not req.user_message or not req.user_message.strip():
        raise HTTPException(status_code=400, detail="user_message is required")
    if req.state:
        state = req.state
    else:
        state = chat_state_store_aicoach.get_state(req.web_no, req.member_no)

    step = 1
    fixed_question = FIXED_QUESTIONS[step]
    state = ChatState(
        step=0,
        fixed_question = fixed_question,
        )

    chat_state_store_aicoach.set_state(req.web_no, req.member_no, state)

    # print_debug("before state", state)
    print_state("BEFORE STATE", state)

    try:
        result = await process_chat_aicoach(req, state)
        print_state("AFTER STATE", result.state)
        # print_debug("result", result)

        chat_state_store_aicoach.set_state(req.web_no, req.member_no, result.state)

        return ChatResponse_aicoach(
            reply=result.reply,
            state=result.state,
            source=result.source or "debug_chat",
        )

    except Exception as e:
        print("DEBUG ERROR =", repr(e))
        return ChatResponse_aicoach(
            reply=f"DEBUG ERROR: {str(e)}",
            state=state,
            source="debug_error",
        )
    
@app.post("/chat/ai-coach")
async def chat(req: ChatRequest_aicoach):
    if not req.user_message or not req.user_message.strip():
        raise HTTPException(status_code=400, detail="user_message is required")
    if req.state:
        state = req.state
    else:
        state = chat_state_store_aicoach.get_state(req.web_no, req.member_no)

    # print_debug("before state", state)
    print_state("BEFORE STATE", state)

    try:
        result = await process_chat_aicoach(req, state)
        print_state("AFTER STATE", result.state)
        # print_debug("result", result)

        chat_state_store_aicoach.set_state(req.web_no, req.member_no, result.state)

        return ChatResponse_aicoach(
            reply=result.reply,
            state=result.state,
            source=result.source or "debug_chat",
        )

    except Exception as e:
        print("DEBUG ERROR =", repr(e))
        return ChatResponse_aicoach(
            reply=f"DEBUG ERROR: {str(e)}",
            state=state,
            source="debug_error",
        )

@app.post("/chat/reset/ai-coach")
async def reset_chat(payload: ResetRequest_model2):
    state = chat_state_store_aicoach.reset_state(payload.web_no, payload.member_no)
    return {"status": "ok", "state": state.model_dump()}

@app.post("/chat/ai-sale", response_model=ChatResponse_aisale)
async def chat_ai_sale(req: ChatRequest_aisale):
    if not req.user_message or not req.user_message.strip():
        raise HTTPException(status_code=400, detail="user_message is required")

    user_message = req.user_message.strip()
    req.user_message = user_message
    conn = get_db_connection()

    if req.state:
        state = req.state
    else:
        state = chat_state_store_aisale.get_state(req.web_no, req.member_no)

    print_state("BEFORE STATE", state)

    try:
        result = await process_chat_aisale(req, state, conn)
        print_state("AFTER STATE", result.state)

        chat_state_store_aisale.set_state(req.web_no, req.member_no, result.state)

        return ChatResponse_aisale(
            reply=result.reply,
            state=result.state,
            source=result.source or "ai_sale",
            # courses=result.courses,
        )

    except Exception as e:
        print("DEBUG ERROR =", repr(e))
        return ChatResponse_aisale(
            reply=f"DEBUG ERROR: {str(e)}",
            state=state,
            source="debug_error",
            courses=[],
        )


@app.post("/chat/reset/ai-sale")
async def reset_chat_ai_sale(payload: ResetRequest_aisale):
    state = chat_state_store_aisale.reset_state(payload.web_no, payload.member_no)
    return {"status": "ok", "state": state.model_dump()}

@app.post("/chat/ai-web", response_model=ChatResponse_aiweb)
async def chat_ai_web(req: ChatRequest_aiweb, request: Request):
    if not req.user_message or not req.user_message.strip():
        raise HTTPException(status_code=400, detail="user_message is required")

    user_message = req.user_message.strip()
    req.user_message = user_message

    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent", "")

    conn_chat = get_mysql_connection()
    conn_script = get_db_connection()   # ตัวเดิมของคุณ

    try:
        ensure_chat_session(
            conn_chat,
            chat_id=req.chat_id,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        insert_request_log(
            conn_chat,
            chat_id=req.chat_id,
            ip_address=ip_address,
        )

        insert_chat_message(
            conn_chat,
            chat_id=req.chat_id,
            sender_type="user",
            message_text=user_message,
        )

        if req.state:
            state = req.state
        else:
            state = load_chat_state(conn_chat, req.chat_id)

        result = await process_chat_aiweb(req, state, conn_script)

        save_chat_state(conn_chat, req.chat_id, result.state)

        insert_chat_message(
            conn_chat,
            chat_id=req.chat_id,
            sender_type="assistant",
            message_text=result.reply,
        )

        return ChatResponse_aiweb(
            reply=result.reply,
            state=result.state,
            source=result.source or "ai_web",
            chat_id=req.chat_id,
        )

    except Exception as e:
        print("DEBUG ERROR =", repr(e))
        return ChatResponse_aiweb(
            reply=f"DEBUG ERROR: {str(e)}",
            state=state if 'state' in locals() else None,
            source="debug_error",
            chat_id=req.chat_id,
        )

    finally:
        try:
            conn_chat.close()
        except Exception:
            pass
        try:
            conn_script.close()
        except Exception:
            pass


@app.post("/chat/reset/ai-web")
async def reset_chat_ai_web(payload: ResetRequest_aiweb):
    conn_chat = get_mysql_connection()
    try:
        reset_chat_state(conn_chat, payload.chat_id)
        return {"status": "ok", "chat_id": payload.chat_id}
    finally:
        conn_chat.close()

@app.post("/chat/ai-self-learning", response_model=ChatResponse_aiselflearning)
async def chat_ai_self_learning(req: ChatRequest_aiselflearning):
    if not req.user_message or not req.user_message.strip():
        raise HTTPException(status_code=400, detail="user_message is required")

    user_message = req.user_message.strip()
    req.user_message = user_message
    conn_mysql = get_mysql_connection()

    if req.state:
        state = req.state
    else:
        state = chat_state_store_aiselflearning.get_state(req.chat_id)

    print_debug("req.user_message", user_message)
    print_debug("before state", state)
    print_state("BEFORE STATE", state)

    try:
        result = await process_chat_aiselflearning(req, state, conn_mysql)

        insert_chat_history_aiselflearning(
        conn=conn_mysql,
        chat_id=req.chat_id,
        course_no=req.OCourse_no,
        user_message=req.user_message,
        ai_reply=result.reply,
        ai_status=result.status,
        ai_reason=result.reason,
)


        print_state("AFTER STATE", result.state)
        print_debug("result", result)

        chat_state_store_aiselflearning.set_state(req.chat_id, result.state)

        return ChatResponse_aiselflearning(
            reply=result.reply,
            state=result.state,
            source=result.source or "ai_self_learning",
            chat_id=req.chat_id,
        )

    except Exception as e:
        print("DEBUG ERROR =", repr(e))
        return ChatResponse_aiselflearning(
            reply=f"DEBUG ERROR: {str(e)}",
            state=state,
            source="debug_error",
            chat_id=req.chat_id,
        )

    finally:
        try:
            conn_mysql.close()
        except Exception:
            pass


@app.post("/chat/reset/ai-self-learning")
async def reset_chat_ai_self_learning(payload: ResetRequest_aiselflearning):
    conn_chat = get_mysql_connection_aisl()
    try:
        reset_chat_state_aisl(conn_chat, payload.chat_id)
        return {"status": "ok", "chat_id": payload.chat_id}
    finally:
        conn_chat.close()


@app.post("/chat/ai-custom", response_model=ChatResponse_aicustom)
async def chat_ai_custom(req: ChatRequest_aicustom):
    if not req.user_message or not req.user_message.strip():
        raise HTTPException(status_code=400, detail="user_message is required")

    user_message = req.user_message.strip()
    req.user_message = user_message
    conn_mysql = get_mysql_connection()

    if req.state:
        state = req.state
    else:
        state = chat_state_store_aicustom.get_state(req.web_no, req.member_no)

    # sync web/member ลง state
    state.web_no = req.web_no
    state.member_no = req.member_no

    # ถ้า frontend ส่ง course_use มา ให้เก็บลง state ทันที
    if req.course_use:
        state.course_use = [str(x).strip() for x in req.course_use if str(x).strip()]

    print_debug("req.user_message", user_message)
    print_debug("before state", state)
    print_state("BEFORE STATE", state)

    try:
        result = await process_chat_aicustom(req, state, conn_mysql)

        print_state("AFTER STATE", result.state)
        print_debug("result", result)

        chat_state_store_aicustom.set_state(req.web_no, req.member_no, result.state)

        return ChatResponse_aicustom(
            reply=result.reply,
            state=result.state,
            source=result.source or "ai_custom",
        )

    except Exception as e:
        print("DEBUG ERROR =", repr(e))
        return ChatResponse_aicustom(
            reply=f"DEBUG ERROR: {str(e)}",
            state=state,
            source="debug_error",
        )

    finally:
        try:
            conn_mysql.close()
        except Exception:
            pass

@app.post("/chat/reset/ai-custom")
async def reset_chat_ai_custom(payload: ResetRequest_aicustom):
    state = chat_state_store_aicustom.reset_state(payload.web_no, payload.member_no)
    return {
        "status": "ok",
        "state": state.model_dump(),
        "web_no": payload.web_no,
        "member_no": payload.member_no,
    }