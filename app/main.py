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

from app.utils.debug_state import print_state, print_debug


#fortest api
from app.services.ai_service import detect_intent
from app.services.learning_service_all import analyze_learning_progress

app = FastAPI(title="Entraining Chat API")
origins = [
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
    print_debug("before state", state)
    print_state("BEFORE STATE", state)

    try:
        result = await process_chat(req, state, conn)
        print_state("AFTER STATE", result.state)
        print_debug("result", result)   

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