from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mysql.connector
import sqlite3


def get_db_connection():
    return sqlite3.connect("ai_idp_script.db")


from app.schemas import ChatRequest, ChatResponse, ChatState, ResetRequest
from app.state_store import chat_state_store
from app.services.chat_flow import process_chat
from app.services.ai_service import detect_intent
from app.services.learning_service import analyze_learning_progress

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

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.user_message or not req.user_message.strip():
        raise HTTPException(status_code=400, detail="user_message is required")

    user_message = req.user_message.strip()
    req.user_message = user_message
    conn = get_db_connection()

    if req.state:
        state = req.state
    else:
        state = chat_state_store.get_state(req.web_no, req.member_no)

    print("DEBUG req.user_message =", user_message)
    print("DEBUG before state =", state)

    print("==== BEFORE STATE ====")
    print(state.model_dump())

    try:
        result = await process_chat(req, state, conn)
        print("==== AFTER STATE ====")
        print(result.state.model_dump())

        print("DEBUG result =", result)

        chat_state_store.set_state(req.web_no, req.member_no, result.state)

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
    

    # result = await process_chat(req, state)
    # chat_state_store.set_state(req.web_no, req.member_no, result.state)

    # return result

@app.post("/chat/reset")
async def reset_chat(payload: ResetRequest):
    state = chat_state_store.reset_state(payload.web_no, payload.member_no)
    return {"status": "ok", "state": state.model_dump()}