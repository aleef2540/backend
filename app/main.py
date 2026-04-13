from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import ChatRequest, ChatResponse, ChatState
from app.state_store import chat_state_store
from app.services.chat_flow import process_chat
from app.services.ai_service import detect_intent

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
    return {"status": "ok"}

@app.post("/detect-intent")
async def detect_intent_route(req: ChatRequest):
    if not req.user_message or not req.user_message.strip():
        raise HTTPException(status_code=400, detail="user_message is required")

    result = await detect_intent(req.user_message)
    return result


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.user_message or not req.user_message.strip():
        raise HTTPException(status_code=400, detail="user_message is required")

    # if req.state:
    #     state = req.state
    # else:
    #     state = chat_state_store.get_state(req.web_no, req.member_no)

    # result = await process_chat(req, state)
    # chat_state_store.set_state(req.web_no, req.member_no, result.state)

    # return result

    state = req.state or ChatState()

    return ChatResponse(
        reply=(
            f"รับค่าแล้ว | "
            f"user_message={req.user_message} | "
            f"web_no={req.web_no} | "
            f"member_no={req.member_no}"
        ),
        state=state,
        source="echo_test",
        debug={
            "received_user_message": req.user_message,
            "received_web_no": req.web_no,
            "received_member_no": req.member_no,
            "received_state": state.model_dump(),
        },
    )


@app.post("/chat/reset")
async def reset_chat(web_no: int | None = None, member_no: int | None = None):
    state = chat_state_store.reset_state(web_no, member_no)
    return {"status": "ok", "state": state}