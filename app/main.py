from fastapi import FastAPI, HTTPException
from app.schemas import ChatRequest, ChatResponse, ChatState
from app.state_store import chat_state_store
from app.services.chat_flow import process_chat

app = FastAPI(title="Entraining Chat API")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.user_message or not req.user_message.strip():
        raise HTTPException(status_code=400, detail="user_message is required")

    # ถ้ามี state ส่งมาใน request ให้ใช้ state นั้นก่อน
    if req.state:
        state = req.state
    else:
        state = chat_state_store.get_state(req.web_no, req.member_no)

    result = await process_chat(req, state)

    # เก็บ state ล่าสุดกลับเข้า store
    chat_state_store.set_state(req.web_no, req.member_no, result.state)

    return result


@app.post("/chat/reset")
async def reset_chat(web_no: int | None = None, member_no: int | None = None):
    state = chat_state_store.reset_state(web_no, member_no)
    return {"status": "ok", "state": state}