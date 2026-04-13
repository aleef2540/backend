from typing import Dict
from app.schemas import ChatState


class InMemoryChatStateStore:
    def __init__(self):
        self._store: Dict[str, ChatState] = {}

    def _make_key(self, web_no: int | None, member_no: int | None) -> str:
        return f"{web_no or 0}:{member_no or 0}"

    def get_state(self, web_no: int | None, member_no: int | None) -> ChatState:
        key = self._make_key(web_no, member_no)
        if key not in self._store:
            self._store[key] = ChatState()
        return self._store[key]

    def set_state(self, web_no: int | None, member_no: int | None, state: ChatState) -> ChatState:
        key = self._make_key(web_no, member_no)
        self._store[key] = state
        return state

    def reset_state(self, web_no: int | None, member_no: int | None) -> ChatState:
        state = ChatState()
        self.set_state(web_no, member_no, state)
        return state


chat_state_store = InMemoryChatStateStore()