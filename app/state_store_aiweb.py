from app.schemas_aiweb import ChatState_aiweb


class ChatStateStoreAIWeb:
    def __init__(self):
        self._store = {}

    def get_state(self, chat_id: str) -> ChatState_aiweb:
        return self._store.get(chat_id, ChatState_aiweb())

    def set_state(self, chat_id: str, state: ChatState_aiweb):
        self._store[chat_id] = state

    def reset_state(self, chat_id: str) -> ChatState_aiweb:
        self._store[chat_id] = ChatState_aiweb()
        return self._store[chat_id]


chat_state_store_aiweb = ChatStateStoreAIWeb()