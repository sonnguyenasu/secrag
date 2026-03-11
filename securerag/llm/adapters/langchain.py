from __future__ import annotations

from securerag.llm.base import SyncToAsyncMixin


class LangChainAdapter(SyncToAsyncMixin):
    def __init__(self, model):
        self._model = model

    def complete(self, prompt: str, *, system: str | None = None, temperature: float = 0.2, max_tokens: int = 512) -> str | None:
        _ = (temperature, max_tokens)
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            messages = []
            if system:
                messages.append(SystemMessage(content=system))
            messages.append(HumanMessage(content=prompt))
            response = self._model.invoke(messages)
            if hasattr(response, "content"):
                return response.content or None
            return str(response).strip() or None
        except Exception:
            return None
