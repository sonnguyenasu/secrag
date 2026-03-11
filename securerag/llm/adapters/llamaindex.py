from __future__ import annotations

from securerag.llm.base import SyncToAsyncMixin


class LlamaIndexAdapter(SyncToAsyncMixin):
    def __init__(self, llm):
        self._llm = llm

    def complete(self, prompt: str, *, system: str | None = None, temperature: float = 0.2, max_tokens: int = 512) -> str | None:
        _ = (temperature, max_tokens)
        try:
            merged = f"{system}\n\n{prompt}" if system else prompt
            response = self._llm.complete(merged)
            return str(response).strip() or None
        except Exception:
            return None
