from __future__ import annotations

from securerag.llm.base import SyncToAsyncMixin


class AnthropicAdapter(SyncToAsyncMixin):
    def __init__(self, model: str = "claude-3-haiku-20240307", *, api_key: str | None = None, **kwargs):
        self.model = model
        self.api_key = api_key
        self.kwargs = kwargs

    def complete(self, prompt: str, *, system: str | None = None, temperature: float = 0.2, max_tokens: int = 512) -> str | None:
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key, **self.kwargs)
            msg = client.messages.create(
                model=self.model,
                system=system or "",
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            if msg.content and hasattr(msg.content[0], "text"):
                return msg.content[0].text or None
            return None
        except Exception:
            return None
