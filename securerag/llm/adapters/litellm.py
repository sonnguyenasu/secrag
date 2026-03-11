from __future__ import annotations

from securerag.llm.base import SyncToAsyncMixin


class LiteLLMAdapter(SyncToAsyncMixin):
    def __init__(self, model: str, **kwargs):
        self.model = model
        self.kwargs = kwargs

    def complete(self, prompt: str, *, system: str | None = None, temperature: float = 0.2, max_tokens: int = 512) -> str | None:
        try:
            import litellm

            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            resp = litellm.completion(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **self.kwargs,
            )
            return resp.choices[0].message.content or None
        except Exception:
            return None
