from __future__ import annotations

from securerag.llm.base import SyncToAsyncMixin


class OpenAIAdapter(SyncToAsyncMixin):
    def __init__(self, model: str = "gpt-4o-mini", *, api_key: str | None = None, **kwargs):
        self.model = model
        self.api_key = api_key
        self.kwargs = kwargs

    def complete(self, prompt: str, *, system: str | None = None, temperature: float = 0.2, max_tokens: int = 512) -> str | None:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key, **self.kwargs)
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            resp = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or None
        except Exception:
            return None
