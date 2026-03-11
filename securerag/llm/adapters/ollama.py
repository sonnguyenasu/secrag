from __future__ import annotations

import logging

import httpx

from securerag.llm.base import SyncToAsyncMixin


class OllamaAdapter(SyncToAsyncMixin):
    def __init__(
        self,
        model: str = "qwen3:0.6b",
        base_url: str = "http://127.0.0.1:11434",
        timeout_s: float = 8.0,
        retries: int = 1,
        enabled: bool = False,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.retries = max(0, retries)
        self.enabled = enabled
        self._logger = logging.getLogger("securerag.llm")
        self._disabled_reason: str | None = None

    def complete(self, prompt: str, *, system: str | None = None, temperature: float = 0.2, max_tokens: int = 512) -> str | None:
        _ = max_tokens
        if not self.enabled or self._disabled_reason is not None:
            return None
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system

        for attempt in range(self.retries + 1):
            try:
                resp = httpx.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout_s)
                resp.raise_for_status()
                text = str(resp.json().get("response", "")).strip()
                return text or None
            except httpx.TimeoutException:
                if attempt < self.retries:
                    continue
                return None
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in {400, 404, 422, 501}:
                    self._disabled_reason = f"status={exc.response.status_code}"
                return None
            except Exception:
                return None
        return None
