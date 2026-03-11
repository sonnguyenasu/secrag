from __future__ import annotations

import logging

import httpx

from securerag.llm.base import SyncToAsyncMixin


class HuggingFaceAdapter(SyncToAsyncMixin):
    def __init__(
        self,
        model: str,
        base_url: str = "https://api-inference.huggingface.co",
        token: str | None = None,
        timeout_s: float = 8.0,
        retries: int = 1,
        enabled: bool = False,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_s = timeout_s
        self.retries = max(0, retries)
        self.enabled = enabled
        self._logger = logging.getLogger("securerag.llm")
        self._disabled_reason: str | None = None

    @staticmethod
    def _extract(payload):
        if isinstance(payload, list) and payload:
            first = payload[0]
            if isinstance(first, dict):
                text = first.get("generated_text") or first.get("summary_text") or first.get("text")
                return str(text).strip() if text else None
            if isinstance(first, str):
                return first.strip() or None
        if isinstance(payload, dict):
            if payload.get("error"):
                return None
            text = payload.get("generated_text") or payload.get("summary_text") or payload.get("text")
            return str(text).strip() if text else None
        if isinstance(payload, str):
            return payload.strip() or None
        return None

    def complete(self, prompt: str, *, system: str | None = None, temperature: float = 0.2, max_tokens: int = 512) -> str | None:
        _ = max_tokens
        if not self.enabled or self._disabled_reason is not None:
            return None
        headers: dict[str, str] = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        merged = prompt if not system else f"System: {system}\n\n{prompt}"
        payload = {"inputs": merged, "parameters": {"temperature": temperature, "return_full_text": False}}
        url = f"{self.base_url}/models/{self.model}"

        for attempt in range(self.retries + 1):
            try:
                resp = httpx.post(url, headers=headers, json=payload, timeout=self.timeout_s)
                resp.raise_for_status()
                return self._extract(resp.json())
            except httpx.TimeoutException:
                if attempt < self.retries:
                    continue
                return None
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in {400, 401, 403, 404, 422, 429, 501}:
                    self._disabled_reason = f"status={exc.response.status_code}"
                return None
            except Exception:
                return None
        return None
