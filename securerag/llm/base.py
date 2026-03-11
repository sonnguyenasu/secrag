from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable
import asyncio


@dataclass(slots=True)
class LLMDecision:
    should_answer: bool
    sub_query: str | None = None


@runtime_checkable
class SecureRAGLLM(Protocol):
    """Minimal LLM contract expected by SecureRAG."""

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str | None:
        ...

    async def acomplete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str | None:
        ...


class SyncToAsyncMixin:
    """Default async wrapper for sync-only adapters."""

    async def acomplete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str | None:
        return await asyncio.to_thread(
            self.complete,
            prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )
