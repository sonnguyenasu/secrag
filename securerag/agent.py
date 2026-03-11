from __future__ import annotations

from dataclasses import dataclass
import asyncio
from typing import Callable

from securerag.config import PrivacyConfig
from securerag.errors import BackendError, BudgetExhaustedError
from securerag.llm import LLMRoles
from securerag.retriever import PrivacyRetriever


@dataclass(slots=True)
class AgentResult:
    answer: str
    context_size: int
    rounds: int


class _CallableLLMAdapter:
    """Wrap a plain callable into the SecureRAGLLM shape."""

    def __init__(self, fn: Callable):
        self._fn = fn

    def complete(self, prompt: str, *, system: str | None = None, temperature: float = 0.2, max_tokens: int = 512):
        try:
            return self._fn(
                prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except TypeError:
            # Support minimal callables that only accept prompt.
            return self._fn(prompt)

    async def acomplete(self, prompt: str, *, system: str | None = None, temperature: float = 0.2, max_tokens: int = 512):
        return await asyncio.to_thread(
            self.complete,
            prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )


class SecureRAGAgent:
    def __init__(self, llm, retriever: PrivacyRetriever, config: PrivacyConfig):
        self.llm = llm
        self._roles = self._coerce_roles(llm)
        self.retriever = retriever
        self.config = config

    @staticmethod
    def _coerce_roles(llm) -> LLMRoles:
        if isinstance(llm, LLMRoles):
            return llm
        if all(hasattr(llm, name) for name in ("decide", "generate")):
            # Backward-compatible path for legacy ModelAgentLLM-like objects.
            class _LegacyPlanner:
                def __init__(self, legacy):
                    self._legacy = legacy

                def decide(self, query: str, context, round_n: int):
                    return self._legacy.decide(query=query, context=context, round=round_n)

            class _LegacyGenerator:
                def __init__(self, legacy):
                    self._legacy = legacy

                def generate(self, query: str, context):
                    return self._legacy.generate(query, context)

            class _LegacyParaphraser:
                def __init__(self, legacy):
                    self._legacy = legacy

                def paraphrase(self, decoys: list[str], source_query: str) -> list[str]:
                    batch = getattr(self._legacy, "paraphrase_decoys", None)
                    if callable(batch):
                        return batch(decoys, source_query)
                    one = getattr(self._legacy, "paraphrase_decoy", None)
                    if callable(one):
                        return [one(d, source_query) for d in decoys]
                    return decoys

            return LLMRoles(
                planner=_LegacyPlanner(llm),
                generator=_LegacyGenerator(llm),
                paraphraser=_LegacyParaphraser(llm),
            )
        if callable(llm) and not hasattr(llm, "complete"):
            return LLMRoles.uniform(_CallableLLMAdapter(llm))
        return LLMRoles.uniform(llm)

    def _dedup(self, new_docs, context):
        existing = {d.doc_id for d in context}
        return [d for d in new_docs if d.doc_id not in existing]

    def _merge_and_rank(self, context, new_docs):
        merged = context + new_docs
        merged.sort(key=lambda d: d.score, reverse=True)
        return merged[: max(self.config.top_k * 2, 8)]

    def _validate_and_return(self, answer: str, context, rounds: int) -> AgentResult:
        return AgentResult(answer=answer, context_size=len(context), rounds=rounds + 1)

    def run(self, query: str) -> AgentResult:
        context = []
        for round_n in range(self.config.max_rounds):
            decision = self._roles.planner.decide(query=query, context=context, round_n=round_n)
            if decision.should_answer:
                return self._validate_and_return(
                    self._roles.generator.generate(query, context), context, round_n
                )

            sub_query = decision.sub_query or query
            if self.retriever.budget.remaining < self.retriever.privacy_cost(sub_query):
                break

            try:
                new_docs = self.retriever.retrieve(sub_query, round_n)
            except BudgetExhaustedError:
                break
            except BackendError as exc:
                if "budget exhausted" in str(exc).lower():
                    break
                raise

            context = self._merge_and_rank(context, self._dedup(new_docs, context))

        return self._validate_and_return(self._roles.generator.generate(query, context), context, round_n)

    def budget_snapshot(self) -> dict:
        return self.retriever.budget.snapshot()

    @classmethod
    def from_config(cls, config, corpus, llm):
        # Ensure protocol implementations self-register.
        import securerag.retrievers  # noqa: F401

        retriever = PrivacyRetriever.from_config(config, corpus)
        agent = cls(llm=llm, retriever=retriever, config=config)
        if hasattr(retriever, "set_runtime_llm"):
            retriever.set_runtime_llm(agent._roles.paraphraser)
        return agent
