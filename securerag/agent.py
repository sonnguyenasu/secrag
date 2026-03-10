from __future__ import annotations

from dataclasses import dataclass

from securerag.config import PrivacyConfig
from securerag.errors import BackendError, BudgetExhaustedError
from securerag.retriever import PrivacyRetriever


@dataclass(slots=True)
class AgentResult:
    answer: str
    context_size: int
    rounds: int


class SecureRAGAgent:
    def __init__(self, llm, retriever: PrivacyRetriever, config: PrivacyConfig):
        self.llm = llm
        self.retriever = retriever
        self.config = config

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
            decision = self.llm.decide(query=query, context=context, round=round_n)
            if decision.should_answer:
                return self._validate_and_return(
                    self.llm.generate(query, context), context, round_n
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

        return self._validate_and_return(self.llm.generate(query, context), context, round_n)

    def budget_snapshot(self) -> dict:
        return self.retriever.budget.snapshot()

    @classmethod
    def from_config(cls, config, corpus, llm):
        # Ensure protocol implementations self-register.
        import securerag.retrievers  # noqa: F401

        retriever = PrivacyRetriever.from_config(config, corpus)
        if hasattr(retriever, "set_runtime_llm"):
            retriever.set_runtime_llm(llm)
        return cls(llm=llm, retriever=retriever, config=config)
