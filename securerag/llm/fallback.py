from __future__ import annotations

from securerag.llm.base import LLMDecision


class DeterministicLLM:
    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str | None:
        _ = (prompt, system, temperature, max_tokens)
        return None

    async def acomplete(self, *args, **kwargs) -> str | None:
        _ = (args, kwargs)
        return None


class DeterministicPlanner:
    def decide(self, query: str, context, round_n: int) -> LLMDecision:
        if round_n >= 2 and len(context) >= 3:
            return LLMDecision(should_answer=True)
        return LLMDecision(should_answer=False, sub_query=self._cot_sub_query(query, context, round_n))

    def _cot_sub_query(self, query: str, context, round_n: int) -> str:
        if round_n <= 0:
            return query
        if round_n == 1:
            return f"{query} evidence details"
        if round_n == 2:
            return f"{query} constraints assumptions risks"

        top_terms: list[str] = []
        for d in context[:2]:
            words = [w.strip(".,:;!?()[]{}\"'").lower() for w in d.text.split()]
            for w in words:
                if len(w) >= 6 and w.isalpha() and w not in top_terms:
                    top_terms.append(w)
                if len(top_terms) >= 3:
                    break
            if len(top_terms) >= 3:
                break
        suffix = " ".join(top_terms) if top_terms else "follow up"
        return f"{query} {suffix}"
