from __future__ import annotations

import json

from securerag.llm.base import LLMDecision, SecureRAGLLM
from securerag.llm.fallback import DeterministicPlanner


class Planner:
    SYSTEM = (
        "You are a retrieval planner for multi-turn RAG. "
        "Output only strict JSON: "
        '{"action": "ANSWER"|"RETRIEVE", "sub_query": "..."}.'
    )

    def __init__(self, llm: SecureRAGLLM, fallback: DeterministicPlanner | None = None):
        self._llm = llm
        self._fallback = fallback or DeterministicPlanner()

    def decide(self, query: str, context, round_n: int) -> LLMDecision:
        raw = self._llm.complete(
            self._build_prompt(query, context, round_n),
            system=self.SYSTEM,
            temperature=0.0,
        )
        if raw:
            out = self._parse(raw)
            if out is not None:
                if out.should_answer and len(context) > 0 and round_n >= 2:
                    return out
                if not out.should_answer:
                    return LLMDecision(False, out.sub_query or self._fallback._cot_sub_query(query, context, round_n))
        return self._fallback.decide(query, context, round_n)

    def _build_prompt(self, query: str, context, round_n: int) -> str:
        return (
            f"Query: {query}\n"
            f"Round: {round_n}\n"
            f"Context count: {len(context)}\n"
            "Context snippets:\n"
            + "\n".join(f"- {d.text[:180]}" for d in context[:3])
            + "\nPlan whether to retrieve or answer. If retrieving, provide one focused follow-up sub_query."
        )

    def _parse(self, raw: str) -> LLMDecision | None:
        text = raw.strip()
        if not text:
            return None
        if text.upper().startswith("ANSWER"):
            return LLMDecision(should_answer=True)
        if text.upper().startswith("RETRIEVE"):
            return LLMDecision(should_answer=False)
        try:
            data = json.loads(text)
            action = str(data.get("action", "")).strip().upper()
            sub_query = str(data.get("sub_query", "")).strip() or None
            if action == "ANSWER":
                return LLMDecision(True)
            if action == "RETRIEVE":
                return LLMDecision(False, sub_query)
        except Exception:
            return None
        return None


class Generator:
    SYSTEM = "You are a concise assistant. Use evidence only."

    def __init__(self, llm: SecureRAGLLM):
        self._llm = llm

    def generate(self, query: str, context) -> str:
        raw = self._llm.complete(
            self._build_prompt(query, context),
            system=self.SYSTEM,
            temperature=0.2,
            max_tokens=512,
        )
        return raw or self._deterministic_answer(query, context)

    def _build_prompt(self, query: str, context) -> str:
        return (
            f"User query: {query}\n\n"
            "Evidence:\n"
            + "\n".join(
                f"[{d.doc_id}] {d.text.strip()[:300]} (score={d.score:.3f})" for d in context[:6]
            )
            + "\n\nProduce a direct answer."
        )

    def _deterministic_answer(self, query: str, context) -> str:
        if not context:
            return "Insufficient context to answer confidently."
        top = "\n".join(
            f"- [{d.doc_id}] {d.text.strip()[:220]} (score={d.score:.3f})" for d in context[:3]
        )
        return f"Answer for: {query}\n\nEvidence:\n{top}"


class Paraphraser:
    SYSTEM = "Rewrite each decoy as a plausible standalone search query."

    def __init__(self, llm: SecureRAGLLM):
        self._llm = llm

    def paraphrase(self, decoys: list[str], source_query: str) -> list[str]:
        _ = source_query
        if not decoys:
            return []
        raw = self._llm.complete(
            "Decoys:\n" + "\n".join(f"- {d}" for d in decoys) + "\n\nReturn one rewritten query per line.",
            system=self.SYSTEM,
            temperature=0.6,
            max_tokens=512,
        )
        if raw:
            lines = [" ".join(line.strip().split()) for line in raw.splitlines() if line.strip()]
            if len(lines) >= len(decoys):
                return lines[: len(decoys)]
        return [self._fallback_decoy(d) for d in decoys]

    @staticmethod
    def _fallback_decoy(decoy: str) -> str:
        snippet = " ".join(decoy.strip().split()[:10])
        if not snippet:
            snippet = "general background"
        return f"Find information related to: {snippet}."


class LLMRoles:
    def __init__(self, planner: Planner, generator: Generator, paraphraser: Paraphraser):
        self.planner = planner
        self.generator = generator
        self.paraphraser = paraphraser

    @classmethod
    def uniform(cls, llm: SecureRAGLLM) -> "LLMRoles":
        return cls(
            planner=Planner(llm),
            generator=Generator(llm),
            paraphraser=Paraphraser(llm),
        )
