from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from typing import Any

import httpx


@dataclass(slots=True)
class LLMDecision:
    should_answer: bool
    sub_query: str | None = None


class ModelAgentLLM:
    """
    Provider-agnostic LLM adapter for RAG planning, answer generation,
    and decoy paraphrasing.

    Supported providers:
    - `ollama`
    - `huggingface` (Inference API compatible endpoints)
    """

    def __init__(
        self,
        model: str = "qwen3:0.6b",
        provider: str = "ollama",
        ollama_model: str | None = None,
        huggingface_model: str | None = None,
        use_ollama: bool | None = None,
        use_huggingface: bool | None = None,
        base_url: str | None = None,
        huggingface_base_url: str | None = None,
        huggingface_token: str | None = None,
        timeout_s: float = 8.0,
        retries: int = 1,
    ):
        self.model = model
        self.provider = str(provider).strip().lower()
        self.ollama_model = ollama_model or model
        self.huggingface_model = huggingface_model or model

        env_use = os.getenv("SECURERAG_USE_OLLAMA", "0") in {"1", "true", "TRUE", "yes", "YES"}
        self.use_ollama = env_use if use_ollama is None else use_ollama

        env_use_hf = os.getenv("SECURERAG_USE_HUGGINGFACE", "0") in {
            "1",
            "true",
            "TRUE",
            "yes",
            "YES",
        }
        self.use_huggingface = env_use_hf if use_huggingface is None else use_huggingface

        self.ollama_base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.hf_base_url = (
            huggingface_base_url
            or os.getenv("HF_INFERENCE_BASE_URL", "https://api-inference.huggingface.co")
        ).rstrip("/")
        self.hf_token = huggingface_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

        self.timeout_s = timeout_s
        self.retries = max(0, retries)
        self._logger = logging.getLogger("securerag.llm")
        self._ollama_disabled_reason: str | None = None
        self._hf_disabled_reason: str | None = None

    def _parse_planner_json(self, raw: str) -> tuple[str | None, str | None]:
        text = raw.strip()
        if not text:
            return None, None

        # Accept either a plain token or JSON object from weaker local models.
        if text.upper().startswith("ANSWER"):
            return "ANSWER", None
        if text.upper().startswith("RETRIEVE"):
            return "RETRIEVE", None

        try:
            data = json.loads(text)
            action = str(data.get("action", "")).strip().upper() or None
            sub_query = str(data.get("sub_query", "")).strip() or None
            return action, sub_query
        except Exception:
            return None, None

    def _fallback_cot_sub_query(self, query: str, context, round: int) -> str:
        if round <= 0:
            return query

        # Deterministic query decomposition for multi-turn retrieval.
        if round == 1:
            return f"{query} evidence details"

        if round == 2:
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

    def _ollama_generate(
        self,
        *,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.2,
    ) -> str | None:
        if not self.use_ollama:
            return None
        if self._ollama_disabled_reason is not None:
            return None

        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system

        for attempt in range(self.retries + 1):
            try:
                resp = httpx.post(
                    f"{self.ollama_base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout_s,
                )
                resp.raise_for_status()
                text = str(resp.json().get("response", "")).strip()
                return text or None
            except httpx.TimeoutException as exc:
                if attempt < self.retries:
                    self._logger.warning(
                        "Ollama request timed out (attempt %d/%d). Retrying...",
                        attempt + 1,
                        self.retries + 1,
                    )
                    continue
                self._logger.warning(
                    "Ollama request timed out after %d attempts. Using deterministic fallback for this call.",
                    self.retries + 1,
                )
                return None
            except httpx.HTTPStatusError as exc:
                details = ""
                try:
                    data = exc.response.json()
                    details = str(data.get("error", "")).strip()
                except Exception:
                    details = exc.response.text.strip()

                reason = (
                    f"Ollama request failed at {self.ollama_base_url}/api/generate "
                    f"(status={exc.response.status_code}, model={self.ollama_model}). {details}"
                )
                # Treat model/route errors as non-recoverable for current run.
                if exc.response.status_code in {400, 404, 422, 501}:
                    self._ollama_disabled_reason = reason
                    self._logger.warning(
                        "%s Falling back to deterministic mode for this run.",
                        reason,
                    )
                else:
                    self._logger.warning(
                        "%s Falling back to deterministic mode for this call.",
                        reason,
                    )
                return None
            except Exception as exc:
                self._logger.warning(
                    "Ollama unavailable at %s: %s. Using deterministic fallback for this call.",
                    self.ollama_base_url,
                    exc,
                )
                return None

        return None

    def _extract_hf_text(self, payload: Any) -> str | None:
        if isinstance(payload, list) and payload:
            first = payload[0]
            if isinstance(first, dict):
                text = first.get("generated_text") or first.get("summary_text") or first.get("text")
                if text:
                    return str(text).strip() or None
            if isinstance(first, str):
                return first.strip() or None

        if isinstance(payload, dict):
            if payload.get("error"):
                return None
            text = payload.get("generated_text") or payload.get("summary_text") or payload.get("text")
            if text:
                return str(text).strip() or None

        if isinstance(payload, str):
            return payload.strip() or None

        return None

    def _huggingface_generate(
        self,
        *,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.2,
    ) -> str | None:
        if not self.use_huggingface:
            return None
        if self._hf_disabled_reason is not None:
            return None

        headers: dict[str, str] = {}
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"

        merged_prompt = prompt if not system else f"System: {system}\n\n{prompt}"
        payload = {
            "inputs": merged_prompt,
            "parameters": {
                "temperature": temperature,
                "return_full_text": False,
            },
        }

        url = f"{self.hf_base_url}/models/{self.huggingface_model}"
        for attempt in range(self.retries + 1):
            try:
                resp = httpx.post(url, headers=headers, json=payload, timeout=self.timeout_s)
                resp.raise_for_status()
                return self._extract_hf_text(resp.json())
            except httpx.TimeoutException:
                if attempt < self.retries:
                    self._logger.warning(
                        "Hugging Face request timed out (attempt %d/%d). Retrying...",
                        attempt + 1,
                        self.retries + 1,
                    )
                    continue
                self._logger.warning(
                    "Hugging Face request timed out after %d attempts. Using deterministic fallback for this call.",
                    self.retries + 1,
                )
                return None
            except httpx.HTTPStatusError as exc:
                details = exc.response.text.strip()
                reason = (
                    f"Hugging Face request failed at {url} "
                    f"(status={exc.response.status_code}, model={self.huggingface_model}). {details}"
                )
                if exc.response.status_code in {400, 401, 403, 404, 422, 429, 501}:
                    self._hf_disabled_reason = reason
                    self._logger.warning(
                        "%s Falling back to deterministic mode for this run.",
                        reason,
                    )
                else:
                    self._logger.warning(
                        "%s Falling back to deterministic mode for this call.",
                        reason,
                    )
                return None
            except Exception as exc:
                self._logger.warning(
                    "Hugging Face unavailable at %s: %s. Using deterministic fallback for this call.",
                    url,
                    exc,
                )
                return None

        return None

    def _model_generate(
        self,
        *,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.2,
    ) -> str | None:
        provider = self.provider
        def _try_ollama() -> str | None:
            return self._ollama_generate(prompt=prompt, system=system, temperature=temperature)

        def _try_hf() -> str | None:
            return self._huggingface_generate(prompt=prompt, system=system, temperature=temperature)

        if provider == "ollama":
            out = _try_ollama()
            if out:
                return out
            if self.use_huggingface:
                self._logger.warning(
                    "Primary provider 'ollama' unavailable; falling back to 'huggingface'."
                )
                return _try_hf()
            return None
        if provider in {"huggingface", "hf"}:
            out = _try_hf()
            if out:
                return out
            if self.use_ollama:
                self._logger.warning(
                    "Primary provider 'huggingface' unavailable; falling back to 'ollama'."
                )
                return _try_ollama()
            return None
        if provider == "auto":
            if self.use_ollama:
                out = _try_ollama()
                if out:
                    return out
            if self.use_huggingface:
                return _try_hf()
            return None

        self._logger.warning(
            "Unknown LLM provider '%s'. Falling back to deterministic mode.",
            provider,
        )
        return None

    def decide(self, query: str, context, round: int) -> LLMDecision:
        llm_decision = self._model_generate(
            system=(
                "You are a retrieval planner for multi-turn RAG. "
                "Think step-by-step internally, but output only strict JSON: "
                "{\"action\": \"ANSWER\"|\"RETRIEVE\", \"sub_query\": \"...\"}."
            ),
            prompt=(
                f"Query: {query}\n"
                f"Round: {round}\n"
                f"Context count: {len(context)}\n"
                "Context snippets:\n"
                + "\n".join(f"- {d.text[:180]}" for d in context[:3])
                + "\n"
                "Plan whether to retrieve or answer. If retrieving, provide one focused follow-up sub_query."
            ),
            temperature=0.0,
        )
        if llm_decision:
            action, planned_sub_query = self._parse_planner_json(llm_decision)
            if action == "ANSWER" and len(context) > 0 and round >= 2:
                return LLMDecision(should_answer=True)
            if action == "RETRIEVE":
                sub_query = planned_sub_query or self._fallback_cot_sub_query(query, context, round)
                return LLMDecision(should_answer=False, sub_query=sub_query)

        if round >= 2 and len(context) >= 3:
            return LLMDecision(should_answer=True)
        return LLMDecision(
            should_answer=False,
            sub_query=self._fallback_cot_sub_query(query, context, round),
        )

    def generate(self, query: str, context) -> str:
        llm_answer = self._model_generate(
            system="You are a concise assistant. Use evidence only.",
            prompt=(
                f"User query: {query}\n\n"
                "Evidence:\n"
                + "\n".join(
                    f"[{d.doc_id}] {d.text.strip()[:300]} (score={d.score:.3f})" for d in context[:6]
                )
                + "\n\nProduce a direct answer."
            ),
            temperature=0.2,
        )
        if llm_answer:
            return llm_answer

        if not context:
            return "Insufficient context to answer confidently."
        top = "\n".join(
            f"- [{d.doc_id}] {d.text.strip()[:220]} (score={d.score:.3f})" for d in context[:3]
        )
        return f"Answer for: {query}\n\nEvidence:\n{top}"

    def paraphrase_decoy(self, decoy: str, source_query: str) -> str:
        """
        Deterministic decoy paraphrasing fallback for MVP.
        Keeps semantics loose while making decoys read like natural queries.
        """
        llm_para = self._model_generate(
            system="Rewrite text as a plausible standalone search query.",
            prompt=(
                f"Decoy source text: {decoy}\n"
                "Output exactly one natural search query sentence, no quotes."
            ),
            temperature=0.6,
        )
        if llm_para:
            return " ".join(llm_para.split())

        snippet = " ".join(decoy.strip().split()[:10])
        if not snippet:
            snippet = "general background"
        return f"Find information related to: {snippet}."

    def paraphrase_decoys(self, decoys: list[str], source_query: str) -> list[str]:
        if not decoys:
            return []

        llm_batch = self._model_generate(
            system=(
                "Rewrite each decoy text as a plausible standalone search query. "
                "Return exactly one line per decoy, in order, with no numbering."
            ),
            prompt=(
                "Decoys:\n"
                + "\n".join(f"- {d}" for d in decoys)
                + "\n\nReturn rewritten decoys, one per line, same count and order."
            ),
            temperature=0.6,
        )

        if llm_batch:
            lines = [" ".join(line.strip().split()) for line in llm_batch.splitlines() if line.strip()]
            if len(lines) >= len(decoys):
                return lines[: len(decoys)]

        # Fallback to per-decoy paraphrasing if batch output is unavailable or malformed.
        return [self.paraphrase_decoy(d, source_query) for d in decoys]


class OllamaLLM(ModelAgentLLM):
    """Backward-compatible alias for Ollama-backed provider."""

    def __init__(
        self,
        model: str = "qwen3:0.6b",
        use_ollama: bool | None = None,
        base_url: str | None = None,
        timeout_s: float = 8.0,
        retries: int = 1,
    ):
        super().__init__(
            model=model,
            provider="ollama",
            ollama_model=model,
            use_ollama=use_ollama,
            base_url=base_url,
            timeout_s=timeout_s,
            retries=retries,
        )


class HuggingFaceLLM(ModelAgentLLM):
    """Convenience wrapper for Hugging Face-backed provider."""

    def __init__(
        self,
        model: str,
        use_huggingface: bool | None = None,
        huggingface_base_url: str | None = None,
        huggingface_token: str | None = None,
        timeout_s: float = 8.0,
        retries: int = 1,
    ):
        super().__init__(
            model=model,
            provider="huggingface",
            huggingface_model=model,
            use_huggingface=use_huggingface,
            huggingface_base_url=huggingface_base_url,
            huggingface_token=huggingface_token,
            timeout_s=timeout_s,
            retries=retries,
        )
