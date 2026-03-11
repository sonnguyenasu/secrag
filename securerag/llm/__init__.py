from __future__ import annotations

import os

from securerag.llm.adapters import (
    AnthropicAdapter,
    HuggingFaceAdapter,
    LangChainAdapter,
    LiteLLMAdapter,
    LlamaIndexAdapter,
    OllamaAdapter,
    OpenAIAdapter,
)
from securerag.llm.base import LLMDecision, SecureRAGLLM
from securerag.llm.fallback import DeterministicLLM
from securerag.llm.roles import Generator, LLMRoles, Paraphraser, Planner


def _build_adapter(model: str, provider: str, **kwargs):
    provider = str(provider).strip().lower()
    retries = int(kwargs.get("retries", 1))
    timeout_s = float(kwargs.get("timeout_s", 8.0))

    env_use_ollama = os.getenv("SECURERAG_USE_OLLAMA", "0") in {"1", "true", "TRUE", "yes", "YES"}
    env_use_hf = os.getenv("SECURERAG_USE_HUGGINGFACE", "0") in {"1", "true", "TRUE", "yes", "YES"}

    if provider == "ollama":
        return OllamaAdapter(
            model=kwargs.get("ollama_model") or model,
            base_url=(kwargs.get("base_url") or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")),
            timeout_s=timeout_s,
            retries=retries,
            enabled=env_use_ollama if kwargs.get("use_ollama") is None else bool(kwargs.get("use_ollama")),
        )
    if provider in {"huggingface", "hf"}:
        return HuggingFaceAdapter(
            model=kwargs.get("huggingface_model") or model,
            base_url=(kwargs.get("huggingface_base_url") or os.getenv("HF_INFERENCE_BASE_URL", "https://api-inference.huggingface.co")),
            token=kwargs.get("huggingface_token") or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            timeout_s=timeout_s,
            retries=retries,
            enabled=env_use_hf if kwargs.get("use_huggingface") is None else bool(kwargs.get("use_huggingface")),
        )
    if provider == "openai":
        return OpenAIAdapter(model=model, api_key=kwargs.get("api_key"))
    if provider == "anthropic":
        return AnthropicAdapter(model=model, api_key=kwargs.get("api_key"))
    if provider == "litellm":
        return LiteLLMAdapter(model=model)
    return DeterministicLLM()


class ModelAgentLLM:
    """Backward-compatible shim over the role/adapters system."""

    def __init__(self, model: str = "qwen3:0.6b", provider: str = "ollama", **kwargs):
        adapter = _build_adapter(model, provider, **kwargs)
        roles = LLMRoles.uniform(adapter)
        self._planner = roles.planner
        self._generator = roles.generator
        self._paraphraser = roles.paraphraser

    def decide(self, query, context, round):
        return self._planner.decide(query, context, round)

    def generate(self, query, context):
        return self._generator.generate(query, context)

    def paraphrase_decoy(self, decoy, source_query):
        return self._paraphraser.paraphrase([decoy], source_query)[0]

    def paraphrase_decoys(self, decoys, source_query):
        return self._paraphraser.paraphrase(decoys, source_query)


class OllamaLLM(ModelAgentLLM):
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


__all__ = [
    "SecureRAGLLM",
    "LLMDecision",
    "DeterministicLLM",
    "Planner",
    "Generator",
    "Paraphraser",
    "LLMRoles",
    "OllamaAdapter",
    "HuggingFaceAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "LangChainAdapter",
    "LlamaIndexAdapter",
    "LiteLLMAdapter",
    "ModelAgentLLM",
    "OllamaLLM",
    "HuggingFaceLLM",
]
