# SecureRAG — LLM Layer Refactor Plan

## Problem Statement

The current `llm.py` rolls its own HTTP client for every provider it supports
(Ollama, HuggingFace Inference API). This creates several compounding problems:

- **Narrow provider support.** OpenAI, Anthropic, Cohere, Gemini, Azure, Bedrock —
  none work without forking the file.
- **Reinvented wheel.** LangChain, LlamaIndex, and LiteLLM already solve provider
  routing, retries, streaming, async, and token counting. The current code duplicates
  a subset of those solutions, worse.
- **Three coupled responsibilities in one class.** `ModelAgentLLM` handles planning
  (`decide`), answer synthesis (`generate`), and decoy paraphrasing
  (`paraphrase_decoys`). These have different latency budgets, temperature profiles,
  and model requirements — they should be separable.
- **No async path.** `PrivacyRetriever` and the benchmark harness would benefit from
  concurrent retrieval + generation, but `httpx` is called synchronously today.
- **Hard-coded fallback logic.** The deterministic fallback (round >= 2 → answer) is
  mixed into `decide()` instead of being an injectable strategy.

---

## Design Principles

1. **SecureRAG defines the contract; popular libraries fill it.**
   We publish a thin `SecureRAGLLM` protocol. Anything that satisfies it works —
   raw callables, LangChain chat models, LlamaIndex LLMs, LiteLLM, a mock for tests.

2. **Separation of concerns.**
   Planning, generation, and paraphrasing become distinct roles. A single model can
   fill all three, but the agent asks for each role by name.

3. **Backward compatibility.**
   `ModelAgentLLM`, `OllamaLLM`, and `HuggingFaceLLM` are kept as thin shims
   over the new adapters. Existing user code does not break.

4. **Zero new required dependencies.**
   All adapter imports are guarded by `try/except ImportError`. The library still
   installs and runs with only `httpx` if the user wants Ollama/HuggingFace.

---

## Target Architecture

```
securerag/
├── llm/
│   ├── __init__.py          re-exports everything; backward compat aliases live here
│   ├── base.py              SecureRAGLLM protocol + LLMRoles dataclass
│   ├── fallback.py          DeterministicLLM  (no network, always works)
│   ├── adapters/
│   │   ├── openai.py        OpenAIAdapter     (openai >= 1.0)
│   │   ├── anthropic.py     AnthropicAdapter  (anthropic >= 0.20)
│   │   ├── langchain.py     LangChainAdapter  (any BaseChatModel / BaseLLM)
│   │   ├── llamaindex.py    LlamaIndexAdapter (any LlamaIndex LLM)
│   │   ├── litellm.py       LiteLLMAdapter    (100+ providers via litellm)
│   │   ├── ollama.py        OllamaAdapter     (replaces current _ollama_generate)
│   │   └── huggingface.py   HuggingFaceAdapter(replaces current _huggingface_generate)
│   └── roles.py             Planner, Generator, Paraphraser role classes
```

The flat `llm.py` is replaced by this package. The existing `__init__.py` exports are
preserved unchanged through `securerag/llm/__init__.py`.

---

## Phase 1 — The Protocol (`llm/base.py`)

Define what SecureRAG actually needs from an LLM. It is intentionally minimal:

```python
# securerag/llm/base.py

from __future__ import annotations
from typing import Protocol, runtime_checkable


@runtime_checkable
class SecureRAGLLM(Protocol):
    """
    Minimal contract any LLM must satisfy to work with SecureRAG.

    Implementors: return None to signal unavailability; the agent
    will fall back to DeterministicLLM automatically.
    """

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str | None:
        """
        Generate a completion for `prompt`.
        Return None if the model is unreachable or unavailable.
        """
        ...

    async def acomplete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str | None:
        """Async variant. Default implementation wraps complete() in a thread."""
        ...
```

Any object with a `complete(prompt, ...)` method satisfies this protocol — including
a simple lambda for tests:

```python
# Tests need zero ceremony
llm = lambda prompt, **_: "ANSWER"
agent = SecureRAGAgent.from_config(config, corpus, llm=llm)
```

---

## Phase 2 — Role Separation (`llm/roles.py`)

Extract the three responsibilities currently fused in `ModelAgentLLM` into
composable role objects. Each role takes a `SecureRAGLLM` and adds its own
prompt templates and fallback logic.

```python
# securerag/llm/roles.py

from securerag.llm.base import SecureRAGLLM
from securerag.llm.fallback import DeterministicLLM


class Planner:
    """
    Decides whether to retrieve more context or generate a final answer.
    Maps to the current `ModelAgentLLM.decide()` method.
    """

    SYSTEM = (
        "You are a retrieval planner for multi-turn RAG. "
        "Output only strict JSON: "
        '{"action": "ANSWER"|"RETRIEVE", "sub_query": "..."}.'
    )

    def __init__(self, llm: SecureRAGLLM, fallback: "Planner | None" = None):
        self._llm = llm
        self._fallback = fallback or DeterministicPlanner()

    def decide(self, query: str, context, round_n: int) -> "LLMDecision":
        raw = self._llm.complete(
            self._build_prompt(query, context, round_n),
            system=self.SYSTEM,
            temperature=0.0,
        )
        if raw:
            decision = self._parse(raw, query, context, round_n)
            if decision:
                return decision
        return self._fallback.decide(query, context, round_n)

    def _build_prompt(self, query, context, round_n) -> str: ...
    def _parse(self, raw, query, context, round_n) -> "LLMDecision | None": ...


class Generator:
    """
    Synthesizes a final answer from retrieved context.
    Maps to the current `ModelAgentLLM.generate()` method.
    """

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

    def _build_prompt(self, query, context) -> str: ...
    def _deterministic_answer(self, query, context) -> str: ...


class Paraphraser:
    """
    Rewrites decoy queries so they read as natural search queries.
    Maps to `ModelAgentLLM.paraphrase_decoys()`.
    Uses a higher temperature; a cheaper/faster model is appropriate here.
    """

    SYSTEM = "Rewrite each decoy as a plausible standalone search query."

    def __init__(self, llm: SecureRAGLLM):
        self._llm = llm

    def paraphrase(self, decoys: list[str], source_query: str) -> list[str]: ...


class LLMRoles:
    """
    Bundles the three roles. Supports using different models per role.

    Quick construction (one model for all roles):
        roles = LLMRoles.uniform(my_llm)

    Fine-grained construction (different model per role):
        roles = LLMRoles(
            planner=Planner(fast_llm),
            generator=Generator(strong_llm),
            paraphraser=Paraphraser(cheap_llm),
        )
    """

    def __init__(
        self,
        planner: Planner,
        generator: Generator,
        paraphraser: Paraphraser,
    ):
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
```

`SecureRAGAgent` receives `LLMRoles` instead of a raw LLM, but still accepts a raw
`SecureRAGLLM` via an auto-wrapping shim for backward compat.

---

## Phase 3 — Adapters (`llm/adapters/`)

Each adapter wraps a third-party LLM and exposes `complete()` / `acomplete()`.
All imports are lazy so missing optional deps don't crash the module.

### 3.1 LangChain

```python
# securerag/llm/adapters/langchain.py

class LangChainAdapter:
    """
    Wraps any LangChain BaseChatModel or BaseLLM.

    Usage:
        from langchain_openai import ChatOpenAI
        from securerag.llm import LangChainAdapter

        llm = LangChainAdapter(ChatOpenAI(model="gpt-4o-mini"))
        agent = SecureRAGAgent.from_config(config, corpus, llm=llm)
    """

    def __init__(self, model):
        # Accepts BaseChatModel, BaseLLM, or any object with .invoke()
        self._model = model

    def complete(self, prompt, *, system=None, temperature=0.2, max_tokens=512):
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            messages = []
            if system:
                messages.append(SystemMessage(content=system))
            messages.append(HumanMessage(content=prompt))
            response = self._model.invoke(messages)
            # Handle both ChatModel (AIMessage) and LLM (str) return types
            if hasattr(response, "content"):
                return response.content or None
            return str(response).strip() or None
        except Exception:
            return None

    async def acomplete(self, prompt, *, system=None, temperature=0.2, max_tokens=512):
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            messages = []
            if system:
                messages.append(SystemMessage(content=system))
            messages.append(HumanMessage(content=prompt))
            response = await self._model.ainvoke(messages)
            if hasattr(response, "content"):
                return response.content or None
            return str(response).strip() or None
        except Exception:
            return None
```

### 3.2 LlamaIndex

```python
# securerag/llm/adapters/llamaindex.py

class LlamaIndexAdapter:
    """
    Wraps any LlamaIndex LLM (llama_index.core.llms.LLM subclass).

    Usage:
        from llama_index.llms.openai import OpenAI
        from securerag.llm import LlamaIndexAdapter

        llm = LlamaIndexAdapter(OpenAI(model="gpt-4o-mini"))
    """

    def __init__(self, llm):
        self._llm = llm

    def complete(self, prompt, *, system=None, temperature=0.2, max_tokens=512):
        try:
            merged = f"{system}\n\n{prompt}" if system else prompt
            response = self._llm.complete(merged)
            return str(response).strip() or None
        except Exception:
            return None

    async def acomplete(self, prompt, *, system=None, temperature=0.2, max_tokens=512):
        try:
            merged = f"{system}\n\n{prompt}" if system else prompt
            response = await self._llm.acomplete(merged)
            return str(response).strip() or None
        except Exception:
            return None
```

### 3.3 LiteLLM (recommended for multi-provider)

```python
# securerag/llm/adapters/litellm.py

class LiteLLMAdapter:
    """
    Exposes 100+ providers (OpenAI, Anthropic, Cohere, Bedrock, Vertex, …)
    through a single adapter using the litellm library.

    Usage:
        from securerag.llm import LiteLLMAdapter

        # OpenAI
        llm = LiteLLMAdapter("gpt-4o-mini")

        # Anthropic
        llm = LiteLLMAdapter("claude-3-haiku-20240307")

        # Ollama (local)
        llm = LiteLLMAdapter("ollama/qwen3:0.6b")

        # AWS Bedrock
        llm = LiteLLMAdapter("bedrock/anthropic.claude-3-haiku-20240307-v1:0")
    """

    def __init__(self, model: str, **litellm_kwargs):
        self._model = model
        self._kwargs = litellm_kwargs

    def complete(self, prompt, *, system=None, temperature=0.2, max_tokens=512):
        try:
            import litellm
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            response = litellm.completion(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **self._kwargs,
            )
            return response.choices[0].message.content or None
        except Exception:
            return None

    async def acomplete(self, prompt, *, system=None, temperature=0.2, max_tokens=512):
        try:
            import litellm
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            response = await litellm.acompletion(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **self._kwargs,
            )
            return response.choices[0].message.content or None
        except Exception:
            return None
```

### 3.4 Native OpenAI / Anthropic adapters

Thin wrappers over the official SDKs for users who don't want LiteLLM:

```python
# securerag/llm/adapters/openai.py
class OpenAIAdapter:
    def __init__(self, model="gpt-4o-mini", *, api_key=None, **kwargs): ...
    def complete(self, prompt, *, system=None, temperature=0.2, max_tokens=512): ...

# securerag/llm/adapters/anthropic.py
class AnthropicAdapter:
    def __init__(self, model="claude-haiku-4-5-20251001", *, api_key=None, **kwargs): ...
    def complete(self, prompt, *, system=None, temperature=0.2, max_tokens=512): ...
```

### 3.5 Existing providers (backward-compat, no behavior change)

```python
# securerag/llm/adapters/ollama.py
class OllamaAdapter:
    """Replaces the _ollama_generate() logic from ModelAgentLLM."""
    def __init__(self, model="qwen3:0.6b", base_url=None, timeout_s=8.0, retries=1): ...
    def complete(self, prompt, *, system=None, temperature=0.2, max_tokens=512): ...

# securerag/llm/adapters/huggingface.py
class HuggingFaceAdapter:
    """Replaces the _huggingface_generate() logic from ModelAgentLLM."""
    def __init__(self, model, base_url=None, token=None, timeout_s=8.0, retries=1): ...
    def complete(self, prompt, *, system=None, temperature=0.2, max_tokens=512): ...
```

---

## Phase 4 — DeterministicLLM (`llm/fallback.py`)

The deterministic fallback is extracted from `ModelAgentLLM` into its own class
so it can be tested, replaced, or composed explicitly:

```python
# securerag/llm/fallback.py

class DeterministicLLM:
    """
    Zero-dependency fallback that never calls any network.
    Used when all configured providers are unavailable.
    Produces rule-based (not model-generated) decisions and answers.
    """

    def complete(self, prompt, *, system=None, temperature=0.2, max_tokens=512):
        # Always returns None → roles apply their own deterministic logic
        return None

    async def acomplete(self, *args, **kwargs):
        return None


class DeterministicPlanner:
    """Rule-based fallback for Planner when LLM is unavailable."""

    def decide(self, query, context, round_n) -> "LLMDecision":
        if round_n >= 2 and len(context) >= 3:
            return LLMDecision(should_answer=True)
        return LLMDecision(
            should_answer=False,
            sub_query=self._cot_sub_query(query, context, round_n),
        )

    def _cot_sub_query(self, query, context, round_n) -> str:
        # Extracted verbatim from current _fallback_cot_sub_query()
        ...
```

---

## Phase 5 — Backward Compatibility (`llm/__init__.py`)

The public API of `securerag.llm` is unchanged:

```python
# securerag/llm/__init__.py

from securerag.llm.base import SecureRAGLLM
from securerag.llm.roles import LLMRoles, Planner, Generator, Paraphraser
from securerag.llm.fallback import DeterministicLLM
from securerag.llm.adapters.ollama import OllamaAdapter
from securerag.llm.adapters.huggingface import HuggingFaceAdapter

# ── Backward-compatible shim ─────────────────────────────────────────────────
# ModelAgentLLM, OllamaLLM, HuggingFaceLLM continue to work unchanged.

class ModelAgentLLM:
    """
    Shim: wraps the new role/adapter system behind the old interface.
    Existing code that constructs ModelAgentLLM(provider="ollama") still works.
    """

    def __init__(self, model="qwen3:0.6b", provider="ollama", **kwargs):
        adapter = _build_adapter(model, provider, **kwargs)
        roles = LLMRoles.uniform(adapter)
        self._planner = roles.planner
        self._generator = roles.generator
        self._paraphraser = roles.paraphraser

    def decide(self, query, context, round_n):
        return self._planner.decide(query, context, round_n)

    def generate(self, query, context):
        return self._generator.generate(query, context)

    def paraphrase_decoy(self, decoy, source_query):
        return self._paraphraser.paraphrase([decoy], source_query)[0]

    def paraphrase_decoys(self, decoys, source_query):
        return self._paraphraser.paraphrase(decoys, source_query)


OllamaLLM = ModelAgentLLM       # aliases preserved
HuggingFaceLLM = ModelAgentLLM  # aliases preserved
```

---

## Phase 6 — `SecureRAGAgent` Integration

`SecureRAGAgent.__init__` accepts either a raw `SecureRAGLLM` or an `LLMRoles`:

```python
# securerag/agent.py  (diff)

class SecureRAGAgent:
    def __init__(self, llm, retriever, config):
        # Accept raw LLM → auto-wrap into uniform LLMRoles
        if isinstance(llm, LLMRoles):
            self._roles = llm
        else:
            from securerag.llm.roles import LLMRoles
            self._roles = LLMRoles.uniform(llm)
        self.llm = llm   # kept for backward compat
        ...

    def run(self, query):
        ...
        decision = self._roles.planner.decide(query, context, round_n)
        ...
        answer = self._roles.generator.generate(query, context)
        ...
```

---

## Migration Examples

### No change needed (existing code)

```python
# Works exactly as before
from securerag.llm import ModelAgentLLM

llm = ModelAgentLLM(model="qwen3:0.6b", provider="ollama")
agent = SecureRAGAgent.from_config(config, corpus, llm=llm)
```

### New: LangChain model

```python
from langchain_openai import ChatOpenAI
from securerag.llm import LangChainAdapter

llm = LangChainAdapter(ChatOpenAI(model="gpt-4o-mini", temperature=0))
agent = SecureRAGAgent.from_config(config, corpus, llm=llm)
```

### New: LiteLLM (any provider by string)

```python
from securerag.llm import LiteLLMAdapter

agent = SecureRAGAgent.from_config(
    config, corpus,
    llm=LiteLLMAdapter("claude-3-haiku-20240307"),
)
```

### New: Different model per role

```python
from langchain_openai import ChatOpenAI
from securerag.llm import LangChainAdapter, LLMRoles, Planner, Generator, Paraphraser

fast  = LangChainAdapter(ChatOpenAI(model="gpt-4o-mini"))
smart = LangChainAdapter(ChatOpenAI(model="gpt-4o"))

agent = SecureRAGAgent.from_config(
    config, corpus,
    llm=LLMRoles(
        planner=Planner(fast),          # cheap model for planning
        generator=Generator(smart),     # strong model for answers
        paraphraser=Paraphraser(fast),  # cheap model for decoys
    ),
)
```

### New: LlamaIndex

```python
from llama_index.llms.anthropic import Anthropic
from securerag.llm import LlamaIndexAdapter

llm = LlamaIndexAdapter(Anthropic(model="claude-haiku-4-5-20251001"))
agent = SecureRAGAgent.from_config(config, corpus, llm=llm)
```

### New: Minimal test stub

```python
# No external deps needed for unit tests
class EchoLLM:
    def complete(self, prompt, **_):
        return "ANSWER"
    async def acomplete(self, prompt, **_):
        return "ANSWER"

agent = SecureRAGAgent.from_config(config, corpus, llm=EchoLLM())
```

---

## Dependency Changes (`pyproject.toml`)

```toml
[project.optional-dependencies]
# existing
dev  = ["pytest>=8.0.0", "grpcio-tools>=1.78.0"]
rust = ["maturin>=1.6.0"]

# new optional groups — none are required for core install
openai     = ["openai>=1.0"]
anthropic  = ["anthropic>=0.20"]
langchain  = ["langchain-core>=0.2", "langchain-openai>=0.1"]
llamaindex = ["llama-index-core>=0.10"]
litellm    = ["litellm>=1.40"]      # enables 100+ providers
```

Core `dependencies` in `[project]` stays unchanged — `httpx` is the only LLM
dependency required for the default Ollama/HuggingFace path.

---

## File Map

```
securerag/
├── llm/                         REPLACES llm.py
│   ├── __init__.py              Public API + backward-compat shims
│   ├── base.py                  SecureRAGLLM protocol (no deps)
│   ├── fallback.py              DeterministicLLM, DeterministicPlanner
│   ├── roles.py                 Planner, Generator, Paraphraser, LLMRoles
│   └── adapters/
│       ├── __init__.py
│       ├── ollama.py            Extracts current _ollama_generate logic
│       ├── huggingface.py       Extracts current _huggingface_generate logic
│       ├── openai.py            New — openai>=1.0
│       ├── anthropic.py         New — anthropic>=0.20
│       ├── langchain.py         New — langchain-core
│       ├── llamaindex.py        New — llama-index-core
│       └── litellm.py           New — litellm (umbrella)
│
├── llm.py                       DELETED (replaced by llm/ package)
│                                (keep as re-export stub for one release cycle)
│
│   (all other files unchanged)
├── agent.py                     Minor: accept LLMRoles | SecureRAGLLM
├── retriever.py                 Minor: call _roles.paraphraser instead of llm
```

---

## Sequencing

### Sprint A — Foundation (no behavior change)
1. Create `llm/base.py` — `SecureRAGLLM` protocol
2. Create `llm/fallback.py` — extract deterministic logic from `ModelAgentLLM`
3. Create `llm/adapters/ollama.py` and `llm/adapters/huggingface.py` — extract HTTP logic
4. Create `llm/__init__.py` — shim `ModelAgentLLM` over new adapters
5. Delete `llm.py`, add re-export stub
6. **All existing tests pass without modification**

### Sprint B — Roles
1. Create `llm/roles.py` — `Planner`, `Generator`, `Paraphraser`, `LLMRoles`
2. Update `agent.py` to accept `LLMRoles | SecureRAGLLM`
3. Update `retriever.py` to use `_roles.paraphraser` when available
4. Write role unit tests with `EchoLLM` (zero deps)

### Sprint C — Popular library adapters
1. `langchain.py`, `llamaindex.py`, `litellm.py`, `openai.py`, `anthropic.py`
2. Integration tests (skipped if optional dep absent)
3. Update `pyproject.toml` optional groups
4. Update README and quickstart examples

---

## What Stays Unchanged

- `securerag.llm.ModelAgentLLM` — still importable, same constructor signature
- `securerag.llm.OllamaLLM` — still importable
- `securerag.llm.HuggingFaceLLM` — still importable
- `LLMDecision` dataclass — same fields
- All Rust backend, gRPC, budget, and protocol code — untouched
- `PrivacyConfig.llm_*` fields — no changes to config shape
- The deterministic fallback behavior — same rules, just in `fallback.py`