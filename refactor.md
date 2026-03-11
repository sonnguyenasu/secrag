# SecureRAG Refactor Plan
## From a DP-RAG Framework to a Privacy-Aware Computation Framework

---

## Vision

The current framework conflates two things: a specific mechanism (differential privacy) with the
general problem it solves (resource-bounded privacy guarantees). The refactor separates them.

**Current framing:** SecureRAG is a DP-RAG framework where privacy = epsilon budget.

**Target framing:** SecureRAG is a privacy-aware retrieval framework where privacy is a
resource management problem. DP is one policy for how that resource depletes. HE noise budgets,
PIR query counts, and obfuscation distinguishability budgets are others. Researchers bring the
policy; the framework handles the tracking, composition, and enforcement.

The design touchstone throughout is PyTorch: every operation has a Python reference
implementation, accelerators (Rust backend) are optional drop-ins, and the Python layer is
expressive enough that researchers never need to go below it.

---

## What Changes and What Stays

**Stays completely unchanged:**
- Rust backend and gRPC service (performance layer, not touched)
- Proto definitions
- `ModelAgentLLM` / `OllamaLLM` / `HuggingFaceLLM` (thin, adequate)
- `Document`, `RawDocument`, `CorpusMeta` models
- `EncryptedSchemePlugin` registry pattern (already correct)
- Existing protocol behavior for BASELINE, OBFUSCATION, DIFF_PRIVACY, ENCRYPTED_SEARCH, PIR

**Refactored significantly:**
- `BudgetManager` → generalized `Budget` + typed `Cost` objects
- `PrivacyConfig` → slimmed down, mechanism-specific config moves to mechanism classes
- `PrivacyProtocol` → gains `required_budget_types` property
- `PrivacyRetriever` → gains `required_dp` flag pattern on corpus/query
- `CorpusBuilder` → gains local-first (no-backend) build path
- `DPMechanismPlugin` → merged into the new `BudgetMechanism` abstraction

**Added new:**
- `Cost` and `BudgetMechanism` base classes
- `PrivacyContext` auto-tracking context manager
- `required_budget` flag on data objects
- Local `EmbeddingIndex` with numpy fallback
- `securerag.benchmarks` standard dataset loaders

---

## Phase 1 — Generalize the Budget

> **Goal:** `BudgetManager` works for any finite resource, not just RDP epsilon.
> **Touches:** `budget.py`, `dp_mechanism.py`, `builtin_mechanisms.py`
> **Backward compatible:** yes — existing `BudgetManager(config)` call still works

### 1.1 Typed Cost objects

Currently budget costs are raw `float` (sigma). This prevents correct composition across
mechanisms — you can't tell if two floats came from Gaussian or Laplace.

```python
# securerag/cost.py  (new file)

from dataclasses import dataclass, field
from typing import Any

class Cost:
    """
    Base class for a privacy cost incurred by one operation.
    Subclasses carry mechanism-specific parameters needed for composition.
    """
    mechanism: str = "unknown"

    def __add__(self, other: "Cost") -> "Cost":
        raise NotImplementedError(
            f"No composition rule defined between {type(self)} and {type(other)}. "
            "Register a composition hook or use the same mechanism for all operations."
        )

@dataclass
class RDPCost(Cost):
    """Cost from a Gaussian or sub-Gaussian mechanism, tracked per Rényi order."""
    mechanism: str = "rdp"
    orders: list[float] = field(default_factory=lambda: [2., 4., 8., 16., 32.])
    values: list[float] = field(default_factory=list)   # one per order

    def __add__(self, other: "RDPCost") -> "RDPCost":
        assert self.orders == other.orders, "RDP orders must match to compose"
        return RDPCost(
            orders=self.orders,
            values=[a + b for a, b in zip(self.values, other.values)]
        )

@dataclass
class PureDPCost(Cost):
    """Cost from a Laplace mechanism — composes by simple addition."""
    mechanism: str = "pure_dp"
    epsilon: float = 0.0

    def __add__(self, other: "PureDPCost") -> "PureDPCost":
        return PureDPCost(epsilon=self.epsilon + other.epsilon)

@dataclass
class CountCost(Cost):
    """Cost for mechanisms with a hard query limit (PIR, k-anonymity)."""
    mechanism: str = "count"
    count: int = 1

    def __add__(self, other: "CountCost") -> "CountCost":
        return CountCost(count=self.count + other.count)

@dataclass
class HENoiseCost(Cost):
    """Noise budget consumed by homomorphic encryption operations."""
    mechanism: str = "he_noise"
    noise_bits: int = 0

    def __add__(self, other: "HENoiseCost") -> "HENoiseCost":
        return HENoiseCost(noise_bits=self.noise_bits + other.noise_bits)
```

### 1.2 BudgetMechanism replaces DPMechanismPlugin

`DPMechanismPlugin` has two responsibilities today: adding noise to embeddings and computing
RDP cost. The refactor makes this the general interface for *any* mechanism.

```python
# securerag/mechanism.py  (replaces dp_mechanism.py)

from abc import ABC, abstractmethod
from securerag.cost import Cost

class BudgetMechanism(ABC):
    """
    A privacy mechanism defines:
      - how noise is applied to data
      - what Cost object one application produces
      - how a Cost converts to a (epsilon, delta)-DP guarantee for reporting

    Researchers implement this to plug in any mechanism.
    DPMechanismPlugin is kept as a backward-compatible alias.
    """

    @abstractmethod
    def apply(self, data: list[float], sensitivity: float, **kwargs) -> list[float]:
        """Apply the mechanism's noise to data. Returns noised data."""
        ...

    @abstractmethod
    def cost(self, sensitivity: float, **kwargs) -> Cost:
        """Return the Cost incurred by one application of this mechanism."""
        ...

    def to_approx_dp(self, accumulated_cost: Cost, delta: float) -> float:
        """
        Convert accumulated cost to an (epsilon, delta)-DP guarantee.
        Used for reporting and budget exhaustion checks.
        Default implementation raises — mechanisms must implement this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement to_approx_dp() "
            "to support budget exhaustion checking."
        )

    # --- backward compatibility shim ---
    def noise(self, embedding, sigma, *, query=""):
        """DPMechanismPlugin-compatible interface. Calls apply() internally."""
        return self.apply(embedding, sensitivity=sigma)

    def rdp_cost(self, sigma, alpha):
        """DPMechanismPlugin-compatible interface."""
        c = self.cost(sensitivity=sigma)
        idx = c.orders.index(alpha) if hasattr(c, "orders") else 0
        return c.values[idx] if hasattr(c, "values") else 0.0

    # --- registry (same pattern as DPMechanismPlugin) ---
    _REGISTRY: dict[str, "BudgetMechanism"] = {}

    @classmethod
    def register(cls, name: str, instance: "BudgetMechanism") -> None:
        cls._REGISTRY[name.lower()] = instance

    @classmethod
    def get(cls, name: str) -> "BudgetMechanism":
        if name.lower() not in cls._REGISTRY:
            raise KeyError(f"No mechanism '{name}'. Available: {sorted(cls._REGISTRY)}")
        return cls._REGISTRY[name.lower()]

# Backward compatibility
DPMechanismPlugin = BudgetMechanism
```

### 1.3 Generalized Budget

```python
# securerag/budget.py  (refactored)

from securerag.cost import Cost
from securerag.errors import BudgetExhaustedError

class Budget:
    """
    A finite resource that depletes through computation.

    Mechanism-agnostic: the composition rule and exhaustion check are
    injected by the mechanism, not hardcoded.

    Backward-compatible: BudgetManager(config) still works via the
    class method Budget.from_config(config).
    """

    def __init__(
        self,
        total: Cost,
        mechanism: "BudgetMechanism",
        delta: float = 1e-5,
    ):
        self._total     = total
        self._spent     = type(total)()       # zero-value of same Cost type
        self._mechanism = mechanism
        self._delta     = delta
        self._round     = 0
        self._ledger: list[tuple[int, Cost]] = []

    def can_consume(self, cost: Cost) -> bool:
        projected = self._spent + cost
        try:
            eps = self._mechanism.to_approx_dp(projected, self._delta)
            limit = self._mechanism.to_approx_dp(self._total, self._delta)
            return eps <= limit
        except NotImplementedError:
            # Mechanisms without to_approx_dp fall back to direct cost comparison
            return self._direct_compare(projected, self._total)

    def consume(self, cost: Cost) -> None:
        if not self.can_consume(cost):
            raise BudgetExhaustedError(str(self._spent), str(self._total))
        self._spent = self._spent + cost
        self._round += 1
        self._ledger.append((self._round, self._spent))

    @property
    def spent(self) -> float:
        try:
            return self._mechanism.to_approx_dp(self._spent, self._delta)
        except NotImplementedError:
            return float(str(self._spent))

    @property
    def remaining(self) -> float:
        total = self._mechanism.to_approx_dp(self._total, self._delta)
        return max(0.0, total - self.spent)

    def snapshot(self) -> dict:
        return {
            "spent":     self.spent,
            "remaining": self.remaining,
            "rounds":    self._round,
            "ledger":    self._ledger,
            "mechanism": type(self._mechanism).__name__,
        }

    # --- factory methods for common mechanisms ---

    @classmethod
    def rdp(cls, epsilon: float, delta: float, sigma: float,
            mechanism: str = "gaussian") -> "Budget":
        """DP via Rényi composition. Backward-compatible with old BudgetManager."""
        from securerag.mechanism import BudgetMechanism
        from securerag.cost import RDPCost
        mech   = BudgetMechanism.get(mechanism)
        orders = [2., 4., 8., 16., 32.]
        total  = RDPCost(orders=orders, values=[epsilon] * len(orders))
        return cls(total=total, mechanism=mech, delta=delta)

    @classmethod
    def query_count(cls, max_queries: int) -> "Budget":
        """Hard query limit — for PIR and k-anonymity."""
        from securerag.mechanism import CountMechanism
        from securerag.cost import CountCost
        return cls(total=CountCost(count=max_queries), mechanism=CountMechanism())

    @classmethod
    def he_noise(cls, max_noise_bits: int) -> "Budget":
        """HE noise budget — depletes with each ciphertext operation."""
        from securerag.mechanism import HENoiseMechanism
        from securerag.cost import HENoiseCost
        return cls(total=HENoiseCost(noise_bits=max_noise_bits), mechanism=HENoiseMechanism())

    @classmethod
    def from_config(cls, config) -> "Budget":
        """Backward-compatible entry point. Replaces BudgetManager(config)."""
        from securerag.mechanism import BudgetMechanism
        import securerag.builtin_mechanisms  # noqa
        return cls.rdp(
            epsilon=config.epsilon,
            delta=config.delta,
            sigma=config.noise_std,
            mechanism=config.dp_mechanism,
        )

# Backward-compatible alias
BudgetManager = Budget
```

---

## Phase 2 — The required_budget Flag

> **Goal:** Data objects carry a flag declaring whether they require budget tracking,
> analogous to `requires_grad=True` in PyTorch.
> **Touches:** `models.py`, `config.py`
> **Backward compatible:** yes — flag defaults to False

### 2.1 Flag on data objects

```python
# securerag/models.py  (additions)

@dataclass(slots=True)
class Document:
    doc_id:          str
    text:            str
    score:           float = 0.0
    metadata:        dict  = field(default_factory=dict)
    required_budget: bool  = False      # True → per-doc budget tracked
    budget_type:     str   = "rdp"      # which Budget factory to use

@dataclass
class PrivateQuery:
    """
    A query that requires privacy protection before touching the server.
    Analogous to a tensor with requires_grad=True.
    """
    text:            str
    required_budget: bool  = True
    mechanism:       str   = "gaussian"   # noise mechanism for embedding
    epsilon:         float = 1.0          # per-query privacy budget
```

### 2.2 Corpus carries budget type declaration

Rather than `PrivacyProtocol` being the single source of truth for all privacy behavior,
the corpus declares which budget types it requires. The protocol becomes a shorthand
that sets sensible defaults.

```python
# securerag/corpus.py  (addition to SecureCorpus)

class SecureCorpus(ABC):
    ...
    # New: explicit budget type, independent of protocol routing
    budget_types: list[str] = field(default_factory=list)
    # e.g. ["rdp"] for DIFF_PRIVACY, ["count"] for PIR, ["he_noise"] for ENCRYPTED_SEARCH
```

---

## Phase 3 — PrivacyContext with Hook System

> **Goal:** Automatic budget tracking across a pipeline, with researcher hooks
> for overriding noise injection and composition.
> **Touches:** new file `context.py`
> **Backward compatible:** purely additive

```python
# securerag/context.py  (new file)

from contextlib import contextmanager
from securerag.cost import Cost
from securerag.budget import Budget

class PrivacyContext:
    """
    Tracks all privacy costs incurred within a with-block.

    Usage:
        with PrivacyContext() as ctx:
            docs   = retriever.retrieve(query)
            answer = llm.generate(query, docs)
        print(ctx.snapshot())

    Hooks:
        @ctx.register_noise_hook("encode")
        def my_noise(embedding, config, budget_state):
            ...
            return noised_embedding, cost

        @ctx.register_budget_hook("retrieve")
        def my_budget(docs, config, corpus_budgets):
            ...
            return total_cost

        @ctx.register_composition_hook
        def my_compose(cost_a, cost_b):
            return my_zcdp_compose(cost_a, cost_b)
    """

    def __init__(self, strict: bool = True):
        """
        strict=True  → raise immediately on incompatible cost composition
        strict=False → warn and attempt best-effort composition (unsafe_compose)
        """
        self._strict             = strict
        self._budgets:   dict[str, Budget]   = {}
        self._noise_hooks:  dict[str, list]  = {}
        self._budget_hooks: dict[str, list]  = {}
        self._composition_hook               = None
        self._active                         = False

    # --- context manager ---

    def __enter__(self):
        self._active = True
        return self

    def __exit__(self, *args):
        self._active = False

    # --- hook registration ---

    def register_noise_hook(self, operation: str):
        """
        Decorator. Hook fires before noise is injected.

        Hook signature:
            fn(embedding: list[float], config, budget_state: dict)
                -> tuple[list[float], Cost]
        """
        def decorator(fn):
            self._noise_hooks.setdefault(operation, []).append(fn)
            return fn
        return decorator

    def register_budget_hook(self, operation: str):
        """
        Decorator. Hook fires after documents are retrieved.

        Hook signature:
            fn(docs: list[Document], config, corpus_budgets: dict[str, Budget])
                -> Cost
        """
        def decorator(fn):
            self._budget_hooks.setdefault(operation, []).append(fn)
            return fn
        return decorator

    def register_composition_hook(self, fn):
        """
        Override how two Cost objects are composed.

        Hook signature:
            fn(cost_a: Cost, cost_b: Cost) -> Cost

        Use this to implement zCDP, f-DP, or any composition
        rule the built-in Cost.__add__ doesn't support.
        """
        self._composition_hook = fn
        return fn

    # --- internal charge interface (called by retrievers/corpus) ---

    def charge(self, budget_key: str, cost: Cost) -> None:
        """Called by framework internals to deduct cost from a named budget."""
        if not self._active:
            return
        if budget_key not in self._budgets:
            return   # no budget registered for this key → silently skip
        budget = self._budgets[budget_key]
        if self._composition_hook:
            # researcher-provided composition
            budget._spent = self._composition_hook(budget._spent, cost)
        else:
            budget.consume(cost)

    def register_budget(self, key: str, budget: Budget) -> None:
        self._budgets[key] = budget

    # --- reporting ---

    def snapshot(self) -> dict:
        return {key: b.snapshot() for key, b in self._budgets.items()}
```

### 3.1 How retrievers integrate with PrivacyContext

Retrievers gain a `_ctx` attribute. When a context is active, operations charge it
automatically. When no context is active, behavior is identical to today.

```python
# securerag/retriever.py  (additions to PrivacyRetriever base)

class PrivacyRetriever(ABC):
    ...
    _ctx: PrivacyContext | None = None

    def with_context(self, ctx: PrivacyContext) -> "PrivacyRetriever":
        """Attach a PrivacyContext. Returns self for chaining."""
        self._ctx = ctx
        return self

    def _charge(self, cost: Cost) -> None:
        if self._ctx:
            self._ctx.charge(self.config.protocol.name, cost)
        else:
            # fall back to existing BudgetManager path
            self.budget.consume_cost(cost)
```

---

## Phase 4 — Local-First Corpus (No Backend Required)

> **Goal:** Researchers can run experiments without starting a backend server.
> The Rust backend becomes an optional accelerator, not a hard dependency.
> **Touches:** `corpus.py`, new `local_index.py`
> **Backward compatible:** yes — existing build() still routes to backend

```python
# securerag/local_index.py  (new file)

import math

class LocalEmbeddingIndex:
    """
    Pure-Python ANN fallback. No server, no Rust required.
    Intended for prototyping and testing — not production scale.
    """

    def __init__(self, chunks: list[dict]):
        self._chunks = chunks
        # Simple TF-IDF vectors as stand-in for embeddings when no encoder available
        self._index  = self._build(chunks)

    def _build(self, chunks):
        # Build term-frequency vectors
        vocab = {}
        for i, c in enumerate(chunks):
            for tok in c["text"].lower().split():
                vocab.setdefault(tok, set()).add(i)
        return vocab

    def search(self, query: str, top_k: int) -> list[dict]:
        q_toks = query.lower().split()
        scores = {}
        for tok in q_toks:
            for idx in self._vocab.get(tok, []):
                scores[idx] = scores.get(idx, 0) + 1
        ranked = sorted(scores, key=lambda i: scores[i], reverse=True)[:top_k]
        return [self._chunks[i] for i in ranked]

    def search_by_embedding(self, embedding: list[float], top_k: int) -> list[dict]:
        """Cosine similarity over stored embeddings (if available)."""
        if not self._chunks or "embedding" not in self._chunks[0]:
            raise ValueError("Chunks have no embeddings. Use search() for text-based retrieval.")
        def cosine(a, b):
            dot = sum(x*y for x,y in zip(a,b))
            return dot / (math.sqrt(sum(x**2 for x in a)) * math.sqrt(sum(x**2 for x in b)) + 1e-9)
        scored = sorted(range(len(self._chunks)),
                        key=lambda i: cosine(embedding, self._chunks[i]["embedding"]),
                        reverse=True)
        return [self._chunks[i] for i in scored[:top_k]]


# securerag/corpus.py  (additions to CorpusBuilder)

class CorpusBuilder:
    ...
    def build_local(self, *, use_rust_if_available: bool = True) -> SecureCorpus:
        """
        Build corpus without a running backend server.

        If Rust extension is available and use_rust_if_available=True,
        delegates to the Rust bridge for fast ANN search.
        Otherwise falls back to LocalEmbeddingIndex (pure Python).

        This is the preferred entry point for research prototyping.
        """
        chunks = self._local_chunk(self._docs, self._chunk_size, self._overlap)
        if self._sanitize:
            chunks = self._local_sanitize(chunks)

        try:
            if use_rust_if_available:
                from securerag.rust_backend import RustLocalBackend
                # existing fast path
                return self._build_with_rust(chunks)
        except ImportError:
            pass

        # Pure Python fallback
        index = LocalEmbeddingIndex(chunks)
        meta  = CorpusMeta(
            doc_count=len(chunks),
            chunk_size=self._chunk_size,
            overlap=self._overlap,
            protocol=self._protocol.name,
        )
        return GenericCorpus(
            protocol=self._protocol,
            meta=meta,
            index_id="local",
            extras={"local_index": index},
        )
```

---

## Phase 5 — Standard Benchmarks

> **Goal:** Reproducible experiments without reimplementing data pipelines.
> Same data Koga et al. and the MURAG paper used, same preprocessing.
> **Touches:** new `securerag/benchmarks/` subpackage
> **Backward compatible:** purely additive

```python
# securerag/benchmarks/__init__.py

from securerag.benchmarks.nq      import NaturalQuestions
from securerag.benchmarks.trivia  import TriviaQA
from securerag.benchmarks.loaders import load_wikipedia_corpus

# Usage:
#   corpus, queries = NaturalQuestions.load(split="dev", n=100)
#   corpus, queries = TriviaQA.load(split="test", n=100)
#   wiki_corpus     = load_wikipedia_corpus(subset="2018-12")
```

Each benchmark loader returns `(SecureCorpus, list[QueryRecord])` where `QueryRecord`
is a new model:

```python
@dataclass
class QueryRecord:
    question:      str
    answers:       list[str]       # acceptable correct answers
    doc_ids:       list[str]       # gold document ids if available
    required_budget: bool = False  # set True to opt into PrivacyContext tracking
```

---

## Phase 6 — Protocol Metadata Cleanup

> **Goal:** `PrivacyProtocol` declares its budget types explicitly rather than
> encoding behavior implicitly in `requires_budget`.
> **Touches:** `protocol.py`
> **Backward compatible:** yes — existing properties kept

```python
# securerag/protocol.py  (additions)

class PrivacyProtocol(Enum):
    BASELINE         = auto()
    OBFUSCATION      = auto()
    DIFF_PRIVACY     = auto()
    ENCRYPTED_SEARCH = auto()
    PIR              = auto()

    @property
    def budget_types(self) -> list[str]:
        """
        Which budget types this protocol consumes.
        Drives PrivacyContext budget registration.
        """
        return {
            PrivacyProtocol.BASELINE:         [],
            PrivacyProtocol.OBFUSCATION:      ["distinguishability"],  # currently untracked
            PrivacyProtocol.DIFF_PRIVACY:     ["rdp"],
            PrivacyProtocol.ENCRYPTED_SEARCH: ["he_noise"],            # currently untracked
            PrivacyProtocol.PIR:              ["count"],
        }[self]

    @property
    def requires_budget(self) -> bool:
        """Backward-compatible. Use budget_types for new code."""
        return len(self.budget_types) > 0
```

---

## File Map

```
securerag/
├── cost.py               NEW   — Cost, RDPCost, PureDPCost, CountCost, HENoiseCost
├── mechanism.py          NEW   — BudgetMechanism (replaces dp_mechanism.py)
│                                 DPMechanismPlugin aliased for backward compat
├── budget.py             REFACTOR — Budget (generalized BudgetManager)
│                                    BudgetManager aliased for backward compat
├── context.py            NEW   — PrivacyContext, hook registration
├── local_index.py        NEW   — LocalEmbeddingIndex (pure Python fallback)
├── models.py             EXTEND — required_budget flag on Document, new PrivateQuery
├── protocol.py           EXTEND — budget_types property
├── corpus.py             EXTEND — build_local() no-server path
├── retriever.py          EXTEND — with_context(), _charge() integration
├── benchmarks/           NEW
│   ├── __init__.py
│   ├── nq.py             NaturalQuestions loader
│   ├── trivia.py         TriviaQA loader
│   └── loaders.py        Wikipedia corpus loader, QueryRecord model
│
│   (unchanged)
├── config.py
├── agent.py
├── llm.py
├── retrievers.py
├── builtin_mechanisms.py (update imports to use mechanism.py)
├── builtin_schemes.py
├── scheme_plugin.py
├── backend_client.py
├── rust_backend.py
├── sim_server.py
├── errors.py
└── proto/
```

---

## Sequencing

### Sprint 1 — Foundation (no behavior change, pure refactor)
1. Write `cost.py` with all Cost types
2. Write `mechanism.py`, alias `DPMechanismPlugin`
3. Refactor `budget.py` to `Budget` with `from_config()` shim, alias `BudgetManager`
4. Update `builtin_mechanisms.py` to extend `BudgetMechanism` and emit typed `Cost`
5. All existing tests pass unchanged

### Sprint 2 — Context and hooks (additive)
1. Write `context.py` with `PrivacyContext`
2. Add `with_context()` + `_charge()` to `PrivacyRetriever`
3. Add `required_budget` flag to `Document` and `PrivateQuery` model
4. Wire `DiffPrivacyRetriever` to charge context when active
5. Write tests for hook override paths

### Sprint 3 — Local-first corpus (additive)
1. Write `local_index.py`
2. Add `build_local()` to `CorpusBuilder` with Rust-if-available dispatch
3. All existing tests still pass (build() unchanged)
4. Write tests that run without any backend server

### Sprint 4 — Protocol cleanup and benchmarks
1. Add `budget_types` to `PrivacyProtocol`
2. Wire `PrivacyContext` to auto-register budgets based on `budget_types`
3. Write `securerag/benchmarks/` loaders
4. Add `QueryRecord` to models

---

## What This Enables for Researchers

After this refactor, a researcher implementing a new private retrieval mechanism needs to:

1. Implement `BudgetMechanism` — define `apply()`, `cost()`, `to_approx_dp()`
2. Register it — `BudgetMechanism.register("my_mechanism", MyMechanism())`
3. Use it — `PrivacyConfig(dp_mechanism="my_mechanism")`

Budget tracking, composition, and exhaustion checking are automatic. To override
any part of the pipeline, register a hook. To compare against baselines, use the
standard benchmark loaders. To prototype without infrastructure, use `build_local()`.

None of this requires touching the Rust backend, the gRPC layer, or any existing
protocol implementation.

Compatibility note: ENCRYPTED_SEARCH protocol uses the encrypted_search plugin flow.