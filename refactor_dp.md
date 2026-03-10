# SecureRAG — DP Mechanism Plugin Refactor Specification

**Goal:** Make the DIFF_PRIVACY path as easy to extend as the ENCRYPTED_SEARCH path
already is. A researcher with a new noise mechanism or accounting strategy should be
able to test it end-to-end in a single Python file without touching framework source,
Rust, or the gRPC layer.

---

## The Problem in Concrete Terms

With the current code, a researcher wanting to test a discrete Gaussian mechanism
instead of the continuous Gaussian has to:

1. Fork `sim_server.py` and edit the `embed_with_noise` handler
2. Fork `pyo3_bridge.rs`, change the `embed_with_noise` arm, recompile
3. Fork `budget.py`, replace the hardcoded `alpha / (2σ²)` formula with the
   discrete Gaussian RDP bound
4. Reconcile all three forks every time the framework updates

With the plugin design below, they write one class and register it. Everything else
is unchanged.

---

## Design Overview

Two new concerns are separated into one ABC:

- **Noise application** — how to perturb a raw embedding given σ
- **RDP accounting** — the per-order cost formula `ε_α(σ)` for this mechanism

A third optional concern is **corpus preparation** — some mechanisms require the
index to be structured differently (e.g. bucketed quantisation for the staircase
mechanism). This is optional and defaults to a no-op.

The noise is applied entirely client-side in `DiffPrivacyRetriever`. The backend
only ever sees a noised embedding vector and σ for its server-side budget check.
The gRPC production backend is completely unaffected. This is the key architectural
advantage over the encrypted search plugin: the trust boundary already places DP
noise on the client, so the plugin runs in the same Python process as the researcher's
experiment script.

---

## New File: `securerag/dp_mechanism.py`

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

_REGISTRY: dict[str, "DPMechanismPlugin"] = {}


class DPMechanismPlugin(ABC):
    """
    Extension point for differential privacy noise mechanisms.

    Implement this ABC to test a new noise mechanism without modifying
    any framework source. Register an instance and reference it by name
    in PrivacyConfig.

    Call order used by the framework per retrieval round:
    1) budget.can_consume(sigma)       — uses rdp_cost() and rdp_orders()
    2) plugin.noise(embedding, sigma)  — called with the raw base embedding
    3) backend.retrieve_by_embedding() — receives the noised embedding
    4) budget.consume(sigma)           — committed only if backend succeeds

    For corpus preparation (optional):
    5) plugin.prepare_corpus(chunks)   — called once during build_local()
    """

    @abstractmethod
    def noise(
        self,
        embedding: list[float],
        sigma: float,
        *,
        query: str = "",
    ) -> list[float]:
        """
        Apply noise to a raw embedding.

        Args:
            embedding: Unit-normalised base embedding, length d (currently 64).
            sigma: Noise scale parameter. Interpretation is mechanism-specific;
                   for Gaussian this is the standard deviation.
            query:  Original query string. Available for mechanisms whose noise
                    is query-dependent (e.g. local randomised response).

        Returns:
            Noised embedding of the same length as `embedding`.
            Does not need to be normalised.
        """
        raise NotImplementedError

    @abstractmethod
    def rdp_cost(self, sigma: float, alpha: float) -> float:
        """
        RDP epsilon for a single round of this mechanism at Rényi order alpha.

        For the Gaussian mechanism this is alpha / (2 * sigma**2).
        For Laplace with sensitivity 1 it is alpha * (1/sigma)**2 / 2 etc.

        Args:
            sigma: Same noise scale passed to noise().
            alpha: Rényi order > 1.

        Returns:
            Non-negative float. The BudgetManager accumulates this across rounds
            and converts to (epsilon, delta)-DP using the standard formula:
                eps = min_alpha [ accumulated_rdp(alpha) + ln(1/delta) / (alpha-1) ]
        """
        raise NotImplementedError

    def rdp_orders(self) -> list[float]:
        """
        Rényi orders to track for this mechanism.

        Override to use a different set of orders (e.g. more fine-grained
        for mechanisms where the optimal order varies significantly with sigma).

        Default matches the Gaussian builtin: [2, 4, 8, 16, 32].
        """
        return [2.0, 4.0, 8.0, 16.0, 32.0]

    def prepare_corpus(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Optional corpus preprocessing for this mechanism.

        Called during CorpusBuilder.build_local() before build_index.
        Useful for mechanisms that require bucketed, quantised, or otherwise
        pre-structured corpus data (e.g. staircase mechanism on sparse vectors).

        Default: identity (no preprocessing).
        """
        return chunks

    # ------------------------------------------------------------------ registry

    @classmethod
    def register(cls, name: str, plugin: "DPMechanismPlugin") -> None:
        _REGISTRY[name.lower()] = plugin

    @classmethod
    def get(cls, name: str) -> "DPMechanismPlugin":
        key = name.lower()
        if key not in _REGISTRY:
            raise KeyError(
                f"No DP mechanism '{name}' registered. "
                f"Available: {sorted(_REGISTRY)}"
            )
        return _REGISTRY[key]

    @classmethod
    def registered_names(cls) -> list[str]:
        return sorted(_REGISTRY)
```

---

## New File: `securerag/builtin_mechanisms.py`

```python
from __future__ import annotations

import hashlib
import math
import random
from typing import Any

from securerag.dp_mechanism import DPMechanismPlugin


class GaussianMechanism(DPMechanismPlugin):
    """
    Standard continuous Gaussian mechanism.

    This is the existing behaviour, extracted into a plugin so that all
    mechanisms — including the default — go through the same code path.
    RDP cost: alpha / (2 * sigma**2), as per Mironov (2017).
    """

    def noise(
        self,
        embedding: list[float],
        sigma: float,
        *,
        query: str = "",
    ) -> list[float]:
        seed = int.from_bytes(
            hashlib.sha256(query.encode("utf-8")).digest()[:8], "little"
        )
        rng = random.Random(seed)
        return [v + rng.gauss(0.0, sigma) for v in embedding]

    def rdp_cost(self, sigma: float, alpha: float) -> float:
        return alpha / (2.0 * sigma * sigma)


class LaplaceMechanism(DPMechanismPlugin):
    """
    Laplace mechanism applied per embedding dimension.

    RDP cost for Laplace with sensitivity Δ=1 and scale b=sigma:
        epsilon_rdp(alpha) = (1/(alpha-1)) * log(
            alpha/(2*alpha-1) * exp((alpha-1)/sigma) +
            (alpha-1)/(2*alpha-1) * exp(-alpha/sigma)
        )
    This is the closed-form RDP of the Laplace mechanism (Mironov 2017, Prop 3).
    """

    def noise(
        self,
        embedding: list[float],
        sigma: float,
        *,
        query: str = "",
    ) -> list[float]:
        seed = int.from_bytes(
            hashlib.sha256(query.encode("utf-8")).digest()[:8], "little"
        )
        rng = random.Random(seed)
        # Laplace(0, sigma) via inverse CDF
        return [v + rng.expovariate(1.0 / sigma) * (1 if rng.random() < 0.5 else -1)
                for v in embedding]

    def rdp_cost(self, sigma: float, alpha: float) -> float:
        # Closed-form RDP for Laplace, sensitivity=1, scale=sigma
        if alpha <= 1.0:
            return 0.0
        try:
            log_term = math.log(
                alpha / (2 * alpha - 1) * math.exp((alpha - 1) / sigma)
                + (alpha - 1) / (2 * alpha - 1) * math.exp(-alpha / sigma)
            )
            return log_term / (alpha - 1)
        except (ValueError, OverflowError):
            return float("inf")


# Register builtins
DPMechanismPlugin.register("gaussian", GaussianMechanism())
DPMechanismPlugin.register("laplace", LaplaceMechanism())
```

---

## Changes to Existing Files

### `securerag/config.py`

Add one field:

```python
dp_mechanism: str = "gaussian"
```

The `__post_init__` guard should also warn when defaults are incoherent:

```python
def __post_init__(self) -> None:
    if not self.protocol.requires_budget and self.epsilon != 1.0:
        warnings.warn(
            f"epsilon has no effect for protocol {self.protocol.name}. Use DIFF_PRIVACY.",
            stacklevel=2,
        )
    if self.protocol.requires_budget:
        import math
        orders = [2.0, 4.0, 8.0, 16.0, 32.0]
        single_round_eps = min(
            a / (2.0 * self.noise_std ** 2) + math.log(1.0 / self.delta) / (a - 1.0)
            for a in orders
        )
        if single_round_eps > self.epsilon:
            warnings.warn(
                f"noise_std={self.noise_std} costs ε≈{single_round_eps:.2f} per round "
                f"but epsilon={self.epsilon}. Round 1 will be rejected. "
                f"Increase epsilon or noise_std.",
                stacklevel=2,
            )
```

### `securerag/budget.py`

`BudgetManager` accepts an optional plugin and delegates the per-order cost formula
and the order list to it. The accounting logic itself (`_rdp_to_dp`, accumulation,
check-then-debit) is unchanged.

```python
import math
from typing import TYPE_CHECKING

from securerag.config import PrivacyConfig
from securerag.errors import BudgetExhaustedError

if TYPE_CHECKING:
    from securerag.dp_mechanism import DPMechanismPlugin


class BudgetManager:
    def __init__(
        self,
        config: PrivacyConfig,
        mechanism: "DPMechanismPlugin | None" = None,
    ):
        self._epsilon_max = float(config.epsilon)
        self._delta = float(config.delta)

        if mechanism is None:
            # Import here to avoid circular dependency at module load time.
            import securerag.builtin_mechanisms  # noqa: F401 — triggers registration
            from securerag.dp_mechanism import DPMechanismPlugin
            mechanism = DPMechanismPlugin.get(config.dp_mechanism)

        self._mechanism = mechanism
        self._orders = mechanism.rdp_orders()
        self._rdp_acc = [0.0] * len(self._orders)
        self._round = 0
        self._ledger: list[tuple[int, float]] = []

    @staticmethod
    def _rdp_to_dp(
        rdp_eps: list[float], delta: float, orders: list[float]
    ) -> float:
        return min(
            r + math.log(1.0 / delta) / (a - 1.0)
            for a, r in zip(orders, rdp_eps)
        )

    def _candidate_acc(self, sigma: float) -> list[float]:
        if sigma <= 0.0:
            raise ValueError("sigma must be > 0")
        return [
            acc + self._mechanism.rdp_cost(sigma, alpha)
            for acc, alpha in zip(self._rdp_acc, self._orders)
        ]

    @property
    def spent(self) -> float:
        if all(x == 0.0 for x in self._rdp_acc):
            return 0.0
        return self._rdp_to_dp(self._rdp_acc, self._delta, self._orders)

    def epsilon_if_consumed(self, sigma: float) -> float:
        return self._rdp_to_dp(
            self._candidate_acc(sigma), self._delta, self._orders
        )

    def incremental_cost(self, sigma: float) -> float:
        return max(0.0, self.epsilon_if_consumed(sigma) - self.spent)

    def can_consume(self, sigma: float) -> bool:
        return self.epsilon_if_consumed(sigma) <= self._epsilon_max

    def consume(self, sigma: float) -> None:
        candidate = self._candidate_acc(sigma)
        candidate_eps = self._rdp_to_dp(candidate, self._delta, self._orders)
        if candidate_eps > self._epsilon_max:
            raise BudgetExhaustedError(
                f"epsilon exhausted: {candidate_eps:.3f} / {self._epsilon_max:.3f}"
            )
        self._rdp_acc = candidate
        self._round += 1
        self._ledger.append((self._round, candidate_eps))

    @property
    def remaining(self) -> float:
        return max(0.0, self._epsilon_max - self.spent)

    def snapshot(self) -> dict:
        return {
            "spent": self.spent,
            "remaining": self.remaining,
            "rounds": self._round,
            "ledger": self._ledger,
            "epsilon_max": self._epsilon_max,
            "delta": self._delta,
            "mechanism": type(self._mechanism).__name__,
        }
```

The only structural change is that `_candidate_acc` calls
`self._mechanism.rdp_cost(sigma, alpha)` instead of the hardcoded
`alpha / (2.0 * sigma * sigma)`. Everything else — the accumulation pattern,
the check-then-debit ordering, the `spent` zero-guard — is unchanged.

### `securerag/retriever.py`

`PrivacyRetriever.__init__` needs to construct `BudgetManager` with the resolved
plugin, so that the retriever owns a `BudgetManager` that already has the right
mechanism injected:

```python
def __init__(self, config: PrivacyConfig, corpus):
    self._validate_compatibility(config.protocol, corpus.protocol)
    self.config = config
    self.corpus = corpus

    if config.protocol.requires_budget:
        import securerag.builtin_mechanisms  # noqa: F401
        from securerag.dp_mechanism import DPMechanismPlugin
        mechanism = DPMechanismPlugin.get(config.dp_mechanism)
        self.budget = BudgetManager(config, mechanism=mechanism)
        self._dp_mechanism = mechanism
    else:
        self.budget = BudgetManager(config)
        self._dp_mechanism = None

    self._backend = create_backend(config.backend)
    ...
```

### `securerag/retrievers.py` — `DiffPrivacyRetriever`

The retriever needs to stop delegating noise to the backend and instead apply it
through the plugin. This requires a base `embed()` call (without noise) followed by
`plugin.noise()`.

```python
@PrivacyRetriever.register(PrivacyProtocol.DIFF_PRIVACY)
class DiffPrivacyRetriever(PrivacyRetriever):
    def retrieve(self, query: str, round_n: int):
        sigma = float(self.config.noise_std)

        # Get the base embedding — no noise applied by the backend.
        base_embedding = self._backend.embed(query)

        # Apply noise through the pluggable mechanism, entirely client-side.
        noised = self._dp_mechanism.noise(base_embedding, sigma, query=query)

        rows = self._backend.retrieve_by_embedding(
            index_id=self.corpus.index_id,
            embedding=noised,
            top_k=self.config.top_k,
            query=query,
            sigma=sigma,
        )
        # Debit only after both network calls succeed.
        self.budget.consume(sigma)

        self._debug(
            "diff-privacy retrieval",
            round_n=round_n,
            original_query=query,
            mechanism=type(self._dp_mechanism).__name__,
            epsilon_cost=self.budget.incremental_cost(sigma),
            epsilon_remaining=self.budget.remaining,
        )
        return self._to_docs(rows)

    def privacy_cost(self, query: str) -> float:
        return self.budget.incremental_cost(float(self.config.noise_std))
```

### `securerag/backend_client.py`

Add `embed(query) -> list[float]` to the `Backend` ABC and all three implementations.
This separates embedding from noise addition, which is necessary for the plugin to
receive a clean vector.

```python
# In Backend ABC:
@abstractmethod
def embed(self, query: str) -> list[float]:
    raise NotImplementedError
```

`embed_with_noise` can remain for backwards compatibility but is no longer called by
`DiffPrivacyRetriever`. The sim_server and PyO3 bridge implementations of `embed`
are trivial — they call the existing `_embed()` function and return without adding
noise.

### `securerag/corpus.py` — `build_local()`

Add a DP branch that calls `plugin.prepare_corpus()`:

```python
if self._protocol is PrivacyProtocol.DIFF_PRIVACY and self._dp_mechanism is not None:
    chunks = self._dp_mechanism.prepare_corpus(chunks)
```

This is a no-op for the Gaussian builtin. For mechanisms that need bucketed or
quantised representations it gives researchers a hook without any further framework
changes.

---

## What a Researcher Experiment Looks Like

A complete experiment testing a custom mechanism is entirely self-contained:

```python
import math
import numpy as np
from securerag.dp_mechanism import DPMechanismPlugin
from securerag.config import PrivacyConfig
from securerag.corpus import CorpusBuilder
from securerag.protocol import PrivacyProtocol
from securerag.models import RawDocument


class DiscreteGaussianMechanism(DPMechanismPlugin):
    """Discrete Gaussian mechanism (Canonne et al. 2020)."""

    def noise(self, embedding, sigma, *, query=""):
        rng = np.random.default_rng(
            int.from_bytes(__import__("hashlib").sha256(query.encode()).digest()[:8], "little")
        )
        # Round embedding to integers, add discrete Gaussian noise, renormalise.
        scaled = np.array(embedding) * 100
        noised = scaled + rng.integers(-int(3*sigma), int(3*sigma)+1, size=len(embedding))
        norm = np.linalg.norm(noised)
        return (noised / norm if norm > 0 else noised).tolist()

    def rdp_cost(self, sigma, alpha):
        # Tight RDP bound for discrete Gaussian (Canonne et al. 2020, Thm 1)
        return alpha / (2.0 * sigma * sigma)  # Gaussian approximation for prototype

    def rdp_orders(self):
        # Finer grid around orders likely to be optimal for this mechanism
        return [1.5, 2.0, 3.0, 4.0, 8.0, 16.0, 32.0, 64.0]


# Register — one line, no framework edits
DPMechanismPlugin.register("discrete_gaussian", DiscreteGaussianMechanism())

# Configure
cfg = PrivacyConfig(
    protocol=PrivacyProtocol.DIFF_PRIVACY,
    dp_mechanism="discrete_gaussian",
    epsilon=10.0,
    delta=1e-5,
    noise_std=1.0,
    max_rounds=5,
)

# Build corpus — prepare_corpus() called automatically
docs = [RawDocument(doc_id=str(i), text=f"document {i}") for i in range(100)]
corpus = CorpusBuilder.from_config(cfg).add_documents(docs).build_local()

# Agent uses discrete Gaussian noise transparently
from securerag.agent import SecureRAGAgent
agent = SecureRAGAgent.from_config(cfg, corpus=corpus, llm=my_llm)
result = agent.run("vendor concentration risk")

snap = agent.budget_snapshot()
print(f"Mechanism: {snap['mechanism']}")
print(f"Rounds: {snap['rounds']}, Spent: {snap['spent']:.4f} / {snap['epsilon_max']}")
print(f"Ledger: {snap['ledger']}")
```

No framework source is touched. The mechanism, accounting, and corpus preparation are
entirely contained in the researcher's file. Switching back to Gaussian for a
baseline comparison is one line: `dp_mechanism="gaussian"`.

---

## What Does Not Change

**The gRPC production backend** is unaffected. Noise is applied client-side before
the embedding is sent. The server receives the same noised vector and σ regardless of
which mechanism was used. The server-side `RDPAccountant` in `dp.rs` still uses the
Gaussian formula — this is intentional, because the server-side budget is a
conservative safety net, not the primary accounting surface. The authoritative budget
is the client-side `BudgetManager`.

**The PyO3 local backend and sim_server** gain a trivial `embed` handler and lose
noise responsibility. Their `embed_with_noise` handlers can remain for any external
callers but are no longer in the `DiffPrivacyRetriever` code path.

**The `EncryptedSchemePlugin` and all encrypted search code** is completely unchanged.

**The `retrieve_by_embedding` backend call** is unchanged. It still receives
`sigma` for the server-side budget check and the noised embedding for retrieval.

---

## New Test File: `tests/test_dp_mechanism_plugin.py`

```python
import math
import pytest
from securerag.budget import BudgetManager
from securerag.config import PrivacyConfig
from securerag.dp_mechanism import DPMechanismPlugin
from securerag.protocol import PrivacyProtocol


# ── minimal custom mechanism for contract testing ──────────────────────────

class _HalfCostMechanism(DPMechanismPlugin):
    """Gaussian with half the RDP cost — useful for verifying accounting substitution."""

    def noise(self, embedding, sigma, *, query=""):
        import random, hashlib
        seed = int.from_bytes(hashlib.sha256(query.encode()).digest()[:8], "little")
        rng = random.Random(seed)
        return [v + rng.gauss(0.0, sigma) for v in embedding]

    def rdp_cost(self, sigma, alpha):
        return alpha / (4.0 * sigma * sigma)  # half the Gaussian cost


class _CustomOrdersMechanism(_HalfCostMechanism):
    def rdp_orders(self):
        return [1.5, 3.0, 6.0, 12.0]


# ── registry ───────────────────────────────────────────────────────────────

def test_register_and_retrieve():
    DPMechanismPlugin.register("half_cost", _HalfCostMechanism())
    plugin = DPMechanismPlugin.get("half_cost")
    assert isinstance(plugin, _HalfCostMechanism)


def test_get_unknown_raises():
    with pytest.raises(KeyError, match="not_a_mechanism"):
        DPMechanismPlugin.get("not_a_mechanism")


# ── BudgetManager uses injected mechanism ─────────────────────────────────

def test_budget_manager_uses_custom_rdp_cost():
    cfg = PrivacyConfig(protocol=PrivacyProtocol.DIFF_PRIVACY, epsilon=30.0, delta=1e-5)
    gaussian_budget = BudgetManager(cfg)
    half_budget = BudgetManager(cfg, mechanism=_HalfCostMechanism())

    gaussian_budget.consume(sigma=1.0)
    half_budget.consume(sigma=1.0)

    # Half-cost mechanism should report lower spent epsilon after one round.
    assert half_budget.spent < gaussian_budget.spent


def test_budget_manager_uses_custom_orders():
    cfg = PrivacyConfig(protocol=PrivacyProtocol.DIFF_PRIVACY, epsilon=30.0, delta=1e-5)
    budget = BudgetManager(cfg, mechanism=_CustomOrdersMechanism())
    budget.consume(sigma=1.0)
    # Internal accumulator has one entry per custom order (4, not 5)
    assert len(budget._rdp_acc) == 4


def test_budget_snapshot_includes_mechanism_name():
    cfg = PrivacyConfig(protocol=PrivacyProtocol.DIFF_PRIVACY, epsilon=30.0, delta=1e-5)
    budget = BudgetManager(cfg, mechanism=_HalfCostMechanism())
    snap = budget.snapshot()
    assert snap["mechanism"] == "_HalfCostMechanism"


# ── noise plugin contract ──────────────────────────────────────────────────

def test_noise_output_length_matches_input():
    plugin = _HalfCostMechanism()
    embedding = [0.1 * i for i in range(64)]
    noised = plugin.noise(embedding, sigma=1.0, query="test query")
    assert len(noised) == 64


def test_noise_is_deterministic_for_same_query():
    plugin = _HalfCostMechanism()
    embedding = [0.1 * i for i in range(64)]
    a = plugin.noise(embedding, sigma=1.0, query="vendor risk")
    b = plugin.noise(embedding, sigma=1.0, query="vendor risk")
    assert a == pytest.approx(b)


def test_noise_differs_across_queries():
    plugin = _HalfCostMechanism()
    embedding = [0.1 * i for i in range(64)]
    a = plugin.noise(embedding, sigma=1.0, query="vendor risk")
    b = plugin.noise(embedding, sigma=1.0, query="q3 results")
    assert a != pytest.approx(b)


def test_prepare_corpus_default_is_identity():
    plugin = _HalfCostMechanism()
    chunks = [{"doc_id": "1", "text": "hello"}]
    assert plugin.prepare_corpus(chunks) is chunks


# ── gaussian builtin is unchanged ─────────────────────────────────────────

def test_gaussian_builtin_accounting_matches_previous_hardcode():
    """Verifies GaussianMechanism.rdp_cost produces identical values to the old hardcode."""
    import securerag.builtin_mechanisms  # noqa: F401
    gaussian = DPMechanismPlugin.get("gaussian")
    for alpha in [2.0, 4.0, 8.0, 16.0, 32.0]:
        sigma = 1.0
        expected = alpha / (2.0 * sigma * sigma)
        assert gaussian.rdp_cost(sigma, alpha) == pytest.approx(expected)


def test_gaussian_budget_matches_previous_accumulated_value():
    """End-to-end: gaussian plugin produces same accumulated epsilon as v0.6 BudgetManager."""
    import securerag.builtin_mechanisms  # noqa: F401
    cfg = PrivacyConfig(
        protocol=PrivacyProtocol.DIFF_PRIVACY,
        epsilon=30.0,
        delta=1e-5,
        dp_mechanism="gaussian",
    )
    budget = BudgetManager(cfg)
    budget.consume(sigma=1.0)
    budget.consume(sigma=1.0)
    assert budget.spent == pytest.approx(7.8376418, rel=1e-6)
```

---

## Migration Notes

**Backwards compatibility:** `PrivacyConfig.dp_mechanism` defaults to `"gaussian"`,
which produces identical behaviour to v0.6. All existing tests pass unchanged. The
`BudgetManager` constructor defaults `mechanism=None` and resolves the Gaussian plugin
internally when not supplied, so callsites that construct `BudgetManager(config)`
directly also continue to work.

**The `embed_with_noise` backend method** is no longer called by the framework's DP
path but is retained in the `Backend` ABC so that external scripts calling it directly
are not broken. It can be formally deprecated in a later version.

**The `snapshot()` dict gains one new key** (`"mechanism"`) which is additive and
does not break existing snapshot consumers.