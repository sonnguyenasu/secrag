# SecureRAG — Differential Privacy Analysis

**Scope:** Full review of the DIFF_PRIVACY protocol path across all three layers:
Python (`budget.py`, `retrievers.py`, `agent.py`), sim_server, and Rust
(`dp.rs`, `pyo3_bridge.rs`). The encrypted search refactor is not in scope here.

---

## Executive Summary

The DP subsystem has three independent budget trackers that use different accounting
methods and cannot communicate with each other. The Rust `RDPAccountant` and the
sim_server (fixed in v0.5) both do correct accumulated-RDP accounting. The Python
`BudgetManager` — which controls whether the agent is allowed to make the next
retrieval call — uses per-round summing instead. The two methods agree only on the
first round. By round 3 Python overcharges by 51–72% depending on σ, causing the
agent to stop retrieving long before the true budget is exhausted. A researcher
reading `budget_snapshot()` to understand how much privacy they spent will see
numbers that do not match what either backend actually consumed.

The gap is large enough to be the dominant source of error in any privacy-utility
experiment on this codebase.

---

## 1. The Three-Budget Architecture

The DP path involves three separate counters that are never reconciled:

**Python `BudgetManager`** (client-side, in `retriever.py`)
Tracks `_spent` as a running scalar sum of per-round epsilon costs. Each call to
`DiffPrivacyRetriever.retrieve()` calls `budget.consume(eps)` where `eps` is the
value returned by `privacy_cost(query)`. This number is the (ε,δ)-DP cost of a
*single* round converted from the single-round RDP, not from the accumulated RDP
over all rounds so far.

**Rust `RDPAccountant`** (server-side, in `dp.rs`, used by `pyo3_bridge.rs`)
Tracks `rdp_epsilons: Vec<f64>` — one accumulator per Rényi order. Each call to
`consume_rdp(sigma)` adds `alpha / (2σ²)` to each order's accumulator, then calls
`rdp_to_dp()` once on the *total accumulated* values. Comparison against budget
happens after this conversion, not before. This is the correct algorithm.

**sim_server** (Python HTTP, in `sim_server.py`)
As of v0.5, uses the same correct accumulated-RDP approach. The `rdp_acc` list is
updated in-place and `_rdp_to_dp()` is called on the running totals. This matches
the Rust accountant.

---

## 2. The Core Accounting Error in `BudgetManager`

RDP composes linearly across rounds. For T rounds of the Gaussian mechanism with
noise σ, the RDP at order α is `T · α/(2σ²)`. The conversion to (ε,δ)-DP is then
applied once to the accumulated value:

```
ε(T) = min_α [ T·α/(2σ²) + ln(1/δ)/(α−1) ]
```

The Python `BudgetManager` instead computes the single-round cost:

```
ε(1) = min_α [ α/(2σ²) + ln(1/δ)/(α−1) ]
```

and then sums `T × ε(1)`. This is not equal to `ε(T)` because the `ln(1/δ)/(α−1)`
term does not scale linearly with T. It is counted once per round instead of once
per composition. The overcharge grows with T:

| Rounds (T) | σ=0.5: true ε | σ=0.5: Python charges | Overcharge |
|------------|--------------|----------------------|------------|
| 1 | 5.64 | 5.64 | 0% |
| 2 | 7.84 | 11.29 | 44% |
| 3 | 9.84 | 16.93 | 72% |
| 5 | 13.84 | 28.22 | 104% |

The practical consequences of this error are:

**The agent stops retrieving earlier than it should.** `agent.py` line 45:

```python
if self.retriever.budget.remaining < self.retriever.privacy_cost(sub_query):
    break
```

`budget.remaining` is depleted faster than correct accounting requires. With σ=8.74
(the minimum noise for true ε=1.0 over 3 rounds), the Python BudgetManager charges
enough for only 1 round before the pre-flight check fails. The agent answers after
a single retrieval instead of three.

**`budget_snapshot()` reports misleading numbers.** `spent` and `remaining` in the
snapshot reflect the overcharged sum. A researcher calling `budget_snapshot()` to
audit how much privacy they consumed will see a number substantially higher than
the actual (ε,δ)-DP cost of the queries they ran.

**PrivacyConfig defaults make the protocol unusable out of the box.** The default
`noise_std=0.1` produces a per-round cost of ε≈111.5, which exceeds the default
`epsilon=1.0` on the very first round. A researcher who creates a `PrivacyConfig`
without tuning these parameters will get zero retrievals and no clear error message
about why.

---

## 3. Debit-Before-Request with No Rollback

In `DiffPrivacyRetriever.retrieve()`:

```python
eps = self.privacy_cost(query)
self.budget.consume(eps)           # ← epsilon debited here
noised = self._backend.embed_with_noise(...)
rows = self._backend.retrieve_by_embedding(...)  # ← can fail with BackendError
return self._to_docs(rows)
```

`budget.consume()` is called before either backend RPC. If `retrieve_by_embedding`
raises a `BackendError` (network failure, or the server's own budget rejecting the
request because the server and client budgets have diverged), the Python budget has
already been decremented. There is no rollback. The epsilon is gone from the client's
perspective but the retrieval never happened.

In `agent.py`, only `BudgetExhaustedError` is caught:

```python
try:
    new_docs = self.retriever.retrieve(sub_query, round_n)
except BudgetExhaustedError:
    break
```

A `BackendError` from the server budget rejecting the request propagates uncaught
through the agent loop entirely, crashing the run rather than stopping cleanly.

---

## 4. Three Budgets That Cannot Agree

Because the Python BudgetManager overcharges and the server budgets use correct
accounting, the three counters will diverge after round 1. This creates an incoherent
privacy guarantee:

- The Python BudgetManager says "budget exhausted at T=2" and stops.
- The Rust server says "budget still has headroom at T=2" and would accept T=3.
- The sim_server says the same as Rust.

The enforced limit is whichever fires first — in practice, always the Python
BudgetManager. The server never actually becomes the binding constraint because
the client stops first. This means the server-side budget tracking, although now
correct after the v0.5 fix, is never actually exercised in normal operation. It would
only matter if a client bypassed the Python layer and called the server directly.

For a researcher designing an experiment, this means the privacy budget they set in
`PrivacyConfig.epsilon` is not what governs the experiment. The Python BudgetManager
will stop the agent at a lower epsilon than specified, and the actual epsilon consumed
(per correct accumulated-RDP accounting) is lower still. No single number in the
codebase tells them the true cost.

---

## 5. `PrivacyConfig` / `CorpusBuilder` Epsilon Mismatch (Unchanged from v0.3)

`PrivacyConfig.epsilon` defaults to `1.0`. `CorpusBuilder._epsilon` defaults to
`1_000_000.0`. These are independent values that are never synchronised. A researcher
who sets `PrivacyConfig(epsilon=5.0, ...)` and passes it to `SecureRAGAgent.from_config`
does not automatically set the server's budget to 5.0. The server was built with
`CorpusBuilder(...).build()` using whatever epsilon was on that builder.

In practice the server epsilon of 1,000,000 is so large that it never binds, which
means the server budget check is cosmetic. But a researcher who calls
`with_privacy_budget(epsilon=5.0)` on the builder to set a tighter server-side limit
will find that it doesn't match `PrivacyConfig.epsilon=1.0` on the agent side — the
agent's client budget will stop it at ε=1.0 even though the server would allow up to
ε=5.0.

---

## 6. Embed Noise Seed Discrepancy (Unchanged from v0.3)

The sim_server uses `random.Random(hash(query) & 0xFFFF)` — a 16-bit seed derived
from Python's built-in `hash()`, which is randomised per-process by PYTHONHASHSEED.
The Rust PyO3 bridge uses `StdRng::seed_from_u64` with a deterministic 64-bit
multiplicative hash. For the same query and σ, the two backends produce different
noised embeddings. The DP guarantee holds independently per backend, but
reproducibility breaks: running the same experiment on sim_server vs the Rust local
backend produces different retrieved documents, making cross-backend comparison
unreliable.

The existing test `test_embed_with_noise_parity_http_vs_rust_sigma_zero` sidesteps
this by using σ=0 (no noise added), which passes trivially but gives no coverage
of the actual noisy case.

---

## 7. What Needs to Change

### Fix 1 — Replace `BudgetManager` with an RDP-accumulating accountant

`BudgetManager` needs to track `rdp_acc: list[float]` (one entry per Rényi order,
mirroring `dp.rs`) and compute the (ε,δ) cost lazily from the accumulated values:

```python
class BudgetManager:
    _ORDERS = [2.0, 4.0, 8.0, 16.0, 32.0]

    def __init__(self, config: PrivacyConfig):
        self._epsilon_max = float(config.epsilon)
        self._delta = float(config.delta)
        self._rdp_acc = [0.0] * len(self._ORDERS)
        self._round = 0
        self._ledger: list[tuple[int, float]] = []

    def _rdp_to_dp(self) -> float:
        return min(
            r + math.log(1.0 / self._delta) / (a - 1.0)
            for a, r in zip(self._ORDERS, self._rdp_acc)
        )

    def consume(self, sigma: float) -> None:
        candidate = [
            acc + a / (2.0 * sigma ** 2)
            for acc, a in zip(self._rdp_acc, self._ORDERS)
        ]
        candidate_eps = min(
            r + math.log(1.0 / self._delta) / (a - 1.0)
            for a, r in zip(self._ORDERS, candidate)
        )
        if candidate_eps > self._epsilon_max:
            raise BudgetExhaustedError(
                f"epsilon exhausted: {candidate_eps:.3f} / {self._epsilon_max:.3f}"
            )
        self._rdp_acc = candidate
        self._round += 1
        self._ledger.append((self._round, candidate_eps))

    @property
    def spent(self) -> float:
        return self._rdp_to_dp()

    @property
    def remaining(self) -> float:
        return self._epsilon_max - self.spent
```

The `consume` signature changes from `consume(eps_round: float)` to
`consume(sigma: float)`. This is a breaking change in the call site:
`DiffPrivacyRetriever.retrieve()` currently calls `budget.consume(eps)` where `eps`
is the output of `privacy_cost()`. It should instead call `budget.consume(sigma)`
directly, and `privacy_cost()` should ask the budget for the current marginal cost
of one more round rather than recomputing from a single-round formula.

The agent pre-flight check (`budget.remaining < retriever.privacy_cost(sub_query)`)
also needs to change: `privacy_cost` should return the *incremental* DP cost of
adding one round to the current accumulated state, not the single-round cost.

### Fix 2 — Move debit after the backend call succeeds

```python
def retrieve(self, query: str, round_n: int):
    noised = self._backend.embed_with_noise(query=query, sigma=self.config.noise_std)
    rows = self._backend.retrieve_by_embedding(...)
    self.budget.consume(self.config.noise_std)   # debit only on success
    return self._to_docs(rows)
```

### Fix 3 — Wire `PrivacyConfig` epsilon into `CorpusBuilder`

Add a `from_config` classmethod or accept a `PrivacyConfig` in the constructor:

```python
class CorpusBuilder:
    def __init__(self, protocol, backend_url="http://127.0.0.1:8099",
                 config: PrivacyConfig | None = None):
        ...
        if config is not None and protocol.requires_budget:
            self._epsilon = config.epsilon
            self._delta = config.delta
```

Or as the simpler interface:
```python
CorpusBuilder.from_config(config, backend_url=...).add_documents(docs).build()
```

### Fix 4 — Unify embed noise seeds

Both sim_server and the Rust bridge should use the same deterministic 64-bit seed
derived from the query. The simplest approach is to hash the query with SHA-256
and take the first 8 bytes as a u64. This is already how the embedding itself is
computed (Sha256 is imported in pyo3_bridge). The sim_server should switch from
`random.Random(hash(query) & 0xFFFF)` to `int.from_bytes(hashlib.sha256(query.encode()).digest()[:8], 'little')` as the seed for its RNG.

---

## 8. Impact on Researchers

A researcher trying to use DIFF_PRIVACY today faces four compounding problems:

1. **Default parameters are incoherent.** `noise_std=0.1` with `epsilon=1.0` allows
   zero retrievals. There is no warning. The agent silently returns an answer from
   an empty context.

2. **The budget accounting is systematically wrong.** Setting parameters that look
   correct (e.g., σ=1.0, ε=20.0 for 3 rounds) will cause the agent to stop after
   1 or 2 rounds because the Python BudgetManager overcharges by 44–72%. The
   researcher has to manually compensate by inflating `epsilon` to account for the
   overcharge — which they have no way of knowing about without reading this analysis.

3. **`budget_snapshot()` is unreliable.** The `spent` value reported is the
   overcharged sum. Using it to audit actual privacy expenditure or to set budgets
   for a follow-up experiment will produce incorrect results.

4. **Cross-backend results are not reproducible.** The same experiment on sim_server
   vs the Rust local backend returns different retrieved documents due to different
   noise seeds. A researcher who develops on sim_server and benchmarks on the Rust
   bridge cannot expect comparable results.

None of these problems affect the encrypted search path, the protocol API surface, or
the refactoring work that was done in v0.4–v0.5. They are pre-existing issues in the
DP subsystem that have not been addressed by the recent refactor cycle.