from __future__ import annotations

from types import SimpleNamespace

import pytest

from securerag.config import PrivacyConfig
from securerag.context import PrivacyContext
from securerag.corpus import CorpusBuilder
from securerag.cost import RDPCost
from securerag.errors import BudgetExhaustedError
from securerag.models import RawDocument
from securerag.protocol import PrivacyProtocol
from securerag.retriever import PrivacyRetriever

# Ensure protocol retrievers are registered for construction.
import securerag.retrievers  # noqa: F401

_ORDERS = [2.0, 4.0, 8.0, 16.0, 32.0]


class _FakeBackend:
    def embed(self, query: str) -> list[float]:
        return [0.1] * 64

    def retrieve_by_embedding(
        self,
        index_id: str,
        embedding: list[float],
        top_k: int,
        query: str | None = None,
        sigma: float | None = None,
    ) -> list[dict]:
        return [
            {
                "doc_id": "d1",
                "text": "retrieved",
                "metadata": {"q": query or ""},
                "score": 1.0,
            }
        ]


def test_noise_and_budget_hooks_are_fired(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("securerag.retriever.create_backend", lambda _url: _FakeBackend())

    cfg = PrivacyConfig(
        protocol=PrivacyProtocol.DIFF_PRIVACY,
        epsilon=50.0,
        delta=1e-5,
        noise_std=1.0,
        backend="fake://dp",
    )
    corpus = SimpleNamespace(protocol=PrivacyProtocol.DIFF_PRIVACY, index_id="idx-hook")
    retriever = PrivacyRetriever.from_config(cfg, corpus)

    ctx = PrivacyContext(strict=True)
    seen: dict[str, bool] = {"noise": False, "budget": False}

    @ctx.register_noise_hook("encode")
    def _noise_hook(embedding, config, budget_state):
        seen["noise"] = True
        return ([x + 0.25 for x in embedding], RDPCost(orders=_ORDERS, values=[0.01] * 5))

    @ctx.register_budget_hook("retrieve")
    def _budget_hook(docs, config, corpus_budgets):
        seen["budget"] = True
        return RDPCost(orders=_ORDERS, values=[0.02] * 5)

    retriever.with_context(ctx)
    with ctx:
        docs = retriever.retrieve("risk", round_n=0)

    assert docs
    assert seen["noise"] is True
    assert seen["budget"] is True
    assert retriever.budget.snapshot()["rounds"] == 1


def test_composition_hook_still_enforces_budget_and_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("securerag.retriever.create_backend", lambda _url: _FakeBackend())

    cfg = PrivacyConfig(
        protocol=PrivacyProtocol.DIFF_PRIVACY,
        epsilon=0.5,
        delta=1e-5,
        noise_std=1.0,
        backend="fake://dp",
    )
    corpus = SimpleNamespace(protocol=PrivacyProtocol.DIFF_PRIVACY, index_id="idx-compose")
    retriever = PrivacyRetriever.from_config(cfg, corpus)

    ctx = PrivacyContext(strict=True)

    @ctx.register_noise_hook("encode")
    def _noise_hook(embedding, config, budget_state):
        return (embedding, RDPCost(orders=_ORDERS, values=[0.3] * 5))

    @ctx.register_composition_hook
    def _compose(a: RDPCost, b: RDPCost) -> RDPCost:
        return a + b

    retriever.with_context(ctx)
    with ctx:
        retriever.retrieve("risk", round_n=0)
        with pytest.raises(BudgetExhaustedError):
            retriever.retrieve("risk again", round_n=1)

    # First round should be recorded in ledger; second should fail cleanly.
    snap = retriever.budget.snapshot()
    assert snap["rounds"] == 1
    assert len(snap["ledger"]) == 1


def test_build_local_does_not_eagerly_require_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    # If create_backend were still eager in __init__, this would raise here.
    monkeypatch.setattr(
        "securerag.corpus.create_backend",
        lambda _url: (_ for _ in ()).throw(RuntimeError("backend must not be created")),
    )

    builder = (
        CorpusBuilder(PrivacyProtocol.BASELINE, backend_url="http://unused")
        .add_documents([RawDocument(doc_id="d1", text="risk evidence")])
    )

    corpus = builder.build_local(use_rust_if_available=False)
    assert corpus.index_id == "local"
    assert "local_index" in corpus.extras
