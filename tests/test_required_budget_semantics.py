from __future__ import annotations

from types import SimpleNamespace

import pytest

from securerag.config import PrivacyConfig
from securerag.context import PrivacyContext
from securerag.models import PrivateQuery
from securerag.protocol import PrivacyProtocol
from securerag.retriever import PrivacyRetriever

# Ensure protocol retrievers are registered.
import securerag.retrievers  # noqa: F401


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
        return [{"doc_id": "d1", "text": "doc", "metadata": {}, "score": 1.0}]


def test_private_query_required_budget_false_skips_budget_and_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("securerag.retriever.create_backend", lambda _url: _FakeBackend())

    cfg = PrivacyConfig(
        protocol=PrivacyProtocol.DIFF_PRIVACY,
        epsilon=5.0,
        delta=1e-5,
        noise_std=1.0,
        backend="fake://dp",
    )
    corpus = SimpleNamespace(protocol=PrivacyProtocol.DIFF_PRIVACY, index_id="idx-rb")
    retriever = PrivacyRetriever.from_config(cfg, corpus)

    ctx = PrivacyContext(strict=True)
    seen = {"noise": 0, "budget": 0}

    @ctx.register_noise_hook("encode")
    def _noise_hook(embedding, config, budget_state):
        seen["noise"] += 1
        return embedding, retriever._dp_mechanism.cost(sensitivity=1.0)

    @ctx.register_budget_hook("retrieve")
    def _budget_hook(docs, config, corpus_budgets):
        seen["budget"] += 1
        return retriever._dp_mechanism.cost(sensitivity=1.0)

    retriever.with_context(ctx)
    with ctx:
        docs = retriever.retrieve(PrivateQuery(text="q", required_budget=False), round_n=0)

    assert docs
    assert seen["noise"] == 0
    assert seen["budget"] == 0
    assert retriever.budget.snapshot()["rounds"] == 0


def test_private_query_required_budget_true_charges_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("securerag.retriever.create_backend", lambda _url: _FakeBackend())

    cfg = PrivacyConfig(
        protocol=PrivacyProtocol.DIFF_PRIVACY,
        epsilon=10.0,
        delta=1e-5,
        noise_std=1.0,
        backend="fake://dp",
    )
    corpus = SimpleNamespace(protocol=PrivacyProtocol.DIFF_PRIVACY, index_id="idx-rb-2")
    retriever = PrivacyRetriever.from_config(cfg, corpus)

    before = retriever.budget.snapshot()["rounds"]
    retriever.retrieve(PrivateQuery(text="q", required_budget=True), round_n=0)
    after = retriever.budget.snapshot()["rounds"]

    assert after == before + 1
