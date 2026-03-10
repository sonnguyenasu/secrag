from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import httpx
import pytest

from securerag.agent import SecureRAGAgent
from securerag.budget import BudgetManager
from securerag.config import PrivacyConfig
from securerag.corpus import CorpusBuilder
from securerag.dp_mechanism import DPMechanismPlugin
from securerag.errors import BackendError
from securerag.models import RawDocument
from securerag.protocol import PrivacyProtocol
from securerag.retriever import PrivacyRetriever


# Ensure protocol retrievers are registered for test construction.
import securerag.retrievers  # noqa: F401


class _FakeDPBackend:
    def __init__(self, *, fail_retrieve: bool = False):
        self.fail_retrieve = fail_retrieve
        self.retrieve_calls = 0

    def embed_with_noise(self, query: str, sigma: float) -> list[float]:
        return [0.01 * (i + 1) for i in range(64)]

    def embed(self, query: str) -> list[float]:
        return [0.01 * (i + 1) for i in range(64)]

    def retrieve_by_embedding(
        self,
        index_id: str,
        embedding: list[float],
        top_k: int,
        query: str | None = None,
        sigma: float | None = None,
    ) -> list[dict]:
        if self.fail_retrieve:
            raise BackendError("transient backend failure")
        self.retrieve_calls += 1
        return [
            {
                "doc_id": f"doc-{self.retrieve_calls}",
                "text": f"evidence for {query}",
                "metadata": {"round": str(self.retrieve_calls)},
                "score": 1.0,
            }
        ]

    def build_index(
        self,
        protocol: str,
        chunks: list[dict],
        *,
        epsilon: float = 1_000_000.0,
        delta: float = 1e-5,
        encrypted_search_scheme: str = "",
        encrypted_search_version: str = "",
    ) -> dict:
        return {"index_id": "fake-index", "doc_count": len(chunks)}


class _AlwaysRetrieveLLM:
    @dataclass
    class _Decision:
        should_answer: bool
        sub_query: str | None = None

    def decide(self, query: str, context: list, round: int):
        return self._Decision(should_answer=False, sub_query=f"{query} round-{round}")

    def generate(self, query: str, context: list) -> str:
        return f"answer with {len(context)} context docs"


class _ConstantNoiseMechanism(DPMechanismPlugin):
    def __init__(self) -> None:
        self.prepare_calls = 0

    def noise(self, embedding: list[float], sigma: float, *, query: str = "") -> list[float]:
        return [v + 1.0 for v in embedding]

    def rdp_cost(self, sigma: float, alpha: float) -> float:
        return 0.01 * alpha

    def rdp_orders(self) -> list[float]:
        return [2.0, 10.0]

    def prepare_corpus(self, chunks: list[dict]) -> list[dict]:
        self.prepare_calls += 1
        return [{**c, "dp_marker": "prepared"} for c in chunks]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for_health(base_url: str, timeout: float = 8.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"{base_url}/health", timeout=0.4)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.1)
    raise RuntimeError(f"sim_server did not become healthy at {base_url}")


def _launch_sim_server(port: int) -> subprocess.Popen:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "securerag.sim_server:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def test_budget_manager_uses_accumulated_rdp_for_research_audits() -> None:
    cfg = PrivacyConfig(protocol=PrivacyProtocol.DIFF_PRIVACY, epsilon=30.0, delta=1e-5)
    budget = BudgetManager(cfg)

    first_round = budget.incremental_cost(sigma=1.0)
    budget.consume(sigma=1.0)
    budget.consume(sigma=1.0)

    # True accumulated-RDP after 2 rounds at sigma=1.0 is ~7.838, not 2 * 5.640.
    assert budget.spent == pytest.approx(7.8376418, rel=1e-6)
    assert budget.spent < 2.0 * first_round

    snap = budget.snapshot()
    assert snap["rounds"] == 2
    assert len(snap["ledger"]) == 2
    assert snap["spent"] == pytest.approx(budget.spent, rel=0.0, abs=1e-12)


def test_dp_retriever_debits_only_after_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeDPBackend(fail_retrieve=True)
    monkeypatch.setattr("securerag.retriever.create_backend", lambda _url: fake)

    cfg = PrivacyConfig(
        protocol=PrivacyProtocol.DIFF_PRIVACY,
        epsilon=20.0,
        delta=1e-5,
        noise_std=1.0,
        backend="fake://dp",
    )
    corpus = SimpleNamespace(protocol=PrivacyProtocol.DIFF_PRIVACY, index_id="idx-1")
    retriever = PrivacyRetriever.from_config(cfg, corpus)

    with pytest.raises(BackendError):
        retriever.retrieve("q3 risk", round_n=0)

    snap = retriever.budget.snapshot()
    assert snap["rounds"] == 0
    assert snap["spent"] == 0.0


def test_agent_runs_expected_dp_rounds_without_manual_epsilon_tuning(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeDPBackend(fail_retrieve=False)
    monkeypatch.setattr("securerag.retriever.create_backend", lambda _url: fake)

    cfg = PrivacyConfig(
        protocol=PrivacyProtocol.DIFF_PRIVACY,
        epsilon=9.0,
        delta=1e-5,
        noise_std=1.0,
        top_k=1,
        max_rounds=5,
        backend="fake://dp",
    )
    corpus = SimpleNamespace(protocol=PrivacyProtocol.DIFF_PRIVACY, index_id="idx-2")
    agent = SecureRAGAgent.from_config(cfg, corpus=corpus, llm=_AlwaysRetrieveLLM())

    result = agent.run("q3 risk")
    snap = agent.budget_snapshot()

    # With correct accumulated-RDP accounting and epsilon=9.0, exactly 2 rounds fit.
    assert fake.retrieve_calls == 2
    assert snap["rounds"] == 2
    assert result.context_size == 2


def test_corpus_builder_from_config_wires_server_dp_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, float | str] = {}

    class _CaptureBackend(_FakeDPBackend):
        def build_index(self, protocol: str, chunks: list[dict], **kwargs) -> dict:
            captured["protocol"] = protocol
            captured["epsilon"] = float(kwargs["epsilon"])
            captured["delta"] = float(kwargs["delta"])
            return {"index_id": "capture-index", "doc_count": len(chunks)}

    backend = _CaptureBackend()

    def _create_backend(url: str):
        captured["backend_url"] = url
        return backend

    monkeypatch.setattr("securerag.corpus.create_backend", _create_backend)

    cfg = PrivacyConfig(
        protocol=PrivacyProtocol.DIFF_PRIVACY,
        epsilon=5.0,
        delta=1e-6,
        noise_std=1.0,
        backend="memory://dp-backend",
    )
    docs = [RawDocument(doc_id="d1", text="risk evidence")]

    corpus = CorpusBuilder.from_config(cfg).add_documents(docs).build_local()

    assert corpus.protocol is PrivacyProtocol.DIFF_PRIVACY
    assert captured["backend_url"] == "memory://dp-backend"
    assert captured["protocol"] == PrivacyProtocol.DIFF_PRIVACY.wire_name
    assert captured["epsilon"] == 5.0
    assert captured["delta"] == pytest.approx(1e-6, rel=0.0, abs=1e-15)


def test_sim_server_dp_noise_is_reproducible_across_restarts() -> None:
    query = "Q3 risk vendor concentration"
    sigma = 0.6

    def _embed_once() -> list[float]:
        port = _free_port()
        base_url = f"http://127.0.0.1:{port}"
        proc = _launch_sim_server(port)
        try:
            _wait_for_health(base_url)
            resp = httpx.post(
                f"{base_url}/rpc",
                json={"operation": "embed_with_noise", "payload": {"query": query, "sigma": sigma}},
                timeout=2.0,
            )
            resp.raise_for_status()
            body = resp.json()
            assert body["ok"] is True
            return [float(x) for x in body["data"]]
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()

    emb_a = _embed_once()
    emb_b = _embed_once()

    assert len(emb_a) == len(emb_b) == 64
    assert emb_a == pytest.approx(emb_b, rel=0.0, abs=1e-12)


def test_custom_dp_mechanism_runs_without_framework_edits(monkeypatch: pytest.MonkeyPatch) -> None:
    mechanism = _ConstantNoiseMechanism()
    DPMechanismPlugin.register("constant_test", mechanism)

    fake = _FakeDPBackend(fail_retrieve=False)
    monkeypatch.setattr("securerag.retriever.create_backend", lambda _url: fake)

    cfg = PrivacyConfig(
        protocol=PrivacyProtocol.DIFF_PRIVACY,
        dp_mechanism="constant_test",
        epsilon=2.0,
        delta=1e-5,
        noise_std=1.0,
        backend="fake://dp",
    )
    corpus = SimpleNamespace(protocol=PrivacyProtocol.DIFF_PRIVACY, index_id="idx-plugin")
    retriever = PrivacyRetriever.from_config(cfg, corpus)

    docs = retriever.retrieve("q3 risk", round_n=0)
    assert docs
    snap = retriever.budget.snapshot()
    assert snap["mechanism"] == "_ConstantNoiseMechanism"
    assert snap["spent"] > 0.0


def test_corpus_builder_runs_dp_prepare_corpus_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    mechanism = _ConstantNoiseMechanism()
    DPMechanismPlugin.register("prep_hook_test", mechanism)

    captured: dict[str, list[dict]] = {}

    class _CaptureBackend(_FakeDPBackend):
        def build_index(self, protocol: str, chunks: list[dict], **kwargs) -> dict:
            captured["chunks"] = chunks
            return {"index_id": "capture-index", "doc_count": len(chunks)}

    monkeypatch.setattr("securerag.corpus.create_backend", lambda _url: _CaptureBackend())

    cfg = PrivacyConfig(
        protocol=PrivacyProtocol.DIFF_PRIVACY,
        dp_mechanism="prep_hook_test",
        epsilon=5.0,
        delta=1e-6,
        noise_std=1.0,
        backend="memory://dp-backend",
    )
    docs = [RawDocument(doc_id="d1", text="risk evidence")]
    CorpusBuilder.from_config(cfg).add_documents(docs).build_local()

    assert mechanism.prepare_calls == 1
    assert captured["chunks"]
    assert all(c.get("dp_marker") == "prepared" for c in captured["chunks"])
