import socket
import subprocess
import sys
import time
import logging

import httpx
import pytest

from securerag import PrivacyConfig, PrivacyProtocol, SecureRAGAgent
from securerag.backend_client import create_backend
from securerag.budget import BudgetManager
from securerag.corpus import CorpusBuilder
from securerag.errors import UnsupportedCapabilityError
from securerag.llm import HuggingFaceLLM, OllamaLLM
from securerag.models import Document, RawDocument


def test_budget_snapshot_shape():
    cfg = PrivacyConfig(protocol=PrivacyProtocol.DIFF_PRIVACY, epsilon=10.0)
    b = BudgetManager(cfg)
    b.consume(2.5)
    snap = b.snapshot()
    assert snap["spent"] == 2.5
    assert snap["remaining"] == 7.5


def test_end_to_end_diffprivacy_localhost_server_required():
    docs = [RawDocument(doc_id="1", text="risk alpha"), RawDocument(doc_id="2", text="risk beta")]
    corpus = CorpusBuilder(PrivacyProtocol.DIFF_PRIVACY).add_documents(docs).build()
    cfg = PrivacyConfig(protocol=PrivacyProtocol.DIFF_PRIVACY, epsilon=100.0, noise_std=0.3, top_k=2)
    agent = SecureRAGAgent.from_config(cfg, corpus=corpus, llm=OllamaLLM())
    result = agent.run("risk")
    assert result.context_size >= 1


def test_verbose_logs_obfuscation_decoys(caplog):
    docs = [RawDocument(doc_id="q3", text="Q3 risk report"), RawDocument(doc_id="p", text="security policy")]
    corpus = CorpusBuilder(PrivacyProtocol.OBFUSCATION).add_documents(docs).build()
    cfg = PrivacyConfig(
        protocol=PrivacyProtocol.OBFUSCATION,
        k_decoys=2,
        top_k=2,
        verbose=True,
    )
    agent = SecureRAGAgent.from_config(cfg, corpus=corpus, llm=OllamaLLM())

    with caplog.at_level(logging.INFO, logger="securerag.retriever"):
        agent.retriever.retrieve("Q3 risk", 0)

    assert "obfuscation retrieval" in caplog.text
    assert "raw_decoy_queries" in caplog.text
    assert "paraphrased_decoy_queries" in caplog.text
    assert "real_query='Q3 risk'" in caplog.text


def test_obfuscation_decoys_are_paraphrased_when_enabled():
    docs = [RawDocument(doc_id="q3", text="Q3 risk report"), RawDocument(doc_id="p", text="security policy")]
    corpus = CorpusBuilder(PrivacyProtocol.OBFUSCATION).add_documents(docs).build()
    cfg = PrivacyConfig(
        protocol=PrivacyProtocol.OBFUSCATION,
        k_decoys=2,
        top_k=2,
        verbose=False,
        paraphrase_decoys=True,
    )
    agent = SecureRAGAgent.from_config(cfg, corpus=corpus, llm=OllamaLLM())

    raw = agent.retriever._backend.generate_decoys(corpus.index_id, "Q3 risk", cfg.k_decoys)
    paraphrased = agent.retriever._paraphrase_decoys(raw, "Q3 risk")

    assert len(paraphrased) == len(raw)
    assert paraphrased != raw


def test_obfuscation_decoys_not_paraphrased_when_disabled():
    docs = [RawDocument(doc_id="q3", text="Q3 risk report"), RawDocument(doc_id="p", text="security policy")]
    corpus = CorpusBuilder(PrivacyProtocol.OBFUSCATION).add_documents(docs).build()
    cfg = PrivacyConfig(
        protocol=PrivacyProtocol.OBFUSCATION,
        k_decoys=2,
        top_k=2,
        verbose=False,
        paraphrase_decoys=False,
    )
    agent = SecureRAGAgent.from_config(cfg, corpus=corpus, llm=OllamaLLM())

    raw = agent.retriever._backend.generate_decoys(corpus.index_id, "Q3 risk", cfg.k_decoys)
    paraphrased = agent.retriever._paraphrase_decoys(raw, "Q3 risk")

    assert paraphrased == raw


def test_obfuscation_paraphrase_does_not_reference_source_query_phrase():
    docs = [RawDocument(doc_id="q3", text="Q3 risk report"), RawDocument(doc_id="p", text="security policy")]
    corpus = CorpusBuilder(PrivacyProtocol.OBFUSCATION).add_documents(docs).build()
    llm = OllamaLLM(use_ollama=False)
    cfg = PrivacyConfig(
        protocol=PrivacyProtocol.OBFUSCATION,
        k_decoys=2,
        top_k=2,
        verbose=False,
        paraphrase_decoys=True,
    )
    agent = SecureRAGAgent.from_config(cfg, corpus=corpus, llm=llm)

    source_query = "Summarize key risks in Q3"
    raw = agent.retriever._backend.generate_decoys(corpus.index_id, source_query, cfg.k_decoys)
    paraphrased = agent.retriever._paraphrase_decoys(raw, source_query)

    joined = " ".join(paraphrased).lower()
    assert "summarize key risks in q3" not in joined


def test_verbose_logs_diffprivacy_noise(caplog):
    docs = [RawDocument(doc_id="q3", text="Q3 risk report"), RawDocument(doc_id="p", text="security policy")]
    corpus = CorpusBuilder(PrivacyProtocol.DIFF_PRIVACY).add_documents(docs).build()
    cfg = PrivacyConfig(
        protocol=PrivacyProtocol.DIFF_PRIVACY,
        epsilon=100.0,
        noise_std=0.2,
        top_k=2,
        verbose=True,
    )
    agent = SecureRAGAgent.from_config(cfg, corpus=corpus, llm=OllamaLLM())

    with caplog.at_level(logging.INFO, logger="securerag.retriever"):
        agent.retriever.retrieve("Q3 risk", 0)

    assert "diff-privacy retrieval" in caplog.text
    assert "original_query='Q3 risk'" in caplog.text
    assert "noised_query_embedding" in caplog.text


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for_health(base_url: str, timeout_s: float = 8.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = httpx.get(f"{base_url}/health", timeout=0.5)
            if r.status_code == 200:
                return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError(f"timed out waiting for backend health at {base_url}")


def _run_top1(
    protocol: PrivacyProtocol,
    backend: str,
    docs: list[RawDocument],
    query: str,
    encrypted_scheme: str | None = None,
) -> str:
    builder = CorpusBuilder(protocol, backend_url=backend).with_chunk_size(120).add_documents(docs)
    if protocol is PrivacyProtocol.ENCRYPTED_SEARCH and encrypted_scheme:
        builder = builder.with_encrypted_search_scheme(encrypted_scheme)
    corpus = builder.build()
    cfg = PrivacyConfig(
        protocol=protocol,
        epsilon=500.0 if protocol is PrivacyProtocol.DIFF_PRIVACY else 1.0,
        noise_std=0.2,
        top_k=3,
        k_decoys=3,
        backend=backend,
        encrypted_search_scheme=encrypted_scheme or "sse",
    )
    agent = SecureRAGAgent.from_config(cfg, corpus=corpus, llm=OllamaLLM())
    return agent.retriever.retrieve(query, 0)[0].doc_id


@pytest.mark.parametrize(
    "protocol,encrypted_scheme",
    [
        (PrivacyProtocol.BASELINE, None),
        (PrivacyProtocol.DIFF_PRIVACY, None),
        (PrivacyProtocol.OBFUSCATION, None),
        (PrivacyProtocol.ENCRYPTED_SEARCH, "sse"),
        (PrivacyProtocol.ENCRYPTED_SEARCH, "structured"),
    ],
)
def test_top1_retrieval_parity_http_vs_rust(protocol: PrivacyProtocol, encrypted_scheme: str | None):
    pytest.importorskip("securerag_rs")

    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
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

    try:
        _wait_for_health(base_url)

        docs = [
            RawDocument(doc_id="q3", text="Q3 risk report highlights vendor concentration and delayed remediation."),
            RawDocument(doc_id="policy", text="Security policy requires quarterly risk treatment tracking and owner assignment."),
            RawDocument(doc_id="ops", text="Operational incidents increased in July due to queue saturation in ingestion service."),
        ]

        query = "Q3 risk"

        http_top = _run_top1(protocol, base_url, docs, query, encrypted_scheme=encrypted_scheme)
        rust_top = _run_top1(protocol, "rust://local", docs, query, encrypted_scheme=encrypted_scheme)

        assert http_top == rust_top == "q3"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_sse_crypto_ops_parity_http_vs_rust():
    pytest.importorskip("securerag_rs")

    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
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

    try:
        _wait_for_health(base_url)

        http_backend = create_backend(base_url)
        rust_backend = create_backend("rust://local")

        text = "Q3 risks include vendor dependencies and delayed remediation plans."
        key = "0123456789abcdef0123456789abcdef"
        chunks = [{"doc_id": "q3", "text": text, "metadata": {"source": "unit"}}]

        generated_key = rust_backend.sse_generate_key()
        assert len(generated_key) == 32

        assert http_backend.sse_encrypt_terms(text, key) == rust_backend.sse_encrypt_terms(text, key)
        assert http_backend.sse_encrypt_structured_terms(
            text,
            key,
            use_bigrams=True,
        ) == rust_backend.sse_encrypt_structured_terms(
            text,
            key,
            use_bigrams=True,
        )

        assert http_backend.sse_prepare_chunks(
            chunks,
            key,
            "sse",
            use_bigrams=True,
        ) == rust_backend.sse_prepare_chunks(
            chunks,
            key,
            "sse",
            use_bigrams=True,
        )
        assert http_backend.sse_prepare_chunks(
            chunks,
            key,
            "structured",
            use_bigrams=True,
        ) == rust_backend.sse_prepare_chunks(
            chunks,
            key,
            "structured",
            use_bigrams=True,
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_top1_retrieval_parity_http_vs_grpc_baseline():
    pytest.importorskip("grpc")

    http_port = _free_port()
    grpc_port = _free_port()
    http_url = f"http://127.0.0.1:{http_port}"
    grpc_target = f"grpc://127.0.0.1:{grpc_port}"

    http_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "securerag.sim_server:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(http_port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    grpc_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "securerag.grpc_server",
            "--host",
            "127.0.0.1",
            "--port",
            str(grpc_port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        _wait_for_health(http_url)
        time.sleep(0.8)

        docs = [
            RawDocument(doc_id="q3", text="Q3 risk report highlights vendor concentration and delayed remediation."),
            RawDocument(doc_id="policy", text="Security policy requires quarterly risk treatment tracking and owner assignment."),
            RawDocument(doc_id="ops", text="Operational incidents increased in July due to queue saturation in ingestion service."),
        ]

        query = "Q3 risk"
        http_top = _run_top1(PrivacyProtocol.BASELINE, http_url, docs, query)
        grpc_top = _run_top1(PrivacyProtocol.BASELINE, grpc_target, docs, query)
        assert grpc_top == http_top == "q3"
    finally:
        http_proc.terminate()
        grpc_proc.terminate()
        try:
            http_proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            http_proc.kill()
        try:
            grpc_proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            grpc_proc.kill()


def test_encrypted_search_unsupported_scheme_raises():
    docs = [RawDocument(doc_id="q3", text="Q3 risk report"), RawDocument(doc_id="p", text="security policy")]
    corpus = CorpusBuilder(PrivacyProtocol.ENCRYPTED_SEARCH).add_documents(docs).build()
    cfg = PrivacyConfig(
        protocol=PrivacyProtocol.ENCRYPTED_SEARCH,
        encrypted_search_scheme="unknown_scheme",
        top_k=2,
    )
    agent = SecureRAGAgent.from_config(cfg, corpus=corpus, llm=OllamaLLM())

    with pytest.raises(UnsupportedCapabilityError):
        agent.retriever.retrieve("q3 risk", 0)


def test_cot_planner_creates_followup_subquery():
    llm = OllamaLLM(use_ollama=False)
    query = "Summarize key risks in Q3"

    d0 = llm.decide(query=query, context=[], round=0)
    assert d0.should_answer is False
    assert d0.sub_query == query

    context = [
        Document(doc_id="q3", text="Q3 risk report highlights vendor concentration and delayed remediation.", score=0.9),
        Document(doc_id="ops", text="Operational incidents increased in July due to queue saturation.", score=0.8),
        Document(doc_id="policy", text="Security policy requires owner assignment and quarterly tracking.", score=0.7),
    ]
    d1 = llm.decide(query=query, context=context, round=1)
    assert d1.should_answer is False
    assert d1.sub_query is not None
    assert d1.sub_query != query

    d2 = llm.decide(query=query, context=context, round=2)
    assert d2.should_answer is True


def test_huggingface_llm_fallback_planner_without_network():
    llm = HuggingFaceLLM(
        model="google/flan-t5-base",
        use_huggingface=False,
    )
    query = "Summarize key risks in Q3"
    d0 = llm.decide(query=query, context=[], round=0)
    d1 = llm.decide(
        query=query,
        context=[Document(doc_id="q3", text="Q3 risk report highlights vendor concentration.", score=0.9)],
        round=1,
    )
    assert d0.should_answer is False
    assert d0.sub_query == query
    assert d1.should_answer is False
    assert d1.sub_query is not None
    assert d1.sub_query != query


def test_ollama_wrapper_is_provider_compatible():
    llm = OllamaLLM(use_ollama=False)
    out = llm.generate(
        "Summarize key risks in Q3",
        [Document(doc_id="q3", text="Q3 risk report highlights vendor concentration.", score=0.9)],
    )
    assert "Q3" in out
