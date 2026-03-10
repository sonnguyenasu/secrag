from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from securerag.backend_client import Backend, create_backend
from securerag.corpus import CorpusBuilder
from securerag.errors import BackendError
from securerag.models import RawDocument
from securerag.protocol import PrivacyProtocol
from securerag.scheme_plugin import EncryptedSchemePlugin


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for_health(base_url: str, timeout: float = 10.0) -> None:
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


def _wait_for_tcp(host: str, port: int, timeout: float = 12.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            s.settimeout(0.5)
            try:
                s.connect((host, port))
                return
            except OSError:
                time.sleep(0.1)
    raise RuntimeError(f"TCP service not ready at {host}:{port}")


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


def _sample_docs() -> list[RawDocument]:
    return [
        RawDocument(
            doc_id="q3",
            text="Q3 risk report highlights vendor concentration and delayed remediation.",
            metadata={"source": "unit"},
        ),
        RawDocument(
            doc_id="policy",
            text="Security policy requires quarterly risk treatment tracking and ownership.",
            metadata={"source": "unit"},
        ),
        RawDocument(
            doc_id="ops",
            text="Operational incidents increased because queue saturation impacted ingestion.",
            metadata={"source": "unit"},
        ),
    ]


def test_markdown_conflict_resolution_prefers_refactor_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    refactor_md = (root / "refactor.md").read_text(encoding="utf-8")
    api_design_path = root / "securerag-api-design.md"
    if api_design_path.exists():
        api_md = api_design_path.read_text(encoding="utf-8")
        # Conflicting legacy API expectations exist in api design doc.
        assert "sse_search" in api_md
        assert "sse_encrypt_query" in api_md or "sse_encrypt_terms" in api_md

    # Refactor contract explicitly replaces them with encrypted_search plugin flow.
    assert "EncryptedSchemePlugin" in refactor_md
    assert "encrypted_search" in refactor_md

    # Runtime/API surface must follow refactor.md, not legacy api design text.
    assert hasattr(Backend, "encrypted_search")
    assert not hasattr(Backend, "sse_search")
    assert not hasattr(Backend, "structured_search")
    assert not hasattr(Backend, "sse_generate_key")

    py_proto = (root / "securerag/proto/secure_retrieval.proto").read_text(encoding="utf-8")
    rs_proto = (root / "securerag-rs/proto/secure_retrieval.proto").read_text(encoding="utf-8")
    for proto in (py_proto, rs_proto):
        assert "rpc EncryptedSearch" in proto
        assert "rpc SseSearch" not in proto
        assert "rpc StructuredSearch" not in proto


def test_plugin_registry_and_builder_contract() -> None:
    assert "sse" in EncryptedSchemePlugin.registered_names()
    assert "structured" in EncryptedSchemePlugin.registered_names()

    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    proc = _launch_sim_server(port)

    docs = _sample_docs()
    try:
        _wait_for_health(base_url)

        builder = (
            CorpusBuilder(PrivacyProtocol.ENCRYPTED_SEARCH, backend_url=base_url)
            .with_encrypted_search_scheme("structured")
            .add_documents(docs)
        )
        corpus = builder.build_local(workers=2)

        assert corpus.extras.get("encrypted_search_scheme") == "structured"
        assert corpus.extras.get("enc_key")
        assert corpus.extras.get("plugin") is not None

        with pytest.raises(KeyError):
            (
                CorpusBuilder(PrivacyProtocol.ENCRYPTED_SEARCH, backend_url=base_url)
                .with_encrypted_search_scheme("not_registered")
                .add_documents(docs)
                .build_local()
            )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_structured_use_bigrams_toggle_is_effective() -> None:
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    proc = _launch_sim_server(port)
    docs = _sample_docs()
    try:
        _wait_for_health(base_url)

        with_bigrams = (
            CorpusBuilder(PrivacyProtocol.ENCRYPTED_SEARCH, backend_url=base_url)
            .with_encrypted_search_scheme("structured", structured_use_bigrams=True)
            .add_documents(docs)
            .build_local(workers=2)
        )
        without_bigrams = (
            CorpusBuilder(PrivacyProtocol.ENCRYPTED_SEARCH, backend_url=base_url)
            .with_encrypted_search_scheme("structured", structured_use_bigrams=False)
            .add_documents(docs)
            .build_local(workers=2)
        )

        assert getattr(with_bigrams.extras["plugin"], "use_bigrams") is True
        assert getattr(without_bigrams.extras["plugin"], "use_bigrams") is False
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_sim_server_rejects_incompatible_encrypted_search_version() -> None:
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    proc = _launch_sim_server(port)

    try:
        _wait_for_health(base_url)
        backend = create_backend(base_url)
        plugin = EncryptedSchemePlugin.get("sse")
        key = "0123456789abcdef0123456789abcdef"
        docs = _sample_docs()
        chunks = [
            {
                "doc_id": d.doc_id,
                "text": d.text,
                "metadata": d.metadata,
                "scheme_data": plugin.prepare_chunk(d.text, key),
            }
            for d in docs
        ]

        idx = backend.build_index(
            "EncryptedSearch",
            chunks,
            encrypted_search_scheme="sse",
            encrypted_search_version="sha256-v0",
        )

        with pytest.raises(BackendError, match="incompatible"):
            backend.encrypted_search(idx["index_id"], plugin.encrypt_query("q3 risk", key), 3)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_sim_server_uses_rdp_accumulation_not_per_round_eps_sum() -> None:
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    proc = _launch_sim_server(port)

    try:
        _wait_for_health(base_url)
        backend = create_backend(base_url)
        docs = _sample_docs()
        chunks = [{"doc_id": d.doc_id, "text": d.text, "metadata": d.metadata} for d in docs]
        idx = backend.build_index(
            "DiffPrivacy",
            chunks,
            epsilon=9.0,
            delta=1e-5,
        )

        emb1 = backend.embed_with_noise("q3 risk", sigma=1.0)
        emb2 = backend.embed_with_noise("vendor concentration", sigma=1.0)
        emb3 = backend.embed_with_noise("delayed remediation", sigma=1.0)

        # Under accumulated-RDP accounting: first two rounds pass, third exhausts.
        backend.retrieve_by_embedding(idx["index_id"], emb1, 2, query="q3 risk", sigma=1.0)
        backend.retrieve_by_embedding(
            idx["index_id"], emb2, 2, query="vendor concentration", sigma=1.0
        )
        with pytest.raises(BackendError, match="DP budget exhausted"):
            backend.retrieve_by_embedding(
                idx["index_id"], emb3, 2, query="delayed remediation", sigma=1.0
            )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_encrypted_search_http_and_rust_local_parity() -> None:
    pytest.importorskip("securerag_rs")

    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    proc = _launch_sim_server(port)

    try:
        _wait_for_health(base_url)

        docs = _sample_docs()
        fixed_key = "0123456789abcdef0123456789abcdef"
        plugin = EncryptedSchemePlugin.get("sse")

        chunks = [
            {
                "doc_id": d.doc_id,
                "text": d.text,
                "metadata": d.metadata,
                "scheme_data": plugin.prepare_chunk(d.text, fixed_key),
            }
            for d in docs
        ]

        http_backend = create_backend(base_url)
        rust_backend = create_backend("rust://local")

        http_idx = http_backend.build_index(
            "EncryptedSearch",
            chunks,
            encrypted_search_scheme="sse",
            encrypted_search_version="hmac-sha256-v1",
        )
        rust_idx = rust_backend.build_index(
            "EncryptedSearch",
            chunks,
            encrypted_search_scheme="sse",
            encrypted_search_version="hmac-sha256-v1",
        )

        encrypted_query = plugin.encrypt_query("q3 risk vendor concentration", fixed_key)

        http_rows = http_backend.encrypted_search(http_idx["index_id"], encrypted_query, 3)
        try:
            rust_rows = rust_backend.encrypted_search(rust_idx["index_id"], encrypted_query, 3)
        except BackendError as exc:
            if "unsupported operation: encrypted_search" in str(exc):
                pytest.skip("installed securerag_rs extension is stale and lacks encrypted_search")
            raise

        assert [r["doc_id"] for r in http_rows] == [r["doc_id"] for r in rust_rows]
        assert http_rows[0]["doc_id"] == "q3"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_encrypted_search_grpc_path_if_server_available() -> None:
    pytest.importorskip("grpc")

    root = Path(__file__).resolve().parents[1]
    grpc_bin = root / "securerag-rs" / "target" / "debug" / "securerag_grpc_server"
    if not grpc_bin.exists():
        pytest.skip("grpc server binary not present")

    grpc_port = _free_port()
    target = f"grpc://127.0.0.1:{grpc_port}"
    proc = subprocess.Popen(
        [str(grpc_bin), "--host", "127.0.0.1", "--port", str(grpc_port)],
        cwd=str(root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        _wait_for_tcp("127.0.0.1", grpc_port)

        backend = create_backend(target)
        plugin = EncryptedSchemePlugin.get("sse")
        fixed_key = "0123456789abcdef0123456789abcdef"
        docs = _sample_docs()

        chunks = [
            {
                "doc_id": d.doc_id,
                "text": d.text,
                "metadata": d.metadata,
                "scheme_data": plugin.prepare_chunk(d.text, fixed_key),
            }
            for d in docs
        ]
        idx = backend.build_index(
            "EncryptedSearch",
            chunks,
            encrypted_search_scheme="sse",
            encrypted_search_version="hmac-sha256-v1",
        )
        try:
            rows = backend.encrypted_search(
                idx["index_id"],
                plugin.encrypt_query("q3 risk vendor concentration", fixed_key),
                3,
            )
        except BackendError as exc:
            if "StatusCode.UNIMPLEMENTED" in str(exc):
                pytest.skip("grpc server binary is stale and lacks EncryptedSearch")
            raise
        assert rows
        assert rows[0]["doc_id"] == "q3"

        legacy_idx = backend.build_index(
            "EncryptedSearch",
            chunks,
            encrypted_search_scheme="sse",
            encrypted_search_version="sha256-v0",
        )
        with pytest.raises(BackendError, match="incompatible"):
            backend.encrypted_search(
                legacy_idx["index_id"],
                plugin.encrypt_query("q3 risk vendor concentration", fixed_key),
                3,
            )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
