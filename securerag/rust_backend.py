from __future__ import annotations

from typing import Any

from securerag.errors import BackendError


class RustBackend:
    """
    Adapter over the optional PyO3 extension module `securerag_rs`.
    The module is intentionally optional so HTTP pseudo-remote remains usable.
    """

    def __init__(self) -> None:
        try:
            import securerag_rs as rs  # type: ignore
        except Exception as exc:
            raise BackendError(
                "Rust backend requested but `securerag_rs` is not available. "
                "Build it with maturin from securerag-rs/."
            ) from exc
        self._bridge = rs.BackendBridge()

    def _call(self, op: str, payload: dict[str, Any]) -> Any:
        try:
            return self._bridge.rpc(op, payload)
        except Exception as exc:
            raise BackendError(f"Rust backend error for {op}: {exc}") from exc

    def chunk(self, docs: list[dict], chunk_size: int, overlap: int) -> list[dict]:
        return self._call("chunk", {"docs": docs, "chunk_size": chunk_size, "overlap": overlap})

    def sanitize(self, chunks: list[dict]) -> list[dict]:
        return self._call("sanitize", {"chunks": chunks})

    def sse_generate_key(self) -> str:
        return self._call("sse_generate_key", {})

    def sse_encrypt_terms(self, text: str, key: str) -> list[str]:
        return self._call("sse_encrypt_terms", {"text": text, "key": key})

    def sse_encrypt_structured_terms(
        self,
        text: str,
        key: str,
        *,
        use_bigrams: bool = True,
    ) -> list[str]:
        return self._call(
            "sse_encrypt_structured_terms",
            {"text": text, "key": key, "use_bigrams": use_bigrams},
        )

    def sse_prepare_chunks(
        self,
        chunks: list[dict],
        key: str,
        scheme: str,
        *,
        use_bigrams: bool = True,
    ) -> list[dict]:
        return self._call(
            "sse_prepare_chunks",
            {
                "chunks": chunks,
                "key": key,
                "scheme": scheme,
                "use_bigrams": use_bigrams,
            },
        )

    def build_index(self, protocol: str, chunks: list[dict]) -> dict:
        return self._call("build_index", {"protocol": protocol, "chunks": chunks})

    def generate_decoys(self, index_id: str, query: str, k: int) -> list[str]:
        return self._call("generate_decoys", {"index_id": index_id, "query": query, "k": k})

    def batch_retrieve(self, index_id: str, queries: list[str], top_k: int) -> list[list[dict]]:
        return self._call(
            "batch_retrieve",
            {"index_id": index_id, "queries": queries, "top_k": top_k},
        )

    def embed_with_noise(self, query: str, sigma: float) -> list[float]:
        return self._call("embed_with_noise", {"query": query, "sigma": sigma})

    def retrieve_by_embedding(
        self,
        index_id: str,
        embedding: list[float],
        top_k: int,
        query: str | None = None,
    ) -> list[dict]:
        return self._call(
            "retrieve_by_embedding",
            {
                "index_id": index_id,
                "embedding": embedding,
                "top_k": top_k,
                "query": query,
            },
        )

    def sse_search(self, index_id: str, enc_terms: list[str], top_k: int) -> list[dict]:
        return self._call(
            "sse_search",
            {
                "index_id": index_id,
                "enc_terms": enc_terms,
                "top_k": top_k,
            },
        )

    def structured_search(self, index_id: str, struct_terms: list[str], top_k: int) -> list[dict]:
        return self._call(
            "structured_search",
            {
                "index_id": index_id,
                "struct_terms": struct_terms,
                "top_k": top_k,
            },
        )
