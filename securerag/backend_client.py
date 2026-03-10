from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import httpx

from securerag.errors import BackendError


class Backend(ABC):
    @abstractmethod
    def chunk(self, docs: list[dict], chunk_size: int, overlap: int) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def sanitize(self, chunks: list[dict]) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def build_index(self, protocol: str, chunks: list[dict]) -> dict:
        raise NotImplementedError

    @abstractmethod
    def generate_decoys(self, index_id: str, query: str, k: int) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def batch_retrieve(self, index_id: str, queries: list[str], top_k: int) -> list[list[dict]]:
        raise NotImplementedError

    @abstractmethod
    def embed_with_noise(self, query: str, sigma: float) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def retrieve_by_embedding(
        self,
        index_id: str,
        embedding: list[float],
        top_k: int,
        query: str | None = None,
    ) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def sse_search(self, index_id: str, enc_terms: list[str], top_k: int) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def structured_search(self, index_id: str, struct_terms: list[str], top_k: int) -> list[dict]:
        raise NotImplementedError


class RemoteBackend(Backend):
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _call(self, operation: str, payload: dict[str, Any]) -> Any:
        body = {"operation": operation, "payload": payload}
        try:
            resp = httpx.post(f"{self.base_url}/rpc", json=body, timeout=30.0)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            raise BackendError(f"Backend call failed for {operation}: {exc}") from exc
        if not data.get("ok", False):
            raise BackendError(data.get("error", f"Unknown backend error in {operation}"))
        return data.get("data")

    def chunk(self, docs: list[dict], chunk_size: int, overlap: int) -> list[dict]:
        return self._call("chunk", {"docs": docs, "chunk_size": chunk_size, "overlap": overlap})

    def sanitize(self, chunks: list[dict]) -> list[dict]:
        return self._call("sanitize", {"chunks": chunks})

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


def create_backend(target: str) -> Backend:
    if target.startswith("http://") or target.startswith("https://"):
        return RemoteBackend(target)

    if target in {"rust://local", "local-rust", "rust-local"}:
        from securerag.rust_backend import RustBackend

        return RustBackend()

    raise BackendError(
        f"Unknown backend target: {target}. Use http(s)://host:port or rust://local."
    )
