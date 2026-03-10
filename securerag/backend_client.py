from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import httpx
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Struct

from securerag.errors import BackendError


class Backend(ABC):
    @abstractmethod
    def chunk(self, docs: list[dict], chunk_size: int, overlap: int) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def sanitize(self, chunks: list[dict]) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def generate_decoys(self, index_id: str, query: str, k: int) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def batch_retrieve(self, index_id: str, queries: list[str], top_k: int) -> list[list[dict]]:
        raise NotImplementedError

    @abstractmethod
    def embed(self, query: str) -> list[float]:
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
        sigma: float | None = None,
    ) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def encrypted_search(self, index_id: str, encrypted_query: dict[str, Any], top_k: int) -> list[dict]:
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
        return self._call(
            "build_index",
            {
                "protocol": protocol,
                "chunks": chunks,
                "epsilon": epsilon,
                "delta": delta,
                "encrypted_search_scheme": encrypted_search_scheme,
                "encrypted_search_version": encrypted_search_version,
            },
        )

    def generate_decoys(self, index_id: str, query: str, k: int) -> list[str]:
        return self._call("generate_decoys", {"index_id": index_id, "query": query, "k": k})

    def batch_retrieve(self, index_id: str, queries: list[str], top_k: int) -> list[list[dict]]:
        return self._call(
            "batch_retrieve",
            {"index_id": index_id, "queries": queries, "top_k": top_k},
        )

    def embed(self, query: str) -> list[float]:
        return self._call("embed_with_noise", {"query": query, "sigma": 0.0})

    def embed_with_noise(self, query: str, sigma: float) -> list[float]:
        return self._call("embed_with_noise", {"query": query, "sigma": sigma})

    def retrieve_by_embedding(
        self,
        index_id: str,
        embedding: list[float],
        top_k: int,
        query: str | None = None,
        sigma: float | None = None,
    ) -> list[dict]:
        return self._call(
            "retrieve_by_embedding",
            {
                "index_id": index_id,
                "embedding": embedding,
                "top_k": top_k,
                "query": query,
                "sigma": sigma,
            },
        )

    def encrypted_search(self, index_id: str, encrypted_query: dict[str, Any], top_k: int) -> list[dict]:
        return self._call(
            "encrypted_search",
            {
                "index_id": index_id,
                "encrypted_query": encrypted_query,
                "top_k": top_k,
            },
        )


class GrpcBackend(Backend):
    def __init__(self, target: str):
        try:
            import grpc
            from securerag.proto import secure_retrieval_pb2 as grpc_pb2
            from securerag.proto import secure_retrieval_pb2_grpc as grpc_pb2_grpc
        except Exception as exc:
            raise BackendError(
                "gRPC backend requested but grpcio is unavailable. Install grpcio and grpcio-tools."
            ) from exc
        self._grpc = grpc
        self._grpc_pb2 = grpc_pb2
        self._channel = grpc.insecure_channel(target)
        self._stub = grpc_pb2_grpc.SecureRetrievalStub(self._channel)

    @staticmethod
    def _dict_to_struct(data: dict) -> Struct:
        out = Struct()
        json_format.ParseDict(data, out)
        return out

    @staticmethod
    def _struct_to_dict(data: Struct) -> dict:
        return json_format.MessageToDict(data)

    def _invoke(self, fn_name: str, req: Any) -> Any:
        fn = getattr(self._stub, fn_name)
        try:
            return fn(req, timeout=30.0)
        except Exception as exc:
            raise BackendError(f"Backend call failed for {fn_name}: {exc}") from exc

    def chunk(self, docs: list[dict], chunk_size: int, overlap: int) -> list[dict]:
        req = self._grpc_pb2.ChunkRequest(
            docs=[self._dict_to_struct(d) for d in docs],
            chunk_size=chunk_size,
            overlap=overlap,
        )
        resp = self._invoke("Chunk", req)
        return [self._struct_to_dict(c) for c in resp.chunks]

    def sanitize(self, chunks: list[dict]) -> list[dict]:
        req = self._grpc_pb2.SanitizeRequest(chunks=[self._dict_to_struct(c) for c in chunks])
        resp = self._invoke("Sanitize", req)
        return [self._struct_to_dict(c) for c in resp.chunks]

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
        req = self._grpc_pb2.BuildIndexRequest(
            protocol=protocol,
            chunks=[self._dict_to_struct(c) for c in chunks],
            epsilon=epsilon,
            delta=delta,
            encrypted_search_scheme=encrypted_search_scheme,
            encrypted_search_version=encrypted_search_version,
        )
        resp = self._invoke("BuildIndex", req)
        return {"index_id": str(resp.index_id), "doc_count": int(resp.doc_count)}

    def generate_decoys(self, index_id: str, query: str, k: int) -> list[str]:
        req = self._grpc_pb2.GenerateDecoysRequest(index_id=index_id, query=query, k=k)
        resp = self._invoke("GenerateDecoys", req)
        return [str(x) for x in resp.decoys]

    def batch_retrieve(self, index_id: str, queries: list[str], top_k: int) -> list[list[dict]]:
        req = self._grpc_pb2.BatchRetrieveRequest(index_id=index_id, queries=queries, top_k=top_k)
        resp = self._invoke("BatchRetrieve", req)
        return [[self._struct_to_dict(r) for r in lst.rows] for lst in resp.rows]

    def embed(self, query: str) -> list[float]:
        req = self._grpc_pb2.EmbedWithNoiseRequest(query=query, sigma=0.0)
        resp = self._invoke("EmbedWithNoise", req)
        return [float(x) for x in resp.embedding]

    def embed_with_noise(self, query: str, sigma: float) -> list[float]:
        req = self._grpc_pb2.EmbedWithNoiseRequest(query=query, sigma=sigma)
        resp = self._invoke("EmbedWithNoise", req)
        return [float(x) for x in resp.embedding]

    def retrieve_by_embedding(
        self,
        index_id: str,
        embedding: list[float],
        top_k: int,
        query: str | None = None,
        sigma: float | None = None,
    ) -> list[dict]:
        req = self._grpc_pb2.RetrieveByEmbeddingRequest(
            index_id=index_id,
            embedding=embedding,
            top_k=top_k,
            query=query or "",
            sigma=float(sigma if sigma is not None else 1.0),
        )
        resp = self._invoke("RetrieveByEmbedding", req)
        return [self._struct_to_dict(r) for r in resp.rows]

    def encrypted_search(self, index_id: str, encrypted_query: dict[str, Any], top_k: int) -> list[dict]:
        req = self._grpc_pb2.EncryptedSearchRequest(
            index_id=index_id,
            encrypted_query=self._dict_to_struct(encrypted_query),
            top_k=top_k,
        )
        resp = self._invoke("EncryptedSearch", req)
        return [self._struct_to_dict(r) for r in resp.rows]


def create_backend(target: str) -> Backend:
    if target.startswith("http://") or target.startswith("https://"):
        return RemoteBackend(target)

    if target.startswith("grpc://"):
        return GrpcBackend(target.removeprefix("grpc://"))

    if target in {"rust://local", "local-rust", "rust-local"}:
        from securerag.rust_backend import RustBackend

        return RustBackend()

    raise BackendError(
        f"Unknown backend target: {target}. Use http(s)://host:port, grpc://host:port, or rust://local."
    )
