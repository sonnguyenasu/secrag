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
    def sse_generate_key(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def sse_encrypt_terms(self, text: str, key: str) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def sse_encrypt_structured_terms(
        self,
        text: str,
        key: str,
        *,
        use_bigrams: bool = True,
    ) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def sse_prepare_chunks(
        self,
        chunks: list[dict],
        key: str,
        scheme: str,
        *,
        use_bigrams: bool = True,
    ) -> list[dict]:
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

    def sse_generate_key(self) -> str:
        resp = self._invoke("SseGenerateKey", self._grpc_pb2.SseGenerateKeyRequest())
        return str(resp.key)

    def sse_encrypt_terms(self, text: str, key: str) -> list[str]:
        resp = self._invoke("SseEncryptTerms", self._grpc_pb2.SseEncryptTermsRequest(text=text, key=key))
        return [str(x) for x in resp.terms]

    def sse_encrypt_structured_terms(
        self,
        text: str,
        key: str,
        *,
        use_bigrams: bool = True,
    ) -> list[str]:
        resp = self._invoke(
            "SseEncryptStructuredTerms",
            self._grpc_pb2.SseEncryptStructuredTermsRequest(
                text=text,
                key=key,
                use_bigrams=use_bigrams,
            ),
        )
        return [str(x) for x in resp.terms]

    def sse_prepare_chunks(
        self,
        chunks: list[dict],
        key: str,
        scheme: str,
        *,
        use_bigrams: bool = True,
    ) -> list[dict]:
        req = self._grpc_pb2.SsePrepareChunksRequest(
            chunks=[self._dict_to_struct(c) for c in chunks],
            key=key,
            scheme=scheme,
            use_bigrams=use_bigrams,
        )
        resp = self._invoke("SsePrepareChunks", req)
        return [self._struct_to_dict(c) for c in resp.chunks]

    def build_index(self, protocol: str, chunks: list[dict]) -> dict:
        req = self._grpc_pb2.BuildIndexRequest(
            protocol=protocol,
            chunks=[self._dict_to_struct(c) for c in chunks],
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
    ) -> list[dict]:
        req = self._grpc_pb2.RetrieveByEmbeddingRequest(
            index_id=index_id,
            embedding=embedding,
            top_k=top_k,
            query=query or "",
        )
        resp = self._invoke("RetrieveByEmbedding", req)
        return [self._struct_to_dict(r) for r in resp.rows]

    def sse_search(self, index_id: str, enc_terms: list[str], top_k: int) -> list[dict]:
        req = self._grpc_pb2.SseSearchRequest(index_id=index_id, enc_terms=enc_terms, top_k=top_k)
        resp = self._invoke("SseSearch", req)
        return [self._struct_to_dict(r) for r in resp.rows]

    def structured_search(self, index_id: str, struct_terms: list[str], top_k: int) -> list[dict]:
        req = self._grpc_pb2.StructuredSearchRequest(index_id=index_id, struct_terms=struct_terms, top_k=top_k)
        resp = self._invoke("StructuredSearch", req)
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
