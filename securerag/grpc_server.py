from __future__ import annotations

import argparse
from concurrent import futures

import grpc
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Struct

from securerag.proto import secure_retrieval_pb2 as pb2
from securerag.proto import secure_retrieval_pb2_grpc as pb2_grpc
from securerag.sim_server import RPCRequest, rpc


class SecureRetrievalService(pb2_grpc.SecureRetrievalServicer):
    @staticmethod
    def _dict_to_struct(data: dict) -> Struct:
        out = Struct()
        json_format.ParseDict(data, out)
        return out

    @staticmethod
    def _struct_to_dict(data: Struct) -> dict:
        return json_format.MessageToDict(data)

    def Chunk(self, request: pb2.ChunkRequest, context):  # noqa: N802
        try:
            data = rpc(
                RPCRequest(
                    operation="chunk",
                    payload={
                        "docs": [self._struct_to_dict(d) for d in request.docs],
                        "chunk_size": int(request.chunk_size),
                        "overlap": int(request.overlap),
                    },
                )
            )
            if not data.get("ok", False):
                raise RuntimeError(str(data.get("error", "chunk failed")))
            return pb2.ChunkResponse(chunks=[self._dict_to_struct(x) for x in data.get("data", [])])
        except Exception as exc:
            context.abort(grpc.StatusCode.INTERNAL, str(exc))

    def Sanitize(self, request: pb2.SanitizeRequest, context):  # noqa: N802
        try:
            data = rpc(
                RPCRequest(
                    operation="sanitize",
                    payload={"chunks": [self._struct_to_dict(c) for c in request.chunks]},
                )
            )
            if not data.get("ok", False):
                raise RuntimeError(str(data.get("error", "sanitize failed")))
            return pb2.SanitizeResponse(chunks=[self._dict_to_struct(x) for x in data.get("data", [])])
        except Exception as exc:
            context.abort(grpc.StatusCode.INTERNAL, str(exc))

    def BuildIndex(self, request: pb2.BuildIndexRequest, context):  # noqa: N802
        try:
            data = rpc(
                RPCRequest(
                    operation="build_index",
                    payload={
                        "protocol": request.protocol,
                        "chunks": [self._struct_to_dict(c) for c in request.chunks],
                    },
                )
            )
            if not data.get("ok", False):
                raise RuntimeError(str(data.get("error", "build_index failed")))
            out = data.get("data", {})
            return pb2.BuildIndexResponse(index_id=str(out.get("index_id", "")), doc_count=int(out.get("doc_count", 0)))
        except Exception as exc:
            context.abort(grpc.StatusCode.INTERNAL, str(exc))

    def GenerateDecoys(self, request: pb2.GenerateDecoysRequest, context):  # noqa: N802
        try:
            data = rpc(
                RPCRequest(
                    operation="generate_decoys",
                    payload={"index_id": request.index_id, "query": request.query, "k": int(request.k)},
                )
            )
            if not data.get("ok", False):
                raise RuntimeError(str(data.get("error", "generate_decoys failed")))
            return pb2.GenerateDecoysResponse(decoys=[str(x) for x in data.get("data", [])])
        except Exception as exc:
            context.abort(grpc.StatusCode.INTERNAL, str(exc))

    def BatchRetrieve(self, request: pb2.BatchRetrieveRequest, context):  # noqa: N802
        try:
            data = rpc(
                RPCRequest(
                    operation="batch_retrieve",
                    payload={
                        "index_id": request.index_id,
                        "queries": list(request.queries),
                        "top_k": int(request.top_k),
                    },
                )
            )
            if not data.get("ok", False):
                raise RuntimeError(str(data.get("error", "batch_retrieve failed")))
            outer = []
            for rows in data.get("data", []):
                outer.append(pb2.RetrievalList(rows=[self._dict_to_struct(r) for r in rows]))
            return pb2.BatchRetrieveResponse(rows=outer)
        except Exception as exc:
            context.abort(grpc.StatusCode.INTERNAL, str(exc))

    def EmbedWithNoise(self, request: pb2.EmbedWithNoiseRequest, context):  # noqa: N802
        try:
            data = rpc(
                RPCRequest(
                    operation="embed_with_noise",
                    payload={"query": request.query, "sigma": float(request.sigma)},
                )
            )
            if not data.get("ok", False):
                raise RuntimeError(str(data.get("error", "embed_with_noise failed")))
            return pb2.EmbedWithNoiseResponse(embedding=[float(x) for x in data.get("data", [])])
        except Exception as exc:
            context.abort(grpc.StatusCode.INTERNAL, str(exc))

    def RetrieveByEmbedding(self, request: pb2.RetrieveByEmbeddingRequest, context):  # noqa: N802
        try:
            data = rpc(
                RPCRequest(
                    operation="retrieve_by_embedding",
                    payload={
                        "index_id": request.index_id,
                        "embedding": [float(x) for x in request.embedding],
                        "top_k": int(request.top_k),
                        "query": request.query,
                    },
                )
            )
            if not data.get("ok", False):
                raise RuntimeError(str(data.get("error", "retrieve_by_embedding failed")))
            return pb2.RetrieveByEmbeddingResponse(rows=[self._dict_to_struct(r) for r in data.get("data", [])])
        except Exception as exc:
            context.abort(grpc.StatusCode.INTERNAL, str(exc))

    def SseGenerateKey(self, request: pb2.SseGenerateKeyRequest, context):  # noqa: N802
        try:
            data = rpc(RPCRequest(operation="sse_generate_key", payload={}))
            if not data.get("ok", False):
                raise RuntimeError(str(data.get("error", "sse_generate_key failed")))
            return pb2.SseGenerateKeyResponse(key=str(data.get("data", "")))
        except Exception as exc:
            context.abort(grpc.StatusCode.INTERNAL, str(exc))

    def SseEncryptTerms(self, request: pb2.SseEncryptTermsRequest, context):  # noqa: N802
        try:
            data = rpc(
                RPCRequest(
                    operation="sse_encrypt_terms",
                    payload={"text": request.text, "key": request.key},
                )
            )
            if not data.get("ok", False):
                raise RuntimeError(str(data.get("error", "sse_encrypt_terms failed")))
            return pb2.SseEncryptTermsResponse(terms=[str(x) for x in data.get("data", [])])
        except Exception as exc:
            context.abort(grpc.StatusCode.INTERNAL, str(exc))

    def SseEncryptStructuredTerms(self, request: pb2.SseEncryptStructuredTermsRequest, context):  # noqa: N802
        try:
            data = rpc(
                RPCRequest(
                    operation="sse_encrypt_structured_terms",
                    payload={"text": request.text, "key": request.key, "use_bigrams": bool(request.use_bigrams)},
                )
            )
            if not data.get("ok", False):
                raise RuntimeError(str(data.get("error", "sse_encrypt_structured_terms failed")))
            return pb2.SseEncryptStructuredTermsResponse(terms=[str(x) for x in data.get("data", [])])
        except Exception as exc:
            context.abort(grpc.StatusCode.INTERNAL, str(exc))

    def SsePrepareChunks(self, request: pb2.SsePrepareChunksRequest, context):  # noqa: N802
        try:
            data = rpc(
                RPCRequest(
                    operation="sse_prepare_chunks",
                    payload={
                        "chunks": [self._struct_to_dict(c) for c in request.chunks],
                        "key": request.key,
                        "scheme": request.scheme,
                        "use_bigrams": bool(request.use_bigrams),
                    },
                )
            )
            if not data.get("ok", False):
                raise RuntimeError(str(data.get("error", "sse_prepare_chunks failed")))
            return pb2.SsePrepareChunksResponse(chunks=[self._dict_to_struct(c) for c in data.get("data", [])])
        except Exception as exc:
            context.abort(grpc.StatusCode.INTERNAL, str(exc))

    def SseSearch(self, request: pb2.SseSearchRequest, context):  # noqa: N802
        try:
            data = rpc(
                RPCRequest(
                    operation="sse_search",
                    payload={
                        "index_id": request.index_id,
                        "enc_terms": list(request.enc_terms),
                        "top_k": int(request.top_k),
                    },
                )
            )
            if not data.get("ok", False):
                raise RuntimeError(str(data.get("error", "sse_search failed")))
            return pb2.SseSearchResponse(rows=[self._dict_to_struct(r) for r in data.get("data", [])])
        except Exception as exc:
            context.abort(grpc.StatusCode.INTERNAL, str(exc))

    def StructuredSearch(self, request: pb2.StructuredSearchRequest, context):  # noqa: N802
        try:
            data = rpc(
                RPCRequest(
                    operation="structured_search",
                    payload={
                        "index_id": request.index_id,
                        "struct_terms": list(request.struct_terms),
                        "top_k": int(request.top_k),
                    },
                )
            )
            if not data.get("ok", False):
                raise RuntimeError(str(data.get("error", "structured_search failed")))
            return pb2.StructuredSearchResponse(rows=[self._dict_to_struct(r) for r in data.get("data", [])])
        except Exception as exc:
            context.abort(grpc.StatusCode.INTERNAL, str(exc))


def serve(host: str = "127.0.0.1", port: int = 50051) -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    pb2_grpc.add_SecureRetrievalServicer_to_server(SecureRetrievalService(), server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    server.wait_for_termination()


def main() -> None:
    parser = argparse.ArgumentParser(description="SecureRAG gRPC server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=50051, type=int)
    args = parser.parse_args()
    serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
