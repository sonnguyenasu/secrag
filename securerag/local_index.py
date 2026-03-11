from __future__ import annotations

import math


class LocalEmbeddingIndex:
    """Small, pure-Python lexical/embedding index for local-only prototyping."""

    def __init__(self, chunks: list[dict]):
        self._chunks = chunks
        self._vocab = self._build_vocab(chunks)

    @staticmethod
    def _build_vocab(chunks: list[dict]) -> dict[str, set[int]]:
        vocab: dict[str, set[int]] = {}
        for idx, chunk in enumerate(chunks):
            for tok in str(chunk.get("text", "")).lower().split():
                vocab.setdefault(tok, set()).add(idx)
        return vocab

    def search(self, query: str, top_k: int) -> list[dict]:
        scores: dict[int, int] = {}
        for tok in query.lower().split():
            for idx in self._vocab.get(tok, set()):
                scores[idx] = scores.get(idx, 0) + 1
        ranked = sorted(scores, key=lambda i: scores[i], reverse=True)[:top_k]
        return [self._chunks[i] for i in ranked]

    def search_by_embedding(self, embedding: list[float], top_k: int) -> list[dict]:
        if not self._chunks or "embedding" not in self._chunks[0]:
            raise ValueError("chunks do not contain embeddings")

        def cosine(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            return dot / (na * nb + 1e-9)

        scored = sorted(
            range(len(self._chunks)),
            key=lambda i: cosine(embedding, self._chunks[i]["embedding"]),
            reverse=True,
        )
        return [self._chunks[i] for i in scored[:top_k]]
