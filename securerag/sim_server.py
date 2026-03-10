from __future__ import annotations

import hashlib
import math
import random
import re
import uuid
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="SecureRAG pseudo-remote server", version="0.1.0")
_INDEXES: dict[str, dict[str, Any]] = {}
LEXICAL_WEIGHT = 0.65
EMBEDDING_WEIGHT = 0.35
ENCRYPTED_SEARCH_VERSION = "hmac-sha256-v1"


def _rdp_to_dp(rdp_eps: list[float], delta: float, orders: list[float]) -> float:
    return min(r + math.log(1.0 / delta) / (a - 1.0) for a, r in zip(orders, rdp_eps))


class RPCRequest(BaseModel):
    operation: str
    payload: dict[str, Any]


def _tokenize(text: str) -> list[str]:
    raw = re.findall(r"[a-z0-9]+", text.lower())
    tokens = []
    for tok in raw:
        if len(tok) > 3 and tok.endswith("s"):
            tok = tok[:-1]
        tokens.append(tok)
    return tokens


def _embed(text: str, dim: int = 64) -> list[float]:
    # Token-hashed bag-of-words embedding for deterministic lexical-semantic behavior.
    vec = [0.0] * dim
    tokens = _tokenize(text)
    if not tokens:
        return vec
    for tok in tokens:
        h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _noise_seed(query: str) -> int:
    digest = hashlib.sha256(query.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _retrieve_lexical(index: dict[str, Any], query: str, top_k: int) -> list[dict]:
    q_tokens = set(_tokenize(query))
    scored = []
    for row in index["rows"]:
        t_tokens = set(_tokenize(row["text"]))
        inter = len(q_tokens & t_tokens)
        union = len(q_tokens | t_tokens) or 1
        score = inter / union
        scored.append(
            {
                "doc_id": row["doc_id"],
                "text": row["text"],
                "metadata": row.get("metadata", {}),
                "score": score,
            }
        )
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def _retrieve_embedding(index: dict[str, Any], emb: list[float], top_k: int, query: str | None = None) -> list[dict]:
    # Blend lexical and embedding signals to improve relevance in the MVP simulation.
    q_tokens = set(_tokenize(query)) if query else set()
    scored = []
    for row in index["rows"]:
        emb_score = _cos(emb, row["embedding"])
        if q_tokens:
            t_tokens = set(_tokenize(row["text"]))
            inter = len(q_tokens & t_tokens)
            union = len(q_tokens | t_tokens) or 1
            lex_score = inter / union
            score = LEXICAL_WEIGHT * lex_score + EMBEDDING_WEIGHT * emb_score
        else:
            score = emb_score
        scored.append(
            {
                "doc_id": row["doc_id"],
                "text": row["text"],
                "metadata": row.get("metadata", {}),
                "score": score,
            }
        )
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/rpc")
def rpc(req: RPCRequest) -> dict[str, Any]:
    try:
        op = req.operation
        p = req.payload

        if op == "chunk":
            docs = p["docs"]
            chunk_size = int(p["chunk_size"])
            overlap = int(p["overlap"])
            step = max(1, chunk_size - overlap)
            chunks = []
            for d in docs:
                text = d["text"]
                if len(text) <= chunk_size:
                    chunks.append(
                        {
                            "doc_id": d["doc_id"],
                            "text": text,
                            "metadata": d.get("metadata", {}),
                        }
                    )
                    continue
                for i in range(0, len(text), step):
                    snippet = text[i : i + chunk_size]
                    if not snippet:
                        continue
                    # Skip tiny trailing fragments to avoid low-quality evidence snippets.
                    if i > 0 and len(snippet) < max(24, chunk_size // 3):
                        continue
                    chunks.append(
                        {
                            "doc_id": d["doc_id"],
                            "text": snippet,
                            "metadata": d.get("metadata", {}),
                        }
                    )
            return {"ok": True, "data": chunks}

        if op == "sanitize":
            chunks = p["chunks"]
            bad = ["ignore previous instructions", "system prompt", "developer instructions"]
            out = []
            for c in chunks:
                text = c["text"]
                for token in bad:
                    text = re.sub(re.escape(token), "", text, flags=re.IGNORECASE)
                out.append({**c, "text": text})
            return {"ok": True, "data": out}

        if op == "build_index":
            protocol = p["protocol"]
            epsilon = float(p.get("epsilon", 1_000_000.0))
            delta = float(p.get("delta", 1e-5))
            chunks = p["chunks"]
            scheme = p.get("encrypted_search_scheme")
            scheme_version = p.get("encrypted_search_version") or "sha256-v0"

            rows = []
            for c in chunks:
                row = {
                    "doc_id": c["doc_id"],
                    "text": c["text"],
                    "metadata": c.get("metadata", {}),
                    "embedding": _embed(c["text"]),
                    "scheme_data": c.get("scheme_data", {}),
                }
                rows.append(row)

            if scheme:
                try:
                    from securerag.scheme_plugin import EncryptedSchemePlugin

                    plugin = EncryptedSchemePlugin.get(scheme)
                    server_index = plugin.build_server_index(rows)
                except KeyError:
                    server_index = rows
            else:
                server_index = rows

            index_id = str(uuid.uuid4())
            _INDEXES[index_id] = {
                "protocol": protocol,
                "server_index": server_index,
                "rows": rows,
                "scheme": scheme,
                "encrypted_search_version": scheme_version,
                "epsilon": epsilon,
                "delta": delta,
                "rdp_acc": [0.0] * 5,
                "spent": 0.0,
            }
            return {"ok": True, "data": {"index_id": index_id, "doc_count": len(rows)}}

        if op == "encrypted_search":
            index = _INDEXES[p["index_id"]]
            encrypted_query = p["encrypted_query"]
            top_k = int(p["top_k"])
            scheme = index.get("scheme")
            version = index.get("encrypted_search_version") or "sha256-v0"

            if scheme and version != ENCRYPTED_SEARCH_VERSION:
                return {
                    "ok": False,
                    "error": (
                        f"Index built with crypto version '{version}' is incompatible with "
                        f"current '{ENCRYPTED_SEARCH_VERSION}'. Rebuild the corpus."
                    ),
                }

            if scheme:
                from securerag.scheme_plugin import EncryptedSchemePlugin

                plugin = EncryptedSchemePlugin.get(scheme)
                results = plugin.search(index["server_index"], encrypted_query, top_k)
            else:
                return {"ok": False, "error": "Index was not built with an encrypted scheme"}

            return {"ok": True, "data": results}

        if op == "generate_decoys":
            index = _INDEXES[p["index_id"]]
            k = int(p["k"])
            query = p["query"]
            rnd = random.Random(hash(query) & 0xFFFF)
            texts = [c["text"] for c in index["rows"]]
            if not texts:
                return {"ok": True, "data": [query] * k}
            return {"ok": True, "data": [texts[rnd.randrange(len(texts))][:80] for _ in range(k)]}

        if op == "batch_retrieve":
            index = _INDEXES[p["index_id"]]
            top_k = int(p["top_k"])
            queries = p["queries"]
            out = [_retrieve_lexical(index, q, top_k) for q in queries]
            return {"ok": True, "data": out}

        if op == "embed_with_noise":
            query = p["query"]
            sigma = float(p["sigma"])
            emb = _embed(query)
            rnd = random.Random(_noise_seed(query))
            noised = [v + rnd.gauss(0.0, sigma) for v in emb]
            return {"ok": True, "data": noised}

        if op == "retrieve_by_embedding":
            index = _INDEXES[p["index_id"]]
            emb = p["embedding"]
            top_k = int(p["top_k"])
            query = p.get("query")
            sigma = float(p.get("sigma", 1.0))
            if index.get("protocol") == "DiffPrivacy" and sigma > 0.0:
                orders = [2.0, 4.0, 8.0, 16.0, 32.0]
                rdp = [a / (2.0 * sigma * sigma) for a in orders]
                delta = float(index.get("delta", 1e-5))
                acc = list(index.get("rdp_acc", [0.0] * len(orders)))
                if len(acc) != len(orders):
                    acc = [0.0] * len(orders)
                candidate_acc = [x + y for x, y in zip(acc, rdp)]
                candidate_eps = _rdp_to_dp(candidate_acc, delta, orders)
                if candidate_eps > float(index.get("epsilon", 1.0)):
                    return {"ok": False, "error": "DP budget exhausted"}
                index["rdp_acc"] = candidate_acc
                index["spent"] = candidate_eps
            out = _retrieve_embedding(index, emb, top_k, query=query)
            return {"ok": True, "data": out}

        return {"ok": False, "error": f"Unsupported operation: {op}"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
