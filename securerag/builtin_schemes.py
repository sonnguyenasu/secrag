from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from typing import Any

from securerag.scheme_plugin import EncryptedSchemePlugin


def _tokenize(text: str) -> list[str]:
    raw = re.findall(r"[a-z0-9]+", text.lower())
    out = []
    for tok in raw:
        if len(tok) > 3 and tok.endswith("s"):
            tok = tok[:-1]
        out.append(tok)
    return out


def _encrypt_token(token: str, key: str) -> str:
    return hmac.new(key.encode(), token.encode(), hashlib.sha256).hexdigest()


class SSEPlugin(EncryptedSchemePlugin):
    def generate_key(self) -> str:
        return secrets.token_hex(16)

    def prepare_chunk(self, text: str, key: str) -> dict[str, Any]:
        return {"enc_terms": [_encrypt_token(t, key) for t in _tokenize(text)]}

    def build_server_index(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        inv: dict[str, list[int]] = {}
        for i, row in enumerate(rows):
            for term in row.get("scheme_data", {}).get("enc_terms", []):
                inv.setdefault(term, []).append(i)
        return {"rows": rows, "inv": inv}

    def encrypt_query(self, query: str, key: str) -> dict[str, Any]:
        return {"enc_terms": [_encrypt_token(t, key) for t in _tokenize(query)]}

    def search(
        self,
        server_index: Any,
        encrypted_query: dict[str, Any],
        top_k: int,
    ) -> list[dict[str, Any]]:
        rows = server_index["rows"]
        inv = server_index["inv"]
        q_terms = encrypted_query.get("enc_terms", [])
        if not q_terms:
            return []

        counts: dict[int, int] = {}
        for term in q_terms:
            for idx in inv.get(term, []):
                counts[idx] = counts.get(idx, 0) + 1

        q_len = len(q_terms)
        scored = []
        for idx, inter in counts.items():
            row = rows[idx]
            doc_terms = row.get("scheme_data", {}).get("enc_terms", [])
            union = q_len + len(doc_terms) - inter
            score = inter / union if union else 0.0
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


class StructuredPlugin(EncryptedSchemePlugin):
    def __init__(self, use_bigrams: bool = True):
        self.use_bigrams = use_bigrams

    def generate_key(self) -> str:
        return secrets.token_hex(16)

    def _make_terms(self, text: str, key: str) -> list[str]:
        tokens = _tokenize(text)
        out = [_encrypt_token(f"tok:{t}", key) for t in tokens]
        if self.use_bigrams and len(tokens) >= 2:
            for i in range(len(tokens) - 1):
                out.append(_encrypt_token(f"bi:{tokens[i]}_{tokens[i + 1]}", key))
        return out

    def prepare_chunk(self, text: str, key: str) -> dict[str, Any]:
        return {"struct_terms": self._make_terms(text, key)}

    def build_server_index(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        inv: dict[str, list[int]] = {}
        for i, row in enumerate(rows):
            for term in row.get("scheme_data", {}).get("struct_terms", []):
                inv.setdefault(term, []).append(i)
        return {"rows": rows, "inv": inv}

    def encrypt_query(self, query: str, key: str) -> dict[str, Any]:
        return {"struct_terms": self._make_terms(query, key)}

    def search(
        self,
        server_index: Any,
        encrypted_query: dict[str, Any],
        top_k: int,
    ) -> list[dict[str, Any]]:
        rows = server_index["rows"]
        inv = server_index["inv"]
        q_terms = encrypted_query.get("struct_terms", [])
        if not q_terms:
            return []

        counts: dict[int, int] = {}
        for term in q_terms:
            for idx in inv.get(term, []):
                counts[idx] = counts.get(idx, 0) + 1

        q_len = len(q_terms)
        scored = []
        for idx, inter in counts.items():
            row = rows[idx]
            doc_terms = row.get("scheme_data", {}).get("struct_terms", [])
            union = q_len + len(doc_terms) - inter
            score = inter / union if union else 0.0
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


EncryptedSchemePlugin.register("sse", SSEPlugin())
EncryptedSchemePlugin.register("structured", StructuredPlugin(use_bigrams=True))
EncryptedSchemePlugin.register("structured_encryption", StructuredPlugin(use_bigrams=True))
