from __future__ import annotations

import hashlib
import secrets


def generate_sse_key() -> str:
    return secrets.token_hex(16)


def tokenize(text: str) -> list[str]:
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]


def encrypt_token(token: str, key: str) -> str:
    digest = hashlib.sha256(f"{key}:{token}".encode("utf-8")).hexdigest()
    return digest[:24]


def encrypt_terms(text: str, key: str) -> list[str]:
    return [encrypt_token(t, key) for t in tokenize(text)]


def encrypt_structured_terms(
    text: str,
    key: str,
    *,
    use_bigrams: bool = True,
) -> list[str]:
    tokens = tokenize(text)
    out: list[str] = []

    # Unigram channel
    out.extend(encrypt_token(f"tok:{t}", key) for t in tokens)

    # Bigram channel adds local-order structure while keeping deterministic searchability.
    if use_bigrams and len(tokens) >= 2:
        for i in range(len(tokens) - 1):
            bg = f"{tokens[i]}_{tokens[i + 1]}"
            out.append(encrypt_token(f"bi:{bg}", key))

    return out
