from __future__ import annotations

from securerag.models import QueryRecord, RawDocument


def load_wikipedia_corpus(subset: str = "2018-12") -> list[RawDocument]:
    # Placeholder loader with a stable API; dataset wiring is intentionally lightweight.
    _ = subset
    return []
