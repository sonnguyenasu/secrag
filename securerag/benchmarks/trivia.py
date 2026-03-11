from __future__ import annotations

from securerag.benchmarks.loaders import QueryRecord
from securerag.models import RawDocument


class TriviaQA:
    @staticmethod
    def load(split: str = "test", n: int = 100) -> tuple[list[RawDocument], list[QueryRecord]]:
        _ = (split, n)
        return [], []
