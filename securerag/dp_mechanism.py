from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


_REGISTRY: dict[str, "DPMechanismPlugin"] = {}


class DPMechanismPlugin(ABC):
    """Extension point for differential privacy noise mechanisms."""

    @abstractmethod
    def noise(self, embedding: list[float], sigma: float, *, query: str = "") -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def rdp_cost(self, sigma: float, alpha: float) -> float:
        raise NotImplementedError

    def rdp_orders(self) -> list[float]:
        return [2.0, 4.0, 8.0, 16.0, 32.0]

    def prepare_corpus(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return chunks

    @classmethod
    def register(cls, name: str, plugin: "DPMechanismPlugin") -> None:
        _REGISTRY[name.lower()] = plugin

    @classmethod
    def get(cls, name: str) -> "DPMechanismPlugin":
        key = name.lower()
        if key not in _REGISTRY:
            raise KeyError(
                f"No DP mechanism '{name}' registered. Available: {sorted(_REGISTRY)}"
            )
        return _REGISTRY[key]

    @classmethod
    def registered_names(cls) -> list[str]:
        return sorted(_REGISTRY)
