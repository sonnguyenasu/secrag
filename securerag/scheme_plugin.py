from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


_REGISTRY: dict[str, "EncryptedSchemePlugin"] = {}


class EncryptedSchemePlugin(ABC):
    @abstractmethod
    def generate_key(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def prepare_chunk(self, text: str, key: Any) -> dict[str, Any]:
        raise NotImplementedError

    def build_server_index(self, rows: list[dict[str, Any]]) -> Any:
        return rows

    @abstractmethod
    def encrypt_query(self, query: str, key: Any) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        server_index: Any,
        encrypted_query: dict[str, Any],
        top_k: int,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    @classmethod
    def register(cls, name: str, plugin: "EncryptedSchemePlugin") -> None:
        _REGISTRY[name.lower()] = plugin

    @classmethod
    def get(cls, name: str) -> "EncryptedSchemePlugin":
        key = name.lower()
        if key not in _REGISTRY:
            raise KeyError(
                f"No encrypted search scheme '{name}' is registered. "
                f"Available: {sorted(_REGISTRY)}"
            )
        return _REGISTRY[key]

    @classmethod
    def registered_names(cls) -> list[str]:
        return sorted(_REGISTRY)
