from __future__ import annotations

from abc import ABC, abstractmethod
import logging

from securerag.backend_client import create_backend
from securerag.budget import BudgetManager
from securerag.config import PrivacyConfig
from securerag.errors import ProtocolMismatchError, UnknownProtocolError
from securerag.models import Document
from securerag.protocol import PrivacyProtocol


class PrivacyRetriever(ABC):
    _registry: dict[PrivacyProtocol, type["PrivacyRetriever"]] = {}

    def __init__(self, config: PrivacyConfig, corpus):
        self._validate_compatibility(config.protocol, corpus.protocol)
        self.config = config
        self.corpus = corpus
        self.budget = BudgetManager(config)
        self._backend = create_backend(config.backend)
        self._logger = logging.getLogger("securerag.retriever")
        self._runtime_llm = None

    @staticmethod
    def _validate_compatibility(rp: PrivacyProtocol, cp: PrivacyProtocol) -> None:
        if rp is not cp:
            raise ProtocolMismatchError(f"Retriever={rp.name} but corpus={cp.name}. They must match.")

    @abstractmethod
    def retrieve(self, query: str, round_n: int) -> list[Document]:
        pass

    @abstractmethod
    def privacy_cost(self, query: str) -> float:
        pass

    @classmethod
    def register(cls, protocol: PrivacyProtocol):
        def decorator(subclass):
            cls._registry[protocol] = subclass
            return subclass

        return decorator

    @classmethod
    def from_config(cls, config: PrivacyConfig, corpus):
        klass = cls._registry.get(config.protocol)
        if klass is None:
            raise UnknownProtocolError(config.protocol)
        return klass(config, corpus)

    @staticmethod
    def _to_docs(rows: list[dict]) -> list[Document]:
        return [
            Document(
                doc_id=row["doc_id"],
                text=row["text"],
                score=float(row.get("score", 0.0)),
                metadata=row.get("metadata", {}),
            )
            for row in rows
        ]

    def _debug(self, message: str, **fields) -> None:
        if not self.config.verbose:
            return
        if fields:
            rendered = ", ".join(f"{k}={v!r}" for k, v in fields.items())
            self._logger.info("%s | %s", message, rendered)
        else:
            self._logger.info("%s", message)

    def set_runtime_llm(self, llm) -> None:
        self._runtime_llm = llm

    def _paraphrase_decoys(self, decoys: list[str], source_query: str) -> list[str]:
        if not self.config.paraphrase_decoys:
            return decoys
        llm = self._runtime_llm
        if llm is None:
            return decoys

        batch_fn = getattr(llm, "paraphrase_decoys", None)
        if callable(batch_fn):
            try:
                out = batch_fn(decoys, source_query)
                if isinstance(out, list) and len(out) == len(decoys):
                    return [str(x) for x in out]
            except Exception:
                return decoys

        one_fn = getattr(llm, "paraphrase_decoy", None)
        if callable(one_fn):
            try:
                return [str(one_fn(d, source_query)) for d in decoys]
            except Exception:
                return decoys

        return decoys
