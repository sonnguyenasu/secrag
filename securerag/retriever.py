from __future__ import annotations

from abc import ABC, abstractmethod
import logging

from securerag.backend_client import create_backend
from securerag.cost import Cost
from securerag.budget import BudgetManager
from securerag.config import PrivacyConfig
from securerag.context import PrivacyContext
from securerag.errors import ProtocolMismatchError, UnknownProtocolError
from securerag.models import Document, PrivateQuery
from securerag.protocol import PrivacyProtocol


class PrivacyRetriever(ABC):
    _registry: dict[PrivacyProtocol, type["PrivacyRetriever"]] = {}

    def __init__(self, config: PrivacyConfig, corpus):
        self._validate_compatibility(config.protocol, corpus.protocol)
        self.config = config
        self.corpus = corpus
        if config.protocol is PrivacyProtocol.DIFF_PRIVACY:
            import securerag.builtin_mechanisms  # noqa: F401
            from securerag.dp_mechanism import DPMechanismPlugin

            mechanism = DPMechanismPlugin.get(config.dp_mechanism)
            self.budget = BudgetManager(config, mechanism=mechanism)
            self._dp_mechanism = mechanism
        else:
            self.budget = BudgetManager(config)
            self._dp_mechanism = None
        self._backend = create_backend(config.backend)
        self._logger = logging.getLogger("securerag.retriever")
        self._runtime_llm = None
        self._ctx: PrivacyContext | None = None

    @staticmethod
    def _validate_compatibility(rp: PrivacyProtocol, cp: PrivacyProtocol) -> None:
        if rp is not cp:
            raise ProtocolMismatchError(f"Retriever={rp.name} but corpus={cp.name}. They must match.")

    @abstractmethod
    def retrieve(self, query: str | PrivateQuery, round_n: int) -> list[Document]:
        pass

    @abstractmethod
    def privacy_cost(self, query: str | PrivateQuery) -> float:
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

    def with_context(self, ctx: PrivacyContext) -> "PrivacyRetriever":
        self._ctx = ctx
        key = self.config.protocol.name
        if key not in ctx.snapshot():
            ctx.register_budget(key, self.budget)
        return self

    def _charge(self, cost: Cost) -> None:
        if self._ctx is not None:
            self._ctx.charge(self.config.protocol.name, cost)
            return
        self.budget.consume(cost)

    @staticmethod
    def _resolve_query(query: str | PrivateQuery) -> tuple[str, bool]:
        if isinstance(query, PrivateQuery):
            return query.text, bool(query.required_budget)
        # Backward-compatible default: plain string queries in DP retrieval are budgeted.
        return str(query), True

    def _paraphrase_decoys(self, decoys: list[str], source_query: str) -> list[str]:
        if not self.config.paraphrase_decoys:
            return decoys
        llm = self._runtime_llm
        if llm is None:
            return decoys

        role_para = getattr(llm, "paraphrase", None)
        if callable(role_para):
            try:
                out = role_para(decoys, source_query)
                if isinstance(out, list) and len(out) == len(decoys):
                    return [str(x) for x in out]
            except Exception:
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
