from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re

from securerag.backend_client import create_backend
from securerag.builtin_schemes import StructuredPlugin
from securerag.config import PrivacyConfig
from securerag.dp_mechanism import DPMechanismPlugin
from securerag.models import CorpusMeta, RawDocument
from securerag.protocol import PrivacyProtocol
from securerag.scheme_plugin import EncryptedSchemePlugin


ENCRYPTED_SEARCH_VERSION = "hmac-sha256-v1"


class SecureCorpus(ABC):
    def __init__(
        self,
        protocol: PrivacyProtocol,
        meta: CorpusMeta,
        index_id: str,
        extras: dict | None = None,
    ):
        self.protocol = protocol
        self.meta = meta
        self.index_id = index_id
        self.extras = extras or {}

    def index_size(self) -> int:
        return self.meta.doc_count

    @abstractmethod
    def save(self, path: str) -> None:
        pass


class GenericCorpus(SecureCorpus):
    def save(self, path: str) -> None:
        Path(path).write_text(
            f"protocol={self.protocol.name}\nindex_id={self.index_id}\n", encoding="utf-8"
        )


class SSECorpus(SecureCorpus):
    @property
    def sse_key(self) -> str:
        return str(self.extras.get("enc_key", ""))

    @property
    def scheme(self) -> str:
        return str(self.extras.get("encrypted_search_scheme", "sse"))

    @property
    def plugin(self) -> EncryptedSchemePlugin:
        return self.extras["plugin"]

    def save(self, path: str) -> None:
        Path(path).write_text(
            f"protocol={self.protocol.name}\nindex_id={self.index_id}\nscheme={self.scheme}\n",
            encoding="utf-8",
        )


class EmbeddingCorpus(SecureCorpus):
    def save(self, path: str) -> None:
        Path(path).write_text(
            f"protocol={self.protocol.name}\nindex_id={self.index_id}\nindex_type=embedding\n",
            encoding="utf-8",
        )


class PIRDatabase(SecureCorpus):
    def save(self, path: str) -> None:
        Path(path).write_text(
            f"protocol={self.protocol.name}\nindex_id={self.index_id}\nindex_type=pir\n",
            encoding="utf-8",
        )


class CorpusBuilder:
    def __init__(
        self,
        protocol: PrivacyProtocol,
        backend_url: str = "http://127.0.0.1:8099",
        config: PrivacyConfig | None = None,
    ):
        self._protocol = protocol
        self._docs: list[RawDocument] = []
        self._chunk_size = 512
        self._overlap = 64
        self._sanitize = True
        self._backend = create_backend(backend_url)
        self._encrypted_search_scheme = "sse"
        self._structured_use_bigrams = True
        self._dp_mechanism_name = "gaussian"
        self._epsilon = 1_000_000.0
        self._delta = 1e-5
        if config is not None and protocol.requires_budget:
            self._dp_mechanism_name = config.dp_mechanism
            self._epsilon = float(config.epsilon)
            self._delta = float(config.delta)

    @classmethod
    def from_config(
        cls,
        config: PrivacyConfig,
        *,
        backend_url: str | None = None,
    ) -> "CorpusBuilder":
        return cls(
            protocol=config.protocol,
            backend_url=backend_url or config.backend,
            config=config,
        )

    def with_chunk_size(self, n: int) -> "CorpusBuilder":
        self._chunk_size = n
        return self

    def with_overlap(self, n: int) -> "CorpusBuilder":
        self._overlap = n
        return self

    def disable_sanitization(self) -> "CorpusBuilder":
        self._sanitize = False
        return self

    def with_encrypted_search_scheme(
        self,
        scheme: str,
        *,
        structured_use_bigrams: bool = True,
    ) -> "CorpusBuilder":
        self._encrypted_search_scheme = str(scheme).lower()
        self._structured_use_bigrams = structured_use_bigrams
        return self

    def add_documents(self, docs: list[RawDocument]) -> "CorpusBuilder":
        self._docs.extend(docs)
        return self

    def with_privacy_budget(self, epsilon: float, delta: float) -> "CorpusBuilder":
        self._epsilon = float(epsilon)
        self._delta = float(delta)
        return self

    def with_dp_mechanism(self, mechanism: str) -> "CorpusBuilder":
        self._dp_mechanism_name = mechanism
        return self

    def add_directory(self, path: str, glob: str = "**/*.txt") -> "CorpusBuilder":
        for file in Path(path).glob(glob):
            if file.is_file():
                text = file.read_text(encoding="utf-8", errors="ignore")
                self._docs.append(RawDocument(doc_id=str(file), text=text))
        return self

    def build(self) -> SecureCorpus:
        docs_payload = [
            {"doc_id": d.doc_id, "text": d.text, "metadata": d.metadata} for d in self._docs
        ]
        chunks = self._backend.chunk(docs_payload, self._chunk_size, self._overlap)
        if self._sanitize:
            chunks = self._backend.sanitize(chunks)

        if self._protocol is PrivacyProtocol.DIFF_PRIVACY:
            import securerag.builtin_mechanisms  # noqa: F401

            mechanism = DPMechanismPlugin.get(self._dp_mechanism_name)
            chunks = mechanism.prepare_corpus(chunks)

        extras: dict = {}
        if self._protocol is PrivacyProtocol.ENCRYPTED_SEARCH:
            if self._encrypted_search_scheme in {"structured", "structured_encryption"}:
                plugin = StructuredPlugin(use_bigrams=self._structured_use_bigrams)
            else:
                plugin = EncryptedSchemePlugin.get(self._encrypted_search_scheme)
            enc_key = plugin.generate_key()
            for chunk in chunks:
                chunk["scheme_data"] = plugin.prepare_chunk(chunk["text"], enc_key)
            extras["enc_key"] = enc_key
            extras["encrypted_search_scheme"] = self._encrypted_search_scheme
            extras["encrypted_search_version"] = ENCRYPTED_SEARCH_VERSION
            extras["plugin"] = plugin

        index_payload = self._backend.build_index(
            self._protocol.wire_name,
            chunks,
            epsilon=self._epsilon,
            delta=self._delta,
            encrypted_search_scheme=self._encrypted_search_scheme
            if self._protocol is PrivacyProtocol.ENCRYPTED_SEARCH
            else "",
            encrypted_search_version=ENCRYPTED_SEARCH_VERSION
            if self._protocol is PrivacyProtocol.ENCRYPTED_SEARCH
            else "",
        )
        meta = CorpusMeta(
            doc_count=index_payload["doc_count"],
            chunk_size=self._chunk_size,
            overlap=self._overlap,
            protocol=self._protocol.name,
        )
        if self._protocol is PrivacyProtocol.ENCRYPTED_SEARCH:
            return SSECorpus(
                protocol=self._protocol,
                meta=meta,
                index_id=index_payload["index_id"],
                extras=extras,
            )
        if self._protocol is PrivacyProtocol.DIFF_PRIVACY:
            return EmbeddingCorpus(
                protocol=self._protocol,
                meta=meta,
                index_id=index_payload["index_id"],
                extras=extras,
            )
        if self._protocol is PrivacyProtocol.PIR:
            return PIRDatabase(
                protocol=self._protocol,
                meta=meta,
                index_id=index_payload["index_id"],
                extras=extras,
            )
        return GenericCorpus(
            protocol=self._protocol,
            meta=meta,
            index_id=index_payload["index_id"],
            extras=extras,
        )

    @staticmethod
    def _local_chunk(docs: list[RawDocument], chunk_size: int, overlap: int) -> list[dict]:
        step = max(1, chunk_size - overlap)
        chunks: list[dict] = []
        for d in docs:
            if len(d.text) <= chunk_size:
                chunks.append({"doc_id": d.doc_id, "text": d.text, "metadata": d.metadata})
                continue
            for i in range(0, len(d.text), step):
                snippet = d.text[i : i + chunk_size]
                if not snippet:
                    continue
                if i > 0 and len(snippet) < max(24, chunk_size // 3):
                    continue
                chunks.append({"doc_id": d.doc_id, "text": snippet, "metadata": d.metadata})
        return chunks

    @staticmethod
    def _local_sanitize(chunks: list[dict]) -> list[dict]:
        bad = ["ignore previous instructions", "system prompt", "developer instructions"]
        out = []
        for c in chunks:
            text = c["text"]
            for token in bad:
                text = re.sub(re.escape(token), "", text, flags=re.IGNORECASE)
            out.append({**c, "text": text})
        return out

    def build_local(self, *, workers: int = 4) -> SecureCorpus:
        chunks = self._local_chunk(self._docs, self._chunk_size, self._overlap)
        if self._sanitize:
            chunks = self._local_sanitize(chunks)

        if self._protocol is PrivacyProtocol.DIFF_PRIVACY:
            import securerag.builtin_mechanisms  # noqa: F401

            mechanism = DPMechanismPlugin.get(self._dp_mechanism_name)
            chunks = mechanism.prepare_corpus(chunks)

        extras: dict = {}
        if self._protocol is PrivacyProtocol.ENCRYPTED_SEARCH:
            if self._encrypted_search_scheme in {"structured", "structured_encryption"}:
                plugin = StructuredPlugin(use_bigrams=self._structured_use_bigrams)
            else:
                plugin = EncryptedSchemePlugin.get(self._encrypted_search_scheme)
            enc_key = plugin.generate_key()

            def _prep(chunk: dict) -> dict:
                return {**chunk, "scheme_data": plugin.prepare_chunk(chunk["text"], enc_key)}

            with ThreadPoolExecutor(max_workers=workers) as ex:
                chunks = list(ex.map(_prep, chunks))

            extras["enc_key"] = enc_key
            extras["encrypted_search_scheme"] = self._encrypted_search_scheme
            extras["encrypted_search_version"] = ENCRYPTED_SEARCH_VERSION
            extras["plugin"] = plugin

        index_payload = self._backend.build_index(
            self._protocol.wire_name,
            chunks,
            epsilon=self._epsilon,
            delta=self._delta,
            encrypted_search_scheme=self._encrypted_search_scheme
            if self._protocol is PrivacyProtocol.ENCRYPTED_SEARCH
            else "",
            encrypted_search_version=ENCRYPTED_SEARCH_VERSION
            if self._protocol is PrivacyProtocol.ENCRYPTED_SEARCH
            else "",
        )
        meta = CorpusMeta(
            doc_count=index_payload["doc_count"],
            chunk_size=self._chunk_size,
            overlap=self._overlap,
            protocol=self._protocol.name,
        )
        if self._protocol is PrivacyProtocol.ENCRYPTED_SEARCH:
            return SSECorpus(
                protocol=self._protocol,
                meta=meta,
                index_id=index_payload["index_id"],
                extras=extras,
            )
        if self._protocol is PrivacyProtocol.DIFF_PRIVACY:
            return EmbeddingCorpus(
                protocol=self._protocol,
                meta=meta,
                index_id=index_payload["index_id"],
                extras=extras,
            )
        if self._protocol is PrivacyProtocol.PIR:
            return PIRDatabase(
                protocol=self._protocol,
                meta=meta,
                index_id=index_payload["index_id"],
                extras=extras,
            )
        return GenericCorpus(
            protocol=self._protocol,
            meta=meta,
            index_id=index_payload["index_id"],
            extras=extras,
        )
