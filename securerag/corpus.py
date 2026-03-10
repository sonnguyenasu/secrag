from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from securerag.backend_client import create_backend
from securerag.models import CorpusMeta, RawDocument
from securerag.protocol import PrivacyProtocol


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
    def __init__(self, protocol: PrivacyProtocol, backend_url: str = "http://127.0.0.1:8099"):
        self._protocol = protocol
        self._docs: list[RawDocument] = []
        self._chunk_size = 512
        self._overlap = 64
        self._sanitize = True
        self._backend = create_backend(backend_url)
        self._encrypted_search_scheme = "sse"
        self._structured_use_bigrams = True

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

        extras: dict = {}
        if self._protocol is PrivacyProtocol.ENCRYPTED_SEARCH:
            scheme = self._encrypted_search_scheme
            enc_key = self._backend.sse_generate_key()
            chunks = self._backend.sse_prepare_chunks(
                chunks=chunks,
                key=enc_key,
                scheme=scheme,
                use_bigrams=self._structured_use_bigrams,
            )
            if scheme in {"structured", "structured_encryption"}:
                scheme = "structured"
            elif scheme != "sse":
                raise ValueError(
                    f"Unknown encrypted search scheme for builder: {scheme}. "
                    "Use 'sse' or 'structured'."
                )

            extras["enc_key"] = enc_key
            extras["encrypted_search_scheme"] = scheme

        index_payload = self._backend.build_index(self._protocol.wire_name, chunks)
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
