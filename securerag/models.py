from dataclasses import dataclass, field


@dataclass(slots=True)
class Document:
    doc_id: str
    text: str
    score: float = 0.0
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class RawDocument:
    doc_id: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class CorpusMeta:
    doc_count: int
    chunk_size: int
    overlap: int
    protocol: str
