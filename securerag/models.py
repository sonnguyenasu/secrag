from dataclasses import dataclass, field


@dataclass(slots=True)
class Document:
    doc_id: str
    text: str
    score: float = 0.0
    metadata: dict[str, str] = field(default_factory=dict)
    required_budget: bool = False
    budget_type: str = "rdp"


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


@dataclass(slots=True)
class PrivateQuery:
    text: str
    required_budget: bool = True
    mechanism: str = "gaussian"
    epsilon: float = 1.0


@dataclass(slots=True)
class QueryRecord:
    question: str
    answers: list[str]
    doc_ids: list[str]
    required_budget: bool = False
