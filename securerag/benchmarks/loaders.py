from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from securerag.corpus import CorpusBuilder, SecureCorpus
from securerag.protocol import PrivacyProtocol
from securerag.models import QueryRecord, RawDocument


def _default_data_dir() -> Path:
    # Data root can be controlled explicitly for reproducible benchmark runs.
    return Path(os.environ.get("SECURERAG_BENCHMARK_DIR", "benchmarks_data"))


def _read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark file not found: {path}. "
            "Provide data_dir or set SECURERAG_BENCHMARK_DIR."
        )

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v)]
    return [str(value)]


def build_query_records(
    rows: list[dict[str, Any]],
    *,
    question_key: str = "question",
    answers_key: str = "answers",
    doc_ids_key: str = "doc_ids",
    required_budget_key: str = "required_budget",
) -> list[QueryRecord]:
    records: list[QueryRecord] = []
    for row in rows:
        question = str(row.get(question_key, "")).strip()
        if not question:
            continue
        records.append(
            QueryRecord(
                question=question,
                answers=_as_list(row.get(answers_key)),
                doc_ids=_as_list(row.get(doc_ids_key)),
                required_budget=bool(row.get(required_budget_key, False)),
            )
        )
    return records


def build_raw_documents(rows: list[dict[str, Any]]) -> list[RawDocument]:
    docs: list[RawDocument] = []
    seen_ids: set[str] = set()

    def _append_doc(doc_id: str, text: str, metadata: dict[str, str] | None = None) -> None:
        if not doc_id or not text or doc_id in seen_ids:
            return
        seen_ids.add(doc_id)
        docs.append(RawDocument(doc_id=doc_id, text=text, metadata=metadata or {}))

    for row in rows:
        # Primary doc shape: one row == one document.
        doc_id = str(row.get("doc_id") or row.get("id") or "").strip()
        text = str(row.get("text") or row.get("context") or "").strip()
        if doc_id and text:
            _append_doc(doc_id, text, {"source": str(row.get("source", "benchmark"))})

        # Alternate shape: question rows carry nested documents list.
        documents = row.get("documents") or row.get("docs") or []
        if isinstance(documents, list):
            for d in documents:
                if not isinstance(d, dict):
                    continue
                nested_id = str(d.get("doc_id") or d.get("id") or "").strip()
                nested_text = str(d.get("text") or d.get("context") or "").strip()
                _append_doc(
                    nested_id,
                    nested_text,
                    {"source": str(d.get("source", "benchmark"))},
                )

    return docs


def build_local_corpus(
    docs: list[RawDocument],
    *,
    protocol: PrivacyProtocol = PrivacyProtocol.BASELINE,
) -> SecureCorpus:
    return (
        CorpusBuilder(protocol)
        .add_documents(docs)
        .build_local(use_rust_if_available=False)
    )


def _resolve_file(data_dir: Path, stem: str, split: str, suffix: str = ".jsonl") -> Path:
    candidates = [
        data_dir / f"{stem}_{split}{suffix}",
        data_dir / stem / f"{split}{suffix}",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Return canonical expected location for clear error text if missing.
    return candidates[0]


def load_qa_dataset(
    *,
    stem: str,
    split: str,
    n: int,
    data_dir: str | None = None,
    protocol: PrivacyProtocol = PrivacyProtocol.BASELINE,
) -> tuple[SecureCorpus, list[QueryRecord]]:
    root = Path(data_dir) if data_dir else _default_data_dir()
    path = _resolve_file(root, stem, split)
    rows = _read_jsonl(path, limit=n)
    queries = build_query_records(rows)
    docs = build_raw_documents(rows)
    return build_local_corpus(docs, protocol=protocol), queries

def load_wikipedia_corpus(subset: str = "2018-12") -> list[RawDocument]:
    root = _default_data_dir()
    path = _resolve_file(root, "wikipedia", subset)
    rows = _read_jsonl(path)
    return build_raw_documents(rows)
