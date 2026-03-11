from __future__ import annotations

import json
from pathlib import Path

from securerag.benchmarks import NaturalQuestions, TriviaQA, load_wikipedia_corpus
from securerag.protocol import PrivacyProtocol


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_nq_loader_builds_corpus_and_queries_from_local_jsonl(tmp_path: Path) -> None:
    rows = [
        {
            "question": "What is SecureRAG?",
            "answers": ["A framework"],
            "doc_ids": ["d1"],
            "documents": [{"doc_id": "d1", "text": "SecureRAG is a privacy-aware retrieval framework."}],
        },
        {
            "question": "Who uses it?",
            "answers": ["Researchers"],
            "doc_ids": ["d2"],
            "documents": [{"doc_id": "d2", "text": "Researchers use it for prototyping."}],
        },
    ]
    _write_jsonl(tmp_path / "nq_dev.jsonl", rows)

    corpus, queries = NaturalQuestions.load(split="dev", n=10, data_dir=str(tmp_path))

    assert corpus.protocol is PrivacyProtocol.BASELINE
    assert corpus.index_id == "local"
    assert len(queries) == 2
    assert queries[0].question == "What is SecureRAG?"
    assert queries[0].answers == ["A framework"]


def test_trivia_loader_respects_n_limit(tmp_path: Path) -> None:
    rows = [
        {
            "question": "q1",
            "answers": ["a1"],
            "doc_ids": ["x1"],
            "documents": [{"doc_id": "x1", "text": "doc1"}],
        },
        {
            "question": "q2",
            "answers": ["a2"],
            "doc_ids": ["x2"],
            "documents": [{"doc_id": "x2", "text": "doc2"}],
        },
    ]
    _write_jsonl(tmp_path / "triviaqa_test.jsonl", rows)

    corpus, queries = TriviaQA.load(split="test", n=1, data_dir=str(tmp_path))

    assert corpus.index_id == "local"
    assert len(queries) == 1
    assert queries[0].question == "q1"


def test_wikipedia_loader_parses_doc_rows(tmp_path: Path, monkeypatch) -> None:
    rows = [
        {"doc_id": "w1", "text": "alpha", "source": "wiki"},
        {"id": "w2", "context": "beta"},
    ]
    _write_jsonl(tmp_path / "wikipedia_2018-12.jsonl", rows)
    monkeypatch.setenv("SECURERAG_BENCHMARK_DIR", str(tmp_path))

    docs = load_wikipedia_corpus("2018-12")

    assert len(docs) == 2
    assert docs[0].doc_id == "w1"
    assert docs[1].doc_id == "w2"
