from __future__ import annotations

import argparse
import random
import string
import time
from dataclasses import asdict, dataclass

from securerag import PrivacyConfig, PrivacyProtocol, SecureRAGAgent
from securerag.corpus import CorpusBuilder
from securerag.llm import ModelAgentLLM
from securerag.models import RawDocument


@dataclass
class PerfResult:
    backend: str
    protocol: str
    docs: int
    chunk_size: int
    overlap: int
    top_k: int
    build_seconds: float
    query_seconds: float
    total_seconds: float
    context_size: int


def _random_words(rng: random.Random, n_words: int) -> str:
    words = []
    for _ in range(n_words):
        length = rng.randint(4, 10)
        words.append("".join(rng.choice(string.ascii_lowercase) for _ in range(length)))
    return " ".join(words)


def build_synthetic_docs(n_docs: int, seed: int, target_keyword: str) -> list[RawDocument]:
    rng = random.Random(seed)
    docs: list[RawDocument] = []

    for i in range(n_docs):
        text = _random_words(rng, n_words=120)
        docs.append(RawDocument(doc_id=f"doc-{i}", text=text, metadata={"split": "synthetic"}))

    # Inject strong signal documents for retrieval sanity.
    hot_spots = [max(0, n_docs // 10), max(1, n_docs // 3), max(2, (2 * n_docs) // 3)]
    for idx, pos in enumerate(hot_spots):
        docs[pos] = RawDocument(
            doc_id=f"target-{idx}",
            text=(
                f"{target_keyword} risk report for quarter {idx + 1}. "
                "Vendor concentration increased and remediation is delayed. "
                "Action owners must track mitigation weekly."
            ),
            metadata={"split": "target"},
        )

    return docs


def run_benchmark(
    n_docs: int,
    backend: str,
    protocol: PrivacyProtocol,
    seed: int,
    chunk_size: int,
    overlap: int,
    top_k: int,
) -> PerfResult:
    query = "summarize q3 risk and vendor concentration"
    docs = build_synthetic_docs(n_docs=n_docs, seed=seed, target_keyword="q3")

    cfg = PrivacyConfig(
        protocol=protocol,
        backend=backend,
        epsilon=500.0 if protocol is PrivacyProtocol.DIFF_PRIVACY else 1_000_000.0,
        delta=1e-5,
        noise_std=0.3,
        top_k=top_k,
        max_rounds=3,
        encrypted_search_scheme="structured" if protocol is PrivacyProtocol.ENCRYPTED_SEARCH else "sse",
        structured_use_bigrams=True,
    )

    t0 = time.perf_counter()

    builder = (
        CorpusBuilder(protocol, backend_url=backend)
        .with_privacy_budget(epsilon=cfg.epsilon, delta=cfg.delta)
        .with_chunk_size(chunk_size)
        .with_overlap(overlap)
        .add_documents(docs)
    )

    if protocol is PrivacyProtocol.ENCRYPTED_SEARCH:
        builder = builder.with_encrypted_search_scheme(
            cfg.encrypted_search_scheme,
            structured_use_bigrams=cfg.structured_use_bigrams,
        )

    b0 = time.perf_counter()
    corpus = builder.build_local(workers=4)
    b1 = time.perf_counter()

    llm = ModelAgentLLM(
        provider="ollama",
        model="qwen3:0.6b",
        use_ollama=False,
        use_huggingface=False,
    )
    agent = SecureRAGAgent.from_config(cfg, corpus=corpus, llm=llm)

    q0 = time.perf_counter()
    result = agent.run(query)
    q1 = time.perf_counter()

    return PerfResult(
        backend=backend,
        protocol=protocol.name,
        docs=n_docs,
        chunk_size=chunk_size,
        overlap=overlap,
        top_k=top_k,
        build_seconds=round(b1 - b0, 4),
        query_seconds=round(q1 - q0, 4),
        total_seconds=round(q1 - t0, 4),
        context_size=result.context_size,
    )


def parse_protocol(value: str) -> PrivacyProtocol:
    normalized = value.strip().upper()
    try:
        return PrivacyProtocol[normalized]
    except KeyError as exc:
        valid = ", ".join(p.name for p in PrivacyProtocol)
        raise argparse.ArgumentTypeError(f"Invalid protocol '{value}'. Choose one of: {valid}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Large corpus performance benchmark for SecureRAG")
    parser.add_argument("--docs", type=int, default=5000, help="Number of synthetic documents")
    parser.add_argument("--backend", type=str, default="rust://local", help="Backend target")
    parser.add_argument("--protocol", type=parse_protocol, default=PrivacyProtocol.BASELINE, help="Privacy protocol")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--chunk-size", type=int, default=220)
    parser.add_argument("--overlap", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    perf = run_benchmark(
        n_docs=args.docs,
        backend=args.backend,
        protocol=args.protocol,
        seed=args.seed,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        top_k=args.top_k,
    )

    print("=== SecureRAG Large Benchmark ===")
    for k, v in asdict(perf).items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
