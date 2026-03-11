from __future__ import annotations

import os

from securerag import PrivacyConfig, PrivacyProtocol, SecureRAGAgent
from securerag.benchmarks import NaturalQuestions, TriviaQA
from securerag.llm import ModelAgentLLM
from securerag.models import PrivateQuery


def main() -> None:
    dataset = os.getenv("SECURERAG_BENCHMARK_DATASET", "nq").lower()
    split = os.getenv("SECURERAG_BENCHMARK_SPLIT", "dev")
    n = int(os.getenv("SECURERAG_BENCHMARK_N", "50"))
    data_dir = os.getenv("SECURERAG_BENCHMARK_DIR")
    protocol = PrivacyProtocol[os.getenv("SECURERAG_PROTOCOL", "DIFF_PRIVACY").upper()]

    if dataset == "triviaqa":
        corpus, queries = TriviaQA.load(split=split, n=n, data_dir=data_dir, protocol=protocol)
    else:
        corpus, queries = NaturalQuestions.load(split=split, n=n, data_dir=data_dir, protocol=protocol)

    if not queries:
        raise RuntimeError("No benchmark queries loaded. Check SECURERAG_BENCHMARK_DIR and dataset files.")

    cfg = PrivacyConfig(
        protocol=protocol,
        backend="http://127.0.0.1:8099",
        max_rounds=3,
        top_k=4,
        epsilon=20.0,
        noise_std=1.0,
    )

    llm = ModelAgentLLM(provider="ollama", model="qwen3:0.6b", use_ollama=False)
    agent = SecureRAGAgent.from_config(cfg, corpus=corpus, llm=llm)

    q = queries[0]
    result = agent.retriever.retrieve(
        PrivateQuery(text=q.question, required_budget=q.required_budget),
        round_n=0,
    )

    print("dataset:", dataset)
    print("split:", split)
    print("protocol:", protocol.name)
    print("query:", q.question)
    print("retrieved docs:", len(result))
    print("budget snapshot:", agent.budget_snapshot())


if __name__ == "__main__":
    main()
