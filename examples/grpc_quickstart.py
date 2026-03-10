import os

from securerag import PrivacyConfig, PrivacyProtocol, SecureRAGAgent
from securerag.corpus import CorpusBuilder
from securerag.llm import ModelAgentLLM
from securerag.models import RawDocument


def main() -> None:
    docs = [
        RawDocument(doc_id="q3", text="Q3 risk report highlights vendor concentration and delayed remediation."),
        RawDocument(doc_id="policy", text="Security policy requires quarterly risk treatment tracking and owner assignment."),
        RawDocument(doc_id="ops", text="Operational incidents increased in July due to queue saturation in ingestion service."),
    ]

    backend = os.getenv("SECURERAG_BACKEND", "grpc://127.0.0.1:50051")

    config = PrivacyConfig(
        protocol=PrivacyProtocol.ENCRYPTED_SEARCH,
        backend=backend,
        top_k=3,
        max_rounds=4,
        encrypted_search_scheme=os.getenv("SECURERAG_ENC_SCHEME", "sse"),
        structured_use_bigrams=os.getenv("SECURERAG_STRUCTURED_BIGRAMS", "1")
        in {"1", "true", "TRUE", "yes", "YES"},
    )

    corpus = (
        CorpusBuilder(config.protocol, backend_url=config.backend)
        .with_encrypted_search_scheme(
            config.encrypted_search_scheme,
            structured_use_bigrams=config.structured_use_bigrams,
        )
        .with_chunk_size(120)
        .add_documents(docs)
        .build()
    )

    llm = ModelAgentLLM(
        provider="ollama",
        model="qwen3:0.6b",
        use_ollama=False,
        use_huggingface=False,
    )

    agent = SecureRAGAgent.from_config(config, corpus=corpus, llm=llm)
    result = agent.run("Summarize key risks in Q3")

    print("backend:", backend)
    print("corpus type:", type(corpus).__name__)
    print(result.answer)
    print(agent.budget_snapshot())


if __name__ == "__main__":
    main()
