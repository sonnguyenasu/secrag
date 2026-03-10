import os

from securerag import PrivacyConfig, PrivacyProtocol, SecureRAGAgent
from securerag.corpus import CorpusBuilder
from securerag.llm import ModelAgentLLM
from securerag.models import RawDocument


def run_protocol(protocol: PrivacyProtocol, backend: str, query: str) -> None:
    docs = [
        RawDocument(doc_id="q3", text="Q3 risk report highlights vendor concentration and delayed remediation."),
        RawDocument(doc_id="policy", text="Security policy requires quarterly risk treatment tracking and owner assignment."),
        RawDocument(doc_id="ops", text="Operational incidents increased in July due to queue saturation in ingestion service."),
    ]

    config = PrivacyConfig(
        protocol=protocol,
        backend=backend,
        epsilon=100.0,
        noise_std=0.2,
        top_k=3,
        k_decoys=2,
        encrypted_search_scheme="structured" if protocol is PrivacyProtocol.ENCRYPTED_SEARCH else "sse",
        structured_use_bigrams=True,
    )

    builder = CorpusBuilder(config.protocol, backend_url=config.backend).with_chunk_size(120).add_documents(docs)
    if protocol is PrivacyProtocol.ENCRYPTED_SEARCH:
        builder = builder.with_encrypted_search_scheme(
            config.encrypted_search_scheme,
            structured_use_bigrams=config.structured_use_bigrams,
        )
    corpus = builder.build()

    llm = ModelAgentLLM(
        provider="ollama",
        model="qwen3:0.6b",
        use_ollama=False,
        use_huggingface=False,
    )
    agent = SecureRAGAgent.from_config(config, corpus=corpus, llm=llm)

    result = agent.run(query)

    print("=" * 72)
    print("protocol:", protocol.name)
    print("backend:", backend)
    print("corpus type:", type(corpus).__name__)
    print("answer:", result.answer)
    print("budget:", agent.budget_snapshot())


def main() -> None:
    backend = os.getenv("SECURERAG_BACKEND", "rust://local")
    query = "Summarize key risks in Q3"

    for protocol in [
        PrivacyProtocol.BASELINE,
        PrivacyProtocol.OBFUSCATION,
        PrivacyProtocol.DIFF_PRIVACY,
        PrivacyProtocol.ENCRYPTED_SEARCH,
    ]:
        run_protocol(protocol, backend, query)


if __name__ == "__main__":
    main()
