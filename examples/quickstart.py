import os
import logging

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

    

    backend = os.getenv("SECURERAG_BACKEND", "rust://local")
    verbose = os.getenv("SECURERAG_VERBOSE", "0") in {"1", "true", "TRUE", "yes", "YES"}
    llm_provider = os.getenv("SECURERAG_LLM_PROVIDER", "ollama").lower()
    use_ollama = os.getenv("SECURERAG_USE_OLLAMA", "0") in {"1", "true", "TRUE", "yes", "YES"}
    use_huggingface = os.getenv("SECURERAG_USE_HUGGINGFACE", "0") in {"1", "true", "TRUE", "yes", "YES"}
    ollama_model = os.getenv("SECURERAG_OLLAMA_MODEL", "qwen3:0.6b")
    huggingface_model = os.getenv("SECURERAG_HF_MODEL", "google/flan-t5-base")
    ollama_timeout_s = float(os.getenv("SECURERAG_OLLAMA_TIMEOUT_S", "60"))
    ollama_retries = int(os.getenv("SECURERAG_OLLAMA_RETRIES", "1"))
    hf_base_url = os.getenv("HF_INFERENCE_BASE_URL", "https://api-inference.huggingface.co")
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    encrypted_search_scheme = os.getenv("SECURERAG_ENC_SCHEME", "sse").lower()
    structured_use_bigrams = os.getenv("SECURERAG_STRUCTURED_BIGRAMS", "1") in {"1", "true", "TRUE", "yes", "YES"}

    if verbose:
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")

    config = PrivacyConfig(
        protocol=PrivacyProtocol.ENCRYPTED_SEARCH,
        epsilon=250.0,
        delta=1e-5,
        max_rounds=4,
        noise_std=0.2,
        top_k=3,
        backend=backend,
        verbose=verbose,
        encrypted_search_scheme=encrypted_search_scheme,
        structured_use_bigrams=structured_use_bigrams,
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

    agent = SecureRAGAgent.from_config(
        config,
        corpus=corpus,
        llm=ModelAgentLLM(
            model=ollama_model if llm_provider == "ollama" else huggingface_model,
            provider=llm_provider,
            ollama_model=ollama_model,
            huggingface_model=huggingface_model,
            use_ollama=use_ollama,
            use_huggingface=use_huggingface,
            huggingface_base_url=hf_base_url,
            huggingface_token=hf_token,
            timeout_s=ollama_timeout_s,
            retries=ollama_retries,
        ),
    )
    result = agent.run("Summarize key risks in Q3")
    print(result.answer)
    print(agent.budget_snapshot())


if __name__ == "__main__":
    main()
