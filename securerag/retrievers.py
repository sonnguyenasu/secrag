from securerag.errors import UnsupportedCapabilityError
from securerag.protocol import PrivacyProtocol
from securerag.retriever import PrivacyRetriever


@PrivacyRetriever.register(PrivacyProtocol.BASELINE)
class BaselineRetriever(PrivacyRetriever):
    def retrieve(self, query: str, round_n: int):
        rows = self._backend.batch_retrieve(
            index_id=self.corpus.index_id,
            queries=[query],
            top_k=self.config.top_k,
        )[0]
        return self._to_docs(rows)

    def privacy_cost(self, query: str) -> float:
        return 0.0


@PrivacyRetriever.register(PrivacyProtocol.OBFUSCATION)
class ObfuscationRetriever(PrivacyRetriever):
    def retrieve(self, query: str, round_n: int):
        raw_decoys = self._backend.generate_decoys(
            index_id=self.corpus.index_id,
            query=query,
            k=self.config.k_decoys,
        )
        decoys = self._paraphrase_decoys(raw_decoys, query)
        self._debug(
            "obfuscation retrieval",
            round_n=round_n,
            real_query=query,
            raw_decoy_queries=raw_decoys,
            paraphrased_decoy_queries=decoys,
            paraphrase_enabled=self.config.paraphrase_decoys,
        )
        results = self._backend.batch_retrieve(
            index_id=self.corpus.index_id,
            queries=[query] + decoys,
            top_k=self.config.top_k,
        )
        return self._to_docs(results[0])

    def privacy_cost(self, query: str) -> float:
        return 0.0


@PrivacyRetriever.register(PrivacyProtocol.DIFF_PRIVACY)
class DiffPrivacyRetriever(PrivacyRetriever):
    def retrieve(self, query: str, round_n: int):
        eps = self.privacy_cost(query)
        self.budget.consume(eps)
        noised = self._backend.embed_with_noise(query=query, sigma=self.config.noise_std)
        self._debug(
            "diff-privacy retrieval",
            round_n=round_n,
            original_query=query,
            noised_query_embedding=noised,
            epsilon_cost=eps,
            epsilon_remaining=self.budget.remaining,
        )
        rows = self._backend.retrieve_by_embedding(
            index_id=self.corpus.index_id,
            embedding=noised,
            top_k=self.config.top_k,
            query=query,
        )
        return self._to_docs(rows)

    def privacy_cost(self, query: str) -> float:
        return (1.0 / self.config.noise_std) ** 2 / 2.0


@PrivacyRetriever.register(PrivacyProtocol.ENCRYPTED_SEARCH)
class EncryptedSearchRetriever(PrivacyRetriever):
    def retrieve(self, query: str, round_n: int):
        scheme = str(getattr(self.config, "encrypted_search_scheme", "sse")).lower()
        if scheme in {"structured_encryption", "structured"}:
            scheme = "structured"

        enc_key = self.corpus.extras.get("enc_key")
        if not enc_key:
            raise UnsupportedCapabilityError("ENCRYPTED_SEARCH requires corpus with client-side SSE key")

        if scheme == "sse":
            enc_terms = self._backend.sse_encrypt_terms(query, enc_key)
            self._debug(
                "encrypted-search retrieval",
                round_n=round_n,
                scheme=scheme,
                encrypted_query_terms=enc_terms,
            )
            rows = self._backend.sse_search(
                index_id=self.corpus.index_id,
                enc_terms=enc_terms,
                top_k=self.config.top_k,
            )
        elif scheme == "structured":
            struct_terms = self._backend.sse_encrypt_structured_terms(
                text=query,
                key=enc_key,
                use_bigrams=self.config.structured_use_bigrams,
            )
            self._debug(
                "encrypted-search retrieval",
                round_n=round_n,
                scheme=scheme,
                encrypted_structured_terms=struct_terms,
            )
            rows = self._backend.structured_search(
                index_id=self.corpus.index_id,
                struct_terms=struct_terms,
                top_k=self.config.top_k,
            )
        else:
            raise UnsupportedCapabilityError(
                f"Encrypted search scheme '{scheme}' is not implemented yet. "
                "Use encrypted_search_scheme='sse' or 'structured'."
            )
        return self._to_docs(rows)

    def privacy_cost(self, query: str) -> float:
        return 0.0


@PrivacyRetriever.register(PrivacyProtocol.PIR)
class PIRRetriever(PrivacyRetriever):
    def retrieve(self, query: str, round_n: int):
        raise UnsupportedCapabilityError("PIR is API-complete but not implemented in this MVP")

    def privacy_cost(self, query: str) -> float:
        return 0.0
