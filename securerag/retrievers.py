from securerag.errors import UnsupportedCapabilityError
from securerag.protocol import PrivacyProtocol
from securerag.retriever import PrivacyRetriever
from securerag.scheme_plugin import EncryptedSchemePlugin


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
        sigma = float(self.config.noise_std)
        eps = self.budget.incremental_cost(sigma)
        cost = self._dp_mechanism.cost(sensitivity=sigma)
        base_embedding = self._backend.embed(query=query)
        noised = self._dp_mechanism.noise(base_embedding, sigma, query=query)
        rows = self._backend.retrieve_by_embedding(
            index_id=self.corpus.index_id,
            embedding=noised,
            top_k=self.config.top_k,
            query=query,
            sigma=sigma,
        )
        self._charge(cost)
        self._debug(
            "diff-privacy retrieval",
            round_n=round_n,
            original_query=query,
            noised_query_embedding=noised,
            mechanism=type(self._dp_mechanism).__name__,
            epsilon_cost=eps,
            epsilon_remaining=self.budget.remaining,
        )
        return self._to_docs(rows)

    def privacy_cost(self, query: str) -> float:
        return self.budget.incremental_cost(float(self.config.noise_std))


@PrivacyRetriever.register(PrivacyProtocol.ENCRYPTED_SEARCH)
class EncryptedSearchRetriever(PrivacyRetriever):
    def retrieve(self, query: str, round_n: int):
        enc_key = self.corpus.extras.get("enc_key")
        if not enc_key:
            raise UnsupportedCapabilityError(
                "ENCRYPTED_SEARCH requires a corpus built with EncryptedSchemePlugin"
            )

        plugin: EncryptedSchemePlugin | None = self.corpus.extras.get("plugin")
        if plugin is None:
            scheme_name = self.corpus.extras.get("encrypted_search_scheme", "sse")
            plugin = EncryptedSchemePlugin.get(scheme_name)

        encrypted_query = plugin.encrypt_query(query, enc_key)

        self._debug(
            "encrypted-search retrieval",
            round_n=round_n,
            scheme=self.corpus.extras.get("encrypted_search_scheme"),
            encrypted_query_keys=list(encrypted_query),
        )

        rows = self._backend.encrypted_search(
            index_id=self.corpus.index_id,
            encrypted_query=encrypted_query,
            top_k=self.config.top_k,
        )
        return self._to_docs(rows)

    def privacy_cost(self, query: str) -> float:
        return 0.0


@PrivacyRetriever.register(PrivacyProtocol.PIR)
class PIRRetriever(PrivacyRetriever):
    def retrieve(self, query: str, round_n: int):
        raise UnsupportedCapabilityError("PIR is API-complete but not implemented in this build")

    def privacy_cost(self, query: str) -> float:
        return 0.0
