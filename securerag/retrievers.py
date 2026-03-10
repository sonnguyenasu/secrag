import math

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
            sigma=self.config.noise_std,
        )
        return self._to_docs(rows)

    def privacy_cost(self, query: str) -> float:
        sigma = self.config.noise_std
        delta = self.config.delta
        orders = [2.0, 4.0, 8.0, 16.0, 32.0]
        rdp_eps = [alpha / (2.0 * sigma**2) for alpha in orders]
        return min(r + math.log(1.0 / delta) / (a - 1.0) for a, r in zip(orders, rdp_eps))


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
