from securerag.errors import UnsupportedCapabilityError
from securerag.models import PrivateQuery
from securerag.protocol import PrivacyProtocol
from securerag.retriever import PrivacyRetriever
from securerag.scheme_plugin import EncryptedSchemePlugin


@PrivacyRetriever.register(PrivacyProtocol.BASELINE)
class BaselineRetriever(PrivacyRetriever):
    def retrieve(self, query: str | PrivateQuery, round_n: int):
        q_text, _ = self._resolve_query(query)
        rows = self._backend.batch_retrieve(
            index_id=self.corpus.index_id,
            queries=[q_text],
            top_k=self.config.top_k,
        )[0]
        return self._to_docs(rows)

    def privacy_cost(self, query: str | PrivateQuery) -> float:
        return 0.0


@PrivacyRetriever.register(PrivacyProtocol.OBFUSCATION)
class ObfuscationRetriever(PrivacyRetriever):
    def retrieve(self, query: str | PrivateQuery, round_n: int):
        q_text, _ = self._resolve_query(query)
        raw_decoys = self._backend.generate_decoys(
            index_id=self.corpus.index_id,
            query=q_text,
            k=self.config.k_decoys,
        )
        decoys = self._paraphrase_decoys(raw_decoys, q_text)
        self._debug(
            "obfuscation retrieval",
            round_n=round_n,
            real_query=q_text,
            raw_decoy_queries=raw_decoys,
            paraphrased_decoy_queries=decoys,
            paraphrase_enabled=self.config.paraphrase_decoys,
        )
        results = self._backend.batch_retrieve(
            index_id=self.corpus.index_id,
            queries=[q_text] + decoys,
            top_k=self.config.top_k,
        )
        return self._to_docs(results[0])

    def privacy_cost(self, query: str | PrivateQuery) -> float:
        return 0.0


@PrivacyRetriever.register(PrivacyProtocol.DIFF_PRIVACY)
class DiffPrivacyRetriever(PrivacyRetriever):
    def retrieve(self, query: str | PrivateQuery, round_n: int):
        q_text, required_budget = self._resolve_query(query)
        sigma = float(self.config.noise_std)
        eps = self.budget.incremental_cost(sigma) if required_budget else 0.0
        cost = self._dp_mechanism.cost(sensitivity=sigma) if required_budget else None
        base_embedding = self._backend.embed(query=q_text)
        noised = (
            self._dp_mechanism.noise(base_embedding, sigma, query=q_text)
            if required_budget
            else base_embedding
        )
        if required_budget and self._ctx is not None:
            noised, cost = self._ctx.apply_noise_hooks(
                "encode",
                noised,
                self.config,
                self.budget.snapshot(),
                cost if cost is not None else self._dp_mechanism.cost(sensitivity=sigma),
            )
        rows = self._backend.retrieve_by_embedding(
            index_id=self.corpus.index_id,
            embedding=noised,
            top_k=self.config.top_k,
            query=q_text,
            sigma=sigma,
        )
        if required_budget and self._ctx is not None:
            docs = self._to_docs(rows)
            cost = self._ctx.apply_budget_hooks(
                "retrieve",
                docs,
                self.config,
                {self.config.protocol.name: self.budget},
                cost if cost is not None else self._dp_mechanism.cost(sensitivity=sigma),
            )
            self._charge(cost)
            self._debug(
                "diff-privacy retrieval",
                round_n=round_n,
                original_query=q_text,
                noised_query_embedding=noised,
                mechanism=type(self._dp_mechanism).__name__,
                epsilon_cost=eps,
                epsilon_remaining=self.budget.remaining,
                required_budget=required_budget,
            )
            return docs
        if required_budget:
            self._charge(cost if cost is not None else self._dp_mechanism.cost(sensitivity=sigma))
        self._debug(
            "diff-privacy retrieval",
            round_n=round_n,
            original_query=q_text,
            noised_query_embedding=noised,
            mechanism=type(self._dp_mechanism).__name__,
            epsilon_cost=eps,
            epsilon_remaining=self.budget.remaining,
            required_budget=required_budget,
        )
        return self._to_docs(rows)

    def privacy_cost(self, query: str | PrivateQuery) -> float:
        _, required_budget = self._resolve_query(query)
        if not required_budget:
            return 0.0
        return self.budget.incremental_cost(float(self.config.noise_std))


@PrivacyRetriever.register(PrivacyProtocol.ENCRYPTED_SEARCH)
class EncryptedSearchRetriever(PrivacyRetriever):
    def retrieve(self, query: str | PrivateQuery, round_n: int):
        q_text, _ = self._resolve_query(query)
        enc_key = self.corpus.extras.get("enc_key")
        if not enc_key:
            raise UnsupportedCapabilityError(
                "ENCRYPTED_SEARCH requires a corpus built with EncryptedSchemePlugin"
            )

        plugin: EncryptedSchemePlugin | None = self.corpus.extras.get("plugin")
        if plugin is None:
            scheme_name = self.corpus.extras.get("encrypted_search_scheme", "sse")
            plugin = EncryptedSchemePlugin.get(scheme_name)

        encrypted_query = plugin.encrypt_query(q_text, enc_key)

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

    def privacy_cost(self, query: str | PrivateQuery) -> float:
        return 0.0


@PrivacyRetriever.register(PrivacyProtocol.PIR)
class PIRRetriever(PrivacyRetriever):
    def retrieve(self, query: str | PrivateQuery, round_n: int):
        raise UnsupportedCapabilityError("PIR is API-complete but not implemented in this build")

    def privacy_cost(self, query: str | PrivateQuery) -> float:
        return 0.0
