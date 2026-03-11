from __future__ import annotations

from securerag.benchmarks.loaders import QueryRecord, load_qa_dataset
from securerag.corpus import SecureCorpus
from securerag.protocol import PrivacyProtocol


class TriviaQA:
    @staticmethod
    def load(
        split: str = "test",
        n: int = 100,
        *,
        data_dir: str | None = None,
        protocol: PrivacyProtocol = PrivacyProtocol.BASELINE,
    ) -> tuple[SecureCorpus, list[QueryRecord]]:
        return load_qa_dataset(
            stem="triviaqa",
            split=split,
            n=n,
            data_dir=data_dir,
            protocol=protocol,
        )
