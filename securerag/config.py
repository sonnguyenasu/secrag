import dataclasses
import json
import warnings
from dataclasses import dataclass

from securerag.protocol import PrivacyProtocol


@dataclass(slots=True)
class PrivacyConfig:
    protocol: PrivacyProtocol = PrivacyProtocol.BASELINE
    epsilon: float = 1.0
    delta: float = 1e-5
    max_rounds: int = 5
    k_decoys: int = 3
    noise_std: float = 0.1
    top_k: int = 5
    backend: str = "http://127.0.0.1:8099"
    verbose: bool = False
    paraphrase_decoys: bool = True
    encrypted_search_scheme: str = "sse"
    structured_use_bigrams: bool = True

    def __post_init__(self) -> None:
        if not self.protocol.requires_budget and self.epsilon != 1.0:
            warnings.warn(
                f"epsilon has no effect for protocol {self.protocol.name}. Use DIFF_PRIVACY.",
                stacklevel=2,
            )

    def for_protocol(self, protocol: PrivacyProtocol) -> "PrivacyConfig":
        return dataclasses.replace(self, protocol=protocol)

    def to_json(self) -> str:
        payload = dataclasses.asdict(self)
        payload["protocol"] = self.protocol.name
        return json.dumps(payload, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "PrivacyConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["protocol"] = PrivacyProtocol[data["protocol"]]
        return cls(**data)
