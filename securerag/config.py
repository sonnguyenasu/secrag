import dataclasses
import json
import math
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
    dp_mechanism: str = "gaussian"
    encrypted_search_scheme: str = "sse"
    structured_use_bigrams: bool = True

    def __post_init__(self) -> None:
        if self.protocol is not PrivacyProtocol.DIFF_PRIVACY and self.epsilon != 1.0:
            warnings.warn(
                f"epsilon has no effect for protocol {self.protocol.name}. Use DIFF_PRIVACY.",
                stacklevel=2,
            )
        if self.protocol is PrivacyProtocol.DIFF_PRIVACY:
            orders = [2.0, 4.0, 8.0, 16.0, 32.0]
            per_order_cost = lambda a: a / (2.0 * self.noise_std**2)
            if self.noise_std <= 0.0:
                warnings.warn(
                    "noise_std must be > 0 for DIFF_PRIVACY.",
                    stacklevel=2,
                )
                return

            try:
                import securerag.builtin_mechanisms  # noqa: F401
                from securerag.dp_mechanism import DPMechanismPlugin

                mechanism = DPMechanismPlugin.get(self.dp_mechanism)
                orders = mechanism.rdp_orders()
                per_order_cost = lambda a: mechanism.rdp_cost(self.noise_std, a)
            except Exception:
                # Keep constructor resilient if mechanism registry is not initialised yet.
                pass

            single_round_eps = min(
                per_order_cost(a) + math.log(1.0 / self.delta) / (a - 1.0)
                for a in orders
            )
            if single_round_eps > self.epsilon:
                warnings.warn(
                    f"noise_std={self.noise_std} costs epsilon~{single_round_eps:.2f} per round "
                    f"but epsilon={self.epsilon}. Round 1 will be rejected. "
                    "Increase epsilon or noise_std.",
                    stacklevel=2,
                )

    def for_protocol(self, protocol: PrivacyProtocol, **kwargs) -> "PrivacyConfig":
        return dataclasses.replace(self, protocol=protocol, **kwargs)

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
