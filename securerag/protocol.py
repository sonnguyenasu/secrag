from enum import Enum, auto


class PrivacyProtocol(Enum):
    BASELINE = auto()
    OBFUSCATION = auto()
    DIFF_PRIVACY = auto()
    ENCRYPTED_SEARCH = auto()
    PIR = auto()

    @property
    def budget_types(self) -> list[str]:
        return {
            PrivacyProtocol.BASELINE: [],
            PrivacyProtocol.OBFUSCATION: ["distinguishability"],
            PrivacyProtocol.DIFF_PRIVACY: ["rdp"],
            PrivacyProtocol.ENCRYPTED_SEARCH: ["he_noise"],
            PrivacyProtocol.PIR: ["count"],
        }[self]

    @property
    def requires_budget(self) -> bool:
        return len(self.budget_types) > 0

    @property
    def is_cryptographic(self) -> bool:
        return self in {PrivacyProtocol.ENCRYPTED_SEARCH, PrivacyProtocol.PIR}

    @property
    def adversary_model(self) -> str:
        return {
            PrivacyProtocol.BASELINE: "None",
            PrivacyProtocol.OBFUSCATION: "Passive observer",
            PrivacyProtocol.DIFF_PRIVACY: "Honest-but-curious server",
            PrivacyProtocol.ENCRYPTED_SEARCH: "Semi-honest (ML inference)",
            PrivacyProtocol.PIR: "Malicious server (full query privacy)",
        }[self]

    @property
    def wire_name(self) -> str:
        return {
            PrivacyProtocol.BASELINE: "Baseline",
            PrivacyProtocol.OBFUSCATION: "Obfuscation",
            PrivacyProtocol.DIFF_PRIVACY: "DiffPrivacy",
            PrivacyProtocol.ENCRYPTED_SEARCH: "EncryptedSearch",
            PrivacyProtocol.PIR: "PIR",
        }[self]
