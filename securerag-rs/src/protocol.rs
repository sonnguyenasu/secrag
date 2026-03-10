#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrivacyProtocol {
    Baseline,
    Obfuscation,
    DiffPrivacy,
    EncryptedSearch,
    Pir,
}

impl PrivacyProtocol {
    pub fn from_wire(s: &str) -> Option<Self> {
        match s {
            "Baseline" => Some(Self::Baseline),
            "Obfuscation" => Some(Self::Obfuscation),
            "DiffPrivacy" => Some(Self::DiffPrivacy),
            "EncryptedSearch" => Some(Self::EncryptedSearch),
            "PIR" => Some(Self::Pir),
            _ => None,
        }
    }
}
