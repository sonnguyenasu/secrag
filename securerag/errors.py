class SecureRAGError(Exception):
    pass


class UnknownProtocolError(SecureRAGError):
    pass


class ProtocolMismatchError(SecureRAGError):
    pass


class BudgetExhaustedError(SecureRAGError):
    pass


class BackendError(SecureRAGError):
    pass


class UnsupportedCapabilityError(SecureRAGError):
    pass
