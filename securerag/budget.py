from securerag.config import PrivacyConfig
from securerag.errors import BudgetExhaustedError


class BudgetManager:
    def __init__(self, config: PrivacyConfig):
        self._epsilon_max = float(config.epsilon)
        self._spent = 0.0
        self._round = 0
        self._ledger: list[tuple[int, float]] = []

    def consume(self, eps_round: float) -> None:
        if self._spent + eps_round > self._epsilon_max:
            raise BudgetExhaustedError(
                f"epsilon exhausted: spent={self._spent:.3f} / {self._epsilon_max:.3f}"
            )
        self._spent += eps_round
        self._round += 1
        self._ledger.append((self._round, self._spent))

    @property
    def remaining(self) -> float:
        return self._epsilon_max - self._spent

    def snapshot(self) -> dict:
        return {
            "spent": self._spent,
            "remaining": self.remaining,
            "rounds": self._round,
            "ledger": self._ledger,
        }
