import math
from typing import TYPE_CHECKING

from securerag.config import PrivacyConfig
from securerag.errors import BudgetExhaustedError

if TYPE_CHECKING:
    from securerag.dp_mechanism import DPMechanismPlugin


class BudgetManager:
    _ORDERS = [2.0, 4.0, 8.0, 16.0, 32.0]

    def __init__(self, config: PrivacyConfig, mechanism: "DPMechanismPlugin | None" = None):
        self._epsilon_max = float(config.epsilon)
        self._delta = float(config.delta)
        if mechanism is None:
            import securerag.builtin_mechanisms  # noqa: F401
            from securerag.dp_mechanism import DPMechanismPlugin

            mechanism = DPMechanismPlugin.get(config.dp_mechanism)
        self._mechanism = mechanism
        self._orders = mechanism.rdp_orders()
        self._rdp_acc = [0.0] * len(self._orders)
        self._round = 0
        self._ledger: list[tuple[int, float]] = []

    @staticmethod
    def _rdp_to_dp(rdp_eps: list[float], delta: float, orders: list[float]) -> float:
        return min(r + math.log(1.0 / delta) / (a - 1.0) for a, r in zip(orders, rdp_eps))

    def _candidate_acc(self, sigma: float) -> list[float]:
        if sigma <= 0.0:
            raise ValueError("sigma must be > 0")
        return [
            acc + self._mechanism.rdp_cost(sigma, a)
            for acc, a in zip(self._rdp_acc, self._orders)
        ]

    @property
    def spent(self) -> float:
        if all(x == 0.0 for x in self._rdp_acc):
            return 0.0
        return self._rdp_to_dp(self._rdp_acc, self._delta, self._orders)

    def epsilon_if_consumed(self, sigma: float) -> float:
        candidate = self._candidate_acc(sigma)
        return self._rdp_to_dp(candidate, self._delta, self._orders)

    def incremental_cost(self, sigma: float) -> float:
        return max(0.0, self.epsilon_if_consumed(sigma) - self.spent)

    def can_consume(self, sigma: float) -> bool:
        return self.epsilon_if_consumed(sigma) <= self._epsilon_max

    def consume(self, sigma: float) -> None:
        candidate = self._candidate_acc(sigma)
        candidate_eps = self._rdp_to_dp(candidate, self._delta, self._orders)
        if candidate_eps > self._epsilon_max:
            raise BudgetExhaustedError(
                f"epsilon exhausted: {candidate_eps:.3f} / {self._epsilon_max:.3f}"
            )
        self._rdp_acc = candidate
        self._round += 1
        self._ledger.append((self._round, candidate_eps))

    @property
    def remaining(self) -> float:
        return max(0.0, self._epsilon_max - self.spent)

    def snapshot(self) -> dict:
        return {
            "spent": self.spent,
            "remaining": self.remaining,
            "rounds": self._round,
            "ledger": self._ledger,
            "epsilon_max": self._epsilon_max,
            "delta": self._delta,
            "mechanism": type(self._mechanism).__name__,
        }
