from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

from securerag.cost import Cost, RDPCost, zero_cost_like
from securerag.errors import BudgetExhaustedError

if TYPE_CHECKING:
    from securerag.config import PrivacyConfig
    from securerag.mechanism import BudgetMechanism


class Budget:
    """Generalized finite-resource budget with backward-compatible DP helpers."""

    def __init__(self, total: Cost, mechanism: "BudgetMechanism", delta: float = 1e-5):
        self._total = total
        self._spent = zero_cost_like(total)
        self._mechanism = mechanism
        self._delta = float(delta)
        self._round = 0
        self._ledger: list[tuple[int, float]] = []

    @staticmethod
    def _rdp_to_dp(rdp_eps: list[float], delta: float, orders: list[float]) -> float:
        return min(r + math.log(1.0 / delta) / (a - 1.0) for a, r in zip(orders, rdp_eps))

    @classmethod
    def from_config(
        cls,
        config: "PrivacyConfig",
        mechanism: "BudgetMechanism | None" = None,
    ) -> "Budget":
        from securerag.mechanism import BudgetMechanism

        if mechanism is None:
            import securerag.builtin_mechanisms  # noqa: F401

            mechanism = BudgetMechanism.get(config.dp_mechanism)

        orders = mechanism.rdp_orders()
        total = RDPCost(orders=orders, values=[float(config.epsilon)] * len(orders))
        return cls(total=total, mechanism=mechanism, delta=float(config.delta))

    @classmethod
    def rdp(
        cls,
        epsilon: float,
        delta: float,
        mechanism: "BudgetMechanism",
    ) -> "Budget":
        orders = mechanism.rdp_orders()
        total = RDPCost(orders=orders, values=[float(epsilon)] * len(orders))
        return cls(total=total, mechanism=mechanism, delta=float(delta))

    def _normalize_cost(self, cost_or_sigma: Cost | float) -> Cost:
        if isinstance(cost_or_sigma, Cost):
            return cost_or_sigma
        sigma = float(cost_or_sigma)
        if sigma <= 0.0:
            raise ValueError("sigma must be > 0")
        return self._mechanism.cost(sensitivity=sigma)

    def _projected_spent(self, cost_or_sigma: Cost | float) -> Cost:
        return self._spent + self._normalize_cost(cost_or_sigma)

    def _effective_value(self, cost: Cost) -> float:
        try:
            return float(self._mechanism.to_approx_dp(cost, self._delta))
        except NotImplementedError:
            if isinstance(cost, RDPCost):
                return self._rdp_to_dp(cost.values, self._delta, cost.orders)
            for attr in ("epsilon", "count", "noise_bits"):
                if hasattr(cost, attr):
                    return float(getattr(cost, attr))
            raise

    @property
    def spent(self) -> float:
        if isinstance(self._spent, RDPCost) and all(x == 0.0 for x in self._spent.values):
            return 0.0
        return self._effective_value(self._spent)

    def epsilon_if_consumed(self, sigma: float) -> float:
        candidate = self._projected_spent(sigma)
        return self._effective_value(candidate)

    def incremental_cost(self, sigma: float) -> float:
        return max(0.0, self.epsilon_if_consumed(sigma) - self.spent)

    def can_consume(self, cost_or_sigma: Cost | float) -> bool:
        candidate = self._projected_spent(cost_or_sigma)
        return self._effective_value(candidate) <= self._effective_value(self._total)

    def consume(
        self,
        cost_or_sigma: Cost | float = 0.0,
        *,
        sigma: float | None = None,
        compose_fn: Callable[[Cost, Cost], Cost] | None = None,
    ) -> None:
        token: Cost | float = sigma if sigma is not None else cost_or_sigma
        cost = self._normalize_cost(token)
        candidate = compose_fn(self._spent, cost) if compose_fn is not None else (self._spent + cost)
        candidate_val = self._effective_value(candidate)
        limit = self._effective_value(self._total)
        if candidate_val > limit:
            raise BudgetExhaustedError(f"epsilon exhausted: {candidate_val:.3f} / {limit:.3f}")
        self._spent = candidate
        self._round += 1
        self._ledger.append((self._round, candidate_val))

    @property
    def remaining(self) -> float:
        return max(0.0, self._effective_value(self._total) - self.spent)

    def snapshot(self) -> dict:
        return {
            "spent": self.spent,
            "remaining": self.remaining,
            "rounds": self._round,
            "ledger": self._ledger,
            "epsilon_max": self._effective_value(self._total),
            "delta": self._delta,
            "mechanism": type(self._mechanism).__name__,
        }


class BudgetManager(Budget):
    """Backward-compatible alias with legacy constructor shape."""

    def __init__(self, config: "PrivacyConfig", mechanism: "BudgetMechanism | None" = None):
        base = Budget.from_config(config, mechanism=mechanism)
        self.__dict__.update(base.__dict__)
