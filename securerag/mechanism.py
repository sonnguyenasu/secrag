from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

from securerag.cost import Cost, CountCost, HENoiseCost, RDPCost


_REGISTRY: dict[str, "BudgetMechanism"] = {}


class BudgetMechanism(ABC):
    """Mechanism interface for privacy/noise application and budget accounting."""

    def apply(self, data: list[float], sensitivity: float, **kwargs: Any) -> list[float]:
        raise NotImplementedError

    def cost(self, sensitivity: float, **kwargs: Any) -> Cost:
        raise NotImplementedError

    def to_approx_dp(self, accumulated_cost: Cost, delta: float) -> float:
        raise NotImplementedError

    # Backward-compatibility shim for legacy DP plugin APIs.
    def noise(self, embedding: list[float], sigma: float, *, query: str = "") -> list[float]:
        return self.apply(embedding, sensitivity=sigma, query=query)

    def rdp_orders(self) -> list[float]:
        c = self.cost(sensitivity=1.0)
        if isinstance(c, RDPCost):
            return list(c.orders)
        return [2.0, 4.0, 8.0, 16.0, 32.0]

    def rdp_cost(self, sigma: float, alpha: float) -> float:
        c = self.cost(sensitivity=sigma)
        if not isinstance(c, RDPCost):
            return 0.0
        try:
            idx = c.orders.index(alpha)
        except ValueError:
            return 0.0
        return float(c.values[idx])

    def prepare_corpus(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return chunks

    @classmethod
    def register(cls, name: str, instance: "BudgetMechanism") -> None:
        _REGISTRY[name.lower()] = instance

    @classmethod
    def get(cls, name: str) -> "BudgetMechanism":
        key = name.lower()
        if key not in _REGISTRY:
            raise KeyError(f"No mechanism '{name}' registered. Available: {sorted(_REGISTRY)}")
        return _REGISTRY[key]

    @classmethod
    def registered_names(cls) -> list[str]:
        return sorted(_REGISTRY)


class CountMechanism(BudgetMechanism):
    def apply(self, data: list[float], sensitivity: float, **kwargs: Any) -> list[float]:
        return data

    def cost(self, sensitivity: float, **kwargs: Any) -> Cost:
        count = int(kwargs.get("count", max(1, int(round(sensitivity)))))
        return CountCost(count=count)

    def to_approx_dp(self, accumulated_cost: Cost, delta: float) -> float:
        if not isinstance(accumulated_cost, CountCost):
            raise NotImplementedError
        return float(accumulated_cost.count)


class HENoiseMechanism(BudgetMechanism):
    def apply(self, data: list[float], sensitivity: float, **kwargs: Any) -> list[float]:
        return data

    def cost(self, sensitivity: float, **kwargs: Any) -> Cost:
        bits = int(kwargs.get("noise_bits", max(0, int(round(sensitivity)))))
        return HENoiseCost(noise_bits=bits)

    def to_approx_dp(self, accumulated_cost: Cost, delta: float) -> float:
        if not isinstance(accumulated_cost, HENoiseCost):
            raise NotImplementedError
        return float(accumulated_cost.noise_bits)


def rdp_cost_to_epsilon(accumulated_cost: RDPCost, delta: float) -> float:
    return min(
        value + math.log(1.0 / delta) / (order - 1.0)
        for order, value in zip(accumulated_cost.orders, accumulated_cost.values)
    )


class DPMechanismPlugin(BudgetMechanism, ABC):
    """Legacy DP extension point retained for backward compatibility."""

    @abstractmethod
    def noise(self, embedding: list[float], sigma: float, *, query: str = "") -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def rdp_cost(self, sigma: float, alpha: float) -> float:
        raise NotImplementedError

    def apply(self, data: list[float], sensitivity: float, **kwargs: Any) -> list[float]:
        return self.noise(data, sensitivity, query=str(kwargs.get("query", "")))

    def cost(self, sensitivity: float, **kwargs: Any) -> Cost:
        orders = self.rdp_orders()
        return RDPCost(
            orders=orders,
            values=[self.rdp_cost(sensitivity, alpha) for alpha in orders],
        )

    def to_approx_dp(self, accumulated_cost: Cost, delta: float) -> float:
        if not isinstance(accumulated_cost, RDPCost):
            raise NotImplementedError
        return rdp_cost_to_epsilon(accumulated_cost, delta)
