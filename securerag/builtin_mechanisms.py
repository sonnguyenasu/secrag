from __future__ import annotations

import hashlib
import math
import random

from securerag.cost import RDPCost
from securerag.mechanism import BudgetMechanism, rdp_cost_to_epsilon


class GaussianMechanism(BudgetMechanism):
    _ORDERS = [2.0, 4.0, 8.0, 16.0, 32.0]

    def apply(self, data: list[float], sensitivity: float, *, query: str = "") -> list[float]:
        seed = int.from_bytes(hashlib.sha256(query.encode("utf-8")).digest()[:8], "little")
        rng = random.Random(seed)
        return [v + rng.gauss(0.0, sensitivity) for v in data]

    def cost(self, sensitivity: float, **kwargs) -> RDPCost:
        orders = self.rdp_orders()
        values = [a / (2.0 * sensitivity * sensitivity) for a in orders]
        return RDPCost(orders=orders, values=values)

    def rdp_orders(self) -> list[float]:
        return list(self._ORDERS)

    def to_approx_dp(self, accumulated_cost, delta: float) -> float:
        if not isinstance(accumulated_cost, RDPCost):
            raise NotImplementedError
        return rdp_cost_to_epsilon(accumulated_cost, delta)


class LaplaceMechanism(BudgetMechanism):
    _ORDERS = [2.0, 4.0, 8.0, 16.0, 32.0]

    def apply(self, data: list[float], sensitivity: float, *, query: str = "") -> list[float]:
        seed = int.from_bytes(hashlib.sha256(query.encode("utf-8")).digest()[:8], "little")
        rng = random.Random(seed)

        def sample_laplace(scale: float) -> float:
            u = rng.random() - 0.5
            sign = -1.0 if u < 0 else 1.0
            return -scale * sign * math.log(1.0 - 2.0 * abs(u))

        return [v + sample_laplace(sensitivity) for v in data]

    def cost(self, sensitivity: float, **kwargs) -> RDPCost:
        values = []
        for alpha in self.rdp_orders():
            if alpha <= 1.0:
                values.append(0.0)
                continue
            try:
                log_term = math.log(
                    alpha / (2 * alpha - 1) * math.exp((alpha - 1) / sensitivity)
                    + (alpha - 1) / (2 * alpha - 1) * math.exp(-alpha / sensitivity)
                )
                values.append(log_term / (alpha - 1))
            except (ValueError, OverflowError):
                values.append(float("inf"))
        return RDPCost(orders=self.rdp_orders(), values=values)

    def rdp_orders(self) -> list[float]:
        return list(self._ORDERS)

    def to_approx_dp(self, accumulated_cost, delta: float) -> float:
        if not isinstance(accumulated_cost, RDPCost):
            raise NotImplementedError
        return rdp_cost_to_epsilon(accumulated_cost, delta)


BudgetMechanism.register("gaussian", GaussianMechanism())
BudgetMechanism.register("laplace", LaplaceMechanism())
