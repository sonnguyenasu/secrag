from __future__ import annotations

import hashlib
import math
import random

from securerag.dp_mechanism import DPMechanismPlugin


class GaussianMechanism(DPMechanismPlugin):
    def noise(self, embedding: list[float], sigma: float, *, query: str = "") -> list[float]:
        seed = int.from_bytes(hashlib.sha256(query.encode("utf-8")).digest()[:8], "little")
        rng = random.Random(seed)
        return [v + rng.gauss(0.0, sigma) for v in embedding]

    def rdp_cost(self, sigma: float, alpha: float) -> float:
        return alpha / (2.0 * sigma * sigma)


class LaplaceMechanism(DPMechanismPlugin):
    def noise(self, embedding: list[float], sigma: float, *, query: str = "") -> list[float]:
        seed = int.from_bytes(hashlib.sha256(query.encode("utf-8")).digest()[:8], "little")
        rng = random.Random(seed)

        def sample_laplace(scale: float) -> float:
            u = rng.random() - 0.5
            sign = -1.0 if u < 0 else 1.0
            return -scale * sign * math.log(1.0 - 2.0 * abs(u))

        return [v + sample_laplace(sigma) for v in embedding]

    def rdp_cost(self, sigma: float, alpha: float) -> float:
        if alpha <= 1.0:
            return 0.0
        try:
            log_term = math.log(
                alpha / (2 * alpha - 1) * math.exp((alpha - 1) / sigma)
                + (alpha - 1) / (2 * alpha - 1) * math.exp(-alpha / sigma)
            )
            return log_term / (alpha - 1)
        except (ValueError, OverflowError):
            return float("inf")


DPMechanismPlugin.register("gaussian", GaussianMechanism())
DPMechanismPlugin.register("laplace", LaplaceMechanism())
