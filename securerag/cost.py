from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass


class Cost:
    """Base class for a privacy cost incurred by one operation."""

    mechanism: str = "unknown"

    def __add__(self, other: "Cost") -> "Cost":
        raise NotImplementedError(
            f"No composition rule defined between {type(self).__name__} and {type(other).__name__}."
        )


@dataclass
class RDPCost(Cost):
    """Cost tracked at fixed Renyi orders."""

    mechanism: str = "rdp"
    orders: list[float] = field(default_factory=lambda: [2.0, 4.0, 8.0, 16.0, 32.0])
    values: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.values:
            self.values = [0.0 for _ in self.orders]
        if len(self.orders) != len(self.values):
            raise ValueError("orders and values must have identical length")

    def __add__(self, other: "Cost") -> "RDPCost":
        if not isinstance(other, RDPCost):
            raise NotImplementedError
        if self.orders != other.orders:
            raise ValueError("RDP orders must match to compose")
        return RDPCost(
            orders=list(self.orders),
            values=[a + b for a, b in zip(self.values, other.values)],
        )


@dataclass
class PureDPCost(Cost):
    mechanism: str = "pure_dp"
    epsilon: float = 0.0

    def __add__(self, other: "Cost") -> "PureDPCost":
        if not isinstance(other, PureDPCost):
            raise NotImplementedError
        return PureDPCost(epsilon=self.epsilon + other.epsilon)


@dataclass
class CountCost(Cost):
    mechanism: str = "count"
    count: int = 1

    def __add__(self, other: "Cost") -> "CountCost":
        if not isinstance(other, CountCost):
            raise NotImplementedError
        return CountCost(count=self.count + other.count)


@dataclass
class HENoiseCost(Cost):
    mechanism: str = "he_noise"
    noise_bits: int = 0

    def __add__(self, other: "Cost") -> "HENoiseCost":
        if not isinstance(other, HENoiseCost):
            raise NotImplementedError
        return HENoiseCost(noise_bits=self.noise_bits + other.noise_bits)


def zero_cost_like(template: Cost) -> Cost:
    """Create a zero-value cost of the same concrete type."""

    if is_dataclass(template):
        kwargs = {}
        for f in fields(template):
            if f.name == "mechanism":
                continue
            if f.name == "orders":
                kwargs[f.name] = list(getattr(template, f.name))
            elif f.type is int or isinstance(getattr(template, f.name), int):
                kwargs[f.name] = 0
            elif f.type is float or isinstance(getattr(template, f.name), float):
                kwargs[f.name] = 0.0
            elif isinstance(getattr(template, f.name), list):
                val = getattr(template, f.name)
                kwargs[f.name] = [0.0 for _ in val]
        return type(template)(**kwargs)

    try:
        return type(template)()
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise TypeError(f"Cannot infer zero value for cost type {type(template).__name__}") from exc
