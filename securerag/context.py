from __future__ import annotations

from securerag.budget import Budget
from securerag.cost import Cost


class PrivacyContext:
    """Context manager for tracking privacy costs across a retrieval pipeline."""

    def __init__(self, strict: bool = True):
        self._strict = strict
        self._budgets: dict[str, Budget] = {}
        self._noise_hooks: dict[str, list] = {}
        self._budget_hooks: dict[str, list] = {}
        self._composition_hook = None
        self._active = False

    def __enter__(self) -> "PrivacyContext":
        self._active = True
        return self

    def __exit__(self, *args) -> None:
        self._active = False

    def register_noise_hook(self, operation: str):
        def decorator(fn):
            self._noise_hooks.setdefault(operation, []).append(fn)
            return fn

        return decorator

    def register_budget_hook(self, operation: str):
        def decorator(fn):
            self._budget_hooks.setdefault(operation, []).append(fn)
            return fn

        return decorator

    def register_composition_hook(self, fn):
        self._composition_hook = fn
        return fn

    def charge(self, budget_key: str, cost: Cost) -> None:
        if not self._active:
            return
        budget = self._budgets.get(budget_key)
        if budget is None:
            return
        if self._composition_hook is not None:
            budget._spent = self._composition_hook(budget._spent, cost)
            return
        budget.consume(cost)

    def register_budget(self, key: str, budget: Budget) -> None:
        self._budgets[key] = budget

    def snapshot(self) -> dict:
        return {k: v.snapshot() for k, v in self._budgets.items()}
