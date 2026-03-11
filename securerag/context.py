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
        budget.consume(cost, compose_fn=self._composition_hook)

    def apply_noise_hooks(
        self,
        operation: str,
        embedding: list[float],
        config,
        budget_state: dict,
        default_cost: Cost,
    ) -> tuple[list[float], Cost]:
        hooks = self._noise_hooks.get(operation, [])
        if not hooks:
            return embedding, default_cost

        current_embedding = embedding
        current_cost = default_cost
        for hook in hooks:
            out = hook(current_embedding, config, budget_state)
            if not isinstance(out, tuple) or len(out) != 2:
                if self._strict:
                    raise TypeError(
                        "Noise hook must return tuple[list[float], Cost]"
                    )
                continue
            current_embedding, current_cost = out
        return current_embedding, current_cost

    def apply_budget_hooks(
        self,
        operation: str,
        docs,
        config,
        corpus_budgets: dict[str, Budget],
        default_cost: Cost,
    ) -> Cost:
        hooks = self._budget_hooks.get(operation, [])
        if not hooks:
            return default_cost

        current_cost = default_cost
        for hook in hooks:
            out = hook(docs, config, corpus_budgets)
            if out is None:
                continue
            if not isinstance(out, Cost):
                if self._strict:
                    raise TypeError("Budget hook must return a Cost object")
                continue
            current_cost = out
        return current_cost

    def register_budget(self, key: str, budget: Budget) -> None:
        self._budgets[key] = budget

    def snapshot(self) -> dict:
        return {k: v.snapshot() for k, v in self._budgets.items()}
