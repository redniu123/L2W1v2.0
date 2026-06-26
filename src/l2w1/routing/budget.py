from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class BudgetControllerConfig:
    """
    OnlineBudgetController configuration (SH-DA++ v4.0).

    Attributes:
        window_size: Sliding window size W for actual call-rate calculation.
        warmup_samples: Warmup sample count, default min(50, window_size).
        k: Proportional coefficient controlling threshold update step size.
        lambda_min: Threshold lower bound.
        lambda_max: Threshold upper bound.
        lambda_init: Initial threshold.
        target_budget: Target call rate B in [0, 1].
    """

    window_size: int = 200
    warmup_samples: int | None = None
    k: float = 0.05
    lambda_min: float = 0.0
    lambda_max: float = 2.0
    lambda_init: float = 0.5
    target_budget: float = 0.2


class OnlineBudgetController:
    """
    SH-DA++ v4.0 online budget controller.

    Dynamically adjusts triage threshold lambda so the actual VLM call rate
    approaches the target budget B.

    Core formula:
        lambda_{t+1} = clip(lambda_t + k(B_bar - B), lambda_min, lambda_max)
    """

    def __init__(self, config: BudgetControllerConfig | None = None):
        self.config = config or BudgetControllerConfig()
        self._warmup_samples = max(
            0,
            min(
                self.config.window_size,
                self.config.warmup_samples
                if self.config.warmup_samples is not None
                else min(50, self.config.window_size),
            ),
        )

        self._lambda = self.config.lambda_init
        self._history: list[bool] = []
        self._sample_count: int = 0
        self._total_upgrades: int = 0
        self._lambda_history: list[float] = [self._lambda]

    @property
    def current_lambda(self) -> float:
        """Current threshold lambda."""
        return self._lambda

    @property
    def is_warmup(self) -> bool:
        """Whether the controller is in the warmup phase."""
        return self._sample_count < self._warmup_samples

    @property
    def actual_budget(self) -> float:
        """
        Actual call rate B_bar.

        Returns the upgrade ratio over the last W samples, or the current
        cumulative ratio if fewer than W samples have been observed.
        """
        if not self._history:
            return 0.0
        return sum(self._history) / len(self._history)

    @property
    def total_budget(self) -> float:
        """Overall call rate from the beginning to the current sample."""
        if self._sample_count == 0:
            return 0.0
        return self._total_upgrades / self._sample_count

    def decide(self, q: float, lambda_override: float | None = None) -> bool:
        """
        Decide whether to upgrade and call VLM.

        Args:
            q: Combined priority score from RuleOnlyScorer.score().
            lambda_override: Optional threshold override for tests.

        Returns:
            True when q >= lambda, otherwise False.
        """
        lam = lambda_override if lambda_override is not None else self._lambda
        return q >= lam

    def update(self, upgrade_decision: bool) -> dict[str, Any]:
        """
        Update controller state after each decision.

        Args:
            upgrade_decision: Current decision result (True=upgrade).

        Returns:
            Update details including lambda_before, lambda_after, and actual_budget.
        """
        self._sample_count += 1
        if upgrade_decision:
            self._total_upgrades += 1

        self._history.append(upgrade_decision)
        if len(self._history) > self.config.window_size:
            self._history.pop(0)

        details: dict[str, Any] = {
            "sample_count": self._sample_count,
            "lambda_before": self._lambda,
            "actual_budget": self.actual_budget,
            "target_budget": self.config.target_budget,
            "is_warmup": self.is_warmup,
            "updated": False,
        }

        if self.is_warmup:
            details["lambda_after"] = self._lambda
            details["reason"] = "warmup"
            return details

        B_bar = self.actual_budget
        B = self.config.target_budget
        k = self.config.k

        delta = k * (B_bar - B)
        lambda_new = self._lambda + delta
        lambda_new = float(np.clip(lambda_new, self.config.lambda_min, self.config.lambda_max))

        details["delta"] = delta
        details["lambda_after"] = lambda_new
        details["updated"] = True

        self._lambda = lambda_new
        self._lambda_history.append(self._lambda)

        return details

    def step(self, q: float) -> tuple[bool, dict[str, Any]]:
        """
        Single step: decide and update.

        Args:
            q: Combined priority score.

        Returns:
            The decision and update details.
        """
        upgrade = self.decide(q)
        details = self.update(upgrade)
        details["q"] = q
        details["upgrade"] = upgrade
        return upgrade, details

    def reset(self) -> None:
        """Reset controller state for a new evaluation epoch or test."""
        self._lambda = self.config.lambda_init
        self._history.clear()
        self._sample_count = 0
        self._total_upgrades = 0
        self._lambda_history = [self._lambda]

    def get_stats(self) -> dict[str, Any]:
        """
        Get controller statistics.

        Returns:
            Stats including lambda_current, lambda_history, actual_budget, and total_budget.
        """
        return {
            "lambda_current": self._lambda,
            "lambda_init": self.config.lambda_init,
            "lambda_min": self.config.lambda_min,
            "lambda_max": self.config.lambda_max,
            "lambda_history_len": len(self._lambda_history),
            "lambda_history_last_10": self._lambda_history[-10:],
            "sample_count": self._sample_count,
            "total_upgrades": self._total_upgrades,
            "actual_budget_window": self.actual_budget,
            "total_budget": self.total_budget,
            "target_budget": self.config.target_budget,
            "is_warmup": self.is_warmup,
            "window_size": self.config.window_size,
            "warmup_samples": self._warmup_samples,
            "k": self.config.k,
        }
