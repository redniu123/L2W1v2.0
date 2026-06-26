from __future__ import annotations

from modules.router.uncertainty_router import BudgetControllerConfig as OldBudgetControllerConfig
from modules.router.uncertainty_router import OnlineBudgetController as OldOnlineBudgetController

from l2w1.routing.budget import BudgetControllerConfig as NewBudgetControllerConfig
from l2w1.routing.budget import OnlineBudgetController as NewOnlineBudgetController


def test_online_budget_controller_warmup_and_steady_state_match_legacy() -> None:
    old = OldOnlineBudgetController(
        OldBudgetControllerConfig(
            window_size=5,
            warmup_samples=3,
            k=0.05,
            lambda_min=0.0,
            lambda_max=2.0,
            lambda_init=0.5,
            target_budget=0.4,
        )
    )
    new = NewOnlineBudgetController(
        NewBudgetControllerConfig(
            window_size=5,
            warmup_samples=3,
            k=0.05,
            lambda_min=0.0,
            lambda_max=2.0,
            lambda_init=0.5,
            target_budget=0.4,
        )
    )

    q_values = [0.10, 0.70, 0.60, 0.30, 0.90, 0.55, 0.45, 0.80, 0.20]

    for q in q_values:
        assert old.current_lambda == new.current_lambda
        assert old.is_warmup == new.is_warmup
        assert old.actual_budget == new.actual_budget
        assert old.total_budget == new.total_budget
        assert old.step(q) == new.step(q)
        assert old.current_lambda == new.current_lambda
        assert old.get_stats() == new.get_stats()


def test_online_budget_controller_reset_and_lambda_override_match_legacy() -> None:
    old = OldOnlineBudgetController(OldBudgetControllerConfig(window_size=4, warmup_samples=1))
    new = NewOnlineBudgetController(NewBudgetControllerConfig(window_size=4, warmup_samples=1))

    assert old.decide(0.4, 0.3) == new.decide(0.4, 0.3)
    assert old.decide(0.2, 0.3) == new.decide(0.2, 0.3)

    for q in [0.6, 0.1, 0.9]:
        assert old.step(q) == new.step(q)

    old.reset()
    new.reset()

    assert old.current_lambda == new.current_lambda
    assert old.get_stats() == new.get_stats()
