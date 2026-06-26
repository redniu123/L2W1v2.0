from __future__ import annotations

from modules.router.circuit_breaker import CircuitBreaker as OldCircuitBreaker
from modules.router.circuit_breaker import CircuitBreakerConfig as OldCircuitBreakerConfig

from l2w1.routing.circuit import CircuitBreaker as NewCircuitBreaker
from l2w1.routing.circuit import CircuitBreakerConfig as NewCircuitBreakerConfig


def test_circuit_breaker_observe_and_cooldown_match_legacy() -> None:
    old = OldCircuitBreaker(
        OldCircuitBreakerConfig(
            enabled=True,
            min_samples=4,
            rejection_rate_threshold=0.5,
            cooldown_steps=3,
        )
    )
    new = NewCircuitBreaker(
        NewCircuitBreakerConfig(
            enabled=True,
            min_samples=4,
            rejection_rate_threshold=0.5,
            cooldown_steps=3,
        )
    )

    for rejected in [False, True, True, False, True, False, False]:
        assert old.allow_upgrade() == new.allow_upgrade()
        assert old.observe(rejected) == new.observe(rejected)
        assert old.get_stats() == new.get_stats()

    for _ in range(4):
        assert old.allow_upgrade() == new.allow_upgrade()
        assert old.step_without_call() == new.step_without_call()
        assert old.get_stats() == new.get_stats()


def test_disabled_circuit_breaker_matches_legacy() -> None:
    old = OldCircuitBreaker(OldCircuitBreakerConfig(enabled=False))
    new = NewCircuitBreaker(NewCircuitBreakerConfig(enabled=False))

    assert old.allow_upgrade() == new.allow_upgrade()
    assert old.observe(True) == new.observe(True)
    assert old.step_without_call() == new.step_without_call()
    assert old.get_stats() == new.get_stats()
