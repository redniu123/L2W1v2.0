#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SH-DA++ 显式熔断器（Circuit Breaker）。"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class CircuitBreakerConfig:
    enabled: bool = True
    min_samples: int = 20
    rejection_rate_threshold: float = 0.60
    cooldown_steps: int = 50


class CircuitBreaker:
    """基于拒改率的轻量熔断器。"""

    def __init__(self, config: CircuitBreakerConfig | None = None):
        self.config = config or CircuitBreakerConfig()
        self.total_calls = 0
        self.total_rejections = 0
        self.cooldown_remaining = 0

    def allow_upgrade(self) -> bool:
        if not self.config.enabled:
            return True
        return self.cooldown_remaining <= 0

    def observe(self, rejected: bool) -> Dict:
        if not self.config.enabled:
            return self.get_stats()

        self.total_calls += 1
        if rejected:
            self.total_rejections += 1

        rejection_rate = self.total_rejections / self.total_calls if self.total_calls else 0.0
        triggered = False
        if self.total_calls >= self.config.min_samples and rejection_rate >= self.config.rejection_rate_threshold:
            self.cooldown_remaining = self.config.cooldown_steps
            triggered = True

        return {
            **self.get_stats(),
            'triggered': triggered,
        }

    def step_without_call(self) -> Dict:
        if not self.config.enabled:
            return self.get_stats()
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
        return self.get_stats()

    def get_stats(self) -> Dict:
        rejection_rate = self.total_rejections / self.total_calls if self.total_calls else 0.0
        return {
            'enabled': self.config.enabled,
            'total_calls': self.total_calls,
            'total_rejections': self.total_rejections,
            'rejection_rate': round(rejection_rate, 6),
            'cooldown_remaining': self.cooldown_remaining,
            'is_open': self.cooldown_remaining > 0,
        }
