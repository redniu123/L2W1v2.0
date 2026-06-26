from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt

FEATURE_NAMES_V51 = ["Mean_Confidence", "Min_Confidence", "b_edge", "drop", "r_d"]
FEATURE_NAMES_V40 = ["v_edge", "b_edge", "v_edge_x_b_edge", "drop"]


@dataclass
class CalibratedScorerConfig:
    """Calibrated scorer configuration."""

    enabled: bool = False
    weights: dict[str, float] | None = None
    bias: float = 0.0
    feature_version: str = "v5.1"
    feature_names: list[str] | None = None

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = {name: 0.2 for name in FEATURE_NAMES_V51}
        if self.feature_names is None:
            if "v_edge" in self.weights:
                self.feature_names = FEATURE_NAMES_V40
                self.feature_version = "v4.0"
            else:
                self.feature_names = FEATURE_NAMES_V51
                self.feature_version = "v5.1"


class CalibratedScorer:
    """
    SH-DA++ v5.1 calibrated scorer.

    Supports v5.1 (5D) and v4.0 (4D) features, inferred from configuration.
    """

    def __init__(self, config: CalibratedScorerConfig):
        self.config = config
        weights = cast(dict[str, float], config.weights)
        self.feature_names = cast(list[str], config.feature_names)
        self.feature_version = config.feature_version

        self.w: npt.NDArray[np.float32] = np.array(
            [weights.get(name, 0.0) for name in self.feature_names],
            dtype=np.float32,
        )
        self.b = float(config.bias)

    def compute_score_v51(
        self,
        mean_conf: float,
        min_conf: float,
        b_edge: float,
        drop: float,
        r_d: float,
    ) -> dict[str, Any]:
        """
        v5.1 scoring interface.

        Args:
            mean_conf: Global confidence mean.
            min_conf: Local minimum confidence.
            b_edge: CTC boundary blank strength.
            drop: Left/right boundary asymmetry.
            r_d: Domain semantic safety anchor.

        Returns:
            Dict containing s_b, logit, and feature values.
        """
        x = np.array([mean_conf, min_conf, b_edge, drop, r_d], dtype=np.float32)
        z = np.dot(self.w, x) + self.b
        s_b = self._sigmoid(z)

        return {
            "s_b": float(s_b),
            "logit": float(z),
            "features": {
                "Mean_Confidence": float(mean_conf),
                "Min_Confidence": float(min_conf),
                "b_edge": float(b_edge),
                "drop": float(drop),
                "r_d": float(r_d),
            },
        }

    def compute_score(
        self,
        mean_conf: float = 0.0,
        min_conf: float = 0.0,
        b_edge: float = 0.0,
        drop: float = 0.0,
        r_d: float = 0.0,
        v_edge: float | None = None,
    ) -> dict[str, Any]:
        """Unified scoring interface, preferring v5.1 while keeping v4.0 compatibility."""
        if self.feature_version == "v4.0" and v_edge is not None:
            x = np.array([v_edge, b_edge, v_edge * b_edge, drop], dtype=np.float32)
        else:
            x = np.array([mean_conf, min_conf, b_edge, drop, r_d], dtype=np.float32)

        if len(x) > len(self.w):
            x = x[: len(self.w)]
        elif len(x) < len(self.w):
            x = np.pad(x, (0, len(self.w) - len(x)))

        z = np.dot(self.w, x) + self.b
        s_b = self._sigmoid(z)

        return {
            "s_b": float(s_b),
            "logit": float(z),
            "feature_version": self.feature_version,
        }

    @staticmethod
    def _sigmoid(z: float | np.floating[Any]) -> float:
        return float(1.0 / (1.0 + np.exp(-np.clip(z, -500, 500))))

    @classmethod
    def from_config_dict(cls, config_dict: dict[str, Any]) -> CalibratedScorer:
        """Create scorer from a configuration dictionary."""
        weights = config_dict.get("weights", {})
        feature_names = config_dict.get("feature_names", None)
        feature_version = config_dict.get("feature_version", "v5.1")

        config = CalibratedScorerConfig(
            enabled=config_dict.get("enabled", False),
            weights=weights,
            bias=config_dict.get("bias", 0.0),
            feature_version=feature_version,
            feature_names=feature_names,
        )
        return cls(config)


class RuleOnlyScorer:
    """Rule-only scorer reused by random and ConfOnly baselines."""

    def __init__(self, a1: float = 0.5, a2: float = 0.3, a3: float = 0.2):
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3

    def compute_score(
        self,
        mean_conf: float = 0.0,
        b_edge: float = 0.0,
        drop: float = 0.0,
        v_edge: float | None = None,
        v_edge_x_b_edge: float | None = None,
    ) -> dict[str, float]:
        """Rule-only score based on confidence and blank boundary."""
        _ = v_edge, v_edge_x_b_edge
        risk = self.a1 * (1.0 - mean_conf) + self.a2 * b_edge + self.a3 * drop
        s_b = float(np.clip(risk, 0.0, 1.0))
        return {"s_b": s_b}
