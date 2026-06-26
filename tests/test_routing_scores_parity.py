from __future__ import annotations

from typing import Any

from modules.router.calibrated_scorer import CalibratedScorer as OldCalibratedScorer
from modules.router.calibrated_scorer import CalibratedScorerConfig as OldCalibratedScorerConfig
from modules.router.calibrated_scorer import RuleOnlyScorer as OldRuleOnlyScorer

from l2w1.routing.scores import CalibratedScorer as NewCalibratedScorer
from l2w1.routing.scores import CalibratedScorerConfig as NewCalibratedScorerConfig
from l2w1.routing.scores import RuleOnlyScorer as NewRuleOnlyScorer


def _assert_score_dict_matches(old_result: dict[str, Any], new_result: dict[str, Any]) -> None:
    assert old_result.keys() == new_result.keys()
    for key, old_value in old_result.items():
        new_value = new_result[key]
        if isinstance(old_value, dict):
            assert old_value == new_value
        else:
            assert old_value == new_value


def test_calibrated_scorer_v51_matches_legacy() -> None:
    weights = {
        "Mean_Confidence": -0.7,
        "Min_Confidence": -0.2,
        "b_edge": 1.3,
        "drop": 0.4,
        "r_d": 0.9,
    }
    old = OldCalibratedScorer(OldCalibratedScorerConfig(enabled=True, weights=weights, bias=-0.15))
    new = NewCalibratedScorer(NewCalibratedScorerConfig(enabled=True, weights=weights, bias=-0.15))

    feature_vectors = [
        (0.95, 0.80, 0.10, 0.00, 0.00),
        (0.45, 0.20, 0.70, 0.30, 0.90),
        (0.00, 0.00, 1.00, 1.00, 1.00),
    ]

    for vector in feature_vectors:
        _assert_score_dict_matches(
            old.compute_score_v51(*vector),
            new.compute_score_v51(*vector),
        )
        _assert_score_dict_matches(
            old.compute_score(
                mean_conf=vector[0],
                min_conf=vector[1],
                b_edge=vector[2],
                drop=vector[3],
                r_d=vector[4],
            ),
            new.compute_score(
                mean_conf=vector[0],
                min_conf=vector[1],
                b_edge=vector[2],
                drop=vector[3],
                r_d=vector[4],
            ),
        )


def test_calibrated_scorer_v40_compatibility_matches_legacy() -> None:
    weights = {
        "v_edge": 0.25,
        "b_edge": 0.50,
        "v_edge_x_b_edge": 0.75,
        "drop": -0.10,
    }
    old = OldCalibratedScorer(OldCalibratedScorerConfig(weights=weights, bias=0.2))
    new = NewCalibratedScorer(NewCalibratedScorerConfig(weights=weights, bias=0.2))

    for v_edge, b_edge, drop in [(0.1, 0.2, 0.3), (1.0, 0.4, 0.0), (2.5, 0.8, 0.6)]:
        _assert_score_dict_matches(
            old.compute_score(b_edge=b_edge, drop=drop, v_edge=v_edge),
            new.compute_score(b_edge=b_edge, drop=drop, v_edge=v_edge),
        )


def test_rule_only_scorer_matches_legacy() -> None:
    old = OldRuleOnlyScorer(a1=0.5, a2=0.3, a3=0.2)
    new = NewRuleOnlyScorer(a1=0.5, a2=0.3, a3=0.2)

    for mean_conf, b_edge, drop in [(1.0, 0.0, 0.0), (0.5, 0.5, 0.5), (0.0, 2.0, 2.0)]:
        assert old.compute_score(mean_conf, b_edge, drop) == new.compute_score(
            mean_conf, b_edge, drop
        )
