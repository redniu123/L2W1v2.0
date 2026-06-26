from __future__ import annotations

from dataclasses import asdict
from typing import Any

from modules.router.backfill import BackfillConfig as OldBackfillConfig
from modules.router.backfill import RouteType as OldRouteType
from modules.router.backfill import apply_strict_backfill as old_apply_strict_backfill

from l2w1.routing.backfill import BackfillConfig as NewBackfillConfig
from l2w1.routing.backfill import RouteType as NewRouteType
from l2w1.routing.backfill import apply_strict_backfill as new_apply_strict_backfill


def _result_payload(result: Any) -> dict[str, Any]:
    payload = asdict(result)
    payload["rejection_reason"] = result.rejection_reason.value
    return payload


def test_strict_backfill_global_and_formatting_rules_match_legacy() -> None:
    cases = [
        ("ABCDE", "WXYZQ", "ed_exceeded"),
        ("ABCD", "ABCDE", "length_exceeded"),
        ("地层（A）", "地层(A)", "formatting_only"),
        ("ABC", "ADC", "accepted"),
        ("", "abc", "empty_base"),
        ("abc", "", "empty_candidate"),
    ]

    for T_A, T_cand, _case_name in cases:
        old_result = old_apply_strict_backfill(T_A, T_cand, OldRouteType.BOTH)
        new_result = new_apply_strict_backfill(T_A, T_cand, NewRouteType.BOTH)

        assert _result_payload(old_result) == _result_payload(new_result)


def test_strict_backfill_configured_path_rules_match_legacy() -> None:
    old_config = OldBackfillConfig(unified_prompt_mode=False)
    new_config = NewBackfillConfig(unified_prompt_mode=False)

    cases = [
        ("ABCDE", "AXCDE", OldRouteType.BOUNDARY, NewRouteType.BOUNDARY, None, None),
        ("ABCDE", "XBCDE", OldRouteType.BOUNDARY, NewRouteType.BOUNDARY, None, None),
        ("ABCDE", "ABXDE", OldRouteType.AMBIGUITY, NewRouteType.AMBIGUITY, 2, ["X", "Y"]),
        ("ABCDE", "ABXDE", OldRouteType.AMBIGUITY, NewRouteType.AMBIGUITY, 1, ["X", "Y"]),
        ("ABCDE", "ABXDE", OldRouteType.AMBIGUITY, NewRouteType.AMBIGUITY, 2, ["Y", "Z"]),
        ("ABCDE", "ABXYE", OldRouteType.AMBIGUITY, NewRouteType.AMBIGUITY, 2, ["X", "Y"]),
    ]

    for T_A, T_cand, old_route, new_route, idx_susp, top2_chars in cases:
        old_result = old_apply_strict_backfill(
            T_A,
            T_cand,
            old_route,
            idx_susp=idx_susp,
            top2_chars=top2_chars,
            config=old_config,
        )
        new_result = new_apply_strict_backfill(
            T_A,
            T_cand,
            new_route,
            idx_susp=idx_susp,
            top2_chars=top2_chars,
            config=new_config,
        )

        assert _result_payload(old_result) == _result_payload(new_result)
