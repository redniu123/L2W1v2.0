from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import Levenshtein

from .scoring import normalize_format

MetricSummaryFn = Callable[[list[dict[str, Any]]], dict[str, Any]]


def select_offline_upgrades(
    scores: Sequence[float], budget: float
) -> tuple[set[int], dict[int, int]]:
    ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    n_upgrades = int(round(len(scores) * budget))
    upgrade_indexes = set(ranked[:n_upgrades])
    rank_map = {index: rank + 1 for rank, index in enumerate(ranked)}
    return upgrade_indexes, rank_map


def _summary_metric(
    metrics: dict[str, Any] | None,
    source_key: str,
) -> Any:
    if metrics is None:
        return None
    return metrics[source_key]


def replay_offline(
    strategy: str,
    budget: float,
    full_rows: Sequence[dict[str, Any]],
    score_map: Sequence[float],
    *,
    prompt_version: str,
    run_id: str = "",
    extended_metrics_fn: MetricSummaryFn | None = None,
    usage_metrics_fn: MetricSummaryFn | None = None,
) -> dict[str, Any]:
    upgrade_indexes, rank_map = select_offline_upgrades(score_map, budget)
    per_sample: list[dict[str, Any]] = []
    cer_num = 0
    gt_len = 0
    n_upgraded = 0
    n_accepted = 0

    for index, item in enumerate(full_rows):
        t_a = item["ocr_text"]
        t_gt = item["gt"]
        selected = index in upgrade_indexes
        final_text = item["final_text_if_upgraded"] if selected else t_a
        if selected:
            n_upgraded += 1
            n_accepted += 1 if final_text != t_a else 0

        cer_num += Levenshtein.distance(normalize_format(final_text), normalize_format(t_gt))
        gt_len += len(normalize_format(t_gt))

        row = dict(item)
        row.update(
            {
                "router_name": strategy,
                "router_score": round(score_map[index], 6),
                "budget": budget,
                "budget_mode": "offline_replay",
                "selected_for_upgrade": selected,
                "replay_rank": rank_map[index],
                "final_text": final_text,
                "vlm_raw_output": item["vlm_raw_output"] if selected else "",
                "latency_ms": item["latency_ms"] if selected else None,
                "token_usage": item["token_usage"] if selected else None,
                "error_type": item["error_type"] if selected else "not_upgraded",
                "backfill_status": "skipped" if selected else "not_upgraded",
                "backfill_reason": "paper1_routeronly" if selected else "not_upgraded",
                "cvr_flag": False,
                "is_correct_final": final_text == t_gt,
                "edit_distance_final": Levenshtein.distance(
                    normalize_format(final_text),
                    normalize_format(t_gt),
                ),
                "prompt_version": prompt_version,
                "run_id": run_id,
            }
        )
        per_sample.append(row)

    extended_metrics = extended_metrics_fn(per_sample) if extended_metrics_fn is not None else None
    usage_metrics = usage_metrics_fn(per_sample) if usage_metrics_fn is not None else None
    actual_call_rate = (n_upgraded / len(full_rows)) if full_rows else 0.0

    return {
        "summary": {
            "run_id": run_id,
            "router_name": strategy,
            "budget": budget,
            "target_call_rate": budget,
            "actual_call_rate": round(actual_call_rate, 4),
            "call_rate_valid": abs(actual_call_rate - budget) <= 0.005,
            "CER": round(cer_num / gt_len, 6) if gt_len else 0.0,
            "BoundaryDeletionRecallAtB": _summary_metric(
                extended_metrics,
                "Boundary_Deletion_Recall@B",
            ),
            "SubstitutionCER": _summary_metric(extended_metrics, "Substitution_CER"),
            "AER": round(n_accepted / n_upgraded, 4) if n_upgraded else 0.0,
            "CVR": 0.0,
            "p95_latency_ms": _summary_metric(usage_metrics, "P95_Latency_MS"),
            "avg_token_usage": _summary_metric(usage_metrics, "Avg_Token_Usage"),
            "agentB_model": full_rows[0]["vlm_model"] if full_rows else "",
            "prompt_version": prompt_version,
            "n_valid": len(full_rows),
        },
        "per_sample": per_sample,
    }
