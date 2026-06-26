from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import Levenshtein

from l2w1.routing.budget import BudgetControllerConfig, OnlineBudgetController

from .scoring import router_score

MetricSummaryFn = Callable[[list[dict[str, Any]]], dict[str, Any]]


def _summary_metric(
    metrics: dict[str, Any] | None,
    source_key: str,
) -> Any:
    if metrics is None:
        return None
    return metrics[source_key]


def replay_online(
    strategy: str,
    target_budget: float,
    all_results: Sequence[dict[str, Any]],
    cached_lookup: Mapping[str, dict[str, Any]],
    budget_cfg: BudgetControllerConfig,
    *,
    run_id: str = "",
    prompt_version: str = "prompt_v1.1",
    agent_b_label: str = "Gemini 3 Flash Preview",
    eta: float = 0.5,
    extended_metrics_fn: MetricSummaryFn | None = None,
    usage_metrics_fn: MetricSummaryFn | None = None,
) -> dict[str, Any]:
    ctrl = OnlineBudgetController(budget_cfg)
    per_sample: list[dict[str, Any]] = []
    validation_logs: list[dict[str, Any]] = []
    cer_num = 0
    gt_len = 0
    n_upgraded = 0
    n_accepted = 0
    missing_cached_rows = 0

    for row in all_results:
        t_a = row["T_A"]
        t_gt = row["T_GT"]
        q = float(router_score(strategy, row, eta=eta))
        upgrade, budget_details = ctrl.step(q)
        cached = cached_lookup.get(row.get("sample_id", ""))
        vlm_raw_output = ""
        latency_ms = None
        token_usage = None
        error_type = "not_upgraded"
        final_text = t_a

        if upgrade:
            n_upgraded += 1
            if cached is None:
                missing_cached_rows += 1
                error_type = "cached_result_missing"
            else:
                vlm_raw_output = cached.get(
                    "vlm_raw_output",
                    cached.get("final_text_if_upgraded", t_a),
                )
                final_text = cached.get("final_text_if_upgraded", t_a) or t_a
                latency_ms = cached.get("latency_ms")
                token_usage = cached.get("token_usage")
                error_type = cached.get("error_type", "none")
                if final_text != t_a:
                    n_accepted += 1

        cer_num += Levenshtein.distance(final_text, t_gt)
        gt_len += len(t_gt)
        item = {
            "sample_id": row.get("sample_id", ""),
            "image_path": row.get("image_path", row.get("img_path", "")),
            "source_image_id": row.get("source_image_id", ""),
            "domain": row.get("domain", "geology"),
            "split": row.get("split", "test"),
            "gt": t_gt,
            "ocr_text": t_a,
            "router_name": strategy,
            "router_score": round(q, 6),
            "budget": target_budget,
            "budget_mode": "online_control_reuse_cache",
            "selected_for_upgrade": upgrade,
            "vlm_model": agent_b_label,
            "prompt_version": prompt_version,
            "vlm_raw_output": vlm_raw_output,
            "latency_ms": latency_ms,
            "token_usage": token_usage,
            "error_type": error_type,
            "has_professional_terms": row.get("has_professional_terms", False),
            "professional_terms": row.get("professional_terms", []),
            "domain_risk_score": round(float(row.get("r_d", 0.0)), 6),
            "cvr_flag": False,
            "replay_rank": None,
            "final_text_if_upgraded": vlm_raw_output if upgrade else "",
            "final_text": final_text,
            "backfill_status": "skipped" if upgrade else "not_upgraded",
            "backfill_reason": "paper1_routeronly_reuse_cache" if upgrade else "not_upgraded",
            "is_correct_ocr": t_a == t_gt,
            "is_correct_final": final_text == t_gt,
            "edit_distance_ocr": Levenshtein.distance(t_a, t_gt),
            "edit_distance_final": Levenshtein.distance(final_text, t_gt),
            "run_id": run_id,
        }
        per_sample.append(item)
        validation_logs.append(
            {
                "sample_id": item["sample_id"],
                "router_name": strategy,
                "target_budget": target_budget,
                "router_score": q,
                "selected_for_upgrade": upgrade,
                "budget_details": budget_details,
                "cached_result_found": cached is not None,
                "run_id": run_id,
            }
        )

    extended_metrics = extended_metrics_fn(per_sample) if extended_metrics_fn is not None else None
    usage_metrics = usage_metrics_fn(per_sample) if usage_metrics_fn is not None else None
    actual_call_rate = (n_upgraded / len(all_results)) if all_results else 0.0

    return {
        "summary": {
            "run_id": run_id,
            "router_name": strategy,
            "budget": target_budget,
            "target_call_rate": target_budget,
            "actual_call_rate": round(actual_call_rate, 4),
            "call_rate_valid": abs(actual_call_rate - target_budget) <= 0.005,
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
            "agentB_model": agent_b_label,
            "prompt_version": prompt_version,
            "n_valid": len(all_results),
            "missing_cached_rows": missing_cached_rows,
        },
        "per_sample": per_sample,
        "validation_logs": validation_logs,
    }
