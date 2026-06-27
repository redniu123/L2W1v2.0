from __future__ import annotations

from scripts.examples import replay_offline_demo


def test_replay_offline_demo_runs_on_synthetic_cache() -> None:
    result = replay_offline_demo.run_demo()

    summary = result["summary"]
    assert summary["run_id"] == "stage9_replay_offline_demo"
    assert summary["router_name"] == "WUR"
    assert summary["actual_call_rate"] == 0.5
    assert summary["CER"] == 0.0

    upgraded = {row["sample_id"] for row in result["per_sample"] if row["selected_for_upgrade"]}
    assert upgraded == {"demo-boundary", "demo-substitution"}
    assert all(row["image_path"].startswith("synthetic/") for row in result["per_sample"])


def test_replay_offline_demo_main_prints_summary(capsys) -> None:
    exit_code = replay_offline_demo.main()

    captured = capsys.readouterr()
    assert exit_code == 0
    assert '"router_name": "WUR"' in captured.out
    assert '"run_id": "stage9_replay_offline_demo"' in captured.out
