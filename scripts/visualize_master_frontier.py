#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ v5.1 Task 2: Multi-Model Pareto Frontier Visualization

输入: results/stage2_v51/master_efficiency_frontier.csv
输出:
  results/stage2_v51/fig1_pareto_frontier.pdf
  results/stage2_v51/fig2_scaling_law.pdf

用法:
  python scripts/visualize_master_frontier.py
  python scripts/visualize_master_frontier.py --csv results/stage2_v51/master_efficiency_frontier.csv
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ============================================================
# 全局样式
# ============================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.linewidth": 1.2,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

# 模型元信息：颜色、marker、参数量、显示标签
MODEL_META = {
    "smolvlm":   {"label": "SmolVLM-500M",       "size_b": 0.5,  "color": "#7f7f7f", "marker": "^",  "ls": ":"},
    "minicpm_v": {"label": "MiniCPM-V-2.6-int4", "size_b": 8.0,  "color": "#2ca02c", "marker": "s",  "ls": "-"},
    "llava":     {"label": "LLaVA-1.5-7B",       "size_b": 7.0,  "color": "#ff7f0e", "marker": "D",  "ls": "-"},
    "qwen2.5_vl":{"label": "Qwen2.5-VL-7B",      "size_b": 7.0,  "color": "#1f77b4", "marker": "o",  "ls": "-"},
    "qwen3.5":   {"label": "Qwen3.5-9B",          "size_b": 9.0,  "color": "#9467bd", "marker": "P",  "ls": "-"},
    "gemini":    {"label": "Gemini-3-Flash",       "size_b": 999,  "color": "#d62728", "marker": "*",  "ls": "-"},
    "baseline":  {"label": "Baseline",             "size_b": 0,    "color": "#8c564b", "marker": "x",  "ls": "--"},
}


def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # 百分比转换
    for col in ["Overall_CER", "AER", "CVR", "Actual_Call_Rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ============================================================
# 图 1: Pareto 效率前沿 (CER vs. Actual_Call_Rate)
# ============================================================
def plot_pareto_frontier(df: pd.DataFrame, output_path: str):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # --- 基线 ---
    baseline_df = df[df["Model"] == "baseline"]

    # AgentA_Only 水平线
    agent_a_row = baseline_df[baseline_df["Strategy"] == "AgentA_Only"]
    if not agent_a_row.empty:
        cer_a = agent_a_row["Overall_CER"].values[0]
        ax.axhline(cer_a, color="#333333", lw=1.5, ls=":",
                   label=f"Agent A Only (CER={cer_a:.3%})")

    # Random 基线曲线
    random_df = baseline_df[baseline_df["Strategy"] == "Random"].sort_values("Actual_Call_Rate")
    if not random_df.empty:
        ax.plot(random_df["Actual_Call_Rate"] * 100, random_df["Overall_CER"] * 100,
                color="#aaaaaa", lw=1.5, ls="--", marker="x", ms=7,
                label="Random@B")

    # ConfOnly 基线曲线
    conf_df = baseline_df[baseline_df["Strategy"] == "ConfOnly"].sort_values("Actual_Call_Rate")
    if not conf_df.empty:
        ax.plot(conf_df["Actual_Call_Rate"] * 100, conf_df["Overall_CER"] * 100,
                color="#888888", lw=1.5, ls="-.", marker="v", ms=7,
                label="ConfOnly@B")

    # --- 各 VLM 模型的 SH-DA++ 曲线 ---
    vlm_keys = [k for k in MODEL_META if k != "baseline"]
    for key in vlm_keys:
        meta = MODEL_META[key]
        model_df = df[
            (df["Model"] == key) & (df["Strategy"] == "SH-DA++")
        ].sort_values("Actual_Call_Rate")
        if model_df.empty:
            continue
        ax.plot(
            model_df["Actual_Call_Rate"] * 100,
            model_df["Overall_CER"] * 100,
            color=meta["color"], lw=2.0, ls=meta["ls"],
            marker=meta["marker"], ms=8, markeredgewidth=1.2,
            label=f"SH-DA++ ({meta['label']})",
        )

    ax.set_xlabel("Actual Agent B Call Rate (%)", fontsize=12)
    ax.set_ylabel("Overall CER (%)", fontsize=12)
    ax.set_title("Fig.1  CER vs. Budget: Pareto Efficiency Frontier", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f%%"))
    ax.grid(True)
    ax.legend(loc="upper right", ncol=1, fontsize=8.5)

    # 标注「越低越好」箭头
    ax.annotate("← Lower is Better", xy=(0.02, 0.08),
                xycoords="axes fraction", fontsize=9, color="#555555",
                style="italic")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [Fig1] Saved: {output_path}")


# ============================================================
# 图 2: Scaling Law — AER & CVR vs. Model Size @ B=0.20
# ============================================================
def plot_scaling_law(df: pd.DataFrame, output_path: str, budget: float = 0.20):
    # 筛选 B=0.20 的 SH-DA++ 数据
    target_df = df[
        (df["Strategy"] == "SH-DA++") &
        (df["Target_Budget"].round(2) == round(budget, 2))
    ].copy()

    if target_df.empty:
        print(f"  [Fig2] No data for B={budget}, skipping.")
        return

    # 补充元信息
    target_df["size_b"] = target_df["Model"].map(
        {k: v["size_b"] for k, v in MODEL_META.items()}
    )
    target_df["label"] = target_df["Model"].map(
        {k: v["label"] for k, v in MODEL_META.items()}
    )
    target_df["color"] = target_df["Model"].map(
        {k: v["color"] for k, v in MODEL_META.items()}
    )
    target_df = target_df.dropna(subset=["size_b"]).sort_values("size_b")

    if target_df.empty:
        print("  [Fig2] No valid rows after merge, skipping.")
        return

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    x = np.arange(len(target_df))
    bar_width = 0.45
    x_labels = target_df["label"].tolist()
    colors = target_df["color"].tolist()

    # 左 Y: AER 柱状图（越高越好）
    bars = ax1.bar(x, target_df["AER"] * 100, width=bar_width,
                   color=colors, alpha=0.75, edgecolor="white",
                   linewidth=1.2, label="AER (higher is better)")

    # 右 Y: CVR 折线图（越低越好）
    ax2.plot(x, target_df["CVR"] * 100,
             color="#d62728", lw=2.0, marker="o", ms=8,
             markeredgewidth=1.5, label="CVR (lower is better)")
    ax2.fill_between(x, target_df["CVR"] * 100, alpha=0.12, color="#d62728")

    # 在柱顶标注 AER 数值
    for bar, aer in zip(bars, target_df["AER"]):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5,
                 f"{aer:.1%}", ha="center", va="bottom", fontsize=8.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, rotation=15, ha="right", fontsize=9)
    ax1.set_ylabel("AER — Accepted Edit Rate (%)", fontsize=11, color="#333333")
    ax2.set_ylabel("CVR — Constraint Violation Rate (%)", fontsize=11, color="#d62728")
    ax2.tick_params(axis="y", colors="#d62728")

    ax1.set_title(
        f"Fig.2  Scaling Law @ B={budget:.0%}: AER vs CVR across Models",
        fontsize=13, fontweight="bold"
    )
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    ax1.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [Fig2] Saved: {output_path}")


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="SH-DA++ v5.1 Pareto Visualization")
    parser.add_argument("--csv", default="results/stage2_v51/master_efficiency_frontier.csv")
    parser.add_argument("--output_dir", default="results/stage2_v51")
    parser.add_argument("--scaling_budget", type=float, default=0.20,
                        help="Budget point for Fig2 scaling law (default 0.20)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        print("  Run run_all_frontiers.py first.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {csv_path}")
    df = load_csv(str(csv_path))
    print(f"  Rows: {len(df)}  Models: {df['Model'].unique().tolist()}")

    print("\nGenerating figures...")
    plot_pareto_frontier(df, str(output_dir / "fig1_pareto_frontier.pdf"))
    plot_scaling_law(df, str(output_dir / "fig2_scaling_law.pdf"),
                     budget=args.scaling_budget)

    # 同时输出 PNG 版本方便预览
    plot_pareto_frontier(df, str(output_dir / "fig1_pareto_frontier.png"))
    plot_scaling_law(df, str(output_dir / "fig2_scaling_law.png"),
                     budget=args.scaling_budget)

    print(f"\nAll figures saved to: {output_dir}")
    print("  fig1_pareto_frontier.pdf  — CER vs Budget Pareto curve")
    print("  fig2_scaling_law.pdf      — AER & CVR vs Model Size")


if __name__ == "__main__":
    main()
