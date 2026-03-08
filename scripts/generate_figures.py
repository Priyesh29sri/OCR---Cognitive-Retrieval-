"""
ICDI-X — Paper Figure Generator
=================================
Generates publication-ready figures from evaluation results JSON files.

Figures produced:
  Fig 1 — Ablation bar chart  (BLEU-4, ROUGE-L, F1 per config)
  Fig 2 — MAB convergence     (expected reward per arm over trials)
  Fig 3 — Latency breakdown   (per-stage timing waterfall)
  Fig 4 — Per-category F1     (grouped bar chart by query category)

Usage:
  python scripts/generate_figures.py --ablation scripts/ablation_results.json
                                     --benchmark scripts/benchmark_results.json
                                     --mab /tmp/mab_state.json
                                     --outdir scripts/figures/
"""

import sys, json, argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


STYLE = {
    "figure.dpi":       300,
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  9,
    "axes.spines.top":  False,
    "axes.spines.right":False,
}
plt.rcParams.update(STYLE)

COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]


def fig1_ablation_bars(ablation: dict, outdir: Path):
    configs = [k for k, v in ablation.items() if v]
    metrics = {
        "BLEU-4":    [ablation[c].get("bleu4", 0) for c in configs],
        "ROUGE-L":   [ablation[c].get("rouge_l_f1", 0) for c in configs],
        "Token F1":  [ablation[c].get("avg_f1_token", 0) for c in configs],
        "KW-Recall": [ablation[c].get("avg_keyword_recall", 0) for c in configs],
    }

    x = np.arange(len(configs))
    n_metrics = len(metrics)
    width = 0.18
    offsets = np.linspace(-(n_metrics - 1) / 2 * width, (n_metrics - 1) / 2 * width, n_metrics)

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, (metric_name, values) in enumerate(metrics.items()):
        bars = ax.bar(x + offsets[idx], values, width, label=metric_name,
                      color=COLORS[idx], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values):
            if val > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Fig 1: ICDI-X Ablation Study — Component Contributions")
    ax.set_ylim(0, min(1.1, max(max(v) for v in metrics.values()) * 1.25))
    ax.legend(loc="upper left", framealpha=0.7)
    ax.axhline(0, color="black", linewidth=0.5)

    out = outdir / "fig1_ablation.pdf"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)


def fig2_mab_convergence(mab_state: dict, outdir: Path):
    history = mab_state.get("history", [])
    if not history:
        print("  MAB history empty — skipping convergence plot")
        return

    arms = ["dense", "sparse", "graph", "hierarchical"]
    arm_rewards: dict = {a: [] for a in arms}
    arm_running: dict = {a: [] for a in arms}
    counts:      dict = {a: 0 for a in arms}
    totals:      dict = {a: 0.0 for a in arms}

    for step in history:
        arm = step.get("arm")
        reward = step.get("reward", 0)
        if arm in arm_rewards:
            counts[arm] += 1
            totals[arm] += reward
            arm_running[arm].append(totals[arm] / counts[arm])
            for other in arms:
                if other != arm:
                    arm_running[other].append(
                        arm_running[other][-1] if arm_running[other] else 0.5
                    )

    fig, ax = plt.subplots(figsize=(9, 4))
    for idx, arm in enumerate(arms):
        if arm_running[arm]:
            ax.plot(arm_running[arm], label=arm.capitalize(), color=COLORS[idx], linewidth=1.5)

    ax.set_xlabel("Trial")
    ax.set_ylabel("Running Average Reward")
    ax.set_title("Fig 2: MAB Arm Convergence (Thompson Sampling)")
    ax.legend(loc="lower right", framealpha=0.7)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Random baseline")

    out = outdir / "fig2_mab_convergence.pdf"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)


def fig3_latency_breakdown(benchmark: dict, outdir: Path):
    system_name = list(benchmark.keys())[0]
    r = benchmark[system_name]
    if not r:
        print("  No benchmark data — skipping latency figure")
        return

    labels  = ["Avg", "Median", "P95"]
    values  = [r.get("avg_latency_ms", 0), r.get("median_latency_ms", 0), r.get("p95_latency_ms", 0)]
    colors  = [COLORS[0], COLORS[1], COLORS[2]]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                f"{val:.0f} ms", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Latency (ms)")
    ax.set_title("Fig 3: ICDI-X Query Latency Distribution")
    ax.set_ylim(0, max(values) * 1.3)

    out = outdir / "fig3_latency.pdf"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)


def fig4_per_category(benchmark: dict, outdir: Path):
    system_name = list(benchmark.keys())[0]
    r = benchmark[system_name]
    by_cat = r.get("by_category", {})
    if not by_cat:
        print("  No per-category data — skipping category figure")
        return

    cats   = list(by_cat.keys())
    f1s    = [by_cat[c].get("avg_f1", 0) for c in cats]
    kwrs   = [by_cat[c].get("avg_kwr", 0) for c in cats]
    counts = [by_cat[c].get("n", 0) for c in cats]

    x = np.arange(len(cats))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    b1 = ax.bar(x - width / 2, f1s, width, label="Token F1", color=COLORS[0], edgecolor="white")
    b2 = ax.bar(x + width / 2, kwrs, width, label="KW Recall", color=COLORS[1], edgecolor="white")

    for bar, val in zip(b1, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(b2, kwrs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={counts[i]})" for i, c in enumerate(cats)], fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Fig 4: Per-Category Performance (F1 and KW Recall)")
    ax.set_ylim(0, 1.15)
    ax.legend(framealpha=0.7)

    out = outdir / "fig4_per_category.pdf"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)


def generate_all(ablation_path: str, benchmark_path: str, mab_path: str, outdir: str):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    ablation_data, benchmark_data, mab_data = {}, {}, {}

    if ablation_path and Path(ablation_path).exists():
        with open(ablation_path) as f:
            ablation_data = json.load(f)
        print(f"[ablation]  loaded {ablation_path}")
    else:
        print(f"[ablation]  not found: {ablation_path}")

    if benchmark_path and Path(benchmark_path).exists():
        with open(benchmark_path) as f:
            benchmark_data = json.load(f)
        print(f"[benchmark] loaded {benchmark_path}")
    else:
        print(f"[benchmark] not found: {benchmark_path}")

    if mab_path and Path(mab_path).exists():
        with open(mab_path) as f:
            mab_data = json.load(f)
        print(f"[mab]       loaded {mab_path}")
    else:
        print(f"[mab]       not found: {mab_path}")

    print(f"\nGenerating figures → {out}/")

    if ablation_data:
        fig1_ablation_bars(ablation_data, out)
    if mab_data:
        fig2_mab_convergence(mab_data, out)
    if benchmark_data:
        fig3_latency_breakdown(benchmark_data, out)
        fig4_per_category(benchmark_data, out)

    print(f"\nAll figures saved to {out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ICDI-X Figure Generator")
    parser.add_argument("--ablation",  default="scripts/ablation_results.json")
    parser.add_argument("--benchmark", default="scripts/benchmark_results.json")
    parser.add_argument("--mab",       default="/tmp/mab_state.json")
    parser.add_argument("--outdir",    default="scripts/figures")
    args = parser.parse_args()

    generate_all(args.ablation, args.benchmark, args.mab, args.outdir)
