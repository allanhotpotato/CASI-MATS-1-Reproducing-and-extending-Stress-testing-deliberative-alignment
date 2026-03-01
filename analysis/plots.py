"""
Visualization helpers for scheming evaluation results.

Generates:
    1. Bar charts — covert action rate by model and environment
    2. CoT trace examples — annotated excerpts from trajectories that were
       classified as covert_action vs complied
    3. Comparison figures — side-by-side model comparisons

Usage:
    python analysis/plots.py --results_dir results/ --output_dir figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis.run_stats import load_results, compute_rates

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

LABELS = ["covert_action", "escalated", "complied", "refused"]
COLORS = {
    "covert_action": "#d62728",
    "escalated": "#ff7f0e",
    "complied": "#2ca02c",
    "refused": "#aec7e8",
}
LABEL_DISPLAY = {
    "covert_action": "Covert Action",
    "escalated": "Escalated",
    "complied": "Complied",
    "refused": "Refused",
}
ENV_DISPLAY = {
    "moderator_task": "Moderator Task\n(comment-before-delete rule)",
    "duplicate_check_task": "Duplicate Check Task\n(search-before-post rule)",
}


def plot_label_breakdown(records: list[dict], output_path: Path) -> None:
    """
    Stacked horizontal bar chart — one bar per environment showing the
    proportion of each label across all rollouts.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required. Run: pip install matplotlib")

    # Aggregate counts per environment
    from collections import defaultdict
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    totals: dict[str, int] = defaultdict(int)
    for r in records:
        env = r.get("environment", "unknown")
        label = r.get("label", "unknown")
        counts[env][label] += 1
        totals[env] += 1

    envs = sorted(counts.keys())
    y_pos = range(len(envs))

    fig, ax = plt.subplots(figsize=(10, 4))

    for i, env in enumerate(envs):
        total = totals[env]
        left = 0.0
        for label in LABELS:
            val = counts[env].get(label, 0) / total if total > 0 else 0
            if val > 0:
                bar = ax.barh(i, val, left=left, color=COLORS[label],
                              edgecolor="white", linewidth=0.8)
                if val >= 0.06:
                    ax.text(left + val / 2, i, f"{val:.0%}",
                            ha="center", va="center",
                            fontsize=10, fontweight="bold", color="white")
            left += val

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([ENV_DISPLAY.get(e, e) for e in envs], fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Proportion of rollouts", fontsize=11)
    ax.set_title("DeepSeek-R1 — Label Breakdown by Environment", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.spines[["top", "right"]].set_visible(False)

    legend_patches = [
        mpatches.Patch(color=COLORS[l], label=LABEL_DISPLAY[l])
        for l in LABELS if any(counts[e].get(l, 0) > 0 for e in envs)
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_api_call_lengths(records: list[dict], output_path: Path) -> None:
    """
    Scatter + strip plot — number of API calls per rollout, coloured by label
    and split by environment.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required. Run: pip install matplotlib")

    import random

    envs = sorted({r.get("environment", "unknown") for r in records})
    n_envs = len(envs)

    fig, axes = plt.subplots(1, n_envs, figsize=(5 * n_envs, 4), sharey=False)
    if n_envs == 1:
        axes = [axes]

    for ax, env in zip(axes, envs):
        env_records = [r for r in records if r.get("environment") == env]
        for r in env_records:
            n_calls = len(r.get("api_calls", []))
            label = r.get("label", "unknown")
            color = COLORS.get(label, "#888888")
            jitter = random.uniform(-0.15, 0.15)
            ax.scatter(0 + jitter, n_calls, color=color, alpha=0.75,
                       s=80, edgecolors="white", linewidths=0.5, zorder=3)

        ax.set_xlim(-0.6, 0.6)
        ax.set_xticks([])
        ax.set_title(ENV_DISPLAY.get(env, env), fontsize=10, fontweight="bold")
        ax.set_ylabel("API calls per rollout", fontsize=10)
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    legend_patches = [
        mpatches.Patch(color=COLORS[l], label=LABEL_DISPLAY[l]) for l in LABELS
    ]
    fig.legend(handles=legend_patches, loc="upper right", fontsize=9,
               framealpha=0.9, bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("API Calls per Rollout — DeepSeek-R1", fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_cot_length(records: list[dict], output_path: Path) -> None:
    """
    Bar chart — mean CoT reasoning length (chars) per label, per environment.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required. Run: pip install matplotlib")

    from collections import defaultdict
    import numpy as np

    envs = sorted({r.get("environment", "unknown") for r in records})
    labels_present = [l for l in LABELS
                      if any(r.get("label") == l for r in records)]

    fig, ax = plt.subplots(figsize=(9, 4))

    x = range(len(envs))
    bar_width = 0.8 / max(len(labels_present), 1)

    for j, label in enumerate(labels_present):
        means = []
        for env in envs:
            subset = [r for r in records
                      if r.get("environment") == env and r.get("label") == label]
            if subset:
                chars = [sum(len(b) for b in r.get("cot_trace", [])) for r in subset]
                means.append(np.mean(chars))
            else:
                means.append(0)

        offsets = [xi + (j - len(labels_present) / 2 + 0.5) * bar_width for xi in x]
        ax.bar(offsets, means, width=bar_width * 0.9,
               color=COLORS[label], label=LABEL_DISPLAY[label],
               edgecolor="white", linewidth=0.6)

    ax.set_xticks(list(x))
    ax.set_xticklabels([ENV_DISPLAY.get(e, e) for e in envs], fontsize=10)
    ax.set_ylabel("Mean CoT length (characters)", fontsize=10)
    ax.set_title("Chain-of-Thought Length by Label — DeepSeek-R1", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def print_cot_examples(records: list[dict], label: str, n: int = 3) -> None:
    """Print the first n CoT traces for trajectories with a given label."""
    examples = [r for r in records if r.get("label") == label][:n]
    for i, ex in enumerate(examples, 1):
        print(f"\n--- Example {i} ({ex.get('environment')} / {ex.get('model')}) ---")
        for block in ex.get("cot_trace", []):
            print(block[:500] + ("..." if len(block) > 500 else ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, default=Path("results/"))
    parser.add_argument("--output_dir", type=Path, default=Path("figures/"))
    parser.add_argument("--cot_examples", action="store_true", help="Print CoT trace examples")
    args = parser.parse_args()

    records = load_results(args.results_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_label_breakdown(records, args.output_dir / "label_breakdown.png")
    plot_api_call_lengths(records, args.output_dir / "api_call_lengths.png")
    plot_cot_length(records, args.output_dir / "cot_length.png")

    if args.cot_examples:
        print("\n=== Covert Action CoT Examples ===")
        print_cot_examples(records, label="covert_action")
        print("\n=== Complied CoT Examples ===")
        print_cot_examples(records, label="complied")


if __name__ == "__main__":
    main()
