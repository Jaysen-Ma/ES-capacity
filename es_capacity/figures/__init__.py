"""Fig 1 pass@k, Fig 2 histogram, Fig 3 coverage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from es_capacity.metrics import (
    accuracy_histogram,
    coverage_contingency,
    paper_style_histogram,
    passk_curve,
    solvable_mask,
    solve_combinations,
    truncate_to_common_n,
)

# Colors chosen to match the Yue et al.-style accuracy histogram (teal/salmon).
_PAPER_COLORS = ["#3d8f86", "#f4978e", "#8c8cf2", "#f2c98e", "#a0c878"]


def plot_passk(
    arms: dict[str, Path],
    out_path: Path,
    *,
    ks: list[int] | None = None,
    title: str = "pass@k on Minerva Math",
) -> dict[str, Any]:
    n_common, loaded = truncate_to_common_n(arms)
    fig, ax = plt.subplots(figsize=(7, 5))
    summary: dict[str, Any] = {"n": n_common, "curves": {}}
    for name, (indices, counts, n) in loaded.items():
        curve = passk_curve(counts, n, ks=ks)
        ks_sorted = sorted(curve)
        ax.plot(ks_sorted, [100 * curve[k] for k in ks_sorted], marker="o", label=name)
        summary["curves"][name] = {str(k): curve[k] for k in ks_sorted}
    ax.set_xscale("log", base=2)
    ax.set_xlabel("k")
    ax.set_ylabel("pass@k (%)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    summary_path = out_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def plot_histogram(
    arms: dict[str, Path],
    out_path: Path,
    *,
    bins: int = 10,
    title: str = "Per-problem accuracy histogram",
) -> dict[str, Any]:
    n_common, loaded = truncate_to_common_n(arms)
    fig, ax = plt.subplots(figsize=(7, 5))
    width = 0.8 / max(len(loaded), 1)
    summary: dict[str, Any] = {"n": n_common, "hists": {}}
    edges = None
    for i, (name, (_, counts, n)) in enumerate(loaded.items()):
        hist = accuracy_histogram(counts, n, bins=bins)
        edges = np.asarray(hist["edges"])
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.bar(
            centers + (i - (len(loaded) - 1) / 2) * width,
            hist["counts"],
            width=width,
            label=name,
            alpha=0.85,
        )
        summary["hists"][name] = hist
    ax.set_xlabel("accuracy (c/n)")
    ax.set_ylabel("# problems")
    ax.set_title(title)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def plot_coverage(
    base_run: Path,
    other_runs: dict[str, Path],
    out_path: Path,
    *,
    threshold: int = 1,
    title: str = "Solvable-set coverage vs base",
) -> dict[str, Any]:
    arms = {"base": base_run, **other_runs}
    n_common, loaded = truncate_to_common_n(arms)
    base_idx, base_counts, n = loaded["base"]
    base_m = solvable_mask(base_counts, threshold=threshold)

    tables = []
    fig, axes = plt.subplots(1, max(len(other_runs), 1), figsize=(5 * max(len(other_runs), 1), 4))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, (name, _) in zip(axes, other_runs.items()):
        _, counts, _ = loaded[name]
        # Align by index
        assert loaded[name][0] == base_idx
        other_m = solvable_mask(counts, threshold=threshold)
        tab = coverage_contingency(base_m, other_m, base_label="base", other_label=name)
        tables.append(tab)
        mat = np.array(
            [
                [tab["both"], tab["other_only"]],
                [tab["base_only"], tab["neither"]],
            ]
        )
        im = ax.imshow(mat, cmap="Blues")
        ax.set_xticks([0, 1], [f"{name}+", f"{name}-"])
        ax.set_yticks([0, 1], ["base+", "base-"])
        for (r, c), v in np.ndenumerate(mat):
            ax.text(c, r, str(v), ha="center", va="center", color="black")
        ax.set_title(f"{name} (other_only={tab['other_only']})")
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f"{title} (n={n_common}, thr={threshold})")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    summary = {"n": n_common, "threshold": threshold, "tables": tables}
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def plot_paper_histogram(
    arms: dict[str, Path],
    out_path: Path,
    *,
    title: str = "Accuracy Histogram",
    colors: list[str] | None = None,
) -> dict[str, Any]:
    """Grouped bar histogram styled after Yue et al. Figure 5: one bar per arm in
    each of the three coarse accuracy bins from `paper_style_histogram`, dual
    Frequency/Percentage y-axes, teal/salmon palette."""
    n_common, loaded = truncate_to_common_n(arms)
    names = list(loaded.keys())
    colors = colors or _PAPER_COLORS

    hists: dict[str, Any] = {}
    labels: list[str] = []
    num_problems = 0
    for name, (_, counts, n) in loaded.items():
        h = paper_style_histogram(counts, n)
        hists[name] = h
        labels = h["labels"]
        num_problems = h["num_problems"]

    x = np.arange(len(labels))
    width = 0.8 / max(len(names), 1)
    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    for i, name in enumerate(names):
        offset = (i - (len(names) - 1) / 2) * width
        ax1.bar(
            x + offset,
            hists[name]["counts"],
            width=width,
            label=name,
            color=colors[i % len(colors)],
            edgecolor="none",
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.set_xlabel("Accuracy Interval")
    ax1.set_ylabel("Frequency")
    ax1.set_title(title)
    ax1.legend(loc="upper right", frameon=True)

    ax2 = ax1.twinx()
    y1_max = ax1.get_ylim()[1]
    ax2.set_ylim(0, 100.0 * y1_max / max(num_problems, 1))
    ax2.set_ylabel("Percentage (%)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    summary = {"n": n_common, "num_problems": num_problems, "hists": hists}
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def plot_solve_table(
    arms: dict[str, Path],
    out_path: Path,
    *,
    threshold: int = 1,
    benchmark_name: str = "Minerva Math",
) -> dict[str, Any]:
    """Booktabs-style solved/unsolved combination table, e.g.:

        Base   GRPO  | Minerva
        ✓      ✓     |  55.5%
        ✓      ✗     |  12.9%
        ✗      ✓     |   3.3%
        ✗      ✗     |  28.3%
    """
    n_common, loaded = truncate_to_common_n(arms)
    names = list(loaded.keys())
    masks = {name: solvable_mask(counts, threshold=threshold) for name, (_, counts, _) in loaded.items()}
    rows = solve_combinations(masks)

    n_names = len(names)
    ncols = n_names + 1
    nrows = len(rows) + 1

    fig, ax = plt.subplots(figsize=(1.7 * ncols + 1.0, 0.55 * nrows + 0.6))
    ax.set_xlim(0, ncols)
    ax.set_ylim(nrows, 0)
    ax.axis("off")

    col_centers = [i + 0.5 for i in range(ncols)]
    for i, name in enumerate(names):
        ax.text(col_centers[i], 0.5, name, ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(col_centers[n_names], 0.5, benchmark_name, ha="center", va="center", fontsize=12, fontweight="bold")

    for r, row in enumerate(rows, start=1):
        for i, name in enumerate(names):
            symbol = "\u2713" if row["combo"][name] else "\u2717"
            ax.text(col_centers[i], r + 0.5, symbol, ha="center", va="center", fontsize=13)
        ax.text(col_centers[n_names], r + 0.5, f"{row['pct']:.1f}%", ha="center", va="center", fontsize=12)

    pad = 0.03
    ax.axhline(0, xmin=pad, xmax=1 - pad, color="black", linewidth=1.8)
    ax.axhline(1, xmin=pad, xmax=1 - pad, color="black", linewidth=0.9)
    ax.axhline(nrows, xmin=pad, xmax=1 - pad, color="black", linewidth=1.8)
    ax.axvline(n_names, ymin=0, ymax=1, color="black", linewidth=0.9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    summary = {"n": n_common, "threshold": threshold, "benchmark_name": benchmark_name, "rows": rows}
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def make_all_figures(
    arms: dict[str, Path],
    out_dir: Path,
    *,
    base_key: str = "base",
    benchmark_name: str = "Minerva Math",
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_passk(arms, out_dir / "fig1_passk.png")
    plot_histogram(arms, out_dir / "fig2_histogram.png")
    others = {k: v for k, v in arms.items() if k != base_key}
    if others and base_key in arms:
        plot_coverage(arms[base_key], others, out_dir / "fig3_coverage.png")
    plot_solve_table(arms, out_dir / "fig4_solve_table.png", benchmark_name=benchmark_name)
    plot_paper_histogram(
        arms, out_dir / "fig5_histogram_paper.png", title=f"{benchmark_name} Accuracy Histogram"
    )
