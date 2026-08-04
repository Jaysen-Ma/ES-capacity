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
    passk_curve,
    solvable_mask,
    truncate_to_common_n,
)


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


def make_all_figures(
    arms: dict[str, Path],
    out_dir: Path,
    *,
    base_key: str = "base",
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_passk(arms, out_dir / "fig1_passk.png")
    plot_histogram(arms, out_dir / "fig2_histogram.png")
    others = {k: v for k, v in arms.items() if k != base_key}
    if others and base_key in arms:
        plot_coverage(arms[base_key], others, out_dir / "fig3_coverage.png")
