#!/usr/bin/env python3
"""
plot_correlation — how the two judges' scores relate, drawn from results/scores.csv.

Reads the CSV that export_scores.py writes and nothing else: no scoring, no API
calls, so the charts can be regenerated and checked against the numbers.

Two charts, because they answer different questions:

  correlation.png  Scatter of Jev (x) against the LLM judge (y), one colour per
                   evaluator, with the y = x line. Answers "do they agree, and
                   where do they part company?" Sorting the rows would not
                   change this chart at all: each point carries its own x.

  ranked.png       Cells sorted by Jev score, plotted against rank. Answers "does
                   the LLM judge preserve Jev's ordering?", which is the question
                   sorting is actually good for. It also makes the LLM's
                   quantisation visible: it can only land on 0, .25, .5, .75, 1.

Correlation is reported as Spearman first (the LLM's scores are five discrete
levels, so rank agreement is the meaningful measure) with Pearson alongside.

Usage:
    python plot_correlation.py                  # both charts into results/
    python plot_correlation.py --batch replication
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent / "results"

# Validated categorical palette, assigned in fixed order so an evaluator keeps
# its colour between charts and between runs. See references in the README.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#dedcd6"
SERIES = {
    "helpfulness": "#2a78d6",
    "clarity": "#eb6834",
    "completeness": "#1baf7a",
    "groundedness": "#eda100",
    "path_efficiency": "#e87ba4",
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_pairs(csv_path: Path, batch: str) -> List[Tuple[str, str, float, float]]:
    """(evaluator, trace_id, jev_median, llm_median) for every cell both scored."""
    by_key: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)
    with csv_path.open() as handle:
        for row in csv.DictReader(handle):
            if row["batch"] != batch or row["skipped"] == "true":
                continue
            by_key[(row["evaluator"], row["trace_id"])][row["side"]] = float(row["median"])
    return [
        (evaluator, trace_id, sides["jev"], sides["llm"])
        for (evaluator, trace_id), sides in by_key.items()
        if "jev" in sides and "llm" in sides
    ]


def _ranks(values: Sequence[float]) -> List[float]:
    """Average ranks, so ties share a rank. The LLM column is mostly ties."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    den = (sum(d * d for d in dx) ** 0.5) * (sum(d * d for d in dy) ** 0.5)
    return sum(a * b for a, b in zip(dx, dy)) / den if den else float("nan")


def correlations(pairs) -> Tuple[float, float]:
    xs = [p[2] for p in pairs]
    ys = [p[3] for p in pairs]
    return _pearson(_ranks(xs), _ranks(ys)), _pearson(xs, ys)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9)


def _titles(fig, title: str, subtitle: str) -> None:
    """Title block above the axes, so it cannot collide with the plot."""
    fig.text(0.055, 0.965, title, color=INK, fontsize=13.5, fontweight="bold", va="top")
    fig.text(0.055, 0.917, subtitle, color=INK_SOFT, fontsize=9.5, va="top")


def plot_scatter(pairs, out: Path, subtitle: str) -> None:
    spearman, pearson = correlations(pairs)
    fig, ax = plt.subplots(figsize=(7.6, 7.4), facecolor=SURFACE)
    fig.subplots_adjust(left=0.115, right=0.97, top=0.865, bottom=0.185)
    _style(ax)

    ax.plot([0, 1], [0, 1], color=INK_SOFT, linewidth=1.5, linestyle=(0, (5, 4)), zorder=1)
    ax.annotate(
        "identical scores",
        xy=(0.275, 0.275), color=INK_SOFT, fontsize=8.5,
        rotation=45, rotation_mode="anchor", ha="left", va="bottom", zorder=2,
    )

    for evaluator, color in SERIES.items():
        pts = [p for p in pairs if p[0] == evaluator]
        if not pts:
            continue
        ax.scatter(
            [p[2] for p in pts], [p[3] for p in pts],
            s=110, color=color, edgecolors=SURFACE, linewidths=2,
            label=evaluator, zorder=3,
        )

    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("Jev score", color=INK, fontsize=10.5, labelpad=8)
    ax.set_ylabel("LLM judge score (gpt-4o-mini)", color=INK, fontsize=10.5, labelpad=8)
    _titles(fig, "Points below the line are where Jev scores an agent lower", subtitle)

    ax.text(
        0.035, 0.965,
        f"Spearman  {spearman:.2f}\nPearson    {pearson:.2f}\nn = {len(pairs)}",
        transform=ax.transAxes, color=INK_SOFT, fontsize=9.5, va="top",
        family="monospace",
        bbox=dict(facecolor=SURFACE, edgecolor=GRID, boxstyle="round,pad=0.55"),
        zorder=5,
    )
    ax.legend(
        frameon=False, fontsize=9.5, labelcolor=INK_SOFT, ncol=3,
        loc="upper center", bbox_to_anchor=(0.5, -0.105),
        handletextpad=0.4, columnspacing=1.6,
    )

    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def plot_ranked(pairs, out: Path, subtitle: str) -> None:
    ordered = sorted(pairs, key=lambda p: p[2])
    xs = list(range(1, len(ordered) + 1))

    fig, ax = plt.subplots(figsize=(9.6, 5.8), facecolor=SURFACE)
    fig.subplots_adjust(left=0.075, right=0.975, top=0.835, bottom=0.215)
    _style(ax)

    ax.plot(xs, [p[2] for p in ordered], color=INK, linewidth=2, zorder=3, label="Jev (sorted)")
    for evaluator, color in SERIES.items():
        idx = [i for i, p in zip(xs, ordered) if p[0] == evaluator]
        if not idx:
            continue
        ax.scatter(
            idx, [ordered[i - 1][3] for i in idx],
            s=95, color=color, edgecolors=SURFACE, linewidths=2, zorder=4,
            label=f"LLM, {evaluator}",
        )

    ax.set_xlim(0.4, len(ordered) + 0.6)
    ax.set_ylim(-0.08, 1.1)
    ax.set_xlabel("cell, ordered by Jev score (low to high)", color=INK, fontsize=10.5, labelpad=8)
    ax.set_ylabel("score", color=INK, fontsize=10.5, labelpad=8)
    _titles(
        fig,
        "The LLM judge lands on five values only, and tracks Jev's ordering loosely",
        subtitle,
    )
    ax.legend(
        frameon=False, fontsize=9, labelcolor=INK_SOFT, ncol=3,
        loc="upper center", bbox_to_anchor=(0.5, -0.125),
        handletextpad=0.4, columnspacing=1.6,
    )

    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    parser.add_argument("--csv", type=Path, default=RESULTS_DIR / "scores.csv")
    parser.add_argument("--batch", default="primary", choices=["primary", "replication"])
    args = parser.parse_args(argv)

    pairs = load_pairs(args.csv, args.batch)
    if not pairs:
        parser.error(f"no paired scores in {args.csv} for batch {args.batch}")

    spearman, pearson = correlations(pairs)
    subtitle = (
        f"{len(pairs)} cells, {args.batch} batch: 5 evaluators over 5 traces, "
        "median of 3 runs each"
    )
    suffix = "" if args.batch == "primary" else f"-{args.batch}"
    plot_scatter(pairs, RESULTS_DIR / f"correlation{suffix}.png", subtitle)
    plot_ranked(pairs, RESULTS_DIR / f"ranked{suffix}.png", subtitle)
    print(f"spearman {spearman:.3f}  pearson {pearson:.3f}  n {len(pairs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
