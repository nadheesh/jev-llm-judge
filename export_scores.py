#!/usr/bin/env python3
"""
export_scores — flatten the run results into a tidy CSV for analysis and plots.

The plotting script reads this, not the JSON, so the numbers behind every chart
are inspectable in a spreadsheet and the chart code has no scoring logic in it.

One row per (side, trace, evaluator). Skipped cells are written with empty
score columns rather than dropped, so the CSV shows what was not scored.

Usage:
    python export_scores.py                     # results/*.json -> results/scores.csv
    python export_scores.py --out /tmp/x.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Optional, Sequence

import judge_common as jc

RESULTS_DIR = Path(__file__).parent / "results"
BATCHES = {
    "primary": RESULTS_DIR / "three-run-scores.json",
    "replication": RESULTS_DIR / "three-run-scores-replication.json",
}

FIELDS = [
    "batch",
    "side",
    "trace_id",
    "trace_note",
    "evaluator",
    "run_1",
    "run_2",
    "run_3",
    "median",
    "spread",
    "skipped",
]


def rows():
    for batch, path in BATCHES.items():
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for key, runs in data.items():
            side, trace_id, evaluator = key.split("|")
            skipped = runs[0] is None
            yield {
                "batch": batch,
                "side": side,
                "trace_id": trace_id,
                "trace_note": jc.INTERESTING_TRACES.get(trace_id, ""),
                "evaluator": evaluator,
                "run_1": "" if skipped else f"{runs[0]:.4f}",
                "run_2": "" if skipped else f"{runs[1]:.4f}",
                "run_3": "" if skipped else f"{runs[2]:.4f}",
                "median": "" if skipped else f"{statistics.median(runs):.4f}",
                "spread": "" if skipped else f"{max(runs) - min(runs):.4f}",
                "skipped": "true" if skipped else "false",
            }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "scores.csv")
    args = parser.parse_args(argv)

    all_rows = sorted(rows(), key=lambda r: (r["batch"], r["side"], r["evaluator"], r["trace_id"]))
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    scored = sum(1 for r in all_rows if r["skipped"] == "false")
    print(f"wrote {args.out} ({len(all_rows)} rows, {scored} scored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
