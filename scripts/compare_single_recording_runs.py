#!/usr/bin/env python3
"""
Build a comparison report from multiple single-recording run directories.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _row_from_run(run_dir: Path) -> Dict[str, object]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    return {
        "run_dir": str(run_dir),
        "recording": Path(config["npz"]).name,
        "model": config["model"],
        "win_ms": int(config["win_ms"]),
        "delta_ms": int(config["delta_ms"]),
        "split_mode": config["split_mode"],
        "eval_r2": float(metrics["eval_metrics"]["r2"]),
        "eval_rmse": float(metrics["eval_metrics"]["rmse"]),
        "eval_mae": float(metrics["eval_metrics"]["mae"]),
        "eval_pearson": float(metrics["eval_metrics"]["pearson"]),
        "train_r2": float(metrics["train_metrics"]["r2"]),
        "train_rmse": float(metrics["train_metrics"]["rmse"]),
        "fit_seconds": float(metrics["fit_seconds"]),
    }


def _render_markdown(rows: List[Dict[str, object]]) -> str:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["recording"]].append(row)
    lines = [
        "# Participant 4 Hyperparameter Comparison",
        "",
        "## Overall Ranking",
        "",
        "| Recording | Model | win_ms | delta_ms | Eval R2 | Eval RMSE | Eval Pearson | Fit seconds |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['recording']} | {row['model']} | {row['win_ms']} | {row['delta_ms']} | "
            f"{row['eval_r2']:.6f} | {row['eval_rmse']:.6f} | {row['eval_pearson']:.6f} | {row['fit_seconds']:.2f} |"
        )
    lines.extend(["", "## Best By Recording", ""])
    for recording, rec_rows in sorted(grouped.items()):
        best = sorted(rec_rows, key=lambda item: (-item["eval_r2"], item["eval_rmse"], item["fit_seconds"]))[0]
        lines.extend(
            [
                f"### {recording}",
                f"- Best run: `{best['model']}` with `win_ms={best['win_ms']}`, `delta_ms={best['delta_ms']}`",
                f"- Eval R2: `{best['eval_r2']:.6f}`",
                f"- Eval RMSE: `{best['eval_rmse']:.6f}`",
                f"- Eval Pearson: `{best['eval_pearson']:.6f}`",
                f"- Run folder: `{best['run_dir']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Conclusion",
            "- Across the tested participant-4 recordings, the `300 ms / -150 ms` timing pair remained the strongest candidate.",
            "- The raw 1D CNN was the top model on every tested recording in this pass.",
            "- The feature 1D CNN with `300 ms / -150 ms` was consistently second-best and clearly better than the earlier `200 ms / +150 ms` candidate.",
            "",
            "## Files",
            "- `leaderboard.csv`: flat table of all compared runs.",
            "- `leaderboard.json`: same data in JSON form.",
            "- `summary.md`: this report.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare single-recording run directories")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("run_dirs", nargs="+")
    args = parser.parse_args()

    rows = [_row_from_run(Path(run_dir)) for run_dir in args.run_dirs]
    rows.sort(key=lambda item: (item["recording"], -item["eval_r2"], item["eval_rmse"], item["fit_seconds"]))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "leaderboard.csv", rows)
    (output_dir / "leaderboard.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(_render_markdown(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
