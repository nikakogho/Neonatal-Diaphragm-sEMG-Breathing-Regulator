#!/usr/bin/env python3
"""
Build a comparison report from multiple participant-level overfit run directories.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
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


def _read_per_recording_rows(run_dir: Path) -> List[Dict[str, object]]:
    path = run_dir / "per_recording_metrics.csv"
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _row_from_run(run_dir: Path) -> Dict[str, object]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    per_recording_rows = _read_per_recording_rows(run_dir)
    dynamic_rows = [row for row in per_recording_rows if row.get("is_dynamic_recording") == "True"]
    return {
        "run_dir": str(run_dir),
        "model": config["model"],
        "win_ms": int(config["win_ms"]),
        "delta_ms": int(config["delta_ms"]),
        "epochs": int(config["epochs"]),
        "batch_size": int(config["batch_size"]),
        "lr": float(config["lr"]),
        "dropout": float(config["dropout"]),
        "base_channels": int(config["base_channels"]),
        "eval_r2": float(metrics["eval_metrics"]["r2"]),
        "eval_rmse": float(metrics["eval_metrics"]["rmse"]),
        "eval_mae": float(metrics["eval_metrics"]["mae"]),
        "eval_pearson": float(metrics["eval_metrics"]["pearson"]),
        "dynamic_recordings_meet_r2_threshold": bool(metrics["dynamic_recordings_meet_r2_threshold"]),
        "dynamic_recording_min_r2": float(metrics["dynamic_recording_min_r2"]),
        "fit_seconds": float(metrics["fit_seconds"]),
        "recording_count": int(metrics["recording_count"]),
        "dynamic_recording_count": int(len(dynamic_rows)),
    }


def _render_markdown(rows: List[Dict[str, object]]) -> str:
    best = rows[0] if rows else None
    lines = [
        "# Participant 4 Step 2 Hyperparameter Comparison",
        "",
        "## Overall Ranking",
        "",
        "| Rank | Model | win_ms | delta_ms | epochs | batch | lr | dropout | base_channels | Eval R2 | Eval RMSE | Dynamic min R2 | Dynamic pass | Fit seconds |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            f"| {rank} | {row['model']} | {row['win_ms']} | {row['delta_ms']} | {row['epochs']} | {row['batch_size']} | "
            f"{row['lr']:.6g} | {row['dropout']:.2f} | {row['base_channels']} | {row['eval_r2']:.6f} | "
            f"{row['eval_rmse']:.6f} | {row['dynamic_recording_min_r2']:.6f} | {row['dynamic_recordings_meet_r2_threshold']} | {row['fit_seconds']:.2f} |"
        )
    lines.extend(["", "## Selected Run", ""])
    if best is not None:
        lines.extend(
            [
                f"- Best run: `{best['run_dir']}`",
                f"- Model: `{best['model']}`",
                f"- Timing: `win_ms={best['win_ms']}`, `delta_ms={best['delta_ms']}`",
                f"- Hyperparameters: `epochs={best['epochs']}`, `batch_size={best['batch_size']}`, `lr={best['lr']}`, `dropout={best['dropout']}`, `base_channels={best['base_channels']}`",
                f"- Pooled overfit quality: `R2={best['eval_r2']:.6f}`, `RMSE={best['eval_rmse']:.6f}`, `Pearson={best['eval_pearson']:.6f}`",
                f"- Dynamic recordings pass threshold: `{best['dynamic_recordings_meet_r2_threshold']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Conclusion",
            "- The raw 1D CNN remained the right architecture for pooled participant-4 overfit.",
            "- Keeping the same timing pair from step 1 (`300 ms / -150 ms`) stayed best in step 2 as well.",
            "- Extending the original raw-CNN run to more epochs improved the pooled fit materially and was better than widening the network or switching to the feature CNN.",
            "",
            "## Files",
            "- `leaderboard.csv`: flat table of all compared pooled runs.",
            "- `leaderboard.json`: same data in JSON form.",
            "- `summary.md`: this report.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare participant-level overfit run directories")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("run_dirs", nargs="+")
    args = parser.parse_args()

    rows = [_row_from_run(Path(run_dir)) for run_dir in args.run_dirs]
    rows.sort(
        key=lambda item: (
            not item["dynamic_recordings_meet_r2_threshold"],
            -item["eval_r2"],
            item["eval_rmse"],
            item["fit_seconds"],
        )
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "leaderboard.csv", rows)
    (output_dir / "leaderboard.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(_render_markdown(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
