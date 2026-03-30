#!/usr/bin/env python3
"""
Build a comparison report from multiple cohort-level overfit run directories.
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


def _participant_title(participant_ids: List[int]) -> str:
    if not participant_ids:
        return "Combined Cohort"
    if len(participant_ids) == 1:
        return f"Participant {participant_ids[0]}"
    if len(participant_ids) == 2:
        return f"Participants {participant_ids[0]} and {participant_ids[1]}"
    return "Participants " + ", ".join(str(item) for item in participant_ids[:-1]) + f", and {participant_ids[-1]}"


def _row_from_run(run_dir: Path) -> Dict[str, object]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    per_recording_rows = _read_per_recording_rows(run_dir)
    return {
        "run_dir": str(run_dir),
        "participant_ids": [int(item) for item in metrics.get("participant_ids", config.get("participant_ids", []))],
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
        "dynamic_recording_count": int(metrics["dynamic_recording_count"]),
        "dynamic_recordings_meeting_r2_threshold_count": int(metrics["dynamic_recordings_meeting_r2_threshold_count"]),
        "dynamic_recordings_meet_r2_threshold": bool(metrics["dynamic_recordings_meet_r2_threshold"]),
        "dynamic_recording_min_r2": float(metrics["dynamic_recording_min_r2"]),
        "fit_seconds": float(metrics["fit_seconds"]),
        "recording_count": int(metrics["recording_count"]),
        "dynamic_rows_in_csv": int(sum(row.get("is_dynamic_recording") == "True" for row in per_recording_rows)),
    }


def _render_markdown(rows: List[Dict[str, object]]) -> str:
    best = rows[0] if rows else None
    participant_sets = {tuple(row["participant_ids"]) for row in rows}
    if len(participant_sets) == 1:
        participant_ids = list(next(iter(participant_sets)))
        title = f"{_participant_title(participant_ids)} Combined Overfit Comparison"
    else:
        title = "Multi-Participant Combined Overfit Comparison"
    lines = [
        f"# {title}",
        "",
        "## Overall Ranking",
        "",
        "| Rank | Model | win_ms | delta_ms | epochs | batch | lr | dropout | base_channels | Eval R2 | Eval RMSE | Dynamic pass count | Dynamic min R2 | Fit seconds |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            f"| {rank} | {row['model']} | {row['win_ms']} | {row['delta_ms']} | {row['epochs']} | {row['batch_size']} | "
            f"{row['lr']:.6g} | {row['dropout']:.2f} | {row['base_channels']} | {row['eval_r2']:.6f} | "
            f"{row['eval_rmse']:.6f} | {row['dynamic_recordings_meeting_r2_threshold_count']}/{row['dynamic_recording_count']} | "
            f"{row['dynamic_recording_min_r2']:.6f} | {row['fit_seconds']:.2f} |"
        )
    lines.extend(["", "## Selected Run", ""])
    if best is not None:
        lines.extend(
            [
                f"- Best run: `{best['run_dir']}`",
                f"- Participants: `{', '.join(f'P{pid}' for pid in best['participant_ids'])}`",
                f"- Model: `{best['model']}`",
                f"- Timing: `win_ms={best['win_ms']}`, `delta_ms={best['delta_ms']}`",
                f"- Hyperparameters: `epochs={best['epochs']}`, `batch_size={best['batch_size']}`, `lr={best['lr']}`, `dropout={best['dropout']}`, `base_channels={best['base_channels']}`",
                f"- Pooled overfit quality: `R2={best['eval_r2']:.6f}`, `RMSE={best['eval_rmse']:.6f}`, `Pearson={best['eval_pearson']:.6f}`",
                f"- Dynamic recordings at target: `{best['dynamic_recordings_meeting_r2_threshold_count']}` / `{best['dynamic_recording_count']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Conclusion",
            "- The raw 1D CNN should remain the default first architecture unless a comparison row above clearly beats it.",
            "- Runs are ranked by dynamic pass count first, then minimum dynamic-recording R2, then pooled R2, then RMSE.",
            "",
            "## Files",
            "- `leaderboard.csv`: flat table of all compared cohort runs.",
            "- `leaderboard.json`: same data in JSON form.",
            "- `summary.md`: this report.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare cohort-level overfit run directories")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("run_dirs", nargs="+")
    args = parser.parse_args()

    rows = [_row_from_run(Path(run_dir)) for run_dir in args.run_dirs]
    rows.sort(
        key=lambda item: (
            -item["dynamic_recordings_meeting_r2_threshold_count"],
            -item["dynamic_recording_min_r2"],
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
