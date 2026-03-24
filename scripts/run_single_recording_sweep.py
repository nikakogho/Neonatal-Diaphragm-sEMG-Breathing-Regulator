#!/usr/bin/env python3
"""
Run a timing/model sweep for one scalar AUX[0] recording experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training import run_single_recording_sweep


def _parse_int_list(value: str):
    return [int(item) for item in value.split(",") if item]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scalar single-recording sweep")
    parser.add_argument("--npz", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument(
        "--models",
        default="constant_mean_scalar,cnn1d_feature_scalar,cnn1d_raw_scalar",
        help="Comma-separated model list",
    )
    parser.add_argument("--win_values", default="100,200,300")
    parser.add_argument("--delta_values", default="-150,-100,-50,0,50,100,150")
    parser.add_argument("--step_ms", type=int, default=50)
    parser.add_argument("--split_mode", choices=["same_windows", "chronological_holdout"], default="same_windows")
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--target_channel", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs_override", type=int, default=None)
    parser.add_argument("--batch_size_override", type=int, default=None)
    args = parser.parse_args()

    result = run_single_recording_sweep(
        npz_path=args.npz,
        output_root=args.output_root,
        models=[item for item in args.models.split(",") if item],
        win_values=_parse_int_list(args.win_values),
        delta_values=_parse_int_list(args.delta_values),
        step_ms=args.step_ms,
        split_mode=args.split_mode,
        train_fraction=args.train_fraction,
        target_channel=args.target_channel,
        seed=args.seed,
        device=args.device,
        epochs_override=args.epochs_override,
        batch_size_override=args.batch_size_override,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
