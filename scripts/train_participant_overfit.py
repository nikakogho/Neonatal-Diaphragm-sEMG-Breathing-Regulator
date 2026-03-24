#!/usr/bin/env python3
"""
Train one pooled participant-level scalar AUX[0] overfit experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training import run_participant_overfit_experiment
from src.training.single_recording import collect_runtime_info, validate_runtime_requirements


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a pooled participant-level EMG -> AUX[0] overfit experiment")
    parser.add_argument("--recordings_dir", required=True)
    parser.add_argument(
        "--model",
        required=True,
        choices=["constant_mean_scalar", "cnn1d_feature_scalar", "cnn1d_raw_scalar", "mlp_feature_scalar"],
    )
    parser.add_argument("--target_channel", type=int, default=0)
    parser.add_argument("--win_ms", type=int, required=True)
    parser.add_argument("--delta_ms", type=int, required=True)
    parser.add_argument("--step_ms", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default=None)
    parser.add_argument("--require_venv", action="store_true")
    parser.add_argument("--require_cuda", action="store_true")
    parser.add_argument("--run_dir", default=None)
    args = parser.parse_args()

    runtime_info = collect_runtime_info(requested_device=args.device)
    validate_runtime_requirements(
        runtime_info,
        require_venv=bool(args.require_venv),
        require_cuda=bool(args.require_cuda),
    )

    result = run_participant_overfit_experiment(
        recordings_dir=args.recordings_dir,
        model_name=args.model,
        win_ms=args.win_ms,
        delta_ms=args.delta_ms,
        step_ms=args.step_ms,
        target_channel=args.target_channel,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        base_channels=args.base_channels,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        device=args.device,
        run_dir=Path(args.run_dir) if args.run_dir else None,
    )
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
