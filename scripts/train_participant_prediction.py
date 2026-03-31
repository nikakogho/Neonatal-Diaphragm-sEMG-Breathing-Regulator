#!/usr/bin/env python3
"""
Run participant-level leave-one-recording-out prediction for participant 4.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training import run_participant_prediction_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Run participant held-out recording prediction benchmark")
    parser.add_argument("--recordings_dir", required=True)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min_epochs", type=int, default=40)
    parser.add_argument("--max_epochs", type=int, default=250)
    parser.add_argument("--device", default=None)
    parser.add_argument("--require_venv", action="store_true")
    parser.add_argument("--require_cuda", action="store_true")
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--no_root_readme_update", action="store_true")
    args = parser.parse_args()

    result = run_participant_prediction_stage(
        recordings_dir=args.recordings_dir,
        run_dir=Path(args.run_dir) if args.run_dir else None,
        device=args.device,
        require_venv=bool(args.require_venv),
        require_cuda=bool(args.require_cuda),
        val_fraction=float(args.val_fraction),
        patience=int(args.patience),
        min_epochs=int(args.min_epochs),
        max_epochs=int(args.max_epochs),
        update_root_readme=not bool(args.no_root_readme_update),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
