#!/usr/bin/env python3
"""
Refresh summary.md for one or more single-recording run directories.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training import refresh_single_recording_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh single-recording summary.md files")
    parser.add_argument("run_dirs", nargs="+")
    args = parser.parse_args()

    for run_dir in args.run_dirs:
        refresh_single_recording_summary(run_dir)


if __name__ == "__main__":
    main()
