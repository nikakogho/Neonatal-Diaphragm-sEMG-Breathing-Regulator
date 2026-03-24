#!/usr/bin/env python3
"""
Print GPU and torch CUDA availability information.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.single_recording import collect_runtime_info


def main() -> None:
    payload = collect_runtime_info()
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
