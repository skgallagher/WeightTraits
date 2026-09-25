#!/usr/bin/env python3
"""Submit one immutable downstream Slurm plan after fail-closed validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from weighttraits.downstream.submission import submit_downstream_plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    args = parser.parse_args()
    receipt = submit_downstream_plan(
        args.config,
        expected_sha256=args.config_sha256,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
