#!/usr/bin/env python3
"""Build receipt-backed EOS sensitivity and leaf-depth behavior summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from weighttraits.behavior.eos_depth import (
    build_eos_depth_analysis,
    write_eos_depth_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--per-leaf-csv", required=True, type=Path)
    parser.add_argument("--per-depth-csv", required=True, type=Path)
    parser.add_argument("--sensitivity-csv", required=True, type=Path)
    parser.add_argument("--sensitivity-summary-csv", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace existing output artifacts after all inputs validate.",
    )
    args = parser.parse_args()

    payload = build_eos_depth_analysis(args.config)
    receipt = write_eos_depth_outputs(
        payload,
        json_path=args.out_json,
        per_leaf_csv_path=args.per_leaf_csv,
        per_depth_csv_path=args.per_depth_csv,
        sensitivity_csv_path=args.sensitivity_csv,
        sensitivity_summary_csv_path=args.sensitivity_summary_csv,
        receipt_path=args.receipt,
        overwrite=args.overwrite,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
