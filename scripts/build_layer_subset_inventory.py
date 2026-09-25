#!/usr/bin/env python3
"""Build a pinned corrected Table 4 inventory from one direct-analysis rollup."""

from __future__ import annotations

import argparse
from pathlib import Path

from weighttraits.paper.layer_subset_table import (
    build_layer_subset_inventory,
    write_layer_subset_inventory_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollup", type=Path, required=True)
    parser.add_argument("--completion-receipt", type=Path, required=True)
    parser.add_argument("--cohort-id", required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    inputs = {
        args.rollup.resolve(),
        args.completion_receipt.resolve(),
        args.base_dir.resolve(),
    }
    if args.out.resolve() in inputs:
        parser.error("--out must not overwrite the rollup, receipt, or base directory")
    payload = build_layer_subset_inventory(
        args.rollup,
        args.completion_receipt,
        cohort_id=args.cohort_id,
        base_dir=args.base_dir,
    )
    write_layer_subset_inventory_json(payload, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
