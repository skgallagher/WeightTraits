#!/usr/bin/env python3
"""Build corrected Table 10 weight/PhyloLM common-46 statistics."""

from __future__ import annotations

import argparse
from pathlib import Path

from weighttraits.paper.phylolm_table import (
    build_phylolm_table,
    write_phylolm_table_csv,
    write_phylolm_table_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    args = parser.parse_args()

    outputs = {args.out.resolve(), args.csv_out.resolve()}
    if len(outputs) != 2:
        parser.error("--out and --csv-out must be different paths")
    if args.config.resolve() in outputs:
        parser.error("outputs must not overwrite the input config")

    payload = build_phylolm_table(args.config, base_dir=args.base_dir)
    inputs = {args.config.resolve()}
    inputs.update(
        Path(spec["path"]).resolve() for spec in payload["truth_manifests"].values()
    )
    for suite_receipts in payload["input_receipts"].values():
        inputs.update(Path(spec["path"]).resolve() for spec in suite_receipts.values())
    collisions = sorted(str(path) for path in outputs & inputs)
    if collisions:
        parser.error(f"outputs must not overwrite input files: {collisions}")

    write_phylolm_table_json(payload, args.out)
    write_phylolm_table_csv(payload["rows"], args.csv_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
