#!/usr/bin/env python3
"""Build corrected direct-weight Table 2 statistics from pinned run-set inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from weighttraits.paper.direct_weight_table import (
    build_direct_weight_table,
    write_direct_weight_table_csv,
    write_direct_weight_table_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML contract pinning every cohort, rollup, representation, and truth hash",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        help="resolve config paths here instead of relative to the config directory",
    )
    parser.add_argument("--out", type=Path, required=True, help="provenance JSON output")
    parser.add_argument("--csv-out", type=Path, required=True, help="flat Table 2 CSV output")
    args = parser.parse_args()

    output_paths = {args.out.resolve(), args.csv_out.resolve()}
    if len(output_paths) != 2:
        parser.error("--out and --csv-out must be different paths")
    if args.config.resolve() in output_paths:
        parser.error("outputs must not overwrite the input config")

    payload = build_direct_weight_table(args.config, base_dir=args.base_dir)
    input_paths = {args.config.resolve()}
    input_paths.update(
        Path(row[field]).resolve()
        for row in payload["rows"]
        for field in ("cohort_contract", "completion_receipt", "source")
    )
    input_paths.update(
        Path(spec["path"]).resolve() for spec in payload["truth_manifests"].values()
    )
    collisions = sorted(str(path) for path in output_paths & input_paths)
    if collisions:
        parser.error(f"outputs must not overwrite input files: {collisions}")
    write_direct_weight_table_json(payload, args.out)
    write_direct_weight_table_csv(payload["rows"], args.csv_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
