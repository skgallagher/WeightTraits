#!/usr/bin/env python3
"""Build corrected Table 4 LayerTrace subset statistics from pinned inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from weighttraits.paper.layer_subset_table import (
    build_layer_subset_table,
    write_layer_subset_table_csv,
    write_layer_subset_table_json,
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

    payload = build_layer_subset_table(args.config, base_dir=args.base_dir)
    inputs = {
        args.config.resolve(),
        Path(payload["analysis_inventory"]).resolve(),
        Path(payload["source_rollup"]).resolve(),
        Path(payload["completion_receipt"]).resolve(),
    }
    inputs.add(Path(payload["table2_receipt"]).resolve())
    inputs.update(
        Path(spec["path"]).resolve() for spec in payload["truth_manifests"].values()
    )
    for tree_files in payload["input_files"].values():
        inputs.update(Path(spec["path"]).resolve() for spec in tree_files.values())
    collisions = sorted(str(path) for path in outputs & inputs)
    if collisions:
        parser.error(f"outputs must not overwrite input files: {collisions}")

    write_layer_subset_table_json(payload, args.out)
    write_layer_subset_table_csv(payload["rows"], args.csv_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
