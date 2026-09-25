#!/usr/bin/env python3
"""Analyze one corrected PhyloLM tree from pinned population receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from weighttraits.behavior.phylolm import analyze_phylolm_tree


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-specs", type=Path, required=True)
    parser.add_argument("--suite-id", required=True)
    parser.add_argument("--tree-id", required=True)
    parser.add_argument("--truth-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    specs = json.loads(args.population_specs.read_text())
    if not isinstance(specs, dict):
        parser.error("--population-specs must be a JSON model-ID-to-pin mapping")
    analyze_phylolm_tree(
        specs,
        suite_id=args.suite_id,
        tree_id=args.tree_id,
        truth_manifest_path=args.truth_manifest,
        output_dir=args.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
