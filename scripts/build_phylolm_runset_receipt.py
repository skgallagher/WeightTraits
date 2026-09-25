#!/usr/bin/env python3
"""Build the exact common-46 corrected PhyloLM run-set receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from weighttraits.behavior.phylolm import (
    build_phylolm_runset_receipt,
    write_phylolm_runset_receipt,
)


def _pin(path: Path) -> dict[str, str]:
    import hashlib

    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree-specs", type=Path, required=True)
    parser.add_argument("--suite-id", required=True)
    parser.add_argument("--completion-receipt", type=Path, required=True)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    specs = json.loads(args.tree_specs.read_text())
    if not isinstance(specs, dict):
        parser.error("--tree-specs must be a JSON tree-ID-to-pin mapping")
    output = args.out.resolve()
    inputs = {
        args.tree_specs.resolve(),
        args.completion_receipt.resolve(),
        args.stage_manifest.resolve(),
    }
    if output in inputs:
        parser.error("--out must not overwrite an input")
    payload = build_phylolm_runset_receipt(
        specs,
        suite_id=args.suite_id,
        completion_receipt=_pin(args.completion_receipt),
        stage_manifest=_pin(args.stage_manifest),
    )
    write_phylolm_runset_receipt(payload, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
