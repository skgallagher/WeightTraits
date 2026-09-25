#!/usr/bin/env python3
"""Analyze an existing leaf-only merged-checkpoint ledger with current diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

from weighttraits.analysis.direct import analyze_training_ledger_direct


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--truth-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metric", action="append", default=[])
    parser.add_argument("--chunk-size", type=int, default=4_000_000)
    args = parser.parse_args()

    analyze_training_ledger_direct(
        args.ledger.resolve(),
        truth_manifest=args.truth_manifest.resolve(),
        out_dir=args.out.resolve(),
        artifact="merged",
        metrics=args.metric or ["cosine"],
        path_base=args.repo_root.resolve(),
        chunk_size=args.chunk_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
