#!/usr/bin/env python3
"""Build the exact deterministic sampled-genome receipt for corrected PhyloLM."""

from __future__ import annotations

import argparse
from pathlib import Path

from weighttraits.behavior.phylolm import build_sampled_genome, write_genome_receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene-pool", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.gene_pool.resolve() == args.out.resolve():
        parser.error("--out must not overwrite --gene-pool")
    write_genome_receipt(build_sampled_genome(args.gene_pool), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
