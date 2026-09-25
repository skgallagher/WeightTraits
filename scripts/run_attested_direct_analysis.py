#!/usr/bin/env python3
"""Produce or independently replay one fail-closed direct analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from weighttraits.analysis.direct_attestation import (
    produce_attested_direct_analysis,
    write_direct_analysis_replay_receipt,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    produce = subparsers.add_parser("produce", help="run a fresh attested direct analysis")
    produce.add_argument("--training-summary", type=Path, required=True)
    produce.add_argument("--completion-receipt", type=Path, required=True)
    produce.add_argument("--stage-manifest", type=Path, required=True)
    produce.add_argument("--cohort-id", required=True)
    produce.add_argument("--tree-id", required=True)
    produce.add_argument(
        "--artifact", choices=("model", "merged", "adapter_chain"), required=True
    )
    produce.add_argument("--metric", action="append", required=True)
    produce.add_argument("--out", type=Path, required=True)
    produce.add_argument("--path-base", type=Path, default=Path("."))
    produce.add_argument("--chunk-size", type=int, default=1_000_000)
    produce.add_argument("--eps", type=float, default=1e-3)
    produce.add_argument("--layer")
    produce.add_argument("--aggregate", choices=("mean", "median"), default="mean")

    replay = subparsers.add_parser(
        "replay", help="recompute every metric/layer and write an independent receipt"
    )
    replay.add_argument("--direct-receipt", type=Path, required=True)
    replay.add_argument("--direct-receipt-sha256")
    replay.add_argument("--out", type=Path, required=True)
    replay.add_argument("--rtol", type=float, default=0.0)
    replay.add_argument("--atol", type=float, default=0.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "produce":
        payload = produce_attested_direct_analysis(
            training_summary_path=args.training_summary,
            completion_receipt_path=args.completion_receipt,
            stage_manifest_path=args.stage_manifest,
            cohort_id=args.cohort_id,
            tree_id=args.tree_id,
            out_dir=args.out,
            artifact=args.artifact,
            metrics=args.metric,
            path_base=args.path_base,
            chunk_size=args.chunk_size,
            eps=args.eps,
            layer=args.layer,
            aggregate=args.aggregate,
        )
    else:
        payload = write_direct_analysis_replay_receipt(
            args.direct_receipt,
            args.out,
            expected_direct_sha256=args.direct_receipt_sha256,
            rtol=args.rtol,
            atol=args.atol,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

