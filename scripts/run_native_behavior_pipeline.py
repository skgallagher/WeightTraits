#!/usr/bin/env python3
"""Build receipt-backed native behavior inputs and terminal study artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from weighttraits.behavior.materialize import audit_prompt_sources, build_prompt_artifacts
from weighttraits.behavior.study import (
    build_behavior_study,
    build_checkpoint_set_receipt,
    build_encoder_receipt,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    prompts = commands.add_parser("prompts")
    prompts.add_argument("--config", type=Path, required=True)
    prompts.add_argument("--out-dir", type=Path, required=True)

    source_audit = commands.add_parser("source-audit")
    source_audit.add_argument("--config", type=Path, required=True)
    source_audit.add_argument("--receipt", type=Path, required=True)
    source_audit.add_argument("--materialization-config", type=Path, required=True)

    checkpoints = commands.add_parser("checkpoints")
    checkpoints.add_argument("--config", type=Path, required=True)
    checkpoints.add_argument("--receipt", type=Path, required=True)

    encoder = commands.add_parser("encoder")
    encoder.add_argument("--config", type=Path, required=True)
    encoder.add_argument("--receipt", type=Path, required=True)

    study = commands.add_parser("study")
    study.add_argument("--config", type=Path, required=True)
    study.add_argument("--out-dir", type=Path, required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    if args.command == "prompts":
        payload = build_prompt_artifacts(args.config, out_dir=args.out_dir)
    elif args.command == "source-audit":
        payload = audit_prompt_sources(
            args.config,
            receipt_path=args.receipt,
            materialization_config_path=args.materialization_config,
        )
    elif args.command == "checkpoints":
        payload = build_checkpoint_set_receipt(args.config, receipt_path=args.receipt)
    elif args.command == "encoder":
        payload = build_encoder_receipt(args.config, receipt_path=args.receipt)
    else:
        payload = build_behavior_study(args.config, out_dir=args.out_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
