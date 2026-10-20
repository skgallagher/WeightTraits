#!/usr/bin/env python3
"""Execute one config-pinned downstream job inside its fixed Slurm wrapper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from weighttraits.downstream.jobs import run_downstream_job
from weighttraits.downstream.submission import JOB_KINDS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=sorted(JOB_KINDS), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--code-manifest", type=Path, required=True)
    parser.add_argument("--code-manifest-sha256", required=True)
    args = parser.parse_args()
    payload = run_downstream_job(
        kind=args.kind,
        config_path=args.config,
        expected_config_sha256=args.config_sha256,
        code_root=args.code_root,
        code_manifest_path=args.code_manifest,
        expected_code_manifest_sha256=args.code_manifest_sha256,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
