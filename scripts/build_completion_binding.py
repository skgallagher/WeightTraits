#!/usr/bin/env python3
"""Bind an immutable strict completion receipt to one exact downstream cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from weighttraits.paper.analysis_contracts import validate_strict_training_completion


SCHEMA = "weighttraits.training_completion_binding.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--cohort-id", required=True)
    parser.add_argument("--artifact-factor", type=int, choices=(2, 4), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve(strict=True)
    output = args.out.resolve()
    if output == source:
        parser.error("--out must not overwrite --source")
    if output.exists():
        parser.error(f"refusing to overwrite existing binding: {output}")
    if len(args.source_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in args.source_sha256
    ):
        parser.error("--source-sha256 must be a lowercase SHA256")
    observed = _sha256(source)
    if observed != args.source_sha256:
        parser.error(
            f"source receipt SHA256 mismatch: expected {args.source_sha256}, observed {observed}"
        )
    if not args.cohort_id or args.cohort_id != args.cohort_id.strip():
        parser.error("--cohort-id must be nonempty without surrounding whitespace")

    payload = json.loads(source.read_text(encoding="utf-8"))
    validate_strict_training_completion(
        payload,
        artifact_factor=args.artifact_factor,
    )
    bound = {
        **payload,
        "schema": SCHEMA,
        "cohort_id": args.cohort_id,
        "artifact_factor": args.artifact_factor,
        "source_completion_receipt": {
            "path": str(source),
            "sha256": observed,
        },
    }
    validate_strict_training_completion(bound, cohort_id=args.cohort_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(bound, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"valid": True, "path": str(output), "sha256": _sha256(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
