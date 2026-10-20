"""Pinned MiniLM snapshot identity used by the frozen WeightTraits study.

Hugging Face snapshots normally contain symlinks into the cache's ``blobs``
directory.  The generic all-node artifact digester intentionally rejects
symlinks, so it cannot be used for this one artifact.  This module is a local,
versioned copy of the exact identity algorithm used to create the immutable
``MINILM_ENCODER.json`` receipt:

``weighttraits.behavior.study._encoder_snapshot_sha256`` from the frozen
WeightTraits source whose ``study.py`` SHA-256 is pinned below.

Do not add path-containment rules or dereference the snapshot before walking
it: either change would produce a different identity from the frozen receipt.
Each discovered file contributes its snapshot-relative name, literal symlink
target (when applicable), resolved size, and resolved content SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEME = "weighttraits.behavior.study._encoder_snapshot_sha256.v1"
REFERENCE_IMPLEMENTATION = "weighttraits.behavior.study._encoder_snapshot_sha256"
REFERENCE_STUDY_SOURCE_SHA256 = (
    "0b451d07cd810d39e57e7a6ea0031eeb8f5739b06154ee7f6eb6f3e5e0fc4d19"
)
_HEX = frozenset("0123456789abcdef")


def encoder_snapshot_sha256(path: str | Path) -> str:
    """Return the exact symlink-aware digest used by WeightTraits.

    This function deliberately mirrors the frozen implementation statement
    for statement.  In particular, a Hugging Face file symlink is represented
    by both its literal link target and the bytes of its resolved blob.
    """

    root = Path(path).resolve(strict=True)
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"encoder snapshot is empty: {root}")
    rows = []
    for item in files:
        resolved = item.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError(f"encoder snapshot entry is not a file: {item}")
        rows.append(
            {
                "path": item.relative_to(root).as_posix(),
                "symlink_target": os.readlink(item) if item.is_symlink() else None,
                "size": resolved.stat().st_size,
                "sha256": _sha256_file(resolved),
            }
        )
    return _canonical_json_sha256(rows)


def require_encoder_snapshot_sha256(
    path: str | Path,
    expected_sha256: str,
) -> str:
    """Recompute a pinned snapshot identity and fail closed on any drift."""

    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in _HEX for character in expected_sha256)
    ):
        raise ValueError("expected encoder snapshot SHA-256 must be lowercase hex")
    observed = encoder_snapshot_sha256(path)
    if observed != expected_sha256:
        raise ValueError(
            "encoder snapshot hash drift: "
            f"expected {expected_sha256}, observed {observed}"
        )
    return observed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "REFERENCE_IMPLEMENTATION",
    "REFERENCE_STUDY_SOURCE_SHA256",
    "SCHEME",
    "encoder_snapshot_sha256",
    "require_encoder_snapshot_sha256",
]
