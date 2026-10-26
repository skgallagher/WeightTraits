"""Runtime provenance replay for the isolated all-node deployment.

The all-node branch consumes both its own small deployment and the pre-existing
downstream WeightTraits snapshot.  A hash of a manifest file is not sufficient
evidence by itself: every entry named by each manifest is replayed before and
after each production job.  This module is deliberately filesystem-only and
has no scheduler or publication API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
from typing import Any, Mapping

from .source_specs import SOURCE_PATH_AUTHORITY
from .symlink_inventory import load_source_path_authority


PRODUCTION_INTERPRETER = "/home/export/sgallagh/.conda/envs/ellmtrees/bin/python"
DOWNSTREAM_CODE_MANIFEST_BASENAME = "DEPLOYED_CODE_SHA256_SUBMISSION_20260831_V2"
DOWNSTREAM_CODE_MANIFEST_SHA256 = (
    "11a95522b1e046731666d4f678a39d39103961bc0b83ae9fe75cfbd4f1a433cb"
)
ALLNODE_CODE_MANIFEST_BASENAME = "ALLNODE_V1_CODE_MANIFEST.json"
SOURCE_RESOLUTION_BASENAME = "ALLNODE_V1_SOURCE_RESOLUTION.json"
ALLNODE_CODE_MANIFEST_SCHEMA = "weighttraits.all_node_code_manifest.v1"
SOURCE_RESOLUTION_SCHEMA = "weighttraits.all_node_source_resolution.v1"
ATTESTATION_SCHEMA = "weighttraits.all_node_runtime_provenance_attestation.v1"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_MANIFEST_LINE = re.compile(r"([0-9a-f]{64})[ \t]+[* ]?(.+)")
_PHASES = frozenset(
    {"preflight", "launch", "before-wrapper", "after-wrapper", "before-runner", "after-runner"}
)


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"provenance artifact is missing, a symlink, or not a file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def production_runtime() -> dict[str, str]:
    """Return and validate the exact interpreter identity used in production."""

    executable = Path(PRODUCTION_INTERPRETER)
    if (
        sys.executable != PRODUCTION_INTERPRETER
        or not executable.exists()
        or not executable.is_file()
    ):
        raise ValueError(
            "production interpreter drift: "
            f"expected {PRODUCTION_INTERPRETER!r}, got {sys.executable!r}"
        )
    return {
        # Return the observed process identity, not a reconstructed command.
        # The equality check above makes this the exact production path while
        # retaining an explicit sys.executable attestation.
        "interpreter": sys.executable,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }


def verify_downstream_code_manifest(
    *, stage_root: str | Path, manifest_path: str | Path, manifest_sha256: str
) -> tuple[dict[str, str], ...]:
    """Verify the immutable initial downstream manifest and every entry."""

    stage = _canonical_directory(stage_root, "stage_root")
    manifest = _canonical_file(manifest_path, "downstream code manifest")
    expected_path = stage / "runtime_receipts" / DOWNSTREAM_CODE_MANIFEST_BASENAME
    if manifest != expected_path:
        raise ValueError("downstream code manifest path drift")
    if _sha(manifest_sha256, "downstream manifest sha256") != DOWNSTREAM_CODE_MANIFEST_SHA256:
        raise ValueError("downstream code manifest declared SHA-256 drift")
    if sha256_file(manifest) != DOWNSTREAM_CODE_MANIFEST_SHA256:
        raise ValueError("downstream code manifest file SHA-256 drift")

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError("cannot read downstream code manifest") from exc
    for line_number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        match = _MANIFEST_LINE.fullmatch(line)
        if match is None:
            raise ValueError(f"invalid downstream manifest line {line_number}")
        digest, relative_text = match.groups()
        relative = Path(relative_text)
        normalized = relative.as_posix()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or normalized != relative_text
            or normalized in seen
        ):
            raise ValueError(f"unsafe or duplicate downstream manifest path: {relative_text!r}")
        seen.add(normalized)
        target = _canonical_file(stage / relative, f"downstream manifest entry {normalized}")
        _strictly_under(target, stage, f"downstream manifest entry {normalized}")
        if sha256_file(target) != digest:
            raise ValueError(f"downstream manifest entry hash drift: {normalized}")
        rows.append({"path": normalized, "sha256": digest})
    if not rows:
        raise ValueError("downstream code manifest is empty")
    return tuple(rows)


def verify_allnode_code_manifest(
    *, deployment_root: str | Path, manifest_path: str | Path, manifest_sha256: str
) -> tuple[dict[str, Any], ...]:
    """Verify the canonical all-node JSON manifest and every source entry."""

    deployment = _canonical_directory(deployment_root, "deployment_root")
    manifest = _canonical_file(manifest_path, "allnode code manifest")
    expected_path = deployment / "runtime_receipts" / ALLNODE_CODE_MANIFEST_BASENAME
    if manifest != expected_path:
        raise ValueError("allnode code manifest path drift")
    expected_sha = _sha(manifest_sha256, "allnode manifest sha256")
    raw = manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("allnode code manifest file SHA-256 drift")
    payload = _json_object(raw, "allnode code manifest")
    canonical = _canonical_bytes(payload)
    if raw != canonical:
        raise ValueError("allnode code manifest is not canonical JSON bytes")
    required = {
        "schema", "valid", "deployment_root", "n_files", "files",
        "files_sha256", "code_manifest_sha256",
    }
    if set(payload) != required:
        raise ValueError("allnode code manifest keys drift")
    files = payload["files"]
    if (
        payload["schema"] != ALLNODE_CODE_MANIFEST_SCHEMA
        or payload["valid"] is not True
        or payload["deployment_root"] != str(deployment)
        or not isinstance(files, list)
        or payload["n_files"] != len(files)
        or not files
    ):
        raise ValueError("allnode code manifest scope drift")
    core = {key: payload[key] for key in payload if key != "code_manifest_sha256"}
    if payload["code_manifest_sha256"] != _canonical_sha256(core):
        raise ValueError("allnode code manifest self-hash drift")
    if payload["files_sha256"] != _canonical_sha256(files):
        raise ValueError("allnode code manifest file-scope hash drift")
    seen: set[str] = set()
    previous: str | None = None
    for index, row in enumerate(files):
        if not isinstance(row, Mapping) or set(row) != {"path", "size_bytes", "sha256"}:
            raise ValueError(f"allnode code manifest row {index} drift")
        relative_text = row["path"]
        relative = Path(relative_text) if isinstance(relative_text, str) else Path("/")
        if (
            not isinstance(relative_text, str)
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != relative_text
            or relative_text in seen
            or (previous is not None and relative_text <= previous)
        ):
            raise ValueError("allnode code manifest paths must be unique sorted relative paths")
        previous = relative_text
        seen.add(relative_text)
        target = _canonical_file(deployment / relative, f"allnode manifest entry {relative_text}")
        _strictly_under(target, deployment, f"allnode manifest entry {relative_text}")
        if (
            isinstance(row["size_bytes"], bool)
            or not isinstance(row["size_bytes"], int)
            or row["size_bytes"] < 0
            or target.stat().st_size != row["size_bytes"]
            or sha256_file(target) != _sha(row["sha256"], f"allnode entry {relative_text} sha256")
        ):
            raise ValueError(f"allnode manifest entry drift: {relative_text}")
    return tuple(dict(row) for row in files)


def verify_source_resolution(
    *, deployment_root: str | Path, receipt_path: str | Path, receipt_sha256: str,
    expected_scope_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Verify the pinned source-resolution receipt and its canonical scope hash."""

    deployment = _canonical_directory(deployment_root, "deployment_root")
    receipt = _canonical_file(receipt_path, "source resolution receipt")
    expected_path = deployment / "runtime_receipts" / SOURCE_RESOLUTION_BASENAME
    if receipt != expected_path:
        raise ValueError("source resolution receipt path drift")
    raw = receipt.read_bytes()
    if hashlib.sha256(raw).hexdigest() != _sha(receipt_sha256, "source resolution sha256"):
        raise ValueError("source resolution receipt SHA-256 drift")
    payload = _json_object(raw, "source resolution receipt")
    if raw != _canonical_bytes(payload):
        raise ValueError("source resolution receipt is not canonical JSON bytes")
    required = {
        "schema", "valid", "read_only", "scheduler_actions", "loads",
        "source_path_authority", "production_resolution", "source_scope_sha256",
    }
    if set(payload) != required:
        raise ValueError("source resolution receipt keys drift")
    core = {key: payload[key] for key in payload if key != "source_scope_sha256"}
    scope = _canonical_sha256(core)
    if (
        payload["schema"] != SOURCE_RESOLUTION_SCHEMA
        or payload["valid"] is not True
        or payload["read_only"] is not True
        or payload["scheduler_actions"] != 0
        or not isinstance(payload["loads"], list)
        or not payload["loads"]
        or not isinstance(payload["production_resolution"], Mapping)
        or payload["source_scope_sha256"] != scope
    ):
        raise ValueError("source resolution receipt contract drift")
    if expected_scope_sha256 is not None and scope != _sha(
        expected_scope_sha256, "expected source scope sha256"
    ):
        raise ValueError("source resolution scope differs from execution contract")
    authority = load_source_path_authority(SOURCE_PATH_AUTHORITY)
    authority_record = authority.to_dict()
    if payload["source_path_authority"] != authority_record:
        raise ValueError("source resolution authority differs from live pinned authority")
    for index, load in enumerate(payload["loads"]):
        if (
            not isinstance(load, Mapping)
            or load.get("source_path_authority") != authority_record
        ):
            raise ValueError(
                f"source resolution load {index} lacks the exact live authority"
            )
    return payload


def attest_runtime_provenance(
    *, phase: str, stage_root: str | Path, deployment_root: str | Path,
    source_resolution_path: str | Path, source_resolution_sha256: str,
    source_scope_sha256: str, allnode_code_manifest_path: str | Path,
    allnode_code_manifest_sha256: str, downstream_code_manifest_path: str | Path,
    downstream_code_manifest_sha256: str, expected_python_version: str,
) -> dict[str, Any]:
    """Replay every runtime pin and return a compact in-memory attestation."""

    if phase not in _PHASES:
        raise ValueError(f"unsupported runtime provenance phase: {phase!r}")
    runtime = production_runtime()
    if runtime["python_version"] != expected_python_version:
        raise ValueError("production Python version drift")
    stage = _canonical_directory(stage_root, "stage_root")
    deployment = _canonical_directory(deployment_root, "deployment_root")
    if deployment != stage / "deployments" / "allnode_v1":
        raise ValueError("runtime deployment root drift")
    source = verify_source_resolution(
        deployment_root=deployment,
        receipt_path=source_resolution_path,
        receipt_sha256=source_resolution_sha256,
        expected_scope_sha256=source_scope_sha256,
    )
    allnode_rows = verify_allnode_code_manifest(
        deployment_root=deployment,
        manifest_path=allnode_code_manifest_path,
        manifest_sha256=allnode_code_manifest_sha256,
    )
    downstream_rows = verify_downstream_code_manifest(
        stage_root=stage,
        manifest_path=downstream_code_manifest_path,
        manifest_sha256=downstream_code_manifest_sha256,
    )
    return {
        "schema": ATTESTATION_SCHEMA,
        "valid": True,
        "phase": phase,
        **runtime,
        "source_scope_sha256": source["source_scope_sha256"],
        "allnode_code_manifest_sha256": _sha(
            allnode_code_manifest_sha256, "allnode code manifest sha256"
        ),
        "downstream_code_manifest_sha256": DOWNSTREAM_CODE_MANIFEST_SHA256,
        "n_allnode_code_entries": len(allnode_rows),
        "n_downstream_code_entries": len(downstream_rows),
    }


def attest_runtime_environment(phase: str) -> dict[str, Any]:
    """Replay the exact launcher-exported runtime contract."""

    required = {
        "WT_ALLNODE_STAGE_ROOT",
        "WT_ALLNODE_DEPLOYMENT_ROOT",
        "WT_ALLNODE_SOURCE_RESOLUTION",
        "WT_ALLNODE_SOURCE_RESOLUTION_SHA256",
        "WT_ALLNODE_SOURCE_SCOPE_SHA256",
        "WT_ALLNODE_CODE_MANIFEST",
        "WT_ALLNODE_CODE_MANIFEST_SHA256",
        "WT_DOWNSTREAM_CODE_MANIFEST",
        "WT_DOWNSTREAM_CODE_MANIFEST_SHA256",
        "WT_ALLNODE_PRODUCTION_INTERPRETER",
        "WT_ALLNODE_PYTHON_VERSION",
    }
    missing = sorted(key for key in required if key not in os.environ)
    if missing:
        raise ValueError(f"missing runtime provenance environment: {missing}")
    if os.environ["WT_ALLNODE_PRODUCTION_INTERPRETER"] != PRODUCTION_INTERPRETER:
        raise ValueError("launcher production interpreter drift")
    if os.environ["WT_DOWNSTREAM_CODE_MANIFEST_SHA256"] != DOWNSTREAM_CODE_MANIFEST_SHA256:
        raise ValueError("launcher downstream code manifest SHA-256 drift")
    return attest_runtime_provenance(
        phase=phase,
        stage_root=os.environ["WT_ALLNODE_STAGE_ROOT"],
        deployment_root=os.environ["WT_ALLNODE_DEPLOYMENT_ROOT"],
        source_resolution_path=os.environ["WT_ALLNODE_SOURCE_RESOLUTION"],
        source_resolution_sha256=os.environ["WT_ALLNODE_SOURCE_RESOLUTION_SHA256"],
        source_scope_sha256=os.environ["WT_ALLNODE_SOURCE_SCOPE_SHA256"],
        allnode_code_manifest_path=os.environ["WT_ALLNODE_CODE_MANIFEST"],
        allnode_code_manifest_sha256=os.environ["WT_ALLNODE_CODE_MANIFEST_SHA256"],
        downstream_code_manifest_path=os.environ["WT_DOWNSTREAM_CODE_MANIFEST"],
        downstream_code_manifest_sha256=os.environ["WT_DOWNSTREAM_CODE_MANIFEST_SHA256"],
        expected_python_version=os.environ["WT_ALLNODE_PYTHON_VERSION"],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=sorted(_PHASES))
    args = parser.parse_args(argv)
    print(json.dumps(attest_runtime_environment(args.phase), sort_keys=True, separators=(",", ":")))
    return 0


def _canonical_directory(raw: str | Path, label: str) -> Path:
    path = Path(raw)
    if not path.is_absolute() or ".." in path.parts or path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be an existing canonical non-symlink directory")
    if path.resolve(strict=True) != path:
        raise ValueError(f"{label} must be canonical")
    return path


def _canonical_file(raw: str | Path, label: str) -> Path:
    path = Path(raw)
    if not path.is_absolute() or ".." in path.parts or path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be an existing canonical non-symlink file")
    if path.resolve(strict=True) != path:
        raise ValueError(f"{label} must be canonical")
    return path


def _strictly_under(path: Path, parent: Path, label: str) -> None:
    if parent not in path.parents:
        raise ValueError(f"{label} must be strictly inside {parent}")


def _sha(raw: object, label: str) -> str:
    if not isinstance(raw, str) or _SHA256.fullmatch(raw) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")
    return raw


def _json_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "ALLNODE_CODE_MANIFEST_BASENAME",
    "ATTESTATION_SCHEMA",
    "DOWNSTREAM_CODE_MANIFEST_BASENAME",
    "DOWNSTREAM_CODE_MANIFEST_SHA256",
    "PRODUCTION_INTERPRETER",
    "SOURCE_RESOLUTION_BASENAME",
    "attest_runtime_environment",
    "attest_runtime_provenance",
    "production_runtime",
    "sha256_file",
    "verify_allnode_code_manifest",
    "verify_downstream_code_manifest",
    "verify_source_resolution",
]


if __name__ == "__main__":
    raise SystemExit(main())
