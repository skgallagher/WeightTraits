"""Fast, versioned identity attestations for frozen checkpoint directories.

The production training stage contains two intentional top-level Llama
symlinks whose checkpoint bytes live outside the frozen stage. Downstream
bindings must never preserve those symlinked paths as executable inputs. This
module distinguishes the frozen ``logical_path`` from ``path``, the canonical
physical checkpoint directory loaded at runtime.

An external physical checkpoint is valid only when an exact, live-attested
``SourcePathAuthority`` proves the logical-prefix to physical-root mapping.
Every identity binds the complete authority (stage-manifest pin, SYMLINKS.txt
pin, exact mapping set, and authority digest) plus the selected mapping.

Large files use a recorded deterministic sampled identity: small files are
hashed in full and large files at five fixed 1 MiB windows. This is an
*identity* digest, not a whole-content digest.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

from .checkpoints import ArtifactPin
from .symlink_inventory import (
    FrozenSymlinkInventory,
    SourcePathAuthority,
    SourcePathAuthoritySpec,
    SourcePathMapping,
    load_source_path_authority,
)


SCHEMA = "weighttraits.all_node_checkpoint_identity.v2"
INDEX_SCHEMA = "weighttraits.all_node_checkpoint_identity_index.v2"
SCHEME = "frozen_fs_full_small_fixed5x1m_large_v1"
FULL_HASH_LIMIT_BYTES = 16 * 1024 * 1024
SAMPLE_BYTES = 1024 * 1024
SAMPLE_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CheckpointIdentity:
    """One canonical physical checkpoint and its frozen logical authority."""

    path: str
    logical_path: str
    logical_prefix: str
    physical_root: str
    frozen_stage_root: str
    stage_manifest_sha256: str
    source_authority: Mapping[str, Any] | None
    source_mapping: Mapping[str, str] | None
    n_files: int
    total_bytes: int
    files: tuple[Mapping[str, Any], ...]
    files_sha256: str
    identity_sha256: str

    @property
    def resolved_path(self) -> str:
        """Compatibility alias: v2 ``path`` is already the physical path."""

        return self.path

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "valid": True,
            "scheme": SCHEME,
            "full_hash_limit_bytes": FULL_HASH_LIMIT_BYTES,
            "sample_bytes": SAMPLE_BYTES,
            "sample_fractions": list(SAMPLE_FRACTIONS),
            "path": self.path,
            "logical_path": self.logical_path,
            "logical_prefix": self.logical_prefix,
            "physical_root": self.physical_root,
            "frozen_stage_root": self.frozen_stage_root,
            "stage_manifest_sha256": self.stage_manifest_sha256,
            "source_authority": (
                None if self.source_authority is None else _deep_dict(self.source_authority)
            ),
            "source_mapping": (
                None if self.source_mapping is None else dict(self.source_mapping)
            ),
            "n_files": self.n_files,
            "total_bytes": self.total_bytes,
            "files": [dict(row) for row in self.files],
            "files_sha256": self.files_sha256,
            "identity_sha256": self.identity_sha256,
        }


def build_checkpoint_identity(
    path: str | Path,
    *,
    frozen_stage_root: str | Path,
    stage_manifest_sha256: str,
    source_authority: SourcePathAuthority | None = None,
    logical_path: str | Path | None = None,
    symlink_inventory: FrozenSymlinkInventory | None = None,
) -> CheckpointIdentity:
    """Inspect one canonical physical checkpoint directory.

    ``path`` is always the physical directory. External physical paths require
    both ``logical_path`` and a live ``source_authority``. The legacy
    ``symlink_inventory`` keyword remains solely to fail with an explicit
    migration error rather than silently trusting the weaker boundary.
    """

    if symlink_inventory is not None:
        raise ValueError(
            "legacy symlink_inventory cannot authorize checkpoint identity v2; "
            "use source_authority with physical path and logical_path"
        )
    root = _canonical_root(frozen_stage_root)
    stage_sha = _sha256(stage_manifest_sha256, "stage_manifest_sha256")
    physical = _canonical_checkpoint_directory(path)
    logical = (
        physical
        if logical_path is None
        else Path(_absolute_normalized(logical_path, "logical checkpoint path"))
    )
    if not _strictly_below(str(logical), str(root)):
        raise ValueError("logical checkpoint path must be strictly below frozen_stage_root")

    authority_record: dict[str, Any] | None = None
    mapping: SourcePathMapping | None = None
    if source_authority is not None:
        authority_record = _authority_record(
            source_authority,
            root=root,
            stage_manifest_sha256=stage_sha,
        )
        mapping = source_authority.authorize(str(logical), str(physical))

    physical_inside_stage = _strictly_below(str(physical), str(root))
    if physical_inside_stage:
        if physical != logical or mapping is not None:
            raise ValueError(
                "physical checkpoint beneath frozen stage must use the identical "
                "non-symlink logical path"
            )
        logical_prefix = str(root)
        physical_root = str(root)
    else:
        if source_authority is None or mapping is None:
            raise ValueError(
                "external physical checkpoint requires exact source-path authority"
            )
        if not _strictly_below(str(logical), mapping.logical_prefix):
            raise ValueError("external checkpoint must be below its logical prefix")
        if not _strictly_below(str(physical), mapping.physical_root):
            raise ValueError("external checkpoint must be below its authorized physical root")
        logical_prefix = mapping.logical_prefix
        physical_root = mapping.physical_root

    rows = _checkpoint_file_rows(physical)
    files_sha = canonical_sha256(rows)
    source_mapping = None if mapping is None else mapping.to_dict()
    core = {
        "scheme": SCHEME,
        "path": str(physical),
        "logical_path": str(logical),
        "logical_prefix": logical_prefix,
        "physical_root": physical_root,
        "frozen_stage_root": str(root),
        "stage_manifest_sha256": stage_sha,
        "source_authority": authority_record,
        "source_mapping": source_mapping,
        "n_files": len(rows),
        "total_bytes": sum(row["size_bytes"] for row in rows),
        "files_sha256": files_sha,
    }
    return CheckpointIdentity(
        path=str(physical),
        logical_path=str(logical),
        logical_prefix=logical_prefix,
        physical_root=physical_root,
        frozen_stage_root=str(root),
        stage_manifest_sha256=stage_sha,
        source_authority=authority_record,
        source_mapping=source_mapping,
        n_files=len(rows),
        total_bytes=core["total_bytes"],
        files=tuple(rows),
        files_sha256=files_sha,
        identity_sha256=canonical_sha256(core),
    )


def validate_checkpoint_identity(
    payload: CheckpointIdentity | Mapping[str, Any],
    *,
    reattest_filesystem: bool = False,
    source_authority: SourcePathAuthority | None = None,
    frozen_stage_root: str | Path | None = None,
    stage_manifest_sha256: str | None = None,
) -> CheckpointIdentity:
    """Validate a serialized identity and optionally repeat live attestation.

    With filesystem reattestation and no supplied authority, the exact
    authority embedded in the identity is reloaded. A many-identity caller
    should reload once with :func:`source_authority_from_identity` and pass it.
    """

    raw = payload.to_dict() if isinstance(payload, CheckpointIdentity) else payload
    if not isinstance(raw, Mapping):
        raise ValueError("checkpoint identity must be a mapping")
    expected_keys = {
        "schema", "valid", "scheme", "full_hash_limit_bytes", "sample_bytes",
        "sample_fractions", "path", "logical_path", "logical_prefix",
        "physical_root", "frozen_stage_root", "stage_manifest_sha256",
        "source_authority", "source_mapping", "n_files", "total_bytes",
        "files", "files_sha256", "identity_sha256",
    }
    if set(raw) != expected_keys:
        raise ValueError("checkpoint identity keys drift")
    if (
        raw["schema"] != SCHEMA
        or raw["valid"] is not True
        or raw["scheme"] != SCHEME
        or raw["full_hash_limit_bytes"] != FULL_HASH_LIMIT_BYTES
        or raw["sample_bytes"] != SAMPLE_BYTES
        or raw["sample_fractions"] != list(SAMPLE_FRACTIONS)
    ):
        raise ValueError("checkpoint identity control fields drift")

    root = _canonical_root(raw["frozen_stage_root"])
    if frozen_stage_root is not None and root != _canonical_root(frozen_stage_root):
        raise ValueError("checkpoint identity frozen_stage_root differs from caller pin")
    stage_sha = _sha256(raw["stage_manifest_sha256"], "stage_manifest_sha256")
    if (
        stage_manifest_sha256 is not None
        and stage_sha
        != _sha256(stage_manifest_sha256, "caller stage_manifest_sha256")
    ):
        raise ValueError("checkpoint identity stage manifest differs from caller pin")

    physical = _absolute_normalized(raw["path"], "physical checkpoint path")
    logical = _absolute_normalized(raw["logical_path"], "logical checkpoint path")
    logical_prefix = _absolute_normalized(raw["logical_prefix"], "logical prefix")
    physical_root = _absolute_normalized(raw["physical_root"], "physical root")
    if not _strictly_below(logical, str(root)):
        raise ValueError("logical checkpoint path is outside frozen stage")

    authority_record = _normalize_authority_record(
        raw["source_authority"],
        root=root,
        stage_manifest_sha256=stage_sha,
    )
    mapping = _normalize_source_mapping(raw["source_mapping"])
    if mapping is None:
        if (
            physical != logical
            or logical_prefix != str(root)
            or physical_root != str(root)
            or not _strictly_below(physical, str(root))
        ):
            raise ValueError("internal checkpoint source-boundary fields drift")
    else:
        if authority_record is None:
            raise ValueError("external checkpoint lacks source authority")
        if mapping.to_dict() not in authority_record["mappings"]:
            raise ValueError("selected source mapping is absent from authority")
        if logical_prefix != mapping.logical_prefix or physical_root != mapping.physical_root:
            raise ValueError("checkpoint source mapping boundary drift")
        if not _strictly_below(logical, logical_prefix):
            raise ValueError("logical checkpoint is outside selected source prefix")
        if not _strictly_below(physical, physical_root):
            raise ValueError("physical checkpoint is outside selected physical root")
        if _strictly_below(physical, str(root)) or physical == str(root):
            raise ValueError("mapped physical checkpoint must be outside frozen stage")
        suffix = os.path.relpath(logical, logical_prefix)
        if os.path.join(physical_root, suffix) != physical:
            raise ValueError("logical/physical checkpoint suffix mapping drift")

    if source_authority is not None:
        supplied_record = _authority_record(
            source_authority,
            root=root,
            stage_manifest_sha256=stage_sha,
        )
        if authority_record != supplied_record:
            raise ValueError("supplied source authority differs from checkpoint identity")
    elif reattest_filesystem and authority_record is not None:
        source_authority = _load_authority_record(authority_record)

    rows = _validate_file_rows(raw["files"])
    if raw["n_files"] != len(rows) or raw["n_files"] <= 0:
        raise ValueError("checkpoint identity file count drift")
    total = sum(row["size_bytes"] for row in rows)
    if raw["total_bytes"] != total or total <= 0:
        raise ValueError("checkpoint identity byte count drift")
    files_sha = canonical_sha256(rows)
    if raw["files_sha256"] != files_sha:
        raise ValueError("checkpoint identity file-manifest hash drift")
    core = {
        "scheme": SCHEME,
        "path": physical,
        "logical_path": logical,
        "logical_prefix": logical_prefix,
        "physical_root": physical_root,
        "frozen_stage_root": str(root),
        "stage_manifest_sha256": stage_sha,
        "source_authority": authority_record,
        "source_mapping": None if mapping is None else mapping.to_dict(),
        "n_files": len(rows),
        "total_bytes": total,
        "files_sha256": files_sha,
    }
    identity_sha = canonical_sha256(core)
    if raw["identity_sha256"] != identity_sha:
        raise ValueError("checkpoint identity hash drift")
    value = CheckpointIdentity(
        path=physical,
        logical_path=logical,
        logical_prefix=logical_prefix,
        physical_root=physical_root,
        frozen_stage_root=str(root),
        stage_manifest_sha256=stage_sha,
        source_authority=authority_record,
        source_mapping=None if mapping is None else mapping.to_dict(),
        n_files=len(rows),
        total_bytes=total,
        files=tuple(rows),
        files_sha256=files_sha,
        identity_sha256=identity_sha,
    )
    if reattest_filesystem:
        observed = build_checkpoint_identity(
            physical,
            frozen_stage_root=root,
            stage_manifest_sha256=stage_sha,
            source_authority=source_authority,
            logical_path=logical,
        )
        if observed != value:
            raise ValueError("checkpoint filesystem differs from pinned identity")
    return value


def attest_checkpoint_identity(
    payload: CheckpointIdentity | Mapping[str, Any],
    *,
    source_authority: SourcePathAuthority | None = None,
    frozen_stage_root: str | Path | None = None,
    stage_manifest_sha256: str | None = None,
) -> CheckpointIdentity:
    """Repeat the complete live authority and sampled-filesystem attestation."""

    return validate_checkpoint_identity(
        payload,
        reattest_filesystem=True,
        source_authority=source_authority,
        frozen_stage_root=frozen_stage_root,
        stage_manifest_sha256=stage_manifest_sha256,
    )


def source_authority_from_identity(
    payload: CheckpointIdentity | Mapping[str, Any],
) -> SourcePathAuthority | None:
    """Reload the exact live authority embedded in an identity, or return null."""

    value = validate_checkpoint_identity(payload)
    if value.source_authority is None:
        return None
    return _load_authority_record(value.source_authority)


def build_identity_index(
    cohorts: Mapping[str, Sequence[CheckpointIdentity | Mapping[str, Any]]],
    *,
    cohort_order: Sequence[str],
    source_authority: SourcePathAuthority | None = None,
) -> dict[str, Any]:
    """Bind an exact ordered cohort grid of physical checkpoint identities."""

    order = tuple(_text(value, "cohort_order") for value in cohort_order)
    if len(order) != len(set(order)) or set(cohorts) != set(order):
        raise ValueError("checkpoint identity index cohort keys/order drift")
    rows: list[dict[str, Any]] = []
    all_logical_paths: set[str] = set()
    for cohort in order:
        values = tuple(
            validate_checkpoint_identity(value, source_authority=source_authority)
            for value in cohorts[cohort]
        )
        if not values:
            raise ValueError(f"checkpoint identity cohort {cohort!r} is empty")
        paths = [value.path for value in values]
        logical_paths = [value.logical_path for value in values]
        if len(paths) != len(set(paths)):
            raise ValueError(f"checkpoint identity cohort {cohort!r} reuses a physical path")
        if len(logical_paths) != len(set(logical_paths)):
            raise ValueError(f"checkpoint identity cohort {cohort!r} reuses a logical path")
        if all_logical_paths.intersection(logical_paths):
            raise ValueError("checkpoint identity index reuses a logical path across cohorts")
        all_logical_paths.update(logical_paths)
        entries = [value.to_dict() for value in values]
        rows.append({
            "cohort_key": cohort,
            "n_checkpoints": len(entries),
            "entries": entries,
            "entries_sha256": canonical_sha256(entries),
        })
    core = {
        "scheme": SCHEME,
        "cohort_order": list(order),
        "n_cohorts": len(order),
        "n_checkpoints": sum(row["n_checkpoints"] for row in rows),
        "cohorts": rows,
    }
    return {
        "schema": INDEX_SCHEMA,
        "valid": True,
        **core,
        "index_sha256": canonical_sha256(core),
    }


def read_identity_index(
    payload: Mapping[str, Any],
    *,
    source_authority: SourcePathAuthority | None = None,
    reattest_filesystem: bool = False,
) -> dict[str, CheckpointIdentity]:
    """Validate an index and return an exact physical-path lookup."""

    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint identity index must be a mapping")
    required = {
        "schema", "valid", "scheme", "cohort_order", "n_cohorts",
        "n_checkpoints", "cohorts", "index_sha256",
    }
    if set(payload) != required:
        raise ValueError("checkpoint identity index keys drift")
    if (
        payload["schema"] != INDEX_SCHEMA
        or payload["valid"] is not True
        or payload["scheme"] != SCHEME
    ):
        raise ValueError("checkpoint identity index control fields drift")
    order = tuple(_text(value, "cohort_order") for value in payload["cohort_order"])
    cohort_rows = payload["cohorts"]
    if isinstance(cohort_rows, (str, bytes)) or not isinstance(cohort_rows, Sequence):
        raise ValueError("checkpoint identity index cohorts must be a sequence")
    if payload["n_cohorts"] != len(order) or len(cohort_rows) != len(order):
        raise ValueError("checkpoint identity index cohort count drift")
    lookup: dict[str, CheckpointIdentity] = {}
    logical_paths: set[str] = set()
    normalized_rows: list[dict[str, Any]] = []
    for expected, row in zip(order, cohort_rows):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"cohort_key", "n_checkpoints", "entries", "entries_sha256"}
        ):
            raise ValueError("checkpoint identity index cohort row drift")
        if row["cohort_key"] != expected:
            raise ValueError("checkpoint identity index cohort order drift")
        entries = tuple(
            validate_checkpoint_identity(
                value,
                source_authority=source_authority,
                reattest_filesystem=reattest_filesystem,
            )
            for value in row["entries"]
        )
        entry_rows = [value.to_dict() for value in entries]
        if (
            row["n_checkpoints"] != len(entries)
            or row["entries_sha256"] != canonical_sha256(entry_rows)
        ):
            raise ValueError("checkpoint identity index entry count/hash drift")
        for value in entries:
            if value.path in lookup:
                raise ValueError("checkpoint identity index reuses a physical path")
            if value.logical_path in logical_paths:
                raise ValueError("checkpoint identity index reuses a logical path")
            lookup[value.path] = value
            logical_paths.add(value.logical_path)
        normalized_rows.append({
            "cohort_key": expected,
            "n_checkpoints": len(entries),
            "entries": entry_rows,
            "entries_sha256": canonical_sha256(entry_rows),
        })
    core = {
        "scheme": SCHEME,
        "cohort_order": list(order),
        "n_cohorts": len(order),
        "n_checkpoints": len(lookup),
        "cohorts": normalized_rows,
    }
    if (
        payload["n_checkpoints"] != len(lookup)
        or payload["index_sha256"] != canonical_sha256(core)
    ):
        raise ValueError("checkpoint identity index total/hash drift")
    return lookup


def _checkpoint_file_rows(source: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in sorted(source.rglob("*")):
        if entry.is_symlink():
            raise ValueError(f"checkpoint contains a symlink: {entry}")
        observed = entry.lstat()
        if stat.S_ISDIR(observed.st_mode):
            continue
        if not stat.S_ISREG(observed.st_mode):
            raise ValueError(f"checkpoint contains a non-regular file: {entry}")
        relative = entry.relative_to(source).as_posix()
        size = observed.st_size
        if size <= FULL_HASH_LIMIT_BYTES:
            row = {
                "path": relative,
                "size_bytes": size,
                "mode": stat.S_IMODE(observed.st_mode),
                "hash_kind": "full_sha256",
                "sha256": _sha256_file(entry),
            }
        else:
            offsets = _sample_offsets(size)
            row = {
                "path": relative,
                "size_bytes": size,
                "mode": stat.S_IMODE(observed.st_mode),
                "hash_kind": "fixed_window_sha256",
                "sample_bytes": SAMPLE_BYTES,
                "sample_offsets": list(offsets),
                "sample_sha256": _sample_sha256(entry, offsets=offsets),
            }
        rows.append(row)
    if not rows:
        raise ValueError(f"checkpoint directory is empty: {source}")
    return rows


def _authority_record(
    authority: SourcePathAuthority,
    *,
    root: Path,
    stage_manifest_sha256: str,
) -> dict[str, Any]:
    if not isinstance(authority, SourcePathAuthority):
        raise ValueError("source_authority type drift")
    value = _normalize_authority_record(
        authority.to_dict(), root=root, stage_manifest_sha256=stage_manifest_sha256
    )
    if value is None:
        raise AssertionError("source authority unexpectedly normalized to null")
    return value


def _normalize_authority_record(
    value: object,
    *,
    root: Path,
    stage_manifest_sha256: str,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint source_authority must be a mapping or null")
    required = {
        "stage_root", "stage_manifest", "symlink_inventory", "mappings",
        "authority_sha256",
    }
    if set(value) != required:
        raise ValueError("checkpoint source_authority keys drift")
    stage_root = _absolute_normalized(value["stage_root"], "authority stage_root")
    if stage_root != str(root):
        raise ValueError("checkpoint source authority stage root drift")
    stage_manifest = _normalize_pin(value["stage_manifest"], "authority stage manifest")
    inventory = _normalize_pin(value["symlink_inventory"], "authority symlink inventory")
    if stage_manifest != {
        "path": f"{root}/.stage_provenance/STAGE_SHA256SUMS",
        "sha256": stage_manifest_sha256,
    }:
        raise ValueError("checkpoint source authority stage manifest pin drift")
    if inventory["path"] != f"{root}/.stage_provenance/SYMLINKS.txt":
        raise ValueError("checkpoint source authority symlink inventory path drift")
    raw_mappings = value["mappings"]
    if isinstance(raw_mappings, (str, bytes)) or not isinstance(raw_mappings, Sequence):
        raise ValueError("checkpoint source authority mappings must be a sequence")
    mappings = [_normalize_source_mapping(row) for row in raw_mappings]
    if any(mapping is None for mapping in mappings):
        raise ValueError("checkpoint source authority mappings cannot contain null")
    mapping_rows = [mapping.to_dict() for mapping in mappings if mapping is not None]
    prefixes = [row["logical_prefix"] for row in mapping_rows]
    if prefixes != sorted(prefixes) or len(prefixes) != len(set(prefixes)):
        raise ValueError("checkpoint source authority mappings must be sorted and unique")
    for row in mapping_rows:
        if not _strictly_below(row["logical_prefix"], str(root)):
            raise ValueError("checkpoint authority logical prefix escapes frozen stage")
        if row["physical_root"] == str(root) or _strictly_below(row["physical_root"], str(root)):
            raise ValueError("checkpoint authority physical root must be outside frozen stage")
    core = {
        "stage_root": stage_root,
        "stage_manifest": stage_manifest,
        "symlink_inventory": inventory,
        "mappings": mapping_rows,
    }
    authority_sha = _sha256(value["authority_sha256"], "source authority sha256")
    if canonical_sha256(core) != authority_sha:
        raise ValueError("checkpoint source authority digest drift")
    return {**core, "authority_sha256": authority_sha}


def _load_authority_record(record: Mapping[str, Any]) -> SourcePathAuthority:
    value = load_source_path_authority(
        SourcePathAuthoritySpec(
            stage_root=record["stage_root"],
            stage_manifest=ArtifactPin(**record["stage_manifest"]),
            symlink_inventory=ArtifactPin(**record["symlink_inventory"]),
            mappings=tuple(
                SourcePathMapping(row["logical_prefix"], row["physical_root"])
                for row in record["mappings"]
            ),
        )
    )
    if value.to_dict() != _deep_dict(record):
        raise ValueError("live source authority differs from checkpoint identity")
    return value


def _normalize_source_mapping(value: object) -> SourcePathMapping | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"logical_prefix", "physical_root"}:
        raise ValueError("checkpoint source_mapping keys drift")
    return SourcePathMapping(
        _absolute_normalized(value["logical_prefix"], "source logical prefix"),
        _absolute_normalized(value["physical_root"], "source physical root"),
    )


def _normalize_pin(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} pin keys drift")
    return {
        "path": _absolute_normalized(value["path"], f"{label} path"),
        "sha256": _sha256(value["sha256"], f"{label} sha256"),
    }


def _validate_file_rows(value: object) -> list[dict[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("checkpoint identity files must be a sequence")
    rows: list[dict[str, Any]] = []
    paths: list[str] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ValueError("checkpoint identity file row must be a mapping")
        row = dict(raw)
        kind = row.get("hash_kind")
        common = {"path", "size_bytes", "mode", "hash_kind"}
        expected = common | (
            {"sha256"}
            if kind == "full_sha256"
            else {"sample_bytes", "sample_offsets", "sample_sha256"}
        )
        if set(row) != expected:
            raise ValueError("checkpoint identity file row keys drift")
        relative = _relative(row["path"])
        if (
            isinstance(row["size_bytes"], bool)
            or not isinstance(row["size_bytes"], int)
            or row["size_bytes"] < 0
        ):
            raise ValueError("checkpoint identity file size drift")
        if (
            isinstance(row["mode"], bool)
            or not isinstance(row["mode"], int)
            or not 0 <= row["mode"] <= 0o7777
        ):
            raise ValueError("checkpoint identity file mode drift")
        if kind == "full_sha256":
            _sha256(row["sha256"], "file sha256")
            if row["size_bytes"] > FULL_HASH_LIMIT_BYTES:
                raise ValueError("large checkpoint file incorrectly uses full-small hash kind")
        elif kind == "fixed_window_sha256":
            if row["size_bytes"] <= FULL_HASH_LIMIT_BYTES or row["sample_bytes"] != SAMPLE_BYTES:
                raise ValueError("sampled checkpoint file size/control drift")
            offsets = _sample_offsets(row["size_bytes"])
            if row["sample_offsets"] != list(offsets):
                raise ValueError("sampled checkpoint offsets drift")
            samples = row["sample_sha256"]
            if not isinstance(samples, list) or len(samples) != len(offsets):
                raise ValueError("sampled checkpoint hashes drift")
            for digest in samples:
                _sha256(digest, "sample sha256")
        else:
            raise ValueError("checkpoint identity hash_kind drift")
        row["path"] = relative
        rows.append(row)
        paths.append(relative)
    if not rows or paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("checkpoint identity files must be nonempty, sorted, and unique")
    return rows


def _sample_offsets(size: int) -> tuple[int, ...]:
    maximum = size - SAMPLE_BYTES
    if maximum < 0:
        raise ValueError("sampled file is smaller than one sample window")
    return tuple(int(round(maximum * fraction)) for fraction in SAMPLE_FRACTIONS)


def _sample_sha256(path: Path, *, offsets: Sequence[int]) -> list[str]:
    hashes: list[str] = []
    with path.open("rb", buffering=0) as handle:
        for offset in offsets:
            handle.seek(offset)
            chunk = handle.read(SAMPLE_BYTES)
            if len(chunk) != SAMPLE_BYTES:
                raise ValueError(f"short checkpoint sample read at {path}:{offset}")
            hashes.append(hashlib.sha256(chunk).hexdigest())
    return hashes


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_root(value: str | Path) -> Path:
    root = Path(_absolute_normalized(value, "frozen_stage_root"))
    if root == Path("/") or root.is_symlink() or not root.is_dir():
        raise ValueError("frozen_stage_root must be a bounded existing non-symlink directory")
    if root.resolve(strict=True) != root:
        raise ValueError("frozen_stage_root must be canonical")
    return root


def _canonical_checkpoint_directory(value: str | Path) -> Path:
    source = Path(_absolute_normalized(value, "physical checkpoint path"))
    if source == Path("/") or source.is_symlink() or not source.is_dir():
        raise ValueError("physical checkpoint path must be a bounded existing directory")
    if source.resolve(strict=True) != source:
        raise ValueError("physical checkpoint path must be canonical and contain no symlink")
    return source


def _absolute_normalized(value: object, label: str) -> str:
    text = str(value) if isinstance(value, Path) else _text(value, label)
    if not text or text.strip() != text:
        raise ValueError(f"{label} must be nonempty trimmed text")
    if not os.path.isabs(text) or os.path.normpath(text) != text:
        raise ValueError(f"{label} must be an absolute normalized path")
    return text


def _strictly_below(path: str, root: str) -> bool:
    try:
        return os.path.commonpath((root, path)) == root and path != root
    except ValueError:
        return False


def _relative(value: object) -> str:
    text = _text(value, "relative file path")
    path = Path(text)
    if (
        path.is_absolute()
        or text in {".", ".."}
        or ".." in path.parts
        or os.path.normpath(text) != text
    ):
        raise ValueError("checkpoint file path must be relative and normalized")
    return path.as_posix()


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be a nonempty trimmed string")
    return value


def _sha256(value: object, label: str) -> str:
    text = _text(value, label)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


def _deep_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        result = json.loads(canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint authority must be canonical JSON data") from exc
    if not isinstance(result, dict):
        raise ValueError("checkpoint authority must be a JSON object")
    return result


__all__ = [
    "CheckpointIdentity", "FULL_HASH_LIMIT_BYTES", "INDEX_SCHEMA", "SAMPLE_BYTES",
    "SCHEMA", "SCHEME", "attest_checkpoint_identity",
    "build_checkpoint_identity", "build_identity_index", "canonical_sha256",
    "read_identity_index", "source_authority_from_identity",
    "validate_checkpoint_identity",
]
