"""Pinned, read-only symlink inventory for the frozen training stage.

The frozen production stage is normally a physical tree.  Its two Llama
output roots are an intentional exception: each is one top-level symlink to a
large, immutable absolute output directory.  This module makes that exception
explicit and content addressed.  A lexical artifact path may cross the frozen
root boundary only through one exact inventory entry, and only while the live
link still contains the exact pinned absolute target string.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .checkpoints import ArtifactPin
from .manifest import canonical_sha256


@dataclass(frozen=True)
class SourcePathMapping:
    """One exact frozen logical prefix to canonical physical output root."""

    logical_prefix: str
    physical_root: str

    @property
    def path(self) -> str:
        return self.logical_prefix

    @property
    def target(self) -> str:
        return self.physical_root

    def to_dict(self) -> dict[str, str]:
        return {
            "logical_prefix": self.logical_prefix,
            "physical_root": self.physical_root,
        }


@dataclass(frozen=True)
class SourcePathAuthoritySpec:
    """Immutable inputs defining the only accepted source-path exceptions."""

    stage_root: str
    stage_manifest: ArtifactPin
    symlink_inventory: ArtifactPin
    mappings: tuple[SourcePathMapping, ...]

    @property
    def authority_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_root": self.stage_root,
            "stage_manifest": _pin_dict(self.stage_manifest),
            "symlink_inventory": _pin_dict(self.symlink_inventory),
            "mappings": [mapping.to_dict() for mapping in self.mappings],
        }


@dataclass(frozen=True)
class SourcePathResolution:
    logical_path: str
    physical_path: str
    mapping: SourcePathMapping | None
    authority_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_path": self.logical_path,
            "physical_path": self.physical_path,
            "mapping": None if self.mapping is None else self.mapping.to_dict(),
            "authority_sha256": self.authority_sha256,
        }


@dataclass(frozen=True)
class SourcePathAuthority:
    """Live-attested central authority for physical source bindings."""

    stage_root: str
    stage_manifest: ArtifactPin
    symlink_inventory: ArtifactPin
    mappings: tuple[SourcePathMapping, ...]
    authority_sha256: str

    @property
    def stage_manifest_sha256(self) -> str:
        return self.stage_manifest.sha256

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_root": self.stage_root,
            "stage_manifest": _pin_dict(self.stage_manifest),
            "symlink_inventory": _pin_dict(self.symlink_inventory),
            "mappings": [mapping.to_dict() for mapping in self.mappings],
            "authority_sha256": self.authority_sha256,
        }

    def spec(self) -> SourcePathAuthoritySpec:
        return SourcePathAuthoritySpec(
            stage_root=self.stage_root,
            stage_manifest=self.stage_manifest,
            symlink_inventory=self.symlink_inventory,
            mappings=self.mappings,
        )

    def reattest(self) -> "SourcePathAuthority":
        observed = load_source_path_authority(self.spec())
        if observed != self:
            raise ValueError("live source-path authority differs from pinned authority")
        return observed

    def mapping_for_logical(self, logical_path: str) -> SourcePathMapping | None:
        logical = _absolute_normalized(logical_path, "logical source path")
        matches = [
            mapping
            for mapping in self.mappings
            if logical == mapping.logical_prefix
            or _strictly_below(logical, mapping.logical_prefix)
        ]
        if len(matches) > 1:
            raise ValueError("logical path matches overlapping source mappings")
        return matches[0] if matches else None

    def mapping_for_physical(self, physical_path: str) -> SourcePathMapping | None:
        physical = _absolute_normalized(physical_path, "physical source path")
        matches = [
            mapping
            for mapping in self.mappings
            if physical == mapping.physical_root
            or _strictly_below(physical, mapping.physical_root)
        ]
        if len(matches) > 1:
            raise ValueError("physical path matches overlapping source mappings")
        return matches[0] if matches else None

    def logical_for_physical(self, physical_path: str) -> str:
        """Invert an authorized physical binding to its exact logical path."""

        physical = _absolute_normalized(physical_path, "physical source path")
        if _strictly_below(physical, self.stage_root):
            if os.path.realpath(physical) != physical:
                raise ValueError("physical stage path unexpectedly crosses a symlink")
            return physical
        mapping = self.mapping_for_physical(physical)
        if mapping is None:
            raise ValueError("physical source path is outside exact source authority")
        _validate_live_mapping(mapping, stage_root=self.stage_root)
        suffix = os.path.relpath(physical, mapping.physical_root)
        logical = (
            mapping.logical_prefix
            if suffix == "."
            else os.path.join(mapping.logical_prefix, suffix)
        )
        resolution = self.resolve(logical)
        if resolution.physical_path != physical:
            raise ValueError("physical source path cannot be inverted exactly")
        return logical

    def resolve(self, logical_path: str) -> SourcePathResolution:
        """Return a canonical physical path without leaving a link in bindings."""

        logical = _absolute_normalized(logical_path, "logical source path")
        if not _strictly_below(logical, self.stage_root):
            raise ValueError("logical source path is not beneath frozen stage")
        observed = os.path.realpath(logical)
        if observed == logical:
            return SourcePathResolution(
                logical_path=logical,
                physical_path=logical,
                mapping=None,
                authority_sha256=self.authority_sha256,
            )
        mapping = self.mapping_for_logical(logical)
        if mapping is None:
            raise ValueError("logical source path crosses an unlisted link")
        _validate_live_mapping(mapping, stage_root=self.stage_root)
        suffix = os.path.relpath(logical, mapping.logical_prefix)
        physical = (
            mapping.physical_root
            if suffix == "."
            else os.path.join(mapping.physical_root, suffix)
        )
        if os.path.normpath(physical) != physical:
            raise AssertionError("authorized source suffix produced a non-normal path")
        if not (
            physical == mapping.physical_root
            or _strictly_below(physical, mapping.physical_root)
        ):
            raise ValueError("source suffix escapes its exact authorized physical root")
        _require_canonical_physical_path(
            physical, physical_root=mapping.physical_root
        )
        if observed != physical:
            raise ValueError("logical resolution differs from exact authorized physical path")
        return SourcePathResolution(
            logical_path=logical,
            physical_path=physical,
            mapping=mapping,
            authority_sha256=self.authority_sha256,
        )

    def authorize(
        self,
        logical_path: str,
        physical_path: str,
    ) -> SourcePathMapping | None:
        physical = _absolute_normalized(physical_path, "physical source path")
        resolution = self.resolve(logical_path)
        if resolution.physical_path != physical:
            raise ValueError("physical source path differs from authority resolution")
        return resolution.mapping


def load_source_path_authority(
    spec: SourcePathAuthoritySpec,
) -> SourcePathAuthority:
    """Load the exact global inventory and attest all live source mappings."""

    if not isinstance(spec, SourcePathAuthoritySpec):
        raise ValueError("source path authority spec type drift")
    root = _canonical_root(spec.stage_root)
    stage_manifest = _verify_physical_file_pin(
        spec.stage_manifest, root=root, label="stage manifest"
    )
    inventory_pin = _verify_physical_file_pin(
        spec.symlink_inventory, root=root, label="symlink inventory"
    )
    expected_inventory_path = f"{root}/.stage_provenance/SYMLINKS.txt"
    if inventory_pin.path != expected_inventory_path:
        raise ValueError("symlink inventory path differs from exact stage location")
    mappings = tuple(_mapping(value, root=root) for value in spec.mappings)
    if len(mappings) != len(set(mapping.logical_prefix for mapping in mappings)):
        raise ValueError("source path authority mappings contain duplicate prefixes")
    if [mapping.logical_prefix for mapping in mappings] != sorted(
        mapping.logical_prefix for mapping in mappings
    ):
        raise ValueError("source path authority mappings must be sorted")
    raw_entries = _parse_inventory(Path(inventory_pin.path).read_bytes(), stage_root=root)
    observed_mappings = tuple(
        SourcePathMapping(entry.path, entry.target) for entry in raw_entries
    )
    if observed_mappings != mappings:
        raise ValueError("pinned symlink inventory differs from exact authority mappings")
    _require_manifest_pin(
        Path(stage_manifest.path),
        artifact=inventory_pin,
        stage_root=root,
    )
    for mapping in mappings:
        _validate_live_mapping(mapping, stage_root=root)
    observed_links = tuple(sorted(_walk_symlinks(Path(root))))
    expected_links = tuple(mapping.logical_prefix for mapping in mappings)
    if observed_links != expected_links:
        raise ValueError("live frozen-stage symlink set differs from exact inventory")
    core = {
        "stage_root": root,
        "stage_manifest": _pin_dict(stage_manifest),
        "symlink_inventory": _pin_dict(inventory_pin),
        "mappings": [mapping.to_dict() for mapping in mappings],
    }
    return SourcePathAuthority(
        stage_root=root,
        stage_manifest=stage_manifest,
        symlink_inventory=inventory_pin,
        mappings=mappings,
        authority_sha256=canonical_sha256(core),
    )


def source_path_authority_from_dict(
    payload: Mapping[str, Any],
) -> SourcePathAuthority:
    """Reload and live-attest one exact serialized source-path authority."""

    required = {
        "stage_root",
        "stage_manifest",
        "symlink_inventory",
        "mappings",
        "authority_sha256",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError("serialized source-path authority keys drift")

    def serialized_pin(value: object, label: str) -> ArtifactPin:
        if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
            raise ValueError(f"serialized {label} pin keys drift")
        return ArtifactPin(
            _absolute_normalized(value["path"], f"serialized {label} path"),
            _sha(value["sha256"], f"serialized {label} sha256"),
        )

    raw_mappings = payload["mappings"]
    if not isinstance(raw_mappings, list):
        raise ValueError("serialized source-path authority mappings must be a list")
    mappings: list[SourcePathMapping] = []
    for row in raw_mappings:
        if not isinstance(row, Mapping) or set(row) != {
            "logical_prefix",
            "physical_root",
        }:
            raise ValueError("serialized source-path authority mapping keys drift")
        mappings.append(
            SourcePathMapping(
                _absolute_normalized(
                    row["logical_prefix"], "serialized logical source prefix"
                ),
                _absolute_normalized(
                    row["physical_root"], "serialized physical source root"
                ),
            )
        )
    value = load_source_path_authority(
        SourcePathAuthoritySpec(
            stage_root=_absolute_normalized(
                payload["stage_root"], "serialized authority stage_root"
            ),
            stage_manifest=serialized_pin(
                payload["stage_manifest"], "authority stage manifest"
            ),
            symlink_inventory=serialized_pin(
                payload["symlink_inventory"], "authority symlink inventory"
            ),
            mappings=tuple(mappings),
        )
    )
    _sha(payload["authority_sha256"], "serialized authority sha256")
    if value.to_dict() != dict(payload):
        raise ValueError("live source-path authority differs from serialized contract")
    return value


@dataclass(frozen=True)
class SymlinkInventoryEntry:
    """One canonical lexical-link/absolute-target pair."""

    path: str
    target: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "target": self.target}


@dataclass(frozen=True)
class FrozenSymlinkInventory:
    """Verified bytes and canonical entries for one frozen-stage inventory."""

    pin: ArtifactPin
    stage_root: str
    entries: tuple[SymlinkInventoryEntry, ...]
    entries_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.pin.path,
            "sha256": self.pin.sha256,
            "n_entries": len(self.entries),
            "entries": [entry.to_dict() for entry in self.entries],
            "entries_sha256": self.entries_sha256,
        }

    def entry(self, path: str) -> SymlinkInventoryEntry | None:
        """Return the unique direct inventory ancestor of ``path``."""

        candidate = _absolute_normalized(path, "artifact path")
        matches = [
            entry
            for entry in self.entries
            if candidate == entry.path or _strictly_below(candidate, entry.path)
        ]
        if len(matches) > 1:
            raise ValueError("artifact path matches overlapping symlink inventory entries")
        return matches[0] if matches else None

    def resolve(self, path: str) -> tuple[str, SymlinkInventoryEntry | None]:
        """Resolve a lexical path under the exact pinned inventory contract."""

        candidate = _absolute_normalized(path, "artifact path")
        if not _strictly_below(candidate, self.stage_root):
            raise ValueError("artifact lexical path is not beneath the frozen stage")
        observed = os.path.realpath(candidate)
        if observed == candidate:
            return candidate, None
        entry = self.entry(candidate)
        if entry is None:
            raise ValueError("artifact path crosses an unlisted symlink")
        _validate_live_entry(entry)
        relative = os.path.relpath(candidate, entry.path)
        expected = entry.target if relative == "." else os.path.join(entry.target, relative)
        if os.path.normpath(expected) != expected:
            raise AssertionError("canonical symlink suffix produced a non-normal path")
        if observed != expected:
            raise ValueError("artifact path resolution differs from pinned symlink target")
        return observed, entry


def load_frozen_symlink_inventory(
    pin: ArtifactPin,
    *,
    stage_root: str,
    allowed_entry_paths: Sequence[str] | None = None,
) -> FrozenSymlinkInventory:
    """Read, hash, parse, and live-attest a pinned inventory.

    ``allowed_entry_paths`` restricts the returned view to the exact entries a
    cohort is permitted to use.  The inventory file hash always covers the
    complete stage-global inventory, including entries outside that view.
    """

    if not isinstance(pin, ArtifactPin):
        raise ValueError("symlink inventory pin must be an ArtifactPin")
    root = _canonical_root(stage_root)
    inventory_path = _absolute_normalized(pin.path, "symlink inventory path")
    if not _strictly_below(inventory_path, root):
        raise ValueError("symlink inventory must be beneath the frozen stage")
    if Path(inventory_path).is_symlink() or os.path.realpath(inventory_path) != inventory_path:
        raise ValueError("symlink inventory path must contain no symlink component")
    source = Path(inventory_path)
    if not source.is_file():
        raise FileNotFoundError(f"required symlink inventory is missing: {source}")
    raw = source.read_bytes()
    observed_sha = hashlib.sha256(raw).hexdigest()
    if observed_sha != _sha(pin.sha256, "symlink inventory sha256"):
        raise ValueError("symlink inventory hash drift")
    entries = _parse_inventory(raw, stage_root=root)
    by_path = {entry.path: entry for entry in entries}
    if len(by_path) != len(entries):
        raise ValueError("symlink inventory contains duplicate lexical paths")
    requested = tuple(by_path) if allowed_entry_paths is None else tuple(
        _absolute_normalized(value, "allowed symlink inventory entry")
        for value in allowed_entry_paths
    )
    if len(requested) != len(set(requested)):
        raise ValueError("allowed symlink inventory entries contain duplicates")
    missing = [path for path in requested if path not in by_path]
    if missing:
        raise ValueError(f"allowed symlink entry is absent from pinned inventory: {missing[0]}")
    selected = tuple(by_path[path] for path in requested)
    for entry in selected:
        _validate_live_entry(entry)
    return FrozenSymlinkInventory(
        pin=ArtifactPin(inventory_path, observed_sha),
        stage_root=root,
        entries=selected,
        entries_sha256=canonical_sha256([entry.to_dict() for entry in selected]),
    )


def inventory_from_attestation(
    *,
    pin: Mapping[str, object],
    entry: Mapping[str, object],
    stage_root: str,
) -> FrozenSymlinkInventory:
    """Reload an exact serialized inventory entry for runtime reattestation."""

    if set(pin) != {"path", "sha256"}:
        raise ValueError("serialized symlink inventory pin keys drift")
    if set(entry) != {"path", "target"}:
        raise ValueError("serialized symlink inventory entry keys drift")
    expected = SymlinkInventoryEntry(
        path=_absolute_normalized(entry["path"], "symlink entry path"),
        target=_canonical_target(entry["target"]),
    )
    value = load_frozen_symlink_inventory(
        ArtifactPin(
            _absolute_normalized(pin["path"], "symlink inventory pin path"),
            _sha(pin["sha256"], "symlink inventory pin sha256"),
        ),
        stage_root=stage_root,
        allowed_entry_paths=(expected.path,),
    )
    if value.entries != (expected,):
        raise ValueError("serialized symlink entry differs from pinned inventory")
    return value


def _parse_inventory(raw: bytes, *, stage_root: str) -> tuple[SymlinkInventoryEntry, ...]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("symlink inventory must be UTF-8") from exc
    if not text or not text.endswith("\n"):
        raise ValueError("symlink inventory must be nonempty and newline terminated")
    lines = text.splitlines()
    if not lines or any(not line for line in lines):
        raise ValueError("symlink inventory must not contain blank lines")
    rows: list[SymlinkInventoryEntry] = []
    for line in lines:
        # Historical stage tooling emitted human-readable ``path -> target``;
        # tab-separated inventories are accepted as the sole machine form.
        if "\t" in line and " -> " not in line:
            parts = line.split("\t")
        elif " -> " in line and "\t" not in line:
            parts = line.split(" -> ")
        else:
            raise ValueError("symlink inventory row delimiter is invalid or ambiguous")
        if len(parts) != 2 or any(not part or part != part.strip() for part in parts):
            raise ValueError("symlink inventory row must contain exact path and target")
        raw_path, raw_target = parts
        if os.path.isabs(raw_path):
            lexical = _absolute_normalized(raw_path, "symlink inventory path")
        else:
            if raw_path in {".", ".."} or os.path.normpath(raw_path) != raw_path:
                raise ValueError("symlink inventory relative path must be normalized")
            lexical = os.path.join(stage_root, raw_path)
        if not _strictly_below(lexical, stage_root):
            raise ValueError("symlink inventory lexical path escapes frozen stage")
        target = _canonical_target(raw_target)
        try:
            target_within_stage = os.path.commonpath((stage_root, target)) == stage_root
        except ValueError:
            target_within_stage = False
        if target_within_stage:
            raise ValueError("symlink inventory target must be outside frozen stage")
        rows.append(SymlinkInventoryEntry(path=lexical, target=target))
    paths = [row.path for row in rows]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("symlink inventory entries must be sorted and unique")
    for index, path in enumerate(paths):
        for other in paths[index + 1 :]:
            if _strictly_below(other, path):
                raise ValueError("symlink inventory entries must not overlap")
    return tuple(rows)


def _validate_live_entry(entry: SymlinkInventoryEntry) -> None:
    if not os.path.islink(entry.path):
        raise ValueError(f"pinned symlink inventory entry is not a live link: {entry.path}")
    observed_target = os.readlink(entry.path)
    if observed_target != entry.target:
        raise ValueError("live symlink target differs from pinned inventory target")
    if not Path(entry.target).is_dir():
        raise ValueError("pinned symlink target must be an existing directory")
    if os.path.realpath(entry.target) != entry.target:
        raise ValueError("pinned symlink target must be canonical and contain no symlink")
    if os.path.realpath(entry.path) != entry.target:
        raise ValueError("live symlink resolution differs from pinned target")


def _mapping(value: object, *, root: str) -> SourcePathMapping:
    if not isinstance(value, SourcePathMapping):
        raise ValueError("source path mapping type drift")
    logical = _absolute_normalized(value.logical_prefix, "logical source prefix")
    physical = _canonical_target(value.physical_root)
    if not _strictly_below(logical, root):
        raise ValueError("logical source prefix is not beneath frozen stage")
    try:
        physical_inside = os.path.commonpath((root, physical)) == root
    except ValueError:
        physical_inside = False
    if physical_inside:
        raise ValueError("authorized physical source root must be outside frozen stage")
    return SourcePathMapping(logical, physical)


def _validate_live_mapping(mapping: SourcePathMapping, *, stage_root: str) -> None:
    logical = Path(mapping.logical_prefix)
    if not logical.is_symlink():
        raise ValueError(f"authorized logical source is not a live symlink: {logical}")
    for parent in logical.parents:
        if str(parent) == stage_root:
            break
        if parent.is_symlink():
            raise ValueError("authorized logical source has a symlink parent")
    else:
        raise ValueError("authorized logical source is not beneath frozen stage")
    observed_target = os.readlink(logical)
    if observed_target != mapping.physical_root:
        raise ValueError("live source link target differs from exact authority")
    if os.path.realpath(logical) != mapping.physical_root:
        raise ValueError("live source link resolution differs from exact authority")
    _require_canonical_physical_path(
        mapping.physical_root,
        physical_root=mapping.physical_root,
    )


def _require_canonical_physical_path(path: str, *, physical_root: str) -> None:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"authorized physical source is missing: {source}")
    root = Path(physical_root)
    if root.is_symlink() or not root.is_dir() or root.resolve(strict=True) != root:
        raise ValueError("authorized physical source root must be canonical directory")
    if source.resolve(strict=True) != source:
        raise ValueError("authorized physical source path contains a nested symlink")


def _verify_physical_file_pin(
    pin: ArtifactPin,
    *,
    root: str,
    label: str,
) -> ArtifactPin:
    if not isinstance(pin, ArtifactPin):
        raise ValueError(f"{label} pin type drift")
    path = _absolute_normalized(pin.path, f"{label} path")
    if not _strictly_below(path, root):
        raise ValueError(f"{label} must be beneath frozen stage")
    source = Path(path)
    if source.is_symlink() or not source.is_file() or source.resolve(strict=True) != source:
        raise ValueError(f"{label} must be an existing canonical physical file")
    observed = hashlib.sha256(source.read_bytes()).hexdigest()
    if observed != _sha(pin.sha256, f"{label} sha256"):
        raise ValueError(f"{label} hash drift")
    return ArtifactPin(path, observed)


def _require_manifest_pin(
    manifest: Path,
    *,
    artifact: ArtifactPin,
    stage_root: str,
) -> None:
    relative = Path(artifact.path).relative_to(stage_root).as_posix()
    matches: list[str] = []
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError("cannot read frozen stage manifest") from exc
    for line in lines:
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError("frozen stage manifest row format drift")
        digest, raw_path = parts
        _sha(digest, "stage manifest row sha256")
        if raw_path.startswith("*"):
            raw_path = raw_path[1:]
        if raw_path.startswith("./"):
            raw_path = raw_path[2:]
        if raw_path == relative:
            matches.append(digest)
    if matches != [artifact.sha256]:
        raise ValueError("symlink inventory is absent or duplicated in stage manifest")


def _walk_symlinks(root: Path) -> list[str]:
    links: list[str] = []
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            raise ValueError(f"cannot enumerate frozen stage: {directory}") from exc
        for entry in entries:
            path = Path(entry.path)
            if entry.is_symlink():
                links.append(str(path))
            elif entry.is_dir(follow_symlinks=False):
                stack.append(path)
    return links


def _pin_dict(pin: ArtifactPin) -> dict[str, str]:
    return {"path": pin.path, "sha256": pin.sha256}


def _canonical_root(value: object) -> str:
    root = _absolute_normalized(value, "stage_root")
    if root == "/" or Path(root).is_symlink() or not Path(root).is_dir():
        raise ValueError("stage_root must be a bounded existing non-symlink directory")
    if os.path.realpath(root) != root:
        raise ValueError("stage_root must be canonical")
    return root


def _canonical_target(value: object) -> str:
    target = _absolute_normalized(value, "symlink target")
    if target == "/":
        raise ValueError("symlink target must be bounded")
    return target


def _absolute_normalized(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be nonempty trimmed text")
    if not os.path.isabs(value) or os.path.normpath(value) != value:
        raise ValueError(f"{label} must be an absolute normalized path")
    return value


def _strictly_below(path: str, root: str) -> bool:
    try:
        return os.path.commonpath((root, path)) == root and path != root
    except ValueError:
        return False


def _sha(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


__all__ = [
    "FrozenSymlinkInventory",
    "SourcePathAuthority",
    "SourcePathAuthoritySpec",
    "SourcePathMapping",
    "SourcePathResolution",
    "SymlinkInventoryEntry",
    "inventory_from_attestation",
    "load_frozen_symlink_inventory",
    "load_source_path_authority",
    "source_path_authority_from_dict",
]
