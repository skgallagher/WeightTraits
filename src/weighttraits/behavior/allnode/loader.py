"""Read-only production filesystem adapter for the all-node resolver.

The pure resolver deliberately accepts already-loaded records.  This module is
the narrow boundary that opens the frozen training stage, verifies every
declared source, and constructs those inputs.  It never writes to the stage and
has no scheduler functionality.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, Union

from .checkpoints import ArtifactPin
from .manifest import canonical_sha256
from .resolver import ResolverArtifactPins, ResolvedAllNodeTree, resolve_all_node_tree
from .symlink_inventory import (
    SourcePathAuthority,
    SourcePathAuthoritySpec,
    load_source_path_authority,
)


DigestPath = Callable[[Union[str, Path]], str]


@dataclass(frozen=True)
class CohortSourceSpec:
    """Pinned entry points and immutable training identity for one cohort."""

    cohort_key: str
    training_cohort_id: str
    stage_root: str
    top_config: ArtifactPin
    training_summary: ArtifactPin
    completion_receipt: ArtifactPin
    expected_method: str
    expected_artifact_name: str
    base_model_id: str
    base_model_revision: str
    completion_receipt_root: str | None = None
    expected_completion_schema: str = "weighttraits.training_completion_binding.v1"
    expected_artifact_factor: int = 2
    source_authority: SourcePathAuthoritySpec | None = None
    allowed_symlink_entries: tuple[str, ...] = ()


@dataclass(frozen=True)
class LoadedCohort:
    """Resolved trees plus a compact receipt for the read-only load."""

    trees: tuple[ResolvedAllNodeTree, ...]
    receipt: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.receipt)


def load_cohort_from_frozen_stage(
    spec: CohortSourceSpec,
    *,
    digest_path: DigestPath | None = None,
    source_authority: SourcePathAuthority | None = None,
) -> LoadedCohort:
    """Load and resolve one cohort after checking its complete source chain.

    ``digest_path`` is injectable so a deployment can use a separately audited
    content-addressed checkpoint index.  Regardless of the implementation, it
    must return the observed lowercase SHA-256 for the named file or directory.
    The default computes a complete content digest and rejects symlinks.
    """

    if not isinstance(spec, CohortSourceSpec):
        raise ValueError("spec must be a CohortSourceSpec")
    digest = sha256_path if digest_path is None else digest_path
    if not callable(digest):
        raise ValueError("digest_path must be callable")
    root = _root(spec.stage_root)
    if spec.source_authority is None:
        if spec.allowed_symlink_entries:
            raise ValueError("allowed symlink entries require a source authority")
        if source_authority is not None:
            raise ValueError("unplanned source authority supplied for physical-only cohort")
        authority = None
    else:
        authority = (
            load_source_path_authority(spec.source_authority)
            if source_authority is None
            else source_authority
        )
        if authority.spec() != spec.source_authority:
            raise ValueError("loaded source authority differs from cohort source spec")
        if authority.stage_root != root:
            raise ValueError("source authority root differs from cohort stage root")
        authority_paths = {mapping.logical_prefix for mapping in authority.mappings}
        if any(path not in authority_paths for path in spec.allowed_symlink_entries):
            raise ValueError("allowed symlink entry is absent from source authority")
    receipt_root = (
        root
        if spec.completion_receipt_root is None
        else _root(spec.completion_receipt_root)
    )
    top_config = _verify_entry_pin(spec.top_config, root=root, digest=digest)
    summary_pin = _verify_entry_pin(spec.training_summary, root=root, digest=digest)
    completion_pin = _verify_entry_pin(
        spec.completion_receipt,
        root=receipt_root,
        digest=digest,
    )

    summary = _json_mapping(Path(summary_pin.path), "training summary")
    completion = _json_mapping(Path(completion_pin.path), "completion receipt")
    summary_rows = _validate_training_summary(
        summary,
        stage_root=root,
        top_config=top_config,
        source_authority=authority,
        allowed_symlink_entries=spec.allowed_symlink_entries,
    )
    declared_path_base = _text(
        completion.get("path_base"),
        "completion receipt path_base",
    )
    effective_path_base = _completion_stage_root(
        declared_path_base,
        expected_stage_root=root,
    )
    completion_rows = _validate_completion_receipt(
        completion,
        stage_root=root,
        expected_summary_path=summary_pin.path,
        expected_schema=spec.expected_completion_schema,
        expected_cohort_id=spec.training_cohort_id,
        expected_artifact_factor=spec.expected_artifact_factor,
        source_authority=authority,
        allowed_symlink_entries=spec.allowed_symlink_entries,
    )
    summary_by_id = {row["tree_id"]: row for row in summary_rows}
    if tuple(summary_by_id) != tuple(row["tree_id"] for row in completion_rows):
        raise ValueError("training summary and completion receipt tree order differ")

    trees: list[ResolvedAllNodeTree] = []
    receipt_rows: list[dict[str, Any]] = []
    used_symlink_entries: set[str] = set()
    for completion_row in completion_rows:
        tree_id = completion_row["tree_id"]
        summary_row = summary_by_id[tree_id]
        _crosscheck_tree_rows(
            summary_row,
            completion_row,
            root=root,
            source_authority=authority,
            allowed_symlink_entries=spec.allowed_symlink_entries,
        )

        run_list_path = _artifact_path(
            completion_row["run_list"], root=root, authority=authority,
            allowed_symlink_entries=spec.allowed_symlink_entries,
        )
        ledger_path = _artifact_path(
            completion_row["ledger"], root=root, authority=authority,
            allowed_symlink_entries=spec.allowed_symlink_entries,
        )
        truth_path = _artifact_path(
            completion_row["manifest"], root=root, authority=authority,
            allowed_symlink_entries=spec.allowed_symlink_entries,
        )
        run_list_pin = _observed_pin(
            run_list_path,
            digest=digest,
            expected_sha256=summary_row["run_list_sha256"],
        )
        ledger_pin = _observed_pin(ledger_path, digest=digest)
        truth_pin = _observed_pin(
            truth_path,
            digest=digest,
            expected_sha256=summary_row["manifest_sha256"],
        )
        run_rows = _jsonl_mappings(Path(run_list_path), "run list")
        ledger_rows = _jsonl_mappings(Path(ledger_path), "ledger")
        truth_rows = _jsonl_mappings(Path(truth_path), "truth manifest")
        artifact_pins, tree_symlink_entries = _run_artifact_pins(
            run_rows,
            root=root,
            artifact_name=spec.expected_artifact_name,
            artifact_factor=spec.expected_artifact_factor,
            digest=digest,
            source_authority=authority,
            allowed_symlink_entries=spec.allowed_symlink_entries,
        )
        used_symlink_entries.update(tree_symlink_entries)
        checkpoint_pins = {
            node_id: pins[spec.expected_artifact_name]
            for node_id, pins in artifact_pins.items()
        }
        observed = ResolverArtifactPins(
            top_config=top_config,
            training_summary=summary_pin,
            completion_receipt=completion_pin,
            run_list=run_list_pin,
            ledger=ledger_pin,
            truth_manifest=truth_pin,
            checkpoints=checkpoint_pins,
        )
        resolved = resolve_all_node_tree(
            cohort_key=spec.cohort_key,
            training_cohort_id=spec.training_cohort_id,
            path_base=root,
            summary_row=completion_row,
            run_list_rows=run_rows,
            ledger_rows=ledger_rows,
            truth_rows=truth_rows,
            observed_pins=observed,
            expected_method=spec.expected_method,
            expected_artifact_name=spec.expected_artifact_name,
            base_model_id=spec.base_model_id,
            base_model_revision=spec.base_model_revision,
            completion_receipt_base=receipt_root,
            source_authority=authority,
            allowed_symlink_entries=spec.allowed_symlink_entries,
        )
        trees.append(resolved)
        receipt_rows.append(
            {
                "tree_id": tree_id,
                "n_nodes": resolved.topology.n_nodes,
                "n_internal": resolved.topology.n_internal,
                "n_leaves": resolved.topology.n_leaves,
                "run_list": _pin_dict(run_list_pin),
                "ledger": _pin_dict(ledger_pin),
                "truth_manifest": _pin_dict(truth_pin),
                "n_verified_artifacts": sum(
                    len(pins) for pins in artifact_pins.values()
                ),
                "symlink_entries_used": list(tree_symlink_entries),
                "artifact_set_sha256": _artifact_set_sha256(artifact_pins),
                "training_log_set_sha256": _training_log_set_sha256(artifact_pins),
                "checkpoint_set_sha256": resolved.checkpoint_manifest.checkpoint_set_sha256,
                "node_covariates_sha256": resolved.node_covariates.covariates_sha256,
                "resolution_sha256": resolved.resolution_sha256,
            }
        )

    allowed_entries = tuple(
        mapping
        for mapping in (() if authority is None else authority.mappings)
        if mapping.logical_prefix in spec.allowed_symlink_entries
    )
    allowed_paths = {entry.logical_prefix for entry in allowed_entries}
    if used_symlink_entries != allowed_paths:
        raise ValueError("cohort did not use its exact allowed symlink inventory entries")
    core = {
        "cohort_key": spec.cohort_key,
        "training_cohort_id": spec.training_cohort_id,
        "stage_root": root,
        "completion_receipt_root": receipt_root,
        "declared_path_base": declared_path_base,
        "effective_path_base": effective_path_base,
        "read_only": True,
        "scheduler_actions": 0,
        "top_config": _pin_dict(top_config),
        "training_summary": _pin_dict(summary_pin),
        "completion_receipt": _pin_dict(completion_pin),
        "expected_method": spec.expected_method,
        "expected_artifact_name": spec.expected_artifact_name,
        "expected_artifact_factor": spec.expected_artifact_factor,
        "expected_completion_schema": spec.expected_completion_schema,
        "base_model_id": spec.base_model_id,
        "base_model_revision": spec.base_model_revision,
        "source_path_authority": (
            None if authority is None else authority.to_dict()
        ),
        "allowed_symlink_entries": [entry.to_dict() for entry in allowed_entries],
        "used_symlink_entries": sorted(used_symlink_entries),
        "n_trees": len(trees),
        "n_nodes": sum(tree.topology.n_nodes for tree in trees),
        "n_internal": sum(tree.topology.n_internal for tree in trees),
        "n_leaves": sum(tree.topology.n_leaves for tree in trees),
        "trees": receipt_rows,
    }
    receipt = {
        "schema": "weighttraits.all_node_frozen_stage_load.v1",
        "valid": True,
        **core,
        "load_sha256": canonical_sha256(core),
    }
    return LoadedCohort(trees=tuple(trees), receipt=receipt)


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    if source.is_symlink():
        raise ValueError(f"source file must not be a symlink: {source}")
    if not source.is_file():
        raise FileNotFoundError(f"required source file is missing: {source}")
    value = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def sha256_path(path: str | Path) -> str:
    """Hash a file or a directory's sorted path/size/content manifest."""

    source = Path(path)
    if source.is_symlink():
        raise ValueError(f"source artifact must not be a symlink: {source}")
    if source.is_file():
        return sha256_file(source)
    if not source.is_dir():
        raise FileNotFoundError(f"required source artifact is missing: {source}")
    root = source.resolve(strict=True)
    entries = sorted(root.rglob("*"))
    symlinks = [item for item in entries if item.is_symlink()]
    if symlinks:
        raise ValueError(f"source artifact contains a symlink: {symlinks[0]}")
    files = [item for item in entries if item.is_file()]
    if not files:
        raise ValueError(f"source artifact directory is empty: {source}")
    rows: list[dict[str, Any]] = []
    for item in files:
        rows.append(
            {
                "path": item.relative_to(root).as_posix(),
                "size": item.stat().st_size,
                "sha256": sha256_file(item),
            }
        )
    return canonical_sha256(rows)


def _validate_training_summary(
    value: Mapping[str, Any],
    *,
    stage_root: str,
    top_config: ArtifactPin,
    source_authority: SourcePathAuthority | None,
    allowed_symlink_entries: Sequence[str],
) -> tuple[Mapping[str, Any], ...]:
    required = {
        "valid",
        "config",
        "config_sha256",
        "n_trees",
        "n_runs",
        "n_errors",
        "n_warnings",
        "trees",
    }
    _require_keys(value, required, "training summary")
    if _artifact_path(
        value["config"], root=stage_root, authority=source_authority,
        allowed_symlink_entries=allowed_symlink_entries,
    ) != top_config.path:
        raise ValueError("training summary config path differs from pinned top_config")
    if _sha(value["config_sha256"], "training summary config_sha256") != top_config.sha256:
        raise ValueError("training summary config_sha256 differs from pinned top_config")
    trees = _mapping_sequence(value["trees"], "training summary trees")
    if value["valid"] is not True or value["n_errors"] != 0 or value["n_warnings"] != 0:
        raise ValueError("training summary is not valid and clean")
    if value["n_trees"] != len(trees):
        raise ValueError("training summary n_trees differs from its tree rows")
    total = 0
    ids: list[str] = []
    for row in trees:
        _require_keys(
            row,
            {
                "tree_id", "valid", "n_runs", "n_errors", "n_warnings",
                "run_list", "run_list_sha256", "ledger", "manifest",
                "manifest_sha256",
            },
            "training summary tree row",
        )
        if row["valid"] is not True or row["n_errors"] != 0 or row["n_warnings"] != 0:
            raise ValueError("training summary contains a non-clean tree row")
        ids.append(_text(row["tree_id"], "summary tree_id"))
        total += _integer(row["n_runs"], "summary tree n_runs", minimum=1)
        _sha(row["run_list_sha256"], "summary run_list_sha256")
        _sha(row["manifest_sha256"], "summary manifest_sha256")
    if len(ids) != len(set(ids)):
        raise ValueError("training summary contains duplicate tree IDs")
    if value["n_runs"] != total:
        raise ValueError("training summary n_runs differs from the sum of tree rows")
    return trees


def _validate_completion_receipt(
    value: Mapping[str, Any],
    *,
    stage_root: str,
    expected_summary_path: str,
    expected_schema: str,
    expected_cohort_id: str,
    expected_artifact_factor: int,
    source_authority: SourcePathAuthority | None,
    allowed_symlink_entries: Sequence[str],
) -> tuple[Mapping[str, Any], ...]:
    required = {
        "schema", "cohort_id", "artifact_factor", "valid", "path_base",
        "summary", "n_trees", "n_ready", "n_failed",
        "n_in_progress", "n_not_started", "n_total_runs", "n_terminal_nodes",
        "n_ok_nodes", "n_failed_nodes", "n_missing_nodes", "n_errors",
        "n_warnings", "rows",
    }
    _require_keys(value, required, "completion receipt")
    if value["schema"] != _text(expected_schema, "expected completion schema"):
        raise ValueError("completion receipt schema differs from its cohort source spec")
    if value["cohort_id"] != _text(expected_cohort_id, "expected training cohort_id"):
        raise ValueError("completion receipt cohort_id differs from its cohort source spec")
    factor = _integer(
        expected_artifact_factor,
        "expected artifact factor",
        minimum=1,
    )
    if value["artifact_factor"] != factor:
        raise ValueError("completion receipt artifact_factor differs from its cohort source spec")
    if value["valid"] is not True:
        raise ValueError("completion receipt is not valid")
    _completion_stage_root(value["path_base"], expected_stage_root=stage_root)
    if _artifact_path(
        value["summary"], root=stage_root, authority=source_authority,
        allowed_symlink_entries=allowed_symlink_entries,
    ) != expected_summary_path:
        raise ValueError("completion receipt summary path differs from its pinned summary")
    rows = _mapping_sequence(value["rows"], "completion receipt rows")
    n_trees = _integer(value["n_trees"], "receipt n_trees", minimum=1)
    n_runs = _integer(value["n_total_runs"], "receipt n_total_runs", minimum=1)
    if n_trees != len(rows) or value["n_ready"] != n_trees:
        raise ValueError("completion receipt does not mark every tree ready")
    for field in ("n_failed", "n_in_progress", "n_not_started", "n_failed_nodes", "n_missing_nodes", "n_errors", "n_warnings"):
        if value[field] != 0:
            raise ValueError(f"completion receipt {field} must be zero")
    if value["n_terminal_nodes"] != n_runs or value["n_ok_nodes"] != n_runs:
        raise ValueError("completion receipt terminal/ok totals differ from n_total_runs")
    observed_runs = 0
    for row in rows:
        _require_keys(
            row,
            {
                "tree_id",
                "valid",
                "ready_for_analysis",
                "n_runs",
                "n_events",
                "n_ledger_nodes",
                "n_terminal_nodes",
                "n_ok_nodes",
                "n_failed_nodes",
                "n_missing_nodes",
                "n_expected_artifacts",
                "n_existing_artifacts",
                "n_errors",
                "n_warnings",
                "status_counts",
                "issue_counts",
                "run_list",
                "ledger",
                "manifest",
            },
            "completion receipt tree row",
        )
        row_runs = _integer(row["n_runs"], "receipt row n_runs", minimum=1)
        observed_runs += row_runs
        if row["valid"] is not True or row["ready_for_analysis"] is not True:
            raise ValueError("completion receipt contains a tree that is not ready")
        if _integer(row["n_events"], "receipt row n_events", minimum=0) < row_runs:
            raise ValueError("completion receipt row n_events must be at least n_runs")
        for field in ("n_ledger_nodes", "n_terminal_nodes", "n_ok_nodes"):
            if row[field] != row_runs:
                raise ValueError(f"completion receipt row {field} must equal n_runs")
        for field in ("n_failed_nodes", "n_missing_nodes", "n_errors", "n_warnings"):
            if row[field] != 0:
                raise ValueError(f"completion receipt row {field} must be zero")
        if row["status_counts"] != {"completed": row_runs}:
            raise ValueError("completion receipt row status_counts must be completed-only")
        if row["issue_counts"] != {}:
            raise ValueError("completion receipt row issue_counts must be empty")
        expected_artifacts = factor * row_runs
        if (
            row["n_expected_artifacts"] != expected_artifacts
            or row["n_existing_artifacts"] != expected_artifacts
        ):
            raise ValueError(
                "completion receipt row artifact counts differ from artifact_factor*n_runs"
            )
    if observed_runs != n_runs:
        raise ValueError("completion receipt row run counts differ from n_total_runs")
    return rows


def _crosscheck_tree_rows(
    summary: Mapping[str, Any],
    completion: Mapping[str, Any],
    *,
    root: str,
    source_authority: SourcePathAuthority | None,
    allowed_symlink_entries: Sequence[str],
) -> None:
    for field in ("tree_id", "n_runs"):
        if summary[field] != completion[field]:
            raise ValueError(f"summary/completion {field} drift")
    for field in ("run_list", "ledger", "manifest"):
        if _artifact_path(
            summary[field], root=root, authority=source_authority,
            allowed_symlink_entries=allowed_symlink_entries,
        ) != _artifact_path(
            completion[field], root=root, authority=source_authority,
            allowed_symlink_entries=allowed_symlink_entries,
        ):
            raise ValueError(f"summary/completion {field} drift")


def _run_artifact_pins(
    run_rows: Sequence[Mapping[str, Any]],
    *,
    root: str,
    artifact_name: str,
    artifact_factor: int,
    digest: DigestPath,
    source_authority: SourcePathAuthority | None,
    allowed_symlink_entries: Sequence[str],
) -> tuple[dict[str, dict[str, ArtifactPin]], tuple[str, ...]]:
    factor = _integer(artifact_factor, "artifact_factor", minimum=1)
    pins: dict[str, dict[str, ArtifactPin]] = {}
    used_symlink_entries: set[str] = set()
    for row in run_rows:
        if not isinstance(row, Mapping):
            raise ValueError("run-list rows must be mappings")
        node_id = _text(row.get("node_id"), "run-list node_id")
        if node_id in pins:
            raise ValueError("run list contains duplicate checkpoint node IDs")
        outer = _canonical_artifact_map(
            row.get("expected_artifacts"),
            label="run-list expected_artifacts",
            root=root,
            source_authority=source_authority,
            allowed_symlink_entries=allowed_symlink_entries,
        )
        job = row.get("job")
        if not isinstance(job, Mapping):
            raise ValueError("run-list row job must be a mapping")
        nested = _canonical_artifact_map(
            job.get("expected_artifacts"),
            label="run-list job.expected_artifacts",
            root=root,
            source_authority=source_authority,
            allowed_symlink_entries=allowed_symlink_entries,
        )
        if outer != nested:
            raise ValueError("run/job expected_artifacts drift")
        if artifact_name not in outer:
            raise ValueError(f"run-list row lacks {artifact_name!r} checkpoint")
        if "training_log" not in outer:
            raise ValueError("run-list row lacks required 'training_log' artifact")
        if len(outer) != factor:
            raise ValueError("run-list artifact map size differs from artifact_factor")
        if source_authority is not None:
            for path in outer.values():
                entry = source_authority.mapping_for_physical(path)
                if entry is not None:
                    used_symlink_entries.add(entry.logical_prefix)
        pins[node_id] = {
            name: _observed_pin(path, digest=digest)
            for name, path in sorted(outer.items())
        }
    if not pins:
        raise ValueError("run list contains no artifacts")
    return pins, tuple(sorted(used_symlink_entries))


def _canonical_artifact_map(
    value: object,
    *,
    label: str,
    root: str,
    source_authority: SourcePathAuthority | None,
    allowed_symlink_entries: Sequence[str],
) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be a nonempty mapping")
    result: dict[str, str] = {}
    for raw_name, raw_path in value.items():
        name = _text(raw_name, f"{label} name")
        if name in result:
            raise ValueError(f"{label} contains duplicate artifact names")
        result[name] = _artifact_path(
            raw_path, root=root, authority=source_authority,
            allowed_symlink_entries=allowed_symlink_entries,
        )
    return result


def _artifact_set_sha256(
    artifacts: Mapping[str, Mapping[str, ArtifactPin]],
) -> str:
    return canonical_sha256(
        [
            {
                "node_id": node_id,
                "artifacts": {
                    name: _pin_dict(pin) for name, pin in sorted(pins.items())
                },
            }
            for node_id, pins in artifacts.items()
        ]
    )


def _training_log_set_sha256(
    artifacts: Mapping[str, Mapping[str, ArtifactPin]],
) -> str:
    return canonical_sha256(
        [
            {"node_id": node_id, "training_log": _pin_dict(pins["training_log"])}
            for node_id, pins in artifacts.items()
        ]
    )


def _verify_entry_pin(pin: ArtifactPin, *, root: str, digest: DigestPath) -> ArtifactPin:
    if not isinstance(pin, ArtifactPin):
        raise ValueError("source entry pins must be ArtifactPin values")
    path = _artifact_path(pin.path, root=root)
    expected = _sha(pin.sha256, "source pin sha256")
    observed = _sha(digest(path), f"observed sha256 for {path}")
    if observed != expected:
        raise ValueError(f"source pin hash drift: {path}")
    return ArtifactPin(path=path, sha256=observed)


def _observed_pin(
    path: str,
    *,
    digest: DigestPath,
    expected_sha256: object | None = None,
) -> ArtifactPin:
    observed = _sha(digest(path), f"observed sha256 for {path}")
    if expected_sha256 is not None and observed != _sha(expected_sha256, "expected sha256"):
        raise ValueError(f"observed source hash differs from its declaration: {path}")
    return ArtifactPin(path=path, sha256=observed)


def _json_mapping(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid {label}: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _jsonl_mappings(path: Path, label: str) -> tuple[Mapping[str, Any], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc
    if not lines or any(not line.strip() for line in lines):
        raise ValueError(f"{label} must be nonempty JSONL without blank rows: {path}")
    rows: list[Mapping[str, Any]] = []
    for index, line in enumerate(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid {label} JSON at row {index}: {path}") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} row {index} is not a JSON object: {path}")
        rows.append(value)
    return tuple(rows)


def _root(value: object) -> str:
    path = _absolute(value, "stage_root")
    if path == "/":
        raise ValueError("stage_root must be bounded")
    try:
        canonical = str(Path(path).resolve(strict=True))
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"stage_root does not resolve to an existing directory: {path}") from exc
    if not Path(canonical).is_dir():
        raise ValueError(f"stage_root must be an existing directory: {path}")
    if canonical == "/":
        raise ValueError("stage_root must be bounded")
    return canonical


def _completion_stage_root(value: object, *, expected_stage_root: str) -> str:
    """Bind a completion receipt's path-base marker to the pinned stage root.

    Production completion bindings use either the absolute frozen-stage root or
    the literal ``.`` marker emitted when the audit was run from that stage.
    The latter is a representation marker, not a request to resolve against the
    loader's current working directory or the separately located receipt.  Its
    meaning is therefore supplied exclusively by ``expected_stage_root``, which
    has already passed the strict absolute/bounded stage-root validation.

    No other relative spelling is accepted.  This preserves the original
    fail-closed scope while supporting both audited receipt encodings.
    """

    raw = _text(value, "completion receipt path_base")
    if raw == ".":
        return expected_stage_root
    if not os.path.isabs(raw):
        raise ValueError(
            "completion receipt relative path_base must be the literal '.' marker"
        )
    observed = os.path.realpath(_absolute(raw, "completion receipt path_base"))
    if observed != expected_stage_root:
        raise ValueError("completion receipt path_base differs from frozen stage root")
    return observed


def _artifact_path(
    value: object,
    *,
    root: str,
    authority: SourcePathAuthority | None = None,
    allowed_symlink_entries: Sequence[str] = (),
) -> str:
    raw = _text(value, "artifact path")
    if os.path.normpath(raw) != raw or raw == ".":
        raise ValueError("artifact path must be normalized")
    candidate = raw if os.path.isabs(raw) else os.path.normpath(os.path.join(root, raw))
    if os.path.normpath(candidate) != candidate:
        raise ValueError("artifact path must be normalized")
    candidate = _canonicalize_stage_prefix(candidate, root=root)
    try:
        within_root = os.path.commonpath((root, candidate)) == root
    except ValueError as exc:
        raise ValueError("artifact path escapes or equals the frozen stage root") from exc
    if not within_root or candidate == root:
        raise ValueError("artifact path escapes or equals the frozen stage root")
    observed = os.path.realpath(candidate)
    if observed == candidate:
        return candidate
    if authority is None:
        raise ValueError("artifact path crosses an unlisted symlink")
    resolution = authority.resolve(candidate)
    mapping = resolution.mapping
    if mapping is None or resolution.physical_path != observed:
        raise ValueError("artifact symlink resolution lacks an exact inventory entry")
    if mapping.logical_prefix not in set(allowed_symlink_entries):
        raise ValueError("artifact path crosses a source mapping not allowed for cohort")
    # Persist only canonical physical paths.  Runtime consumers therefore do
    # not traverse the logical link; the receipt records the exact mapping.
    return resolution.physical_path


def _canonicalize_stage_prefix(candidate: str, *, root: str) -> str:
    """Normalize only a platform alias that names the stage root itself.

    macOS temporary directories are commonly spelled below ``/var`` while
    their canonical path is below ``/private/var``.  Test and local audit
    fixtures may therefore pin the former even though ``_root`` canonicalizes
    the stage.  Only a whole parent prefix resolving *exactly* to the stage
    root is rewritten; symlinks below that boundary remain visible and are
    subject to the pinned inventory contract.
    """

    source = Path(candidate)
    for parent in source.parents:
        if os.path.realpath(parent) != root:
            continue
        suffix = source.relative_to(parent)
        normalized = str(Path(root) / suffix)
        if os.path.normpath(normalized) != normalized:
            raise AssertionError("canonical stage-prefix rewrite is not normalized")
        return normalized
    return candidate


def _absolute(value: object, label: str) -> str:
    path = _text(value, label)
    if not os.path.isabs(path) or os.path.normpath(path) != path:
        raise ValueError(f"{label} must be an absolute normalized path")
    return path


def _pin_dict(pin: ArtifactPin) -> dict[str, str]:
    return {"path": pin.path, "sha256": pin.sha256}


def _mapping_sequence(value: object, label: str) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a sequence")
    rows = tuple(value)
    if not rows or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError(f"{label} must contain mapping rows")
    return rows  # type: ignore[return-value]


def _require_keys(value: Mapping[str, Any], required: set[str], label: str) -> None:
    missing = required - set(value)
    if missing:
        raise ValueError(f"{label} lacks required keys {sorted(missing)!r}")


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a nonempty trimmed string")
    return value


def _integer(value: object, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _sha(value: object, label: str) -> str:
    text = _text(value, label)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


__all__ = [
    "CohortSourceSpec",
    "LoadedCohort",
    "load_cohort_from_frozen_stage",
    "sha256_file",
    "sha256_path",
]
