"""Pure all-node resolver from already-loaded, externally pinned records.

The resolver performs no I/O.  It consumes already-loaded production summary,
run-list, ledger-event, and enriched truth-manifest mappings plus observed
artifact pins, then produces the exact ancestor-aware topology and checkpoint
manifest used by inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Mapping, Sequence

from .checkpoints import (
    ArtifactPin,
    INTERNAL_NODE_TREATMENT,
    NodeCheckpointBinding,
    POPULATION,
    TreeCheckpointManifest,
    build_tree_checkpoint_manifest,
)
from .covariates import (
    NodeCovariate,
    TreeNodeCovariates,
    build_tree_node_covariates,
)
from .manifest import NodeSpec, TreeTopology, build_tree_topology, canonical_sha256
from .symlink_inventory import SourcePathAuthority


ALLOWED_COHORT_KEYS = frozenset(
    {"llama_full", "llama_notrans", "flan_full", "flan_notrans"}
)
PLANNED_MAX_STEPS = 2000
_METHOD_ARTIFACT = {"full": "model", "lora": "merged"}


@dataclass(frozen=True)
class ResolverArtifactPins:
    """Observed pins for all supplied records and resolved checkpoints."""

    top_config: ArtifactPin
    training_summary: ArtifactPin
    completion_receipt: ArtifactPin
    run_list: ArtifactPin
    ledger: ArtifactPin
    truth_manifest: ArtifactPin
    checkpoints: Mapping[str, ArtifactPin]


@dataclass(frozen=True)
class ResolvedAllNodeTree:
    """One external behavior cohort mapped to one shared training cohort."""

    cohort_key: str
    training_cohort_id: str
    population: str
    internal_node_treatment: str
    topology: TreeTopology
    checkpoint_manifest: TreeCheckpointManifest
    node_covariates: TreeNodeCovariates
    resolution_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "weighttraits.all_node_resolved_tree.v1",
            "cohort_key": self.cohort_key,
            "training_cohort_id": self.training_cohort_id,
            "population": self.population,
            "internal_node_treatment": self.internal_node_treatment,
            "topology": self.topology.to_dict(),
            "checkpoint_manifest": self.checkpoint_manifest.to_dict(),
            "node_covariates": self.node_covariates.to_dict(),
            "resolution_sha256": self.resolution_sha256,
        }


def resolve_all_node_tree(
    *,
    cohort_key: str,
    training_cohort_id: str,
    path_base: str,
    summary_row: Mapping[str, Any],
    run_list_rows: Sequence[Mapping[str, Any]],
    ledger_rows: Sequence[Mapping[str, Any]],
    truth_rows: Sequence[Mapping[str, Any]],
    observed_pins: ResolverArtifactPins,
    expected_method: str,
    expected_artifact_name: str,
    base_model_id: str,
    base_model_revision: str,
    completion_receipt_base: str | None = None,
    source_authority: SourcePathAuthority | None = None,
    allowed_symlink_entries: Sequence[str] = (),
) -> ResolvedAllNodeTree:
    """Resolve one exact completed tree in run-list order."""

    key = _text(cohort_key, "cohort_key")
    if key not in ALLOWED_COHORT_KEYS:
        raise ValueError(f"unknown external cohort_key {key!r}")
    training_cohort = _text(training_cohort_id, "training_cohort_id")
    method = _text(expected_method, "expected_method")
    artifact_name = _text(expected_artifact_name, "expected_artifact_name")
    if _METHOD_ARTIFACT.get(method) != artifact_name:
        raise ValueError("expected method/artifact pairing is invalid")
    model_id = _text(base_model_id, "base_model_id")
    revision = _text(base_model_revision, "base_model_revision")
    stage_root = _stage_root(path_base)
    receipt_root = (
        stage_root
        if completion_receipt_base is None
        else _stage_root(completion_receipt_base)
    )
    pins = _pins(
        observed_pins,
        path_base=stage_root,
        completion_receipt_base=receipt_root,
        source_authority=source_authority,
        allowed_symlink_entries=allowed_symlink_entries,
    )
    summary = _summary(
        summary_row,
        pins=pins,
        path_base=stage_root,
        source_authority=source_authority,
        allowed_symlink_entries=allowed_symlink_entries,
    )
    tree_id = summary["tree_id"]

    runs = tuple(
        _run(
            row,
            path_base=stage_root,
            source_authority=source_authority,
            allowed_symlink_entries=allowed_symlink_entries,
        )
        for row in run_list_rows
    )
    if not runs:
        raise ValueError("run list is empty")
    run_ids = [row["node_id"] for row in runs]
    if "root" in run_ids:
        raise ValueError("untrained root must not appear in the run list")
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("run list contains duplicate node IDs")
    for ordinal, row in enumerate(runs):
        if row["array_index"] != ordinal:
            raise ValueError("run-list array_index must equal exact input order")
        if row["method"] != method:
            raise ValueError(f"run {row['node_id']!r} method/artifact drift")
        if artifact_name not in row["artifacts"]:
            raise ValueError(
                f"run {row['node_id']!r} lacks expected {artifact_name!r} artifact"
            )
        if row["base_model_id"] != model_id:
            raise ValueError(f"run {row['node_id']!r} base model ID drift")
        if row["base_model_revision"] != revision:
            raise ValueError(f"run {row['node_id']!r} base model revision drift")
        if row["max_steps"] != PLANNED_MAX_STEPS:
            raise ValueError(
                f"run {row['node_id']!r} trainer must be exactly max_steps=2000"
            )

    truths = tuple(_truth(row) for row in truth_rows)
    truth_ids = [row["node_id"] for row in truths]
    if "root" in truth_ids:
        raise ValueError("untrained root must not appear in the trained truth population")
    if len(truth_ids) != len(set(truth_ids)):
        raise ValueError("truth population contains duplicate node IDs")
    if set(truth_ids) != set(run_ids):
        raise ValueError("truth and run-list trained-node populations differ")
    truth_by_id = {row["node_id"]: row for row in truths}
    for run in runs:
        truth = truth_by_id[run["node_id"]]
        if truth["tree_id"] != tree_id:
            raise ValueError(f"truth tree_id drift for node {run['node_id']!r}")
        for field in ("parent_id", "depth", "path", "dataset_id", "task_family"):
            if truth[field] != run[field]:
                raise ValueError(f"run/truth {field} drift for node {run['node_id']!r}")

    if summary["n_runs"] != len(runs):
        raise ValueError("summary n_runs differs from exact run-list population")
    checkpoint_ids = set(pins.checkpoints)
    if checkpoint_ids != set(run_ids):
        raise ValueError("checkpoint pins and run-list node populations differ")

    latest = _latest_ledger(ledger_rows)
    if set(latest) != set(run_ids):
        raise ValueError("latest ledger and run-list node populations differ")
    for run in runs:
        node_id = run["node_id"]
        event = latest[node_id]
        if event["status"] != "completed":
            raise ValueError(
                f"latest ledger state for node {node_id!r} is not terminal completed"
            )
        extra = event["extra"]
        if "execution_overrides" not in extra:
            raise ValueError(
                f"completed ledger event for node {node_id!r} lacks execution_overrides"
            )
        overrides = extra["execution_overrides"]
        if not isinstance(overrides, Mapping) or dict(overrides) != {}:
            raise ValueError(
                f"completed ledger event for node {node_id!r} execution_overrides must be empty"
            )
        artifacts = extra.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ValueError(
                f"completed ledger event for node {node_id!r} lacks artifacts"
            )
        artifact_paths = _artifact_paths(
            artifacts,
            "ledger.extra.artifacts",
            path_base=stage_root,
            source_authority=source_authority,
            allowed_symlink_entries=allowed_symlink_entries,
        )
        checkpoint = pins.checkpoints[node_id]
        if run["artifacts"][artifact_name] != checkpoint.path:
            raise ValueError(f"run-list checkpoint path drift for node {node_id!r}")
        if artifact_paths.get(artifact_name) != checkpoint.path:
            raise ValueError(f"latest ledger checkpoint artifact drift for node {node_id!r}")
        if artifact_paths != run["artifacts"]:
            raise ValueError(
                f"latest ledger full artifact map drift for node {node_id!r}"
            )

    children = {node_id: [] for node_id in run_ids}
    for run in runs:
        parent = run["parent_id"]
        if parent != "root":
            if parent not in children:
                raise ValueError(f"node {run['node_id']!r} has missing trained parent")
            children[parent].append(run["node_id"])
    nodes = [
        NodeSpec(
            node_ordinal=ordinal,
            array_index=run["array_index"],
            node_id=run["node_id"],
            parent_id=run["parent_id"],
            depth=run["depth"],
            is_leaf=not children[run["node_id"]],
        )
        for ordinal, run in enumerate(runs)
    ]
    topology = build_tree_topology(tree_id, nodes)
    bindings = [
        NodeCheckpointBinding(
            population=POPULATION,
            internal_node_treatment=INTERNAL_NODE_TREATMENT,
            cohort_id=training_cohort,
            tree_id=tree_id,
            node_ordinal=node.node_ordinal,
            array_index=node.array_index,
            node_id=node.node_id,
            parent_id=node.parent_id,
            depth=node.depth,
            is_leaf=node.is_leaf,
            node_role=node.node_role,
            status="completed",
            method=method,
            artifact_name=artifact_name,
            checkpoint=pins.checkpoints[node.node_id],
            base_model_id=model_id,
            base_model_revision=revision,
            run_list=pins.run_list,
            ledger=pins.ledger,
            truth_manifest=pins.truth_manifest,
            top_config=pins.top_config,
            training_summary=pins.training_summary,
            completion_receipt=pins.completion_receipt,
        )
        for node in topology.nodes
    ]
    manifest = build_tree_checkpoint_manifest(
        topology,
        bindings,
        cohort_id=training_cohort,
    )
    covariates = build_tree_node_covariates(
        topology,
        manifest,
        [
            NodeCovariate(
                node_id=node.node_id,
                parent_id=node.parent_id,
                depth=node.depth,
                path=runs[node.node_ordinal]["path"],
                dataset_id=runs[node.node_ordinal]["dataset_id"],
                task_family=runs[node.node_ordinal]["task_family"],
                role=node.node_role,
            )
            for node in topology.nodes
        ],
        source_manifest=pins.truth_manifest,
    )
    core = {
        "cohort_key": key,
        "training_cohort_id": training_cohort,
        "population": POPULATION,
        "internal_node_treatment": INTERNAL_NODE_TREATMENT,
        "topology_sha256": topology.topology_sha256,
        "checkpoint_set_sha256": manifest.checkpoint_set_sha256,
        "node_covariates_sha256": covariates.covariates_sha256,
    }
    return ResolvedAllNodeTree(
        cohort_key=key,
        training_cohort_id=training_cohort,
        population=POPULATION,
        internal_node_treatment=INTERNAL_NODE_TREATMENT,
        topology=topology,
        checkpoint_manifest=manifest,
        node_covariates=covariates,
        resolution_sha256=canonical_sha256(core),
    )


def _summary(
    value: Mapping[str, Any],
    *,
    pins: ResolverArtifactPins,
    path_base: str,
    source_authority: SourcePathAuthority | None,
    allowed_symlink_entries: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("summary_row must be a mapping")
    required = {
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
    }
    _require_keys(value, required, "summary_row")
    if value["valid"] is not True or value["ready_for_analysis"] is not True:
        raise ValueError("summary row must be valid and ready_for_analysis")
    n_runs = _integer(value["n_runs"], "n_runs", minimum=1)
    counts = {
        field: _integer(value[field], field, minimum=0)
        for field in (
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
        )
    }
    if counts["n_events"] < n_runs:
        raise ValueError("summary n_events must be at least n_runs")
    for field in ("n_ledger_nodes", "n_terminal_nodes", "n_ok_nodes"):
        if counts[field] != n_runs:
            raise ValueError(f"summary {field} must equal n_runs")
    if any(counts[field] != 0 for field in (
        "n_failed_nodes",
        "n_missing_nodes",
        "n_errors",
        "n_warnings",
    )):
        raise ValueError("summary row must have zero errors and warnings")
    if counts["n_existing_artifacts"] != counts["n_expected_artifacts"]:
        raise ValueError("summary existing/expected artifact counts differ")
    status_counts = value["status_counts"]
    if not isinstance(status_counts, Mapping) or dict(status_counts) != {"completed": n_runs}:
        raise ValueError("summary status_counts must be completed-only for all runs")
    issue_counts = value["issue_counts"]
    if not isinstance(issue_counts, Mapping) or dict(issue_counts) != {}:
        raise ValueError("summary issue_counts must be empty")
    path_pins = {
        "run_list": pins.run_list,
        "ledger": pins.ledger,
        "manifest": pins.truth_manifest,
    }
    for field, pin in path_pins.items():
        if _source_path(
            value[field],
            f"summary.{field}",
            path_base=path_base,
            source_authority=source_authority,
            allowed_symlink_entries=allowed_symlink_entries,
        ) != pin.path:
            raise ValueError(f"summary row {field} path differs from observed pin")
    return {
        "tree_id": _text(value["tree_id"], "tree_id"),
        "n_runs": n_runs,
    }


def _run(
    value: Mapping[str, Any],
    *,
    path_base: str,
    source_authority: SourcePathAuthority | None,
    allowed_symlink_entries: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("run-list row must be a mapping")
    required = {
        "array_index",
        "node_id",
        "parent_id",
        "depth",
        "dataset_id",
        "task_family",
        "method",
        "expected_artifacts",
        "job",
    }
    _require_keys(value, required, "run-list row")
    if "execution_overrides" in value:
        raise ValueError("run-list row must not contain execution_overrides")
    job = value["job"]
    if not isinstance(job, Mapping):
        raise ValueError("run job must be a mapping")
    job_required = {
        "node_id",
        "parent_id",
        "depth",
        "path",
        "dataset_id",
        "task_family",
        "method",
        "expected_artifacts",
        "base_model",
        "base_model_revision",
        "trainer",
    }
    _require_keys(job, job_required, "run job")
    if "execution_overrides" in job:
        raise ValueError("run job must not contain execution_overrides")
    trainer = job["trainer"]
    if not isinstance(trainer, Mapping) or "max_steps" not in trainer:
        raise ValueError("run job trainer lacks max_steps")
    outer_artifacts = _artifact_paths(
        value["expected_artifacts"],
        "expected_artifacts",
        path_base=path_base,
        source_authority=source_authority,
        allowed_symlink_entries=allowed_symlink_entries,
    )
    job_artifacts = _artifact_paths(
        job["expected_artifacts"],
        "job.expected_artifacts",
        path_base=path_base,
        source_authority=source_authority,
        allowed_symlink_entries=allowed_symlink_entries,
    )
    if outer_artifacts != job_artifacts:
        raise ValueError("run/job expected_artifacts drift")
    outer = {
        "node_id": _text(value["node_id"], "node_id"),
        "parent_id": _text(value["parent_id"], "parent_id"),
        "depth": _integer(value["depth"], "depth", minimum=1),
        "dataset_id": _text(value["dataset_id"], "dataset_id"),
        "task_family": _text(value["task_family"], "task_family"),
        "method": _text(value["method"], "method"),
    }
    nested = {
        "node_id": _text(job["node_id"], "job.node_id"),
        "parent_id": _text(job["parent_id"], "job.parent_id"),
        "depth": _integer(job["depth"], "job.depth", minimum=1),
        "dataset_id": _text(job["dataset_id"], "job.dataset_id"),
        "task_family": _text(job["task_family"], "job.task_family"),
        "method": _text(job["method"], "job.method"),
    }
    for field in outer:
        if outer[field] != nested[field]:
            raise ValueError(f"run/job {field} drift")
    path = _node_path(job["path"], node_id=outer["node_id"], label="job.path")
    if path[-2] != outer["parent_id"] or len(path) != outer["depth"] + 1:
        raise ValueError("run job path is inconsistent with parent/depth")
    return {
        "array_index": _integer(value["array_index"], "array_index", minimum=0),
        **outer,
        "path": path,
        "artifacts": outer_artifacts,
        "base_model_id": _text(job["base_model"], "job.base_model"),
        "base_model_revision": _text(
            job["base_model_revision"], "job.base_model_revision"
        ),
        "max_steps": _integer(trainer["max_steps"], "job.trainer.max_steps", minimum=1),
    }


def _truth(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("truth row must be a mapping")
    required = {
        "tree_id",
        "node_id",
        "parent_id",
        "depth",
        "path",
        "dataset_id",
        "task_family",
    }
    _require_keys(value, required, "truth row")
    node_id = _text(value["node_id"], "node_id")
    parent_id = _text(value["parent_id"], "parent_id")
    depth = _integer(value["depth"], "depth", minimum=1)
    path = _node_path(value["path"], node_id=node_id, label="truth.path")
    if path[-2] != parent_id or len(path) != depth + 1:
        raise ValueError("truth path is inconsistent with parent/depth")
    return {
        "tree_id": _text(value["tree_id"], "tree_id"),
        "node_id": node_id,
        "parent_id": parent_id,
        "depth": depth,
        "path": path,
        "dataset_id": _text(value["dataset_id"], "dataset_id"),
        "task_family": _text(value["task_family"], "task_family"),
    }


def _latest_ledger(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    if not rows:
        raise ValueError("ledger is empty")
    latest: dict[str, dict[str, Any]] = {}
    for value in rows:
        if not isinstance(value, Mapping):
            raise ValueError("ledger row must be a mapping")
        required = {"node_id", "status", "extra"}
        _require_keys(value, required, "ledger row")
        extra = value["extra"]
        if not isinstance(extra, Mapping):
            raise ValueError("ledger extra must be a mapping")
        node_id = _text(value["node_id"], "ledger.node_id")
        if node_id == "root":
            raise ValueError("untrained root must not appear in the ledger")
        latest[node_id] = {
            "status": _text(value["status"], "ledger.status"),
            "extra": dict(extra),
        }
    return latest


def _pins(
    value: ResolverArtifactPins,
    *,
    path_base: str,
    completion_receipt_base: str,
    source_authority: SourcePathAuthority | None,
    allowed_symlink_entries: Sequence[str],
) -> ResolverArtifactPins:
    if not isinstance(value, ResolverArtifactPins):
        raise ValueError("observed_pins must be ResolverArtifactPins")
    checkpoints = {
        _text(node_id, "checkpoint node_id"): _pin_under(
            pin,
            "checkpoint",
            path_base=path_base,
            source_authority=source_authority,
            allowed_symlink_entries=allowed_symlink_entries,
            allow_authorized_physical=True,
        )
        for node_id, pin in value.checkpoints.items()
    }
    if len(checkpoints) != len(value.checkpoints):
        raise ValueError("checkpoint pins contain duplicate normalized node IDs")
    return ResolverArtifactPins(
        top_config=_pin_under(value.top_config, "top_config", path_base=path_base),
        training_summary=_pin_under(
            value.training_summary, "training_summary", path_base=path_base
        ),
        completion_receipt=_pin_under(
            value.completion_receipt,
            "completion_receipt",
            path_base=completion_receipt_base,
        ),
        run_list=_pin_under(
            value.run_list,
            "run_list",
            path_base=path_base,
            source_authority=source_authority,
            allowed_symlink_entries=allowed_symlink_entries,
            allow_authorized_physical=True,
        ),
        ledger=_pin_under(
            value.ledger,
            "ledger",
            path_base=path_base,
            source_authority=source_authority,
            allowed_symlink_entries=allowed_symlink_entries,
            allow_authorized_physical=True,
        ),
        truth_manifest=_pin_under(
            value.truth_manifest,
            "truth_manifest",
            path_base=path_base,
            source_authority=source_authority,
            allowed_symlink_entries=allowed_symlink_entries,
            allow_authorized_physical=True,
        ),
        checkpoints=checkpoints,
    )


def _pin(value: object, label: str) -> ArtifactPin:
    if not isinstance(value, ArtifactPin):
        raise ValueError(f"{label} must be an ArtifactPin")
    # Checkpoint manifest construction applies strict path/SHA validation too;
    # perform minimal normalization here before comparing summary declarations.
    _absolute_path(value.path, f"{label}.path")
    if len(value.sha256) != 64 or any(ch not in "0123456789abcdef" for ch in value.sha256):
        raise ValueError(f"{label}.sha256 must be a lowercase SHA-256")
    return value


def _pin_under(
    value: object,
    label: str,
    *,
    path_base: str,
    source_authority: SourcePathAuthority | None = None,
    allowed_symlink_entries: Sequence[str] = (),
    allow_authorized_physical: bool = False,
) -> ArtifactPin:
    pin = _pin(value, label)
    if _is_strictly_below(pin.path, path_base):
        _stage_path(pin.path, f"{label}.path", path_base=path_base)
    elif not allow_authorized_physical or source_authority is None:
        _stage_path(pin.path, f"{label}.path", path_base=path_base)
    else:
        mapping = source_authority.mapping_for_physical(pin.path)
        if (
            mapping is None
            or mapping.logical_prefix not in set(allowed_symlink_entries)
        ):
            raise ValueError(f"{label}.path is outside exact source authority")
        if os.path.realpath(pin.path) != pin.path:
            raise ValueError(f"{label}.path must be canonical physical path")
    return pin


def _artifact_paths(
    value: object,
    label: str,
    *,
    path_base: str,
    source_authority: SourcePathAuthority | None,
    allowed_symlink_entries: Sequence[str],
) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be a nonempty mapping")
    return {
        _text(name, f"{label} name"): _source_path(
            path,
            f"{label}.{name}",
            path_base=path_base,
            source_authority=source_authority,
            allowed_symlink_entries=allowed_symlink_entries,
        )
        for name, path in value.items()
    }


def _node_path(value: object, *, node_id: str, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a node sequence")
    path = tuple(_text(item, f"{label} item") for item in value)
    if len(path) < 2 or path[0] != "root" or path[-1] != node_id:
        raise ValueError(f"{label} must run from root through its declared node")
    if len(path) != len(set(path)):
        raise ValueError(f"{label} must not repeat nodes")
    return path


def _require_keys(value: Mapping[str, Any], required: set[str], label: str) -> None:
    missing = required - set(value)
    if missing:
        raise ValueError(f"{label} lacks required keys {sorted(missing)!r}")


def _absolute_path(value: object, label: str) -> str:
    import os

    path = _text(value, label)
    if not os.path.isabs(path) or os.path.normpath(path) != path:
        raise ValueError(f"{label} must be an absolute normalized path")
    return path


def _stage_root(value: object) -> str:
    root = _absolute_path(value, "path_base")
    if root == "/":
        raise ValueError("path_base must be a bounded frozen stage root")
    return root


def _stage_path(value: object, label: str, *, path_base: str) -> str:
    import os

    raw = _text(value, label)
    if os.path.isabs(raw):
        if os.path.normpath(raw) != raw:
            raise ValueError(f"{label} absolute path must be normalized")
        resolved = raw
    else:
        if os.path.normpath(raw) != raw or raw == ".":
            raise ValueError(f"{label} relative path must be normalized")
        resolved = os.path.normpath(os.path.join(path_base, raw))
    try:
        within_root = os.path.commonpath((path_base, resolved)) == path_base
    except ValueError as exc:
        raise ValueError(f"{label} is not under path_base") from exc
    if not within_root or resolved == path_base:
        raise ValueError(f"{label} escapes or does not name an artifact under path_base")
    return resolved


def _source_path(
    value: object,
    label: str,
    *,
    path_base: str,
    source_authority: SourcePathAuthority | None,
    allowed_symlink_entries: Sequence[str],
) -> str:
    logical = _stage_path(value, label, path_base=path_base)
    observed = os.path.realpath(logical)
    if observed == logical:
        return logical
    if source_authority is None:
        raise ValueError(f"{label} crosses an unlisted symlink")
    resolution = source_authority.resolve(logical)
    mapping = resolution.mapping
    if (
        mapping is None
        or mapping.logical_prefix not in set(allowed_symlink_entries)
        or resolution.physical_path != observed
    ):
        raise ValueError(f"{label} lacks an exact allowed source mapping")
    return resolution.physical_path


def _is_strictly_below(path: str, root: str) -> bool:
    try:
        return os.path.commonpath((root, path)) == root and path != root
    except ValueError:
        return False


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a nonempty string without surrounding whitespace")
    return value


def _integer(value: object, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value
