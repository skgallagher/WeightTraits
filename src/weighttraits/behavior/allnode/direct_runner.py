"""Strict filesystem runner for the production all-node direct endpoint.

One self-hashed job config binds the exact fifty resolved trees, the sampled
checkpoint-identity index, and the two frozen WeightTraits source files used
to open checkpoints and build streaming distance cubes.  The job computes
only the three direct all-node matrices and their topology-derived leaf
slices.  Each tree is published as one atomic directory; the cohort becomes
terminal only after all fifty tree directories validate byte-for-byte.

Checkpoint directories use the pinned sampled-identity scheme: the index is
validated structurally at job start, then every checkpoint in a fresh tree is
re-attested immediately before readers open it and again after the tree's
distance computation, before any output is published.  The index reader is
injectable for tests, but the production index file is always SHA-256 verified
and must contain the full authority-bearing identities.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Callable, Mapping, Sequence
import uuid

from .checkpoint_identity import (
    CheckpointIdentity,
    read_identity_index,
    source_authority_from_identity,
    validate_checkpoint_identity,
)
from .checkpoints import (
    ArtifactPin,
    INTERNAL_NODE_TREATMENT,
    POPULATION,
    PRODUCTION_COHORT_IDS,
    PRODUCTION_N_INTERNAL,
    PRODUCTION_N_LEAVES,
    PRODUCTION_N_NODES,
    PRODUCTION_N_TREES,
    read_tree_checkpoint_manifest,
)
from .covariates import read_tree_node_covariates
from .direct_execution import (
    DIRECT_ARTIFACT_RECEIPT_SCHEMA,
    DIRECT_EXECUTION_SCHEMA,
    DIRECT_METRICS,
    DIRECT_VALUES_SCHEMA,
    DISTANCE_ENGINE,
    LAYER_AGGREGATION,
    REPRESENTATION,
    execute_direct_distances,
    pinned_snapshot_adapter,
    plan_direct_artifacts,
)
from .direct_compatibility import (
    LeafCompatibilityAPI,
    LeafCompatibilitySpec,
    RECEIPT_BASENAME as LEAF_COMPATIBILITY_BASENAME,
    execute_leaf_compatibility,
    read_leaf_compatibility_spec,
    validate_leaf_compatibility_receipt,
)
from .manifest import canonical_sha256, read_tree_topology
from .matrices import validate_matrix_receipt
from .production import PRODUCTION_TREE_IDS
from .provenance import attest_runtime_environment
from .resolver import ResolvedAllNodeTree
from .symlink_inventory import (
    SourcePathAuthority,
    source_path_authority_from_dict,
)


PLAN_SCHEMA = "weighttraits.all_node_direct_cohort_plan.v2"
JOB_CONFIG_SCHEMA = "weighttraits.all_node_direct_job_config.v1"
TREE_TERMINAL_SCHEMA = "weighttraits.all_node_direct_tree_terminal.v1"
COHORT_TERMINAL_SCHEMA = "weighttraits.all_node_direct_cohort_terminal.v1"
CLI_STATUS_SCHEMA = "weighttraits.all_node_direct_cli_status.v1"
PLAN_BASENAME = "direct_plan.json"
PLAN_RELATIVE_ROOT = Path("deployments/allnode_v1/runtime_inputs")
JOB_CONFIG_RELATIVE_ROOT = Path("deployments/allnode_v1/configs/jobs")
OUTPUT_RELATIVE_ROOT = Path("outputs/allnode_v1/direct")
TREE_TERMINAL_BASENAME = "terminal_receipt.json"
COHORT_TERMINAL_BASENAME = "terminal_receipt.json"
LEAF_COMPATIBILITY_DIRNAME = "leaf_compatibility"
DEFAULT_CHUNK_SIZE = 1_000_000

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True)
class DirectCohortPlan:
    cohort_key: str
    training_cohort_id: str
    source_stage_root: str
    output_dir: str
    snapshot_root: str
    readers_source: ArtifactPin
    streaming_source: ArtifactPin
    checkpoint_identity_index: ArtifactPin
    source_path_authority: SourcePathAuthority | None
    leaf_compatibility: LeafCompatibilitySpec
    trees: tuple[ResolvedAllNodeTree, ...]
    resolution_scope_sha256: str
    checkpoint_scope_sha256: str
    node_covariate_scope_sha256: str
    chunk_size: int
    direct_plan_sha256: str

    def to_dict(self) -> dict[str, Any]:
        core = _plan_core(
            cohort_key=self.cohort_key,
            training_cohort_id=self.training_cohort_id,
            source_stage_root=self.source_stage_root,
            output_dir=self.output_dir,
            snapshot_root=self.snapshot_root,
            readers_source=self.readers_source,
            streaming_source=self.streaming_source,
            checkpoint_identity_index=self.checkpoint_identity_index,
            source_path_authority=self.source_path_authority,
            leaf_compatibility=self.leaf_compatibility,
            trees=self.trees,
            resolution_scope_sha256=self.resolution_scope_sha256,
            checkpoint_scope_sha256=self.checkpoint_scope_sha256,
            node_covariate_scope_sha256=self.node_covariate_scope_sha256,
            chunk_size=self.chunk_size,
        )
        return {**core, "direct_plan_sha256": self.direct_plan_sha256}


@dataclass(frozen=True)
class DirectJobConfig:
    cohort_key: str
    logical_id: str
    output_dir: str
    direct_plan: ArtifactPin
    job_config_sha256: str

    def to_dict(self) -> dict[str, Any]:
        core = {
            "schema": JOB_CONFIG_SCHEMA,
            "valid": True,
            "kind": "direct_allnode_cpu",
            "cohort_key": self.cohort_key,
            "logical_id": self.logical_id,
            "output_dir": self.output_dir,
            "direct_plan": self.direct_plan.to_dict(),
            "execution_overrides": {},
        }
        return {**core, "job_config_sha256": self.job_config_sha256}


@dataclass(frozen=True)
class WeightTraitsDirectAPI:
    reader_from_path: Callable[..., object]
    build_distance_cube: Callable[..., object]


IdentityIndexReader = Callable[[Mapping[str, Any]], Mapping[str, object]]


@dataclass(frozen=True)
class FilesystemDirectResult:
    cache_hit: bool
    plan: DirectCohortPlan
    plan_path: Path
    plan_file_sha256: str
    job_config_path: Path
    job_config_file_sha256: str
    output_dir: Path
    terminal_receipt: Mapping[str, Any]

    def status_dict(self) -> dict[str, Any]:
        return {
            "schema": CLI_STATUS_SCHEMA,
            "valid": True,
            "status": "cached" if self.cache_hit else "completed",
            "cache_hit": self.cache_hit,
            "population": POPULATION,
            "internal_node_treatment": INTERNAL_NODE_TREATMENT,
            "cohort_key": self.plan.cohort_key,
            "training_cohort_id": self.plan.training_cohort_id,
            "direct_plan": str(self.plan_path),
            "direct_plan_file_sha256": self.plan_file_sha256,
            "direct_plan_sha256": self.plan.direct_plan_sha256,
            "job_config": str(self.job_config_path),
            "job_config_file_sha256": self.job_config_file_sha256,
            "output_dir": str(self.output_dir),
            "cohort_terminal_sha256": self.terminal_receipt[
                "cohort_terminal_sha256"
            ],
            "n_trees": self.terminal_receipt["n_trees"],
            "n_nodes": self.terminal_receipt["n_nodes"],
            "n_internal": self.terminal_receipt["n_internal"],
            "n_leaves": self.terminal_receipt["n_leaves"],
            "execution_overrides": {},
        }


def build_direct_cohort_plan(
    *,
    cohort_key: str,
    source_stage_root: str,
    output_dir: str,
    snapshot_root: str,
    readers_source: ArtifactPin | Mapping[str, Any],
    streaming_source: ArtifactPin | Mapping[str, Any],
    checkpoint_identity_index: ArtifactPin | Mapping[str, Any],
    source_path_authority: SourcePathAuthority | None = None,
    leaf_compatibility: LeafCompatibilitySpec | Mapping[str, Any],
    trees: Sequence[ResolvedAllNodeTree],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> DirectCohortPlan:
    """Build the sole exact 50-tree/641-node direct job contract."""

    key = _text(cohort_key, "cohort_key")
    if key not in PRODUCTION_COHORT_IDS:
        raise ValueError(f"unknown production cohort_key {key!r}")
    source_root = _root(source_stage_root, "source_stage_root")
    destination = _root(output_dir, "output_dir")
    if tuple(Path(destination).parts[-4:]) != ("outputs", "allnode_v1", "direct", key):
        raise ValueError(
            "output_dir must end with outputs/allnode_v1/direct/<cohort_key>"
        )
    snapshot = _root(snapshot_root, "snapshot_root")
    readers_pin = _pin(readers_source, "readers_source")
    streaming_pin = _pin(streaming_source, "streaming_source")
    for label, pin in (
        ("readers_source", readers_pin),
        ("streaming_source", streaming_pin),
    ):
        if not _strictly_below(pin.path, snapshot):
            raise ValueError(f"{label} must be strictly below snapshot_root")
    identity_pin = _pin(checkpoint_identity_index, "checkpoint_identity_index")
    authority = source_path_authority
    if authority is not None:
        if not isinstance(authority, SourcePathAuthority):
            raise ValueError("source_path_authority type drift")
        authority = authority.reattest()
        if authority.stage_root != source_root:
            raise ValueError("source_path_authority stage root drift")
    compatibility = (
        leaf_compatibility
        if isinstance(leaf_compatibility, LeafCompatibilitySpec)
        else read_leaf_compatibility_spec(leaf_compatibility)
    )
    compatibility = read_leaf_compatibility_spec(compatibility.to_dict())
    if compatibility.comparison_available:
        for label, pin in (
            ("reconstruct_source", compatibility.reconstruct_source),
            ("recovery_source", compatibility.recovery_source),
            ("splits_source", compatibility.splits_source),
        ):
            assert pin is not None
            if not _strictly_below(pin.path, snapshot):
                raise ValueError(f"leaf compatibility {label} escapes snapshot_root")
    verified = tuple(_resolved(value, cohort_key=key) for value in trees)
    if len(verified) != PRODUCTION_N_TREES:
        raise ValueError(f"direct plan requires exactly {PRODUCTION_N_TREES} trees")
    if tuple(tree.topology.tree_id for tree in verified) != PRODUCTION_TREE_IDS:
        raise ValueError("direct plan tree order must be exactly trees 001..050")
    training_ids = {tree.training_cohort_id for tree in verified}
    if len(training_ids) != 1:
        raise ValueError("direct plan mixes training cohort identities")
    totals = (
        sum(tree.topology.n_nodes for tree in verified),
        sum(tree.topology.n_internal for tree in verified),
        sum(tree.topology.n_leaves for tree in verified),
    )
    expected = (PRODUCTION_N_NODES, PRODUCTION_N_INTERNAL, PRODUCTION_N_LEAVES)
    if totals != expected:
        raise ValueError(
            f"direct plan all-node totals differ: observed={totals}, expected={expected}"
        )
    checkpoint_paths: list[str] = []
    for tree in verified:
        for binding in tree.checkpoint_manifest.bindings:
            path = binding.checkpoint.path
            if _strictly_below(path, source_root):
                if authority is not None:
                    logical = authority.logical_for_physical(path)
                    if logical != path or authority.authorize(logical, path) is not None:
                        raise ValueError(
                            "physical stage checkpoint unexpectedly uses a source mapping"
                        )
            else:
                if authority is None:
                    raise ValueError(
                        "checkpoint binding escapes source_stage_root without authority"
                    )
                logical = authority.logical_for_physical(path)
                mapping = authority.authorize(logical, path)
                if mapping is None or not _strictly_below(path, mapping.physical_root):
                    raise ValueError(
                        "external checkpoint lacks a bounded exact source-path mapping"
                    )
            checkpoint_paths.append(path)
    if len(checkpoint_paths) != len(set(checkpoint_paths)):
        raise ValueError("direct plan reuses a checkpoint path")
    size = _integer(chunk_size, "chunk_size", minimum=1)
    resolution_scope = [
        {
            "tree_id": tree.topology.tree_id,
            "resolution_sha256": tree.resolution_sha256,
            "topology_sha256": tree.topology.topology_sha256,
            "pair_index_sha256": tree.topology.pair_index_sha256,
            "leaf_subset_sha256": tree.topology.leaf_subset_sha256,
        }
        for tree in verified
    ]
    checkpoint_scope = [
        tree.checkpoint_manifest.checkpoint_set_sha256 for tree in verified
    ]
    covariate_scope = [tree.node_covariates.covariates_sha256 for tree in verified]
    resolution_sha = canonical_sha256(resolution_scope)
    checkpoint_sha = canonical_sha256(checkpoint_scope)
    covariate_sha = canonical_sha256(covariate_scope)
    core = _plan_core(
        cohort_key=key,
        training_cohort_id=next(iter(training_ids)),
        source_stage_root=source_root,
        output_dir=destination,
        snapshot_root=snapshot,
        readers_source=readers_pin,
        streaming_source=streaming_pin,
        checkpoint_identity_index=identity_pin,
        source_path_authority=authority,
        leaf_compatibility=compatibility,
        trees=verified,
        resolution_scope_sha256=resolution_sha,
        checkpoint_scope_sha256=checkpoint_sha,
        node_covariate_scope_sha256=covariate_sha,
        chunk_size=size,
    )
    return DirectCohortPlan(
        cohort_key=key,
        training_cohort_id=next(iter(training_ids)),
        source_stage_root=source_root,
        output_dir=destination,
        snapshot_root=snapshot,
        readers_source=readers_pin,
        streaming_source=streaming_pin,
        checkpoint_identity_index=identity_pin,
        source_path_authority=authority,
        leaf_compatibility=compatibility,
        trees=verified,
        resolution_scope_sha256=resolution_sha,
        checkpoint_scope_sha256=checkpoint_sha,
        node_covariate_scope_sha256=covariate_sha,
        chunk_size=size,
        direct_plan_sha256=canonical_sha256(core),
    )


def read_direct_cohort_plan(payload: Mapping[str, Any]) -> DirectCohortPlan:
    if not isinstance(payload, Mapping):
        raise ValueError("direct cohort plan must be a mapping")
    required = {
        "schema", "valid", "population", "internal_node_treatment",
        "exclude_root", "direct_matrix_only", "neighbor_joining", "tree_builder",
        "cohort_key", "training_cohort_id", "source_stage_root", "output_dir",
        "snapshot_root", "readers_source", "streaming_source",
        "checkpoint_identity_index", "source_path_authority", "leaf_compatibility",
        "n_trees", "n_nodes", "n_internal",
        "n_leaves", "tree_ids", "metrics", "distance_engine", "representation",
        "layer_aggregation", "chunk_size", "trees", "resolution_scope_sha256",
        "checkpoint_scope_sha256", "node_covariate_scope_sha256",
        "execution_overrides", "direct_plan_sha256",
    }
    if set(payload) != required:
        raise ValueError(f"direct cohort plan keys must be exactly {sorted(required)!r}")
    if payload.get("schema") != PLAN_SCHEMA:
        raise ValueError("direct cohort plan schema drift")
    raw_trees = payload.get("trees")
    if not isinstance(raw_trees, list):
        raise ValueError("direct cohort plan trees must be a list")
    trees = tuple(_read_resolved_tree(value) for value in raw_trees)
    raw_authority = payload.get("source_path_authority")
    authority = (
        None
        if raw_authority is None
        else source_path_authority_from_dict(raw_authority)
    )
    rebuilt = build_direct_cohort_plan(
        cohort_key=payload.get("cohort_key"),
        source_stage_root=payload.get("source_stage_root"),
        output_dir=payload.get("output_dir"),
        snapshot_root=payload.get("snapshot_root"),
        readers_source=payload.get("readers_source"),
        streaming_source=payload.get("streaming_source"),
        checkpoint_identity_index=payload.get("checkpoint_identity_index"),
        source_path_authority=authority,
        leaf_compatibility=payload.get("leaf_compatibility"),
        trees=trees,
        chunk_size=payload.get("chunk_size"),
    )
    if canonical_sha256(dict(payload)) != canonical_sha256(rebuilt.to_dict()):
        raise ValueError("saved direct cohort plan differs from recomputed contract")
    return rebuilt


def build_direct_job_config(
    *,
    cohort_key: str,
    output_dir: str,
    direct_plan: ArtifactPin | Mapping[str, Any],
) -> DirectJobConfig:
    key = _text(cohort_key, "cohort_key")
    if key not in PRODUCTION_COHORT_IDS:
        raise ValueError("direct job config cohort_key drift")
    destination = _root(output_dir, "output_dir")
    if tuple(Path(destination).parts[-4:]) != (
        "outputs", "allnode_v1", "direct", key
    ):
        raise ValueError("direct job config output_dir drift")
    plan_pin = _pin(direct_plan, "direct_plan")
    if tuple(Path(plan_pin.path).parts[-5:]) != (
        "deployments", "allnode_v1", "runtime_inputs", key, PLAN_BASENAME
    ):
        raise ValueError("direct job config plan pin path drift")
    core = {
        "schema": JOB_CONFIG_SCHEMA,
        "valid": True,
        "kind": "direct_allnode_cpu",
        "cohort_key": key,
        "logical_id": f"direct_{key}",
        "output_dir": destination,
        "direct_plan": plan_pin.to_dict(),
        "execution_overrides": {},
    }
    return DirectJobConfig(
        cohort_key=key,
        logical_id=f"direct_{key}",
        output_dir=destination,
        direct_plan=plan_pin,
        job_config_sha256=canonical_sha256(core),
    )


def read_direct_job_config(payload: Mapping[str, Any]) -> DirectJobConfig:
    required = {
        "schema", "valid", "kind", "cohort_key", "logical_id",
        "output_dir", "direct_plan", "execution_overrides", "job_config_sha256",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError("direct job config keys drift")
    if payload.get("schema") != JOB_CONFIG_SCHEMA:
        raise ValueError("direct job config schema drift")
    rebuilt = build_direct_job_config(
        cohort_key=payload.get("cohort_key"),
        output_dir=payload.get("output_dir"),
        direct_plan=payload.get("direct_plan"),
    )
    if canonical_sha256(dict(payload)) != canonical_sha256(rebuilt.to_dict()):
        raise ValueError("direct job config differs from recomputed contract")
    return rebuilt


def load_direct_cohort_plan(
    path: str | Path,
) -> tuple[DirectCohortPlan, Path, Path, str]:
    raw_path = Path(path)
    absolute = Path(os.path.abspath(str(raw_path)))
    resolved = raw_path.resolve(strict=True)
    if absolute != resolved or resolved.is_symlink() or not resolved.is_file():
        raise ValueError("direct config path must be a canonical regular file without symlinks")
    if resolved.name != PLAN_BASENAME:
        raise ValueError("direct runtime plan basename drift")
    if (
        resolved.parent.parent.name != "runtime_inputs"
        or resolved.parent.parent.parent.name != "allnode_v1"
        or resolved.parent.parent.parent.parent.name != "deployments"
    ):
        raise ValueError(
            "direct plan must be at deployments/allnode_v1/runtime_inputs/"
            "<cohort>/direct_plan.json"
        )
    stage_root = resolved.parent.parent.parent.parent.parent
    raw = resolved.read_bytes()
    plan = read_direct_cohort_plan(_json_object(raw, "direct cohort plan"))
    if resolved.parent.name != plan.cohort_key:
        raise ValueError("direct plan directory differs from cohort_key")
    if Path(plan.output_dir) != stage_root / OUTPUT_RELATIVE_ROOT / plan.cohort_key:
        raise ValueError("direct plan output_dir differs from fixed stage output")
    if raw != _canonical_bytes(plan.to_dict()):
        raise ValueError("direct runtime plan is not canonical bytes")
    return plan, resolved, stage_root, hashlib.sha256(raw).hexdigest()


def load_direct_job_config(
    path: str | Path,
) -> tuple[DirectJobConfig, DirectCohortPlan, Path, str, Path, str]:
    raw_path = Path(path)
    absolute = Path(os.path.abspath(str(raw_path)))
    resolved = raw_path.resolve(strict=True)
    if absolute != resolved or resolved.is_symlink() or not resolved.is_file():
        raise ValueError("direct job config must be canonical and contain no symlinks")
    if (
        resolved.parent.name != "jobs"
        or resolved.parent.parent.name != "configs"
        or resolved.parent.parent.parent.name != "allnode_v1"
        or resolved.parent.parent.parent.parent.name != "deployments"
    ):
        raise ValueError(
            "direct job config must be at deployments/allnode_v1/configs/jobs/"
            "direct_<cohort>.json"
        )
    raw = resolved.read_bytes()
    config = read_direct_job_config(_json_object(raw, "direct job config"))
    if resolved.name != f"direct_{config.cohort_key}.json":
        raise ValueError("direct job config filename differs from cohort_key")
    if raw != _canonical_bytes(config.to_dict()):
        raise ValueError("direct job config is not canonical bytes")
    _verify_file_pin(config.direct_plan)
    plan, plan_path, stage_root, plan_file_sha = load_direct_cohort_plan(
        config.direct_plan.path
    )
    if (
        plan.cohort_key != config.cohort_key
        or plan.output_dir != config.output_dir
        or plan_file_sha != config.direct_plan.sha256
    ):
        raise ValueError("direct job config and runtime plan scope drift")
    if resolved.parent.parent.parent.parent.parent != stage_root:
        raise ValueError("direct job config and runtime plan stage roots differ")
    return (
        config,
        plan,
        resolved,
        hashlib.sha256(raw).hexdigest(),
        plan_path,
        plan_file_sha,
    )


def execute_filesystem_direct(
    config_path: str | Path,
    *,
    api: WeightTraitsDirectAPI | None = None,
    identity_index_reader: IdentityIndexReader | None = None,
    compatibility_api: LeafCompatibilityAPI | None = None,
) -> FilesystemDirectResult:
    """Execute or validate one exact cohort, with no scheduler side effects."""

    (
        job_config,
        plan,
        job_config_path,
        job_config_file_sha,
        saved_path,
        plan_file_sha,
    ) = load_direct_job_config(config_path)
    _verify_file_pin(plan.readers_source)
    _verify_file_pin(plan.streaming_source)
    identity_lookup = _load_identity_lookup(
        plan.checkpoint_identity_index,
        reader=identity_index_reader,
    )
    _verify_checkpoint_scope(plan, identity_lookup)
    source_authority = _load_runtime_source_authority(identity_lookup)
    if source_authority != plan.source_path_authority:
        raise ValueError(
            "checkpoint identity source authority differs from direct plan"
        )

    output_root = Path(plan.output_dir)
    if os.path.lexists(output_root) and output_root.is_symlink():
        raise ValueError("direct output root must not be a symlink")
    output_root.mkdir(parents=True, exist_ok=True)
    if output_root.resolve(strict=True) != output_root:
        raise ValueError("direct output root must be canonical")
    cohort_terminal_path = output_root / COHORT_TERMINAL_BASENAME

    existing_dirs = _validate_output_entries(plan, output_root)
    if cohort_terminal_path.exists():
        if len(existing_dirs) != PRODUCTION_N_TREES:
            raise ValueError("terminal direct cohort is missing tree directories")
        tree_rows = tuple(
            _validate_tree_cache(
                plan,
                tree,
                output_root / tree.topology.tree_id,
                plan_path=saved_path,
                plan_file_sha256=plan_file_sha,
            )
            for tree in plan.trees
        )
        compatibility = _execute_or_validate_compatibility(
            plan,
            output_root=output_root,
            api=compatibility_api,
        )
        terminal = _validate_cohort_terminal(
            cohort_terminal_path,
            plan=plan,
            plan_path=saved_path,
            plan_file_sha256=plan_file_sha,
            tree_rows=tree_rows,
            compatibility=compatibility,
        )
        return FilesystemDirectResult(
            cache_hit=True,
            plan=plan,
            plan_path=saved_path,
            plan_file_sha256=plan_file_sha,
            job_config_path=job_config_path,
            job_config_file_sha256=job_config_file_sha,
            output_dir=output_root,
            terminal_receipt=terminal,
        )

    selected_api: WeightTraitsDirectAPI | None = api
    tree_rows: list[Mapping[str, Any]] = []
    any_fresh = False
    for tree in plan.trees:
        tree_dir = output_root / tree.topology.tree_id
        if tree.topology.tree_id in existing_dirs:
            tree_rows.append(
                _validate_tree_cache(
                    plan,
                    tree,
                    tree_dir,
                    plan_path=saved_path,
                    plan_file_sha256=plan_file_sha,
                )
            )
            continue
        if selected_api is None:
            selected_api = _load_pinned_weighttraits_api(plan)
        adapter = pinned_snapshot_adapter(
            snapshot_root=plan.snapshot_root,
            readers_source=plan.readers_source,
            streaming_source=plan.streaming_source,
            reader_from_path=selected_api.reader_from_path,
            build_distance_cube=selected_api.build_distance_cube,
            source_pins_verified=True,
        )
        before_identity_scope = _reattest_tree_checkpoint_identities(
            tree,
            identity_lookup,
            source_authority=source_authority,
        )
        execution = execute_direct_distances(
            tree,
            adapter=adapter,
            metrics=DIRECT_METRICS,
            chunk_size=plan.chunk_size,
        )
        after_identity_scope = _reattest_tree_checkpoint_identities(
            tree,
            identity_lookup,
            source_authority=source_authority,
        )
        if after_identity_scope != before_identity_scope:
            raise ValueError(
                f"checkpoint identity scope changed while computing {tree.topology.tree_id}"
            )
        artifact_plan = plan_direct_artifacts(execution, output_dir=str(tree_dir))
        _publish_tree_directory(
            plan,
            tree,
            artifact_plan.writes,
            execution_sha256=execution.receipt.payload["execution_sha256"],
            output_root=output_root,
            plan_path=saved_path,
            plan_file_sha256=plan_file_sha,
        )
        any_fresh = True
        tree_rows.append(
            _validate_tree_cache(
                plan,
                tree,
                tree_dir,
                plan_path=saved_path,
                plan_file_sha256=plan_file_sha,
            )
        )

    terminal = _cohort_terminal_payload(
        plan,
        plan_path=saved_path,
        plan_file_sha256=plan_file_sha,
        tree_rows=tuple(tree_rows),
        compatibility=_execute_or_validate_compatibility(
            plan,
            output_root=output_root,
            api=compatibility_api,
        ),
    )
    _publish_exclusive_file(cohort_terminal_path, _canonical_bytes(terminal))
    terminal = _validate_cohort_terminal(
        cohort_terminal_path,
        plan=plan,
        plan_path=saved_path,
        plan_file_sha256=plan_file_sha,
        tree_rows=tuple(tree_rows),
        compatibility=_load_compatibility_receipt(output_root),
    )
    return FilesystemDirectResult(
        cache_hit=not any_fresh,
        plan=plan,
        plan_path=saved_path,
        plan_file_sha256=plan_file_sha,
        job_config_path=job_config_path,
        job_config_file_sha256=job_config_file_sha,
        output_dir=output_root,
        terminal_receipt=terminal,
    )


def validate_direct_cohort_terminal(
    config_path: str | Path,
) -> FilesystemDirectResult:
    """Read-only validation boundary for downstream semantic consumers.

    This never creates an output or imports model/NJ code.  It validates the
    thin gate config, its pinned runtime plan, all 50 atomic tree directories,
    every matrix/receipt byte, the separate compatibility receipt hash, and
    the exact cohort terminal contract.
    """

    (
        _,
        plan,
        job_config_path,
        job_config_file_sha,
        plan_path,
        plan_file_sha,
    ) = load_direct_job_config(config_path)
    output_root = Path(plan.output_dir)
    if output_root.is_symlink() or not output_root.is_dir() or output_root.resolve() != output_root:
        raise ValueError("terminal direct output root is missing/noncanonical/symlink")
    existing = _validate_output_entries(plan, output_root)
    if existing != set(PRODUCTION_TREE_IDS):
        raise ValueError("terminal direct output lacks the exact fifty tree directories")
    tree_rows = tuple(
        _validate_tree_cache(
            plan,
            tree,
            output_root / tree.topology.tree_id,
            plan_path=plan_path,
            plan_file_sha256=plan_file_sha,
        )
        for tree in plan.trees
    )
    compatibility = _load_compatibility_receipt(output_root)
    expected_status = (
        "completed"
        if plan.leaf_compatibility.comparison_available
        else "comparison_unavailable"
    )
    if (
        compatibility.get("status") != expected_status
        or compatibility.get("comparison_available")
        is not plan.leaf_compatibility.comparison_available
        or compatibility.get("spec") != plan.leaf_compatibility.to_dict()
        or compatibility.get("direct_plan_sha256") != plan.direct_plan_sha256
        or compatibility.get("cohort_key") != plan.cohort_key
    ):
        raise ValueError("leaf compatibility receipt differs from the direct plan")
    terminal = _validate_cohort_terminal(
        output_root / COHORT_TERMINAL_BASENAME,
        plan=plan,
        plan_path=plan_path,
        plan_file_sha256=plan_file_sha,
        tree_rows=tree_rows,
        compatibility=compatibility,
    )
    return FilesystemDirectResult(
        cache_hit=True,
        plan=plan,
        plan_path=plan_path,
        plan_file_sha256=plan_file_sha,
        job_config_path=job_config_path,
        job_config_file_sha256=job_config_file_sha,
        output_dir=output_root,
        terminal_receipt=terminal,
    )


def _publish_tree_directory(
    plan: DirectCohortPlan,
    tree: ResolvedAllNodeTree,
    writes: Sequence[object],
    *,
    execution_sha256: str,
    output_root: Path,
    plan_path: Path,
    plan_file_sha256: str,
) -> None:
    final = output_root / tree.topology.tree_id
    if os.path.lexists(final):
        raise FileExistsError(f"direct tree output already exists: {final}")
    temporary = output_root / f".{tree.topology.tree_id}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir(mode=0o700)
    try:
        artifact_rows: list[dict[str, Any]] = []
        for raw in writes:
            path = Path(getattr(raw, "path"))
            if path.parent != final:
                raise ValueError("direct artifact plan escaped its final tree directory")
            destination = temporary / path.name
            content = getattr(raw, "content")
            digest = hashlib.sha256(content).hexdigest()
            if digest != getattr(raw, "content_sha256") or len(content) != getattr(raw, "size_bytes"):
                raise ValueError("direct artifact write bytes drift")
            _write_new_file(destination, content)
            artifact_rows.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "sha256": digest,
                    "size_bytes": len(content),
                }
            )
        terminal = _tree_terminal_payload(
            plan,
            tree,
            output_dir=final,
            plan_path=plan_path,
            plan_file_sha256=plan_file_sha256,
            execution_sha256=execution_sha256,
            artifacts=artifact_rows,
        )
        _write_new_file(temporary / TREE_TERMINAL_BASENAME, _canonical_bytes(terminal))
        _fsync_directory(temporary)
        os.rename(temporary, final)
        _fsync_directory(output_root)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _validate_tree_cache(
    plan: DirectCohortPlan,
    tree: ResolvedAllNodeTree,
    tree_dir: Path,
    *,
    plan_path: Path,
    plan_file_sha256: str,
) -> Mapping[str, Any]:
    if tree_dir.is_symlink() or not tree_dir.is_dir() or tree_dir.resolve() != tree_dir:
        raise ValueError(f"direct tree cache is not a canonical directory: {tree_dir}")
    terminal_path = tree_dir / TREE_TERMINAL_BASENAME
    if terminal_path.is_symlink() or not terminal_path.is_file():
        raise ValueError(f"direct tree cache lacks a terminal receipt: {tree_dir}")
    terminal = _json_file(terminal_path, "direct tree terminal")
    expected_keys = {
        "schema", "valid", "population", "internal_node_treatment",
        "direct_matrix_only", "neighbor_joining", "cohort_key",
        "training_cohort_id", "tree_id", "resolution_sha256",
        "topology_sha256", "checkpoint_set_sha256", "node_covariates_sha256",
        "direct_plan", "direct_plan_file_sha256", "direct_plan_sha256",
        "output_dir", "execution_sha256", "n_artifacts", "artifacts",
        "tree_terminal_sha256",
    }
    if set(terminal) != expected_keys or terminal.get("schema") != TREE_TERMINAL_SCHEMA or terminal.get("valid") is not True:
        raise ValueError("direct tree terminal schema/keys/validity drift")
    artifacts = terminal.get("artifacts")
    if not isinstance(artifacts, list) or terminal.get("n_artifacts") != len(artifacts):
        raise ValueError("direct tree terminal artifact count drift")
    core = {key: value for key, value in terminal.items() if key != "tree_terminal_sha256"}
    if terminal.get("tree_terminal_sha256") != canonical_sha256(core):
        raise ValueError("direct tree terminal hash drift")
    fixed = {
        "population": POPULATION,
        "internal_node_treatment": INTERNAL_NODE_TREATMENT,
        "direct_matrix_only": True,
        "neighbor_joining": False,
        "cohort_key": plan.cohort_key,
        "training_cohort_id": plan.training_cohort_id,
        "tree_id": tree.topology.tree_id,
        "resolution_sha256": tree.resolution_sha256,
        "topology_sha256": tree.topology.topology_sha256,
        "checkpoint_set_sha256": tree.checkpoint_manifest.checkpoint_set_sha256,
        "node_covariates_sha256": tree.node_covariates.covariates_sha256,
        "direct_plan": str(plan_path),
        "direct_plan_file_sha256": plan_file_sha256,
        "direct_plan_sha256": plan.direct_plan_sha256,
        "output_dir": str(tree_dir),
    }
    for field, expected in fixed.items():
        if terminal.get(field) != expected:
            raise ValueError(f"direct tree terminal {field} drift")
    expected_names = [
        name
        for metric in DIRECT_METRICS
        for name in (
            f"{tree.topology.tree_id}.{metric}.all_nodes.matrix.json",
            f"{tree.topology.tree_id}.{metric}.all_nodes.receipt.json",
        )
    ] + [f"{tree.topology.tree_id}.direct_execution.receipt.json"]
    if len(artifacts) != len(expected_names):
        raise ValueError("direct tree terminal artifact cardinality drift")
    normalized_rows: list[dict[str, Any]] = []
    for name, row in zip(expected_names, artifacts):
        if not isinstance(row, Mapping) or set(row) != {"name", "path", "sha256", "size_bytes"}:
            raise ValueError("direct tree terminal artifact row drift")
        if row.get("name") != name or row.get("path") != str(tree_dir / name):
            raise ValueError("direct tree terminal artifact name/path drift")
        artifact_path = tree_dir / name
        if artifact_path.is_symlink() or not artifact_path.is_file():
            raise ValueError("direct tree artifact is missing or a symlink")
        raw = artifact_path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if row.get("sha256") != digest or row.get("size_bytes") != len(raw):
            raise ValueError("direct tree artifact digest/size drift")
        normalized_rows.append(dict(row))
    expected_entries = set(expected_names) | {TREE_TERMINAL_BASENAME}
    if {entry.name for entry in tree_dir.iterdir()} != expected_entries:
        raise ValueError("direct tree cache contains missing or unexpected files")
    execution_path = tree_dir / expected_names[-1]
    execution = _validate_execution_receipt(
        _json_file(execution_path, "direct execution receipt"),
        plan=plan,
        tree=tree,
    )
    if terminal.get("execution_sha256") != execution["execution_sha256"]:
        raise ValueError("direct tree terminal execution hash drift")
    for metric_index, metric in enumerate(DIRECT_METRICS):
        values_path = tree_dir / expected_names[2 * metric_index]
        receipt_path = tree_dir / expected_names[2 * metric_index + 1]
        values = _json_file(values_path, f"{metric} matrix values")
        metric_receipt = _json_file(receipt_path, f"{metric} artifact receipt")
        _validate_metric_artifacts(
            values,
            metric_receipt,
            plan=plan,
            tree=tree,
            metric=metric,
            matrix_artifact=normalized_rows[2 * metric_index],
            execution_sha256=execution["execution_sha256"],
        )
    return terminal


def _validate_metric_artifacts(
    values: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    plan: DirectCohortPlan,
    tree: ResolvedAllNodeTree,
    metric: str,
    matrix_artifact: Mapping[str, Any],
    execution_sha256: str,
) -> None:
    value_keys = {
        "schema", "tree_id", "metric", "population", "internal_node_treatment",
        "node_covariates_sha256", "topology_sha256", "pair_index_sha256",
        "ordered_node_ids", "matrix", "leaf_sensitivity_indices",
        "leaf_sensitivity_ids", "leaf_sensitivity_matrix",
    }
    if set(values) != value_keys or values.get("schema") != DIRECT_VALUES_SCHEMA:
        raise ValueError("direct matrix values schema/keys drift")
    for field, expected in {
        "tree_id": tree.topology.tree_id,
        "metric": metric,
        "population": POPULATION,
        "internal_node_treatment": INTERNAL_NODE_TREATMENT,
        "node_covariates_sha256": tree.node_covariates.covariates_sha256,
        "topology_sha256": tree.topology.topology_sha256,
        "pair_index_sha256": tree.topology.pair_index_sha256,
        "ordered_node_ids": list(tree.topology.ordered_node_ids),
        "leaf_sensitivity_indices": list(tree.topology.leaf_sensitivity_indices),
        "leaf_sensitivity_ids": [
            tree.topology.ordered_node_ids[index]
            for index in tree.topology.leaf_sensitivity_indices
        ],
    }.items():
        if values.get(field) != expected:
            raise ValueError(f"direct matrix values {field} drift")
    receipt_keys = {
        "schema", "valid", "matrix_receipt", "matrix_receipt_sha256",
        "matrix_artifact", "execution_sha256", "node_covariates_sha256",
        "artifact_receipt_sha256",
    }
    if set(receipt) != receipt_keys or receipt.get("schema") != DIRECT_ARTIFACT_RECEIPT_SCHEMA or receipt.get("valid") is not True:
        raise ValueError("direct artifact receipt schema/keys/validity drift")
    validated = validate_matrix_receipt(
        receipt.get("matrix_receipt"),
        tree.topology,
        metric=metric,
        ordered_node_ids=values.get("ordered_node_ids"),
        matrix=values.get("matrix"),
    )
    if values.get("leaf_sensitivity_matrix") != [
        list(row) for row in validated.leaf_sensitivity_matrix
    ]:
        raise ValueError("saved leaf sensitivity matrix is not the derived full-matrix slice")
    matrix_receipt = validated.receipt.to_dict()
    if receipt.get("matrix_receipt_sha256") != canonical_sha256(matrix_receipt):
        raise ValueError("direct matrix receipt hash drift")
    expected_artifact = {
        "path": matrix_artifact["path"],
        "content_sha256": matrix_artifact["sha256"],
        "size_bytes": matrix_artifact["size_bytes"],
        "exclusive_create": True,
    }
    if receipt.get("matrix_artifact") != expected_artifact:
        raise ValueError("direct matrix artifact binding drift")
    if receipt.get("execution_sha256") != execution_sha256 or receipt.get("node_covariates_sha256") != tree.node_covariates.covariates_sha256:
        raise ValueError("direct artifact execution/covariate binding drift")
    core = {key: value for key, value in receipt.items() if key not in {"schema", "valid", "artifact_receipt_sha256"}}
    if receipt.get("artifact_receipt_sha256") != canonical_sha256(core):
        raise ValueError("direct artifact receipt hash drift")


def _validate_execution_receipt(
    payload: Mapping[str, Any],
    *,
    plan: DirectCohortPlan,
    tree: ResolvedAllNodeTree,
) -> Mapping[str, Any]:
    if payload.get("schema") != DIRECT_EXECUTION_SCHEMA or payload.get("valid") is not True:
        raise ValueError("direct execution receipt schema/validity drift")
    required = {
        "distance_engine": DISTANCE_ENGINE,
        "population": POPULATION,
        "internal_node_treatment": INTERNAL_NODE_TREATMENT,
        "direct_matrix_only": True,
        "neighbor_joining": False,
        "tree_builder": None,
        "internal_nodes_as_tips": False,
        "cohort_key": plan.cohort_key,
        "training_cohort_id": plan.training_cohort_id,
        "tree_id": tree.topology.tree_id,
        "resolution_sha256": tree.resolution_sha256,
        "checkpoint_set_sha256": tree.checkpoint_manifest.checkpoint_set_sha256,
        "node_covariates": tree.node_covariates.to_dict(),
        "node_covariates_sha256": tree.node_covariates.covariates_sha256,
        "topology_sha256": tree.topology.topology_sha256,
        "pair_index_sha256": tree.topology.pair_index_sha256,
        "leaf_subset_sha256": tree.topology.leaf_subset_sha256,
        "ordered_node_ids": list(tree.topology.ordered_node_ids),
        "ordered_node_ids_sha256": canonical_sha256(list(tree.topology.ordered_node_ids)),
        "n_nodes": tree.topology.n_nodes,
        "n_internal": tree.topology.n_internal,
        "n_leaves": tree.topology.n_leaves,
        "metrics": list(DIRECT_METRICS),
        "chunk_size": plan.chunk_size,
        "representation": REPRESENTATION,
        "layer_aggregation": LAYER_AGGREGATION,
    }
    for field, expected in required.items():
        if payload.get(field) != expected:
            raise ValueError(f"direct execution receipt {field} drift")
    metric_rows = payload.get("metric_results")
    if not isinstance(metric_rows, list) or [row.get("metric") for row in metric_rows if isinstance(row, Mapping)] != list(DIRECT_METRICS):
        raise ValueError("direct execution metric scope drift")
    core = {key: value for key, value in payload.items() if key not in {"schema", "valid", "execution_sha256"}}
    if payload.get("execution_sha256") != canonical_sha256(core):
        raise ValueError("direct execution receipt hash drift")
    return payload


def _tree_terminal_payload(
    plan: DirectCohortPlan,
    tree: ResolvedAllNodeTree,
    *,
    output_dir: Path,
    plan_path: Path,
    plan_file_sha256: str,
    execution_sha256: str,
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    core = {
        "schema": TREE_TERMINAL_SCHEMA,
        "valid": True,
        "population": POPULATION,
        "internal_node_treatment": INTERNAL_NODE_TREATMENT,
        "direct_matrix_only": True,
        "neighbor_joining": False,
        "cohort_key": plan.cohort_key,
        "training_cohort_id": plan.training_cohort_id,
        "tree_id": tree.topology.tree_id,
        "resolution_sha256": tree.resolution_sha256,
        "topology_sha256": tree.topology.topology_sha256,
        "checkpoint_set_sha256": tree.checkpoint_manifest.checkpoint_set_sha256,
        "node_covariates_sha256": tree.node_covariates.covariates_sha256,
        "direct_plan": str(plan_path),
        "direct_plan_file_sha256": plan_file_sha256,
        "direct_plan_sha256": plan.direct_plan_sha256,
        "output_dir": str(output_dir),
        "execution_sha256": _sha(execution_sha256, "execution_sha256"),
        "n_artifacts": len(artifacts),
        "artifacts": [dict(row) for row in artifacts],
    }
    return {**core, "tree_terminal_sha256": canonical_sha256(core)}


def _cohort_terminal_payload(
    plan: DirectCohortPlan,
    *,
    plan_path: Path,
    plan_file_sha256: str,
    tree_rows: Sequence[Mapping[str, Any]],
    compatibility: Mapping[str, Any],
) -> dict[str, Any]:
    if len(tree_rows) != PRODUCTION_N_TREES:
        raise ValueError("cannot publish cohort terminal before all fifty trees")
    rows = [
        {
            "tree_id": tree.topology.tree_id,
            "tree_terminal": str(Path(plan.output_dir) / tree.topology.tree_id / TREE_TERMINAL_BASENAME),
            "tree_terminal_sha256": row["tree_terminal_sha256"],
            "execution_sha256": row["execution_sha256"],
            "resolution_sha256": tree.resolution_sha256,
            "n_nodes": tree.topology.n_nodes,
            "n_internal": tree.topology.n_internal,
            "n_leaves": tree.topology.n_leaves,
        }
        for tree, row in zip(plan.trees, tree_rows)
    ]
    core = {
        "schema": COHORT_TERMINAL_SCHEMA,
        "valid": True,
        "terminal": True,
        "population": POPULATION,
        "internal_node_treatment": INTERNAL_NODE_TREATMENT,
        "direct_matrix_only": True,
        "neighbor_joining": False,
        "cohort_key": plan.cohort_key,
        "training_cohort_id": plan.training_cohort_id,
        "direct_plan": str(plan_path),
        "direct_plan_file_sha256": plan_file_sha256,
        "direct_plan_sha256": plan.direct_plan_sha256,
        "checkpoint_identity_index": plan.checkpoint_identity_index.to_dict(),
        "output_dir": plan.output_dir,
        "metrics": list(DIRECT_METRICS),
        "n_trees": PRODUCTION_N_TREES,
        "n_nodes": PRODUCTION_N_NODES,
        "n_internal": PRODUCTION_N_INTERNAL,
        "n_leaves": PRODUCTION_N_LEAVES,
        "trees": rows,
        "tree_scope_sha256": canonical_sha256(rows),
        "execution_scope_sha256": canonical_sha256([row["execution_sha256"] for row in rows]),
        "leaf_compatibility": {
            "path": str(
                Path(plan.output_dir)
                / LEAF_COMPATIBILITY_DIRNAME
                / LEAF_COMPATIBILITY_BASENAME
            ),
            "status": compatibility["status"],
            "comparison_available": compatibility["comparison_available"],
            "sha256": compatibility["leaf_compatibility_sha256"],
        },
    }
    return {**core, "cohort_terminal_sha256": canonical_sha256(core)}


def _validate_cohort_terminal(
    path: Path,
    *,
    plan: DirectCohortPlan,
    plan_path: Path,
    plan_file_sha256: str,
    tree_rows: Sequence[Mapping[str, Any]],
    compatibility: Mapping[str, Any],
) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("direct cohort terminal is missing or a symlink")
    observed = _json_file(path, "direct cohort terminal")
    expected = _cohort_terminal_payload(
        plan,
        plan_path=plan_path,
        plan_file_sha256=plan_file_sha256,
        tree_rows=tree_rows,
        compatibility=compatibility,
    )
    if canonical_sha256(observed) != canonical_sha256(expected):
        raise ValueError("direct cohort terminal differs from exact recomputed receipt")
    if path.read_bytes() != _canonical_bytes(expected):
        raise ValueError("direct cohort terminal is not canonical bytes")
    return expected


def _execute_or_validate_compatibility(
    plan: DirectCohortPlan,
    *,
    output_root: Path,
    api: LeafCompatibilityAPI | None,
) -> Mapping[str, Any]:
    compatibility_dir = output_root / LEAF_COMPATIBILITY_DIRNAME
    receipt_path = compatibility_dir / LEAF_COMPATIBILITY_BASENAME

    def matrix_loader(tree_id: str, metric: str) -> Mapping[str, Any]:
        if tree_id not in PRODUCTION_TREE_IDS or metric not in DIRECT_METRICS:
            raise ValueError("leaf compatibility requested an unknown coordinate")
        path = output_root / tree_id / f"{tree_id}.{metric}.all_nodes.matrix.json"
        return _json_file(path, f"primary direct matrix {tree_id}/{metric}")

    expected = execute_leaf_compatibility(
        plan.leaf_compatibility,
        cohort_key=plan.cohort_key,
        direct_plan_sha256=plan.direct_plan_sha256,
        matrix_loader=matrix_loader,
        snapshot_root=plan.snapshot_root,
        api=api,
    )
    if os.path.lexists(compatibility_dir):
        if compatibility_dir.is_symlink() or not compatibility_dir.is_dir():
            raise ValueError("leaf compatibility output is not a directory")
        if {entry.name for entry in compatibility_dir.iterdir()} != {
            LEAF_COMPATIBILITY_BASENAME
        }:
            raise ValueError("leaf compatibility directory contains unexpected files")
        observed = _json_file(receipt_path, "leaf compatibility receipt")
        validate_leaf_compatibility_receipt(observed, expected=expected)
        if receipt_path.read_bytes() != _canonical_bytes(expected):
            raise ValueError("leaf compatibility receipt is not canonical bytes")
        return expected

    temporary = output_root / f".{LEAF_COMPATIBILITY_DIRNAME}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir(mode=0o700)
    try:
        _write_new_file(
            temporary / LEAF_COMPATIBILITY_BASENAME,
            _canonical_bytes(expected),
        )
        _fsync_directory(temporary)
        os.rename(temporary, compatibility_dir)
        _fsync_directory(output_root)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return expected


def _load_compatibility_receipt(output_root: Path) -> Mapping[str, Any]:
    path = (
        output_root
        / LEAF_COMPATIBILITY_DIRNAME
        / LEAF_COMPATIBILITY_BASENAME
    )
    payload = _json_file(path, "leaf compatibility receipt")
    core = {
        key: value
        for key, value in payload.items()
        if key != "leaf_compatibility_sha256"
    }
    if payload.get("leaf_compatibility_sha256") != canonical_sha256(core):
        raise ValueError("leaf compatibility receipt hash drift")
    return payload


def _load_identity_lookup(
    pin: ArtifactPin,
    *,
    reader: IdentityIndexReader | None,
) -> dict[str, CheckpointIdentity | str]:
    _verify_file_pin(pin)
    payload = _json_file(Path(pin.path), "checkpoint identity index")
    production_reader = reader is None
    raw_lookup = read_identity_index(payload) if production_reader else reader(payload)
    if not isinstance(raw_lookup, Mapping):
        raise ValueError("checkpoint identity index reader must return a mapping")
    lookup: dict[str, CheckpointIdentity | str] = {}
    for raw_path, raw_identity in raw_lookup.items():
        path = _root(raw_path, "checkpoint identity path")
        if isinstance(raw_identity, CheckpointIdentity):
            if raw_identity.path != path:
                raise ValueError("checkpoint identity lookup key/path drift")
            verified_identity = validate_checkpoint_identity(
                raw_identity,
                reattest_filesystem=False,
            )
            lookup[path] = verified_identity
            continue
        elif isinstance(raw_identity, str):
            if production_reader:
                raise ValueError(
                    "production checkpoint identity index must contain full identities"
                )
            digest = raw_identity
        elif isinstance(raw_identity, Mapping):
            if production_reader:
                raise ValueError(
                    "production checkpoint identity index must contain full identities"
                )
            digest = raw_identity.get("identity_sha256")
        else:
            raise ValueError("checkpoint identity lookup value type drift")
        lookup[path] = _sha(digest, "checkpoint identity digest")
    return lookup


def _identity_digest(value: CheckpointIdentity | str) -> str:
    return value.identity_sha256 if isinstance(value, CheckpointIdentity) else value


def _verify_checkpoint_scope(
    plan: DirectCohortPlan,
    lookup: Mapping[str, CheckpointIdentity | str],
) -> None:
    expected_paths: list[str] = []
    for tree in plan.trees:
        for binding in tree.checkpoint_manifest.bindings:
            path = binding.checkpoint.path
            expected_paths.append(path)
            identity = lookup.get(path)
            if identity is None or _identity_digest(identity) != binding.checkpoint.sha256:
                raise ValueError(
                    f"checkpoint identity index mismatch for {tree.topology.tree_id}/{binding.node_id}"
                )
    if len(expected_paths) != PRODUCTION_N_NODES:
        raise AssertionError("direct checkpoint population cardinality corruption")


def _load_runtime_source_authority(
    lookup: Mapping[str, CheckpointIdentity | str],
) -> SourcePathAuthority | None:
    """Live-load the exact frozen source authority once for this direct job."""

    identities = tuple(
        value for value in lookup.values() if isinstance(value, CheckpointIdentity)
    )
    if not identities:
        # Digest-only lookup values are an explicit unit-test seam.  The
        # production index reader above refuses them.
        return None
    authority = source_authority_from_identity(identities[0])
    for identity in identities[1:]:
        if identity.source_authority != identities[0].source_authority:
            raise ValueError("checkpoint identity source-authority scope drift")
    return authority


def _reattest_tree_checkpoint_identities(
    tree: ResolvedAllNodeTree,
    lookup: Mapping[str, CheckpointIdentity | str],
    *,
    source_authority: SourcePathAuthority | None,
) -> str:
    """Re-sample one exact tree immediately around checkpoint consumption.

    The returned digest is deterministic and is compared across the before
    and after boundaries.  No tree artifact may be planned or published if
    either boundary fails.
    """

    rows: list[dict[str, object]] = []
    for binding in tree.checkpoint_manifest.bindings:
        path = binding.checkpoint.path
        identity = lookup.get(path)
        if identity is None or _identity_digest(identity) != binding.checkpoint.sha256:
            raise ValueError(
                f"checkpoint identity index mismatch for {tree.topology.tree_id}/{binding.node_id}"
            )
        if isinstance(identity, CheckpointIdentity):
            verified = validate_checkpoint_identity(
                identity,
                reattest_filesystem=True,
                source_authority=source_authority,
            )
            if (
                verified.path != path
                or verified.identity_sha256 != binding.checkpoint.sha256
            ):
                raise ValueError(
                    f"checkpoint runtime identity drift for {tree.topology.tree_id}/{binding.node_id}"
                )
        rows.append(
            {
                "node_ordinal": binding.node_ordinal,
                "node_id": binding.node_id,
                "checkpoint_path": path,
                "identity_sha256": binding.checkpoint.sha256,
            }
        )
    if len(rows) != tree.topology.n_nodes:
        raise AssertionError("direct tree identity cardinality corruption")
    return canonical_sha256(rows)


def _load_pinned_weighttraits_api(plan: DirectCohortPlan) -> WeightTraitsDirectAPI:
    readers = importlib.import_module("weighttraits.distances.readers")
    streaming = importlib.import_module("weighttraits.distances.streaming")
    for module, pin, label in (
        (readers, plan.readers_source, "readers"),
        (streaming, plan.streaming_source, "streaming"),
    ):
        module_path = Path(getattr(module, "__file__", "")).resolve(strict=True)
        if str(module_path) != pin.path or module_path.is_symlink():
            raise ValueError(f"imported WeightTraits {label} module path differs from pin")
    reader_from_path = getattr(readers, "reader_from_path", None)
    build_distance_cube = getattr(streaming, "build_distance_cube", None)
    if not callable(reader_from_path) or not callable(build_distance_cube):
        raise ValueError("pinned WeightTraits direct API is incomplete")
    return WeightTraitsDirectAPI(reader_from_path, build_distance_cube)


def _plan_core(
    *,
    cohort_key: str,
    training_cohort_id: str,
    source_stage_root: str,
    output_dir: str,
    snapshot_root: str,
    readers_source: ArtifactPin,
    streaming_source: ArtifactPin,
    checkpoint_identity_index: ArtifactPin,
    source_path_authority: SourcePathAuthority | None,
    leaf_compatibility: LeafCompatibilitySpec,
    trees: Sequence[ResolvedAllNodeTree],
    resolution_scope_sha256: str,
    checkpoint_scope_sha256: str,
    node_covariate_scope_sha256: str,
    chunk_size: int,
) -> dict[str, Any]:
    return {
        "schema": PLAN_SCHEMA,
        "valid": True,
        "population": POPULATION,
        "internal_node_treatment": INTERNAL_NODE_TREATMENT,
        "exclude_root": True,
        "direct_matrix_only": True,
        "neighbor_joining": False,
        "tree_builder": None,
        "cohort_key": cohort_key,
        "training_cohort_id": training_cohort_id,
        "source_stage_root": source_stage_root,
        "output_dir": output_dir,
        "snapshot_root": snapshot_root,
        "readers_source": readers_source.to_dict(),
        "streaming_source": streaming_source.to_dict(),
        "checkpoint_identity_index": checkpoint_identity_index.to_dict(),
        "source_path_authority": (
            None if source_path_authority is None else source_path_authority.to_dict()
        ),
        "leaf_compatibility": leaf_compatibility.to_dict(),
        "n_trees": PRODUCTION_N_TREES,
        "n_nodes": PRODUCTION_N_NODES,
        "n_internal": PRODUCTION_N_INTERNAL,
        "n_leaves": PRODUCTION_N_LEAVES,
        "tree_ids": list(PRODUCTION_TREE_IDS),
        "metrics": list(DIRECT_METRICS),
        "distance_engine": DISTANCE_ENGINE,
        "representation": REPRESENTATION,
        "layer_aggregation": LAYER_AGGREGATION,
        "chunk_size": chunk_size,
        "trees": [_resolved_to_dict(tree) for tree in trees],
        "resolution_scope_sha256": resolution_scope_sha256,
        "checkpoint_scope_sha256": checkpoint_scope_sha256,
        "node_covariate_scope_sha256": node_covariate_scope_sha256,
        "execution_overrides": {},
    }


def _resolved_to_dict(value: ResolvedAllNodeTree) -> dict[str, Any]:
    return {
        "cohort_key": value.cohort_key,
        "training_cohort_id": value.training_cohort_id,
        "population": value.population,
        "internal_node_treatment": value.internal_node_treatment,
        "topology": value.topology.to_dict(),
        "checkpoint_manifest": value.checkpoint_manifest.to_dict(),
        "node_covariates": value.node_covariates.to_dict(),
        "resolution_sha256": value.resolution_sha256,
    }


def _read_resolved_tree(raw: object) -> ResolvedAllNodeTree:
    required = {
        "cohort_key", "training_cohort_id", "population",
        "internal_node_treatment", "topology", "checkpoint_manifest",
        "node_covariates", "resolution_sha256",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("direct resolved tree row keys drift")
    topology = read_tree_topology(raw["topology"])
    manifest = read_tree_checkpoint_manifest(raw["checkpoint_manifest"])
    covariates = read_tree_node_covariates(raw["node_covariates"], topology, manifest)
    value = ResolvedAllNodeTree(
        cohort_key=raw["cohort_key"],
        training_cohort_id=raw["training_cohort_id"],
        population=raw["population"],
        internal_node_treatment=raw["internal_node_treatment"],
        topology=topology,
        checkpoint_manifest=manifest,
        node_covariates=covariates,
        resolution_sha256=raw["resolution_sha256"],
    )
    return _resolved(value, cohort_key=value.cohort_key)


def _resolved(value: object, *, cohort_key: str) -> ResolvedAllNodeTree:
    if not isinstance(value, ResolvedAllNodeTree):
        raise ValueError("direct plan trees must be ResolvedAllNodeTree objects")
    if value.cohort_key != cohort_key:
        raise ValueError("direct resolved tree cohort_key drift")
    topology = read_tree_topology(value.topology.to_dict())
    manifest = read_tree_checkpoint_manifest(value.checkpoint_manifest.to_dict())
    covariates = read_tree_node_covariates(
        value.node_covariates.to_dict(), topology, manifest
    )
    if value.population != POPULATION or value.internal_node_treatment != INTERNAL_NODE_TREATMENT:
        raise ValueError("direct resolved tree population/treatment drift")
    if manifest.cohort_id != value.training_cohort_id or manifest.tree_id != topology.tree_id:
        raise ValueError("direct resolved tree manifest scope drift")
    core = {
        "cohort_key": value.cohort_key,
        "training_cohort_id": value.training_cohort_id,
        "population": POPULATION,
        "internal_node_treatment": INTERNAL_NODE_TREATMENT,
        "topology_sha256": topology.topology_sha256,
        "checkpoint_set_sha256": manifest.checkpoint_set_sha256,
        "node_covariates_sha256": covariates.covariates_sha256,
    }
    if value.resolution_sha256 != canonical_sha256(core):
        raise ValueError("direct resolved tree resolution hash drift")
    return value


def _validate_output_entries(plan: DirectCohortPlan, root: Path) -> set[str]:
    allowed_trees = set(PRODUCTION_TREE_IDS)
    observed: set[str] = set()
    for entry in root.iterdir():
        if entry.name == COHORT_TERMINAL_BASENAME:
            if entry.is_symlink() or not entry.is_file():
                raise ValueError("direct cohort terminal path type drift")
            continue
        if entry.name == LEAF_COMPATIBILITY_DIRNAME:
            if entry.is_symlink() or not entry.is_dir():
                raise ValueError("leaf compatibility output path type drift")
            continue
        if entry.name not in allowed_trees or entry.is_symlink() or not entry.is_dir():
            raise ValueError(f"direct output root contains an unexpected entry: {entry.name}")
        observed.add(entry.name)
    return observed


def _verify_file_pin(pin: ArtifactPin) -> None:
    path = Path(pin.path)
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise ValueError(f"pinned source is missing, noncanonical, or a symlink: {path}")
    if _sha256_file(path) != pin.sha256:
        raise ValueError(f"pinned source SHA-256 drift: {path}")


def _publish_exclusive_file(path: Path, content: bytes) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"immutable terminal path already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_new_file(path: Path, content: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ) + "\n"
    ).encode("utf-8")


def _json_file(path: Path, label: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is missing or a symlink")
    return _json_object(path.read_bytes(), label)


def _json_object(raw: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _pin(raw: ArtifactPin | Mapping[str, Any], label: str) -> ArtifactPin:
    if isinstance(raw, ArtifactPin):
        value = raw
    elif isinstance(raw, Mapping) and set(raw) == {"path", "sha256"}:
        value = ArtifactPin(path=raw["path"], sha256=raw["sha256"])
    else:
        raise ValueError(f"{label} must be an exact path+sha256 pin")
    _root(value.path, f"{label}.path")
    _sha(value.sha256, f"{label}.sha256")
    return value


def _strictly_below(path: str, root: str) -> bool:
    try:
        return os.path.commonpath((path, root)) == root and path != root
    except ValueError:
        return False


def _root(raw: object, label: str) -> str:
    if not isinstance(raw, (str, os.PathLike)):
        raise ValueError(f"{label} must be an absolute path")
    value = os.path.normpath(os.fspath(raw))
    if not os.path.isabs(value) or value == "/" or value != os.fspath(raw):
        raise ValueError(f"{label} must be a normalized bounded absolute path")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be nonempty normalized text")
    if _COMPONENT.fullmatch(value) is None:
        raise ValueError(f"{label} must be a safe component")
    return value


def _integer(value: object, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_launcher_environment(
    plan: DirectCohortPlan,
    *,
    config_path: Path,
    config_sha256: str,
) -> None:
    expected = {
        "WT_ALLNODE_COHORT_KEY": plan.cohort_key,
        "WT_ALLNODE_JOB_CONFIG": str(config_path),
        "WT_ALLNODE_JOB_CONFIG_SHA256": config_sha256,
        "WT_ALLNODE_KIND": "direct_allnode_cpu",
        "WT_ALLNODE_LOGICAL_ID": f"direct_{plan.cohort_key}",
        "WT_ALLNODE_OUTPUT_DIR": plan.output_dir,
    }
    for name, value in expected.items():
        if os.environ.get(name) != value:
            raise ValueError(f"launcher environment {name} drift")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    attest_runtime_environment("before-runner")
    try:
        _, plan, config_path, config_sha, _, _ = load_direct_job_config(args.config)
        _validate_launcher_environment(
            plan, config_path=config_path, config_sha256=config_sha
        )
        result = execute_filesystem_direct(config_path)
    finally:
        attest_runtime_environment("after-runner")
    print(json.dumps(result.status_dict(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CLI_STATUS_SCHEMA", "COHORT_TERMINAL_SCHEMA", "DirectCohortPlan",
    "DirectJobConfig", "FilesystemDirectResult", "JOB_CONFIG_SCHEMA",
    "PLAN_SCHEMA", "TREE_TERMINAL_SCHEMA", "WeightTraitsDirectAPI",
    "build_direct_cohort_plan", "build_direct_job_config",
    "execute_filesystem_direct", "load_direct_cohort_plan",
    "load_direct_job_config", "read_direct_cohort_plan",
    "read_direct_job_config", "validate_direct_cohort_terminal",
]
