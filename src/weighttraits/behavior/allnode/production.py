"""Production dry-run aggregation for the four all-node semantic cohorts.

This module performs no I/O or scheduling.  It turns already-resolved tree
contracts into one fail-closed production scope receipt.  External analysis
cohort keys remain distinct from training cohort IDs, which may legitimately
be shared by matched analyses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .checkpoints import (
    INTERNAL_NODE_TREATMENT,
    POPULATION,
    PRODUCTION_COHORT_IDS,
    PRODUCTION_N_INTERNAL,
    PRODUCTION_N_LEAVES,
    PRODUCTION_N_NODES,
    PRODUCTION_N_TREES,
    read_tree_checkpoint_manifest,
    validate_production_checkpoint_manifests,
)
from .manifest import canonical_sha256, read_tree_topology
from .covariates import validate_tree_node_covariates
from .resolver import ResolvedAllNodeTree


PRODUCTION_TREE_IDS = tuple(
    f"confirm_paper_tree_{index:03d}" for index in range(1, PRODUCTION_N_TREES + 1)
)


@dataclass(frozen=True)
class ProductionResolutionReceipt:
    """Exact non-executing scope contract for the all-node production branch."""

    payload: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


def validate_production_resolutions(
    cohorts: Mapping[str, Sequence[ResolvedAllNodeTree]],
) -> ProductionResolutionReceipt:
    """Require four cohorts x 50 trees x 641 observed trained checkpoints."""

    if not isinstance(cohorts, Mapping) or set(cohorts) != set(PRODUCTION_COHORT_IDS):
        raise ValueError(
            f"external cohort keys must be exactly {PRODUCTION_COHORT_IDS!r}"
        )
    verified_by_cohort: dict[str, tuple[ResolvedAllNodeTree, ...]] = {}
    manifests: dict[str, tuple[Any, ...]] = {}
    cohort_rows: list[dict[str, Any]] = []
    for cohort_key in PRODUCTION_COHORT_IDS:
        trees = tuple(cohorts[cohort_key])
        if len(trees) != PRODUCTION_N_TREES:
            raise ValueError(
                f"cohort {cohort_key!r} must contain exactly {PRODUCTION_N_TREES} trees"
            )
        verified = tuple(_verify_resolution(tree, cohort_key=cohort_key) for tree in trees)
        tree_ids = tuple(tree.topology.tree_id for tree in verified)
        if tree_ids != PRODUCTION_TREE_IDS:
            raise ValueError(
                f"cohort {cohort_key!r} tree order must be exactly trees 001..050"
            )
        training_ids = {tree.training_cohort_id for tree in verified}
        if len(training_ids) != 1:
            raise ValueError(
                f"cohort {cohort_key!r} mixes training cohort identities across trees"
            )
        n_nodes = sum(tree.topology.n_nodes for tree in verified)
        n_internal = sum(tree.topology.n_internal for tree in verified)
        n_leaves = sum(tree.topology.n_leaves for tree in verified)
        observed = (n_nodes, n_internal, n_leaves)
        expected = (PRODUCTION_N_NODES, PRODUCTION_N_INTERNAL, PRODUCTION_N_LEAVES)
        if observed != expected:
            raise ValueError(
                f"cohort {cohort_key!r} all-node totals differ: "
                f"observed={observed}, expected={expected}"
            )
        scope = [
            {
                "tree_id": tree.topology.tree_id,
                "resolution_sha256": tree.resolution_sha256,
                "topology_sha256": tree.topology.topology_sha256,
                "pair_index_sha256": tree.topology.pair_index_sha256,
                "leaf_subset_sha256": tree.topology.leaf_subset_sha256,
                "checkpoint_set_sha256": tree.checkpoint_manifest.checkpoint_set_sha256,
                "node_covariates_sha256": tree.node_covariates.covariates_sha256,
            }
            for tree in verified
        ]
        cohort_rows.append(
            {
                "cohort_key": cohort_key,
                "training_cohort_id": next(iter(training_ids)),
                "n_trees": PRODUCTION_N_TREES,
                "n_nodes": n_nodes,
                "n_internal": n_internal,
                "n_leaves": n_leaves,
                "tree_ids": list(tree_ids),
                "resolution_scope": scope,
                "resolution_scope_sha256": canonical_sha256(scope),
                "node_covariate_scope_sha256": canonical_sha256(
                    [tree.node_covariates.covariates_sha256 for tree in verified]
                ),
            }
        )
        verified_by_cohort[cohort_key] = verified
        manifests[cohort_key] = tuple(tree.checkpoint_manifest for tree in verified)

    # Reuse the checkpoint layer's independently recomputed 50/641/276/365
    # and cross-cohort identical-topology contract.
    checkpoint_contract = validate_production_checkpoint_manifests(
        manifests,
        expected_cohort_ids=PRODUCTION_COHORT_IDS,
        expected_tree_ids=PRODUCTION_TREE_IDS,
    )
    topology_scopes = [
        [
            (
                tree.topology.topology_sha256,
                tree.topology.pair_index_sha256,
                tree.topology.leaf_subset_sha256,
            )
            for tree in verified_by_cohort[cohort_key]
        ]
        for cohort_key in PRODUCTION_COHORT_IDS
    ]
    if any(scope != topology_scopes[0] for scope in topology_scopes[1:]):
        raise ValueError("external cohorts do not share the exact ordered topology scope")

    core = {
        "population": POPULATION,
        "internal_node_treatment": INTERNAL_NODE_TREATMENT,
        "exclude_root": True,
        "dry_run_only": True,
        "scheduler_actions": 0,
        "cohort_keys": list(PRODUCTION_COHORT_IDS),
        "tree_ids": list(PRODUCTION_TREE_IDS),
        "n_cohorts": len(PRODUCTION_COHORT_IDS),
        "n_trees_per_cohort": PRODUCTION_N_TREES,
        "n_nodes_per_cohort": PRODUCTION_N_NODES,
        "n_internal_per_cohort": PRODUCTION_N_INTERNAL,
        "n_leaves_per_cohort": PRODUCTION_N_LEAVES,
        "cohorts": cohort_rows,
        "checkpoint_contract_sha256": canonical_sha256(checkpoint_contract),
        "topology_scope_sha256": checkpoint_contract["topology_scope_sha256"],
    }
    payload = {
        "schema": "weighttraits.all_node_production_resolution.v1",
        "valid": True,
        **core,
        "production_resolution_sha256": canonical_sha256(core),
    }
    return ProductionResolutionReceipt(payload=payload)


def _verify_resolution(
    value: ResolvedAllNodeTree, *, cohort_key: str
) -> ResolvedAllNodeTree:
    if not isinstance(value, ResolvedAllNodeTree):
        raise ValueError("production tree entries must be ResolvedAllNodeTree objects")
    if value.cohort_key != cohort_key:
        raise ValueError(f"resolved tree external cohort key drift for {cohort_key!r}")
    if (
        value.population != POPULATION
        or value.internal_node_treatment != INTERNAL_NODE_TREATMENT
    ):
        raise ValueError("resolved tree population or internal-node treatment drift")
    topology = read_tree_topology(value.topology.to_dict())
    manifest = read_tree_checkpoint_manifest(value.checkpoint_manifest.to_dict())
    covariates = validate_tree_node_covariates(
        value.node_covariates, topology, manifest
    )
    if manifest.cohort_id != value.training_cohort_id:
        raise ValueError("checkpoint manifest training cohort identity drift")
    if manifest.tree_id != topology.tree_id:
        raise ValueError("checkpoint manifest tree identity differs from topology")
    for field in (
        "topology_sha256",
        "nodes_sha256",
        "leaf_subset_sha256",
        "pair_index_sha256",
    ):
        if getattr(manifest, field) != getattr(topology, field):
            raise ValueError(f"checkpoint manifest {field} differs from topology")
    if topology.root_id in topology.ordered_node_ids:
        raise ValueError("untrained root is present in the production population")
    if any(binding.node_id == topology.root_id for binding in manifest.bindings):
        raise ValueError("untrained root has a production checkpoint binding")
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
        raise ValueError("resolved tree hash differs from its recomputed contract")
    return value
