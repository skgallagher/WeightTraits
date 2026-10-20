"""Tree-unit semantic regressions for observed trained ancestors and leaves.

The historical behavior bridge was leaf-only by construction: it selected
``is_leaf`` manifest rows, loaded leaf-indexed matrices, and enumerated only
``combinations(common_leaves, 2)``.  This module replaces that population
contract.  Its primary estimand is otherwise the inherited one: within each
tree, regress semantic similarity (one minus paired cosine distance) on direct
weight cosine distance.

All trained internal checkpoints are observed ancestors, not inferred tips.
The full all-node result is primary; the leaf-only result is derived from the
same validated full matrices.  Pair rows are descriptive dependent dyads.
Inference is across trees.  The equal-tree result never uses pair counts as a
sample size, and the robust DerSimonian--Laird result uses delete-one-node
jackknife variances.  The inherited pooled OLS/HC3 regressions and
``1 / (n_pairs - 3)`` DL variance are also reproduced exactly, but only under
explicit ``legacy_comparability`` labels because they treat overlapping pairs
as independent.

This module performs no I/O, model inference, scheduler action, neighbor
joining, or topology reconstruction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from statistics import NormalDist
from typing import Any, Callable, Iterable, Mapping, Sequence

from .checkpoints import (
    INTERNAL_NODE_TREATMENT,
    POPULATION,
    PRODUCTION_N_INTERNAL,
    PRODUCTION_N_LEAVES,
    PRODUCTION_N_NODES,
    TreeCheckpointManifest,
    read_tree_checkpoint_manifest,
)
from .covariates import (
    TreeNodeCovariates,
    task_path_jaccard,
    validate_tree_node_covariates,
)
from .manifest import TreeTopology, canonical_sha256, read_tree_topology
from .matrices import ValidatedDistanceMatrix, validate_matrix_receipt
from .production import PRODUCTION_TREE_IDS


DIRECT_ARTIFACT_KIND = "direct_weight_cosine_distance"
SEMANTIC_ARTIFACT_KIND = "paired_semantic_cosine_distance"
DIRECT_METRIC = "cosine"
SEMANTIC_METRIC = "mean_paired_response_embedding_cosine_distance"
OUTCOME = "semantic_similarity=1-paired_semantic_cosine_distance"
PREDICTOR = "direct_weight_cosine_distance"
LEGACY_OUTCOME = "semantic_sim"
LEGACY_PREDICTOR = "cosine_dist"
INFERENCE_UNIT = "tree"
CORRELATION_CLIP = 0.999

PAIR_ROLE_CLASSES = ("internal_internal", "internal_leaf", "leaf_leaf")
STRATA = (
    "all_trained_node_pairs",
    "ancestor_descendant",
    "parent_child",
    "collateral",
    "pairs_involving_internal",
    "internal_internal",
    "internal_leaf",
    "trained_leaf_pairs",
)
PRIMARY_STRATUM = "all_trained_node_pairs"
LEAF_SENSITIVITY_STRATUM = "trained_leaf_pairs"
ADJUSTED_MODEL_ID = "all_nodes_internal_ancestry_adjusted"
ADJUSTED_COVARIATES = (
    "pair_has_internal",
    "ancestor_descendant",
    "parent_child",
)
CONTROLLED_MODEL_ID = "all_nodes_legacy_controlled"
CONTROLLED_COVARIATES = (
    "depth_i",
    "depth_j",
    "lca_depth",
    "task_jaccard",
)
TREE_SCHEMA = "weighttraits.all_node_semantic_tree_analysis.v1"
RUNSET_SCHEMA = "weighttraits.all_node_semantic_runset_analysis.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class BoundDistanceArtifact:
    """A validated matrix bound to one checkpoint set and source receipt."""

    artifact_kind: str
    endpoint_id: str
    checkpoint_set_sha256: str
    source_receipt_sha256: str
    matrix: ValidatedDistanceMatrix

    def provenance_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind,
            "endpoint_id": self.endpoint_id,
            "checkpoint_set_sha256": self.checkpoint_set_sha256,
            "source_receipt_sha256": self.source_receipt_sha256,
            "matrix_receipt": self.matrix.receipt.to_dict(),
        }


@dataclass(frozen=True)
class SemanticPairRow:
    """One topology-indexed dependent dyad used descriptively within a tree."""

    pair_ordinal: int
    left_ordinal: int
    right_ordinal: int
    left_node_id: str
    right_node_id: str
    left_role: str
    right_role: str
    pair_role_class: str
    pair_has_internal: bool
    relationship: str
    is_parent_child: bool
    topological_distance: int
    depth_difference: int
    left_depth: int
    right_depth: int
    lca_depth: int
    left_dataset_id: str
    right_dataset_id: str
    left_task_family: str
    right_task_family: str
    same_dataset: bool
    same_task_family: bool
    dataset_path_jaccard: float
    task_family_path_jaccard: float
    direct_distance: float
    semantic_distance: float
    semantic_similarity: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Coefficient:
    term: str
    estimate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TreeRegression:
    """One within-tree descriptive regression with no pair-level standard error."""

    model_id: str
    stratum: str
    formula: str
    covariates: tuple[str, ...]
    n_pairs: int
    n_unique_nodes: int
    defined: bool
    coefficients: tuple[Coefficient, ...]
    direct_slope: float | None
    standardized_direct_slope: float | None
    pearson_r: float | None
    r_squared: float | None
    jackknife_variance_fisher_z: float | None
    jackknife_n_nodes: int
    undefined_reason: str | None
    coefficient_inference: str = "descriptive_within_tree_only"
    jackknife_unit: str = "trained_node_delete_one_incident_dyads"

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["covariates"] = list(self.covariates)
        row["coefficients"] = [value.to_dict() for value in self.coefficients]
        return row


@dataclass(frozen=True)
class TreeSemanticReceipt:
    schema: str
    valid: bool
    cohort_key: str
    training_cohort_id: str
    endpoint_id: str
    tree_id: str
    population: str
    internal_node_treatment: str
    inference_unit: str
    root_excluded: bool
    n_nodes: int
    n_internal: int
    n_leaves: int
    n_pairs: int
    n_leaf_sensitivity_pairs: int
    topology_sha256: str
    pair_index_sha256: str
    leaf_subset_sha256: str
    checkpoint_set_sha256: str
    checkpoint_bindings_sha256: str
    covariate_source_manifest_sha256: str
    node_covariates_sha256: str
    direct_metric: str
    direct_matrix_sha256: str
    direct_source_receipt_sha256: str
    semantic_metric: str
    semantic_matrix_sha256: str
    semantic_source_receipt_sha256: str
    ordered_node_ids_sha256: str
    pair_rows_sha256: str
    pair_role_counts: tuple[tuple[str, int], ...]
    relationship_counts: tuple[tuple[str, int], ...]
    internal_vs_leaf_indicator: str
    strata: tuple[TreeRegression, ...]
    controlled_regression: TreeRegression
    adjusted_regression: TreeRegression
    strata_sha256: str
    neighbor_joining: bool
    tree_builder: None
    internal_nodes_as_tips: bool
    analysis_sha256: str

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["pair_role_counts"] = [list(item) for item in self.pair_role_counts]
        row["relationship_counts"] = [list(item) for item in self.relationship_counts]
        row["strata"] = [value.to_dict() for value in self.strata]
        row["controlled_regression"] = self.controlled_regression.to_dict()
        row["adjusted_regression"] = self.adjusted_regression.to_dict()
        return row


@dataclass(frozen=True)
class TreeSemanticAnalysis:
    topology: TreeTopology
    checkpoint_manifest: TreeCheckpointManifest
    node_covariates: TreeNodeCovariates
    direct: BoundDistanceArtifact
    semantic: BoundDistanceArtifact
    pair_rows: tuple[SemanticPairRow, ...]
    receipt: TreeSemanticReceipt


@dataclass(frozen=True)
class DLSummary:
    defined: bool
    n_trees: int
    fisher_z: float | None
    correlation: float | None
    se_z: float | None
    ci_low_r: float | None
    ci_high_r: float | None
    p_value: float | None
    q: float | None
    q_df: int | None
    q_p_value: float | None
    tau2: float | None
    i2: float | None
    undefined_reason: str | None
    within_tree_variance: str = "delete_one_trained_node_jackknife_on_fisher_z"
    pair_count_variance_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunSetStratum:
    stratum: str
    n_trees: int
    n_defined_trees: int
    n_undefined_trees: int
    n_pairs_total_descriptive_only: int
    equal_tree_fisher_r: float | None
    empirical_tree_se_z: float | None
    empirical_tree_ci_low_r: float | None
    empirical_tree_ci_high_r: float | None
    empirical_tree_p_value: float | None
    dl: DLSummary
    legacy_comparability_dl: DLSummary
    pair_count_used_as_independent_sample_size: bool = False

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["dl"] = self.dl.to_dict()
        row["legacy_comparability_dl"] = self.legacy_comparability_dl.to_dict()
        return row


@dataclass(frozen=True)
class SlopeSummary:
    model_id: str
    n_trees: int
    n_defined_trees: int
    mean_direct_slope_equal_tree_weight: float | None
    sample_sd_across_trees: float | None
    standard_error_across_trees: float | None
    ci_low: float | None
    ci_high: float | None
    p_value: float | None
    inference_unit: str = INFERENCE_UNIT
    pair_level_standard_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClusteredRegressionSummary:
    """One pooled tree-fixed-effect OLS with tree-clustered inference."""

    model_id: str
    stratum: str
    formula: str
    covariates: tuple[str, ...]
    defined: bool
    n_trees: int
    n_pairs: int
    n_parameters: int
    direct_slope: float | None
    tree_clustered_se: float | None
    ci_low: float | None
    ci_high: float | None
    p_value_normal_approximation: float | None
    r_squared: float | None
    covariance_estimator: str
    undefined_reason: str | None
    pair_level_standard_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["covariates"] = list(self.covariates)
        return row


@dataclass(frozen=True)
class LegacyHC3RegressionSummary:
    """Exact inherited pooled OLS/HC3 result, retained for comparability only.

    Pair rows overlap at nodes, so this covariance is not the primary
    inferential result.  The point estimate is useful for strict old-vs-new
    comparability; tree-clustered and equal-tree results remain separately
    labeled above.
    """

    model_id: str
    stratum: str
    formula: str
    covariates: tuple[str, ...]
    defined: bool
    n_trees: int
    n_pairs: int
    n_parameters: int
    direct_slope: float | None
    hc3_standard_error: float | None
    ci_low: float | None
    ci_high: float | None
    p_value_normal_approximation: float | None
    r_squared: float | None
    adjusted_r_squared: float | None
    covariance_estimator: str
    undefined_reason: str | None
    legacy_comparability_only: bool = True
    pair_count_used_as_independent_sample_size: bool = True

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["covariates"] = list(self.covariates)
        return row


@dataclass(frozen=True)
class RunSetSemanticReceipt:
    schema: str
    valid: bool
    cohort_key: str
    training_cohort_id: str
    endpoint_id: str
    population: str
    internal_node_treatment: str
    inference_unit: str
    primary_aggregation: str
    root_excluded: bool
    n_trees: int
    n_nodes: int
    n_internal: int
    n_leaves: int
    per_tree_node_counts: tuple[int, ...]
    tree_ids: tuple[str, ...]
    tree_analysis_sha256: tuple[str, ...]
    strata: tuple[RunSetStratum, ...]
    raw_slope_summaries: tuple[SlopeSummary, ...]
    clustered_regressions: tuple[ClusteredRegressionSummary, ...]
    legacy_hc3_regressions: tuple[LegacyHC3RegressionSummary, ...]
    pair_count_used_as_independent_sample_size: bool
    legacy_pair_count_inference_present: bool
    leaf_sensitivity_derived_from_full_matrix: bool
    dl_variance_estimator: str
    strata_sha256: str
    analysis_sha256: str

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["per_tree_node_counts"] = list(self.per_tree_node_counts)
        row["tree_ids"] = list(self.tree_ids)
        row["tree_analysis_sha256"] = list(self.tree_analysis_sha256)
        row["strata"] = [value.to_dict() for value in self.strata]
        row["raw_slope_summaries"] = [value.to_dict() for value in self.raw_slope_summaries]
        row["clustered_regressions"] = [value.to_dict() for value in self.clustered_regressions]
        row["legacy_hc3_regressions"] = [
            value.to_dict() for value in self.legacy_hc3_regressions
        ]
        return row


@dataclass(frozen=True)
class _PooledModelSpec:
    clustered_model_id: str
    legacy_hc3_model_id: str | None
    stratum: str
    covariates: tuple[str, ...]


@dataclass(frozen=True)
class _WithinTreeCenter:
    n_pairs: int
    mean_predictors: tuple[float, ...]
    mean_outcome: float


@dataclass
class _StreamingModelState:
    spec: _PooledModelSpec
    gram: list[list[float]]
    target: list[float]
    within_outcome_ss: float
    n_pairs: int
    outcome_sum: float
    outcome_square_sum: float
    groups: list[_WithinTreeCenter]

    @classmethod
    def create(cls, spec: _PooledModelSpec) -> "_StreamingModelState":
        width = 1 + len(spec.covariates)
        return cls(
            spec=spec,
            gram=[[0.0] * width for _ in range(width)],
            target=[0.0] * width,
            within_outcome_ss=0.0,
            n_pairs=0,
            outcome_sum=0.0,
            outcome_square_sum=0.0,
            groups=[],
        )

    def add_tree(self, rows: Sequence[SemanticPairRow]) -> None:
        pairs = _select_stratum(rows, self.spec.stratum)
        if not pairs:
            raise ValueError(
                f"tree has no pairs for pooled stratum {self.spec.stratum!r}"
            )
        width = 1 + len(self.spec.covariates)
        means = tuple(
            math.fsum(
                _pooled_predictors(pair, self.spec.covariates)[column]
                for pair in pairs
            )
            / len(pairs)
            for column in range(width)
        )
        mean_y = math.fsum(pair.semantic_similarity for pair in pairs) / len(pairs)
        self.groups.append(_WithinTreeCenter(len(pairs), means, mean_y))
        self.n_pairs += len(pairs)
        self.outcome_sum += math.fsum(pair.semantic_similarity for pair in pairs)
        self.outcome_square_sum += math.fsum(
            pair.semantic_similarity * pair.semantic_similarity for pair in pairs
        )
        for pair in pairs:
            predictors = _pooled_predictors(pair, self.spec.covariates)
            outcome = pair.semantic_similarity
            centered = [value - mean for value, mean in zip(predictors, means)]
            centered_y = outcome - mean_y
            self.within_outcome_ss += centered_y * centered_y
            for left in range(width):
                self.target[left] += centered[left] * centered_y
                for right in range(width):
                    self.gram[left][right] += centered[left] * centered[right]


@dataclass
class _StreamingModelFit:
    state: _StreamingModelState
    coefficients: list[float] | None
    inverse_gram: list[list[float]] | None
    r_squared: float | None
    adjusted_r_squared: float | None
    cluster_scores: list[list[float]]
    hc3_meat: list[list[float]]
    hc3_invalid_reason: str | None


def bind_distance_artifact(
    matrix: ValidatedDistanceMatrix,
    *,
    artifact_kind: str,
    endpoint_id: str,
    checkpoint_set_sha256: str,
    source_receipt_sha256: str,
) -> BoundDistanceArtifact:
    """Construct an explicit matrix/checkpoint/source binding."""

    return BoundDistanceArtifact(
        artifact_kind=_text(artifact_kind, "artifact_kind"),
        endpoint_id=_text(endpoint_id, "endpoint_id"),
        checkpoint_set_sha256=_sha256(checkpoint_set_sha256, "checkpoint_set_sha256"),
        source_receipt_sha256=_sha256(source_receipt_sha256, "source_receipt_sha256"),
        matrix=matrix,
    )


def analyze_tree_semantic_bridge(
    topology: TreeTopology,
    checkpoint_manifest: TreeCheckpointManifest,
    *,
    cohort_key: str,
    endpoint_id: str,
    node_covariates: TreeNodeCovariates,
    direct: BoundDistanceArtifact,
    semantic: BoundDistanceArtifact,
) -> TreeSemanticAnalysis:
    """Build the full all-node and derived leaf semantic regressions for one tree."""

    tree, manifest = _verified_scope(topology, checkpoint_manifest)
    key = _text(cohort_key, "cohort_key")
    endpoint = _text(endpoint_id, "endpoint_id")
    covariates = _verified_covariates(node_covariates, topology=tree, manifest=manifest)
    direct_bound = _verified_artifact(
        direct,
        topology=tree,
        manifest=manifest,
        artifact_kind=DIRECT_ARTIFACT_KIND,
        endpoint_id="weight_cosine",
        expected_metric=DIRECT_METRIC,
    )
    semantic_bound = _verified_artifact(
        semantic,
        topology=tree,
        manifest=manifest,
        artifact_kind=SEMANTIC_ARTIFACT_KIND,
        endpoint_id=endpoint,
        expected_metric=SEMANTIC_METRIC,
    )
    if direct_bound.matrix.receipt.ordered_node_ids != semantic_bound.matrix.receipt.ordered_node_ids:
        raise ValueError("direct and semantic ordered node IDs differ")

    rows = _pair_rows(
        tree,
        covariates,
        direct_bound.matrix,
        semantic_bound.matrix,
    )
    expected_pairs = tree.n_nodes * (tree.n_nodes - 1) // 2
    if len(rows) != expected_pairs:
        raise AssertionError("all-node pair cardinality corruption")
    leaf_pairs = tree.n_leaves * (tree.n_leaves - 1) // 2
    strata = tuple(
        _simple_regression(name, _select_stratum(rows, name)) for name in STRATA
    )
    controlled = _controlled_regression(rows)
    adjusted = _adjusted_regression(rows)
    pair_rows_sha = canonical_sha256([row.to_dict() for row in rows])
    strata_payload = {
        "strata": [row.to_dict() for row in strata],
        "controlled_regression": controlled.to_dict(),
        "adjusted_regression": adjusted.to_dict(),
    }
    role_counts = tuple(
        (role, sum(row.pair_role_class == role for row in rows))
        for role in PAIR_ROLE_CLASSES
    )
    relationship_counts = tuple(
        (name, sum(row.relationship == name for row in rows))
        for name in ("ancestor_descendant", "collateral")
    )
    core = {
        "schema": TREE_SCHEMA,
        "valid": True,
        "cohort_key": key,
        "training_cohort_id": manifest.cohort_id,
        "endpoint_id": endpoint,
        "tree_id": tree.tree_id,
        "population": POPULATION,
        "internal_node_treatment": INTERNAL_NODE_TREATMENT,
        "inference_unit": INFERENCE_UNIT,
        "root_excluded": True,
        "n_nodes": tree.n_nodes,
        "n_internal": tree.n_internal,
        "n_leaves": tree.n_leaves,
        "n_pairs": len(rows),
        "n_leaf_sensitivity_pairs": leaf_pairs,
        "topology_sha256": tree.topology_sha256,
        "pair_index_sha256": tree.pair_index_sha256,
        "leaf_subset_sha256": tree.leaf_subset_sha256,
        "checkpoint_set_sha256": manifest.checkpoint_set_sha256,
        "checkpoint_bindings_sha256": manifest.bindings_sha256,
        "covariate_source_manifest_sha256": covariates.source_manifest.sha256,
        "node_covariates_sha256": covariates.covariates_sha256,
        "direct_metric": direct_bound.matrix.receipt.metric,
        "direct_matrix_sha256": direct_bound.matrix.receipt.matrix_sha256,
        "direct_source_receipt_sha256": direct_bound.source_receipt_sha256,
        "semantic_metric": semantic_bound.matrix.receipt.metric,
        "semantic_matrix_sha256": semantic_bound.matrix.receipt.matrix_sha256,
        "semantic_source_receipt_sha256": semantic_bound.source_receipt_sha256,
        "ordered_node_ids_sha256": direct_bound.matrix.receipt.ordered_node_ids_sha256,
        "pair_rows_sha256": pair_rows_sha,
        "pair_role_counts": [list(item) for item in role_counts],
        "relationship_counts": [list(item) for item in relationship_counts],
        "internal_vs_leaf_indicator": "pair_has_internal=(pair_role_class!=leaf_leaf)",
        "strata": [row.to_dict() for row in strata],
        "controlled_regression": controlled.to_dict(),
        "adjusted_regression": adjusted.to_dict(),
        "strata_sha256": canonical_sha256(strata_payload),
        "neighbor_joining": False,
        "tree_builder": None,
        "internal_nodes_as_tips": False,
    }
    receipt = TreeSemanticReceipt(
        **{key_: value for key_, value in core.items() if key_ not in {"strata", "controlled_regression", "adjusted_regression", "pair_role_counts", "relationship_counts"}},
        pair_role_counts=role_counts,
        relationship_counts=relationship_counts,
        strata=strata,
        controlled_regression=controlled,
        adjusted_regression=adjusted,
        analysis_sha256=canonical_sha256(core),
    )
    return TreeSemanticAnalysis(
        topology=tree,
        checkpoint_manifest=manifest,
        node_covariates=covariates,
        direct=direct_bound,
        semantic=semantic_bound,
        pair_rows=rows,
        receipt=receipt,
    )


def aggregate_semantic_runset(
    analyses: Sequence[TreeSemanticAnalysis],
    *,
    expected_tree_ids: Sequence[str],
    expected_n_nodes: int,
    expected_n_internal: int,
    expected_n_leaves: int,
) -> RunSetSemanticReceipt:
    """In-memory convenience wrapper over the canonical streaming estimator."""

    values = tuple(analyses)
    return aggregate_semantic_runset_streaming(
        lambda: iter(values),
        expected_tree_ids=expected_tree_ids,
        expected_n_nodes=expected_n_nodes,
        expected_n_internal=expected_n_internal,
        expected_n_leaves=expected_n_leaves,
    )


def aggregate_production_semantic_runset(
    analyses: Sequence[TreeSemanticAnalysis],
    *,
    expected_nodes_per_tree: Sequence[int],
) -> RunSetSemanticReceipt:
    """Enforce production scope against manifest-derived per-tree counts."""

    values = tuple(analyses)
    return aggregate_semantic_runset_streaming(
        lambda: iter(values),
        expected_tree_ids=PRODUCTION_TREE_IDS,
        expected_n_nodes=PRODUCTION_N_NODES,
        expected_n_internal=PRODUCTION_N_INTERNAL,
        expected_n_leaves=PRODUCTION_N_LEAVES,
        expected_nodes_per_tree=expected_nodes_per_tree,
    )


def aggregate_semantic_runset_streaming(
    analysis_factory: Callable[[], Iterable[TreeSemanticAnalysis]],
    *,
    expected_tree_ids: Sequence[str],
    expected_n_nodes: int,
    expected_n_internal: int,
    expected_n_leaves: int,
    expected_nodes_per_tree: Sequence[int] | None = None,
) -> RunSetSemanticReceipt:
    """Aggregate without retaining pair rows from more than one tree.

    ``analysis_factory`` must be replayable: it is called exactly twice and
    must yield the same canonical tree sequence on each pass.  The first pass
    stores only per-tree receipts plus centered cross-products.  The second
    pass computes tree-cluster scores and exact HC3 leverage contributions.
    At no point does this function retain another tree's ``SemanticPairRow``
    objects.  The factory may reconstruct analyses from immutable artifacts or
    spool them externally; this pure analysis module performs no I/O itself.
    """

    if not callable(analysis_factory):
        raise ValueError("analysis_factory must be callable and replayable")
    expected_ids = tuple(_text(value, "expected_tree_id") for value in expected_tree_ids)
    if not expected_ids or len(expected_ids) != len(set(expected_ids)):
        raise ValueError("expected tree IDs must be nonempty and unique")
    specs = _pooled_model_specs()
    states = tuple(_StreamingModelState.create(spec) for spec in specs)
    receipts: list[TreeSemanticReceipt] = []
    for tree_index, raw in enumerate(analysis_factory()):
        if tree_index >= len(expected_ids):
            raise ValueError("streaming semantic factory yielded an extra tree")
        analysis = _verified_tree_analysis(raw)
        if analysis.receipt.tree_id != expected_ids[tree_index]:
            raise ValueError("semantic tree order differs from the expected contract")
        receipts.append(analysis.receipt)
        for state in states:
            state.add_tree(analysis.pair_rows)
    if len(receipts) != len(expected_ids):
        raise ValueError(
            "streaming semantic factory yielded fewer trees than expected: "
            f"observed={len(receipts)}, expected={len(expected_ids)}"
        )
    receipt_values = tuple(receipts)
    per_tree_node_counts = tuple(receipt.n_nodes for receipt in receipt_values)
    if expected_nodes_per_tree is not None:
        declared_per_tree = tuple(
            _integer(value, "expected_nodes_per_tree", minimum=1)
            for value in expected_nodes_per_tree
        )
        if declared_per_tree != per_tree_node_counts:
            raise ValueError(
                "semantic per-tree node cardinalities differ: "
                f"observed={per_tree_node_counts}, expected={declared_per_tree}"
            )
    totals = _validate_runset_receipts(
        receipt_values,
        expected_ids=expected_ids,
        expected_n_nodes=expected_n_nodes,
        expected_n_internal=expected_n_internal,
        expected_n_leaves=expected_n_leaves,
    )
    fits = tuple(_fit_streaming_model(state, len(expected_ids)) for state in states)

    second_pass_count = 0
    for tree_index, raw in enumerate(analysis_factory()):
        if tree_index >= len(expected_ids):
            raise ValueError("replayed semantic factory yielded an extra tree")
        analysis = _verified_tree_analysis(raw)
        expected_receipt = receipt_values[tree_index]
        if analysis.receipt != expected_receipt:
            raise ValueError(
                "replayed semantic factory differs from first-pass tree receipt"
            )
        for fit in fits:
            _update_streaming_inference(fit, analysis.pair_rows, tree_index)
        second_pass_count += 1
    if second_pass_count != len(expected_ids):
        raise ValueError(
            "analysis_factory is not replayable over the exact expected tree set"
        )

    summaries = tuple(
        _aggregate_stratum(
            stratum,
            tuple(receipt.strata[index] for receipt in receipt_values),
        )
        for index, stratum in enumerate(STRATA)
    )
    slope_summaries = (
        _slope_summary(
            PRIMARY_STRATUM, tuple(receipt.strata[0] for receipt in receipt_values)
        ),
        _slope_summary(
            CONTROLLED_MODEL_ID,
            tuple(receipt.controlled_regression for receipt in receipt_values),
        ),
        _slope_summary(
            ADJUSTED_MODEL_ID,
            tuple(receipt.adjusted_regression for receipt in receipt_values),
        ),
        _slope_summary(
            LEAF_SENSITIVITY_STRATUM,
            tuple(
                receipt.strata[STRATA.index(LEAF_SENSITIVITY_STRATUM)]
                for receipt in receipt_values
            ),
        ),
    )
    clustered = tuple(_streaming_clustered_summary(fit) for fit in fits)
    legacy_hc3 = tuple(
        _streaming_hc3_summary(fit)
        for fit in fits
        if fit.state.spec.legacy_hc3_model_id is not None
    )
    tree_hashes = tuple(receipt.analysis_sha256 for receipt in receipt_values)
    strata_payload = {
        "strata": [value.to_dict() for value in summaries],
        "raw_slope_summaries": [value.to_dict() for value in slope_summaries],
        "clustered_regressions": [value.to_dict() for value in clustered],
        "legacy_hc3_regressions": [value.to_dict() for value in legacy_hc3],
    }
    first = receipt_values[0]
    core = {
        "schema": RUNSET_SCHEMA,
        "valid": True,
        "cohort_key": first.cohort_key,
        "training_cohort_id": first.training_cohort_id,
        "endpoint_id": first.endpoint_id,
        "population": POPULATION,
        "internal_node_treatment": INTERNAL_NODE_TREATMENT,
        "inference_unit": INFERENCE_UNIT,
        "primary_aggregation": "equal_tree_fisher_z_with_empirical_across_tree_se",
        "root_excluded": True,
        "n_trees": len(receipt_values),
        "n_nodes": totals[0],
        "n_internal": totals[1],
        "n_leaves": totals[2],
        "per_tree_node_counts": list(per_tree_node_counts),
        "tree_ids": list(expected_ids),
        "tree_analysis_sha256": list(tree_hashes),
        **strata_payload,
        "pair_count_used_as_independent_sample_size": False,
        "legacy_pair_count_inference_present": True,
        "leaf_sensitivity_derived_from_full_matrix": True,
        "dl_variance_estimator": "delete_one_trained_node_jackknife_on_fisher_z",
        "strata_sha256": canonical_sha256(strata_payload),
    }
    return RunSetSemanticReceipt(
        **{
            key: value
            for key, value in core.items()
            if key
            not in {
                "tree_ids",
                "per_tree_node_counts",
                "tree_analysis_sha256",
                "strata",
                "raw_slope_summaries",
                "clustered_regressions",
                "legacy_hc3_regressions",
            }
        },
        per_tree_node_counts=per_tree_node_counts,
        tree_ids=expected_ids,
        tree_analysis_sha256=tree_hashes,
        strata=summaries,
        raw_slope_summaries=slope_summaries,
        clustered_regressions=clustered,
        legacy_hc3_regressions=legacy_hc3,
        analysis_sha256=canonical_sha256(core),
    )


def aggregate_production_semantic_runset_streaming(
    analysis_factory: Callable[[], Iterable[TreeSemanticAnalysis]],
    *,
    expected_nodes_per_tree: Sequence[int],
) -> RunSetSemanticReceipt:
    """Streaming production aggregation with manifest-derived tree sizes."""

    return aggregate_semantic_runset_streaming(
        analysis_factory,
        expected_tree_ids=PRODUCTION_TREE_IDS,
        expected_n_nodes=PRODUCTION_N_NODES,
        expected_n_internal=PRODUCTION_N_INTERNAL,
        expected_n_leaves=PRODUCTION_N_LEAVES,
        expected_nodes_per_tree=expected_nodes_per_tree,
    )


def _pooled_model_specs() -> tuple[_PooledModelSpec, ...]:
    return (
        _PooledModelSpec(
            "legacy_practical_all_nodes",
            "legacy_practical_all_nodes_hc3",
            PRIMARY_STRATUM,
            (),
        ),
        _PooledModelSpec(
            "legacy_controlled_all_nodes",
            "legacy_controlled_all_nodes_hc3",
            PRIMARY_STRATUM,
            CONTROLLED_COVARIATES,
        ),
        _PooledModelSpec(
            "internal_ancestry_adjusted_all_nodes",
            None,
            PRIMARY_STRATUM,
            ADJUSTED_COVARIATES,
        ),
        _PooledModelSpec(
            "legacy_practical_trained_leaf_sensitivity",
            "legacy_practical_trained_leaf_sensitivity_hc3",
            LEAF_SENSITIVITY_STRATUM,
            (),
        ),
        _PooledModelSpec(
            "legacy_controlled_trained_leaf_sensitivity",
            "legacy_controlled_trained_leaf_sensitivity_hc3",
            LEAF_SENSITIVITY_STRATUM,
            CONTROLLED_COVARIATES,
        ),
    )


def _validate_runset_receipts(
    receipts: Sequence[TreeSemanticReceipt],
    *,
    expected_ids: tuple[str, ...],
    expected_n_nodes: int,
    expected_n_internal: int,
    expected_n_leaves: int,
) -> tuple[int, int, int]:
    if tuple(receipt.tree_id for receipt in receipts) != expected_ids:
        raise ValueError("semantic tree order differs from the expected contract")
    totals = (
        sum(receipt.n_nodes for receipt in receipts),
        sum(receipt.n_internal for receipt in receipts),
        sum(receipt.n_leaves for receipt in receipts),
    )
    expected = (
        _integer(expected_n_nodes, "expected_n_nodes", minimum=1),
        _integer(expected_n_internal, "expected_n_internal", minimum=0),
        _integer(expected_n_leaves, "expected_n_leaves", minimum=1),
    )
    if totals != expected or totals[1] + totals[2] != totals[0]:
        raise ValueError(
            f"semantic runset node totals differ: observed={totals}, expected={expected}"
        )
    for receipt in receipts:
        if (
            receipt.valid is not True
            or receipt.population != POPULATION
            or receipt.internal_node_treatment != INTERNAL_NODE_TREATMENT
            or receipt.inference_unit != INFERENCE_UNIT
            or receipt.root_excluded is not True
            or tuple(value.stratum for value in receipt.strata) != STRATA
        ):
            raise ValueError("tree semantic receipt scope or stratum contract drift")
    for field in (
        "cohort_key",
        "training_cohort_id",
        "endpoint_id",
        "direct_metric",
        "semantic_metric",
    ):
        if len({getattr(receipt, field) for receipt in receipts}) != 1:
            raise ValueError(f"semantic runset mixes {field}")
    return totals


def _pooled_predictors(
    pair: SemanticPairRow, covariates: Sequence[str]
) -> tuple[float, ...]:
    return (
        pair.direct_distance,
        *(_pair_covariate(pair, name) for name in covariates),
    )


def _fit_streaming_model(
    state: _StreamingModelState, n_trees: int
) -> _StreamingModelFit:
    width = len(state.gram)
    n_parameters = n_trees + width
    inverse = _inverse(state.gram) if state.n_pairs > n_parameters else None
    if inverse is None:
        return _StreamingModelFit(
            state=state,
            coefficients=None,
            inverse_gram=None,
            r_squared=None,
            adjusted_r_squared=None,
            cluster_scores=[],
            hc3_meat=[[0.0] * width for _ in range(width)],
            hc3_invalid_reason=None,
        )
    coefficients = [
        math.fsum(inverse[row][column] * state.target[column] for column in range(width))
        for row in range(width)
    ]
    residual_ss = (
        state.within_outcome_ss
        - 2.0
        * math.fsum(value * target for value, target in zip(coefficients, state.target))
        + math.fsum(
            coefficients[left]
            * state.gram[left][right]
            * coefficients[right]
            for left in range(width)
            for right in range(width)
        )
    )
    if residual_ss < 0.0 and residual_ss > -1e-10:
        residual_ss = 0.0
    total_ss = state.outcome_square_sum - state.outcome_sum**2 / state.n_pairs
    if residual_ss < 0.0 or total_ss <= 0.0:
        return _StreamingModelFit(
            state=state,
            coefficients=None,
            inverse_gram=None,
            r_squared=None,
            adjusted_r_squared=None,
            cluster_scores=[],
            hc3_meat=[[0.0] * width for _ in range(width)],
            hc3_invalid_reason="invalid_streaming_residual_or_total_sum_of_squares",
        )
    r_squared = 1.0 - residual_ss / total_ss
    adjusted = 1.0 - (1.0 - r_squared) * (
        (state.n_pairs - 1.0) / (state.n_pairs - n_parameters)
    )
    return _StreamingModelFit(
        state=state,
        coefficients=coefficients,
        inverse_gram=inverse,
        r_squared=r_squared,
        adjusted_r_squared=adjusted,
        cluster_scores=[],
        hc3_meat=[[0.0] * width for _ in range(width)],
        hc3_invalid_reason=None,
    )


def _update_streaming_inference(
    fit: _StreamingModelFit,
    rows: Sequence[SemanticPairRow],
    tree_index: int,
) -> None:
    state = fit.state
    pairs = _select_stratum(rows, state.spec.stratum)
    if tree_index >= len(state.groups):
        raise ValueError("streaming inference tree index exceeds first-pass groups")
    center = state.groups[tree_index]
    if len(pairs) != center.n_pairs:
        raise ValueError("replayed pooled-regression pair count drift")
    width = 1 + len(state.spec.covariates)
    score = [0.0] * width
    if fit.coefficients is None or fit.inverse_gram is None:
        fit.cluster_scores.append(score)
        return
    for pair in pairs:
        predictors = _pooled_predictors(pair, state.spec.covariates)
        centered = [
            value - mean for value, mean in zip(predictors, center.mean_predictors)
        ]
        residual = (
            pair.semantic_similarity
            - center.mean_outcome
            - math.fsum(
                coefficient * value
                for coefficient, value in zip(fit.coefficients, centered)
            )
        )
        for column in range(width):
            score[column] += centered[column] * residual
        if fit.hc3_invalid_reason is not None:
            continue
        leverage = 1.0 / center.n_pairs + math.fsum(
            centered[left]
            * fit.inverse_gram[left][right]
            * centered[right]
            for left in range(width)
            for right in range(width)
        )
        denominator = 1.0 - leverage
        if not math.isfinite(leverage) or denominator <= 1e-12:
            fit.hc3_invalid_reason = "invalid_hc3_leverage"
            continue
        scale = (residual / denominator) ** 2
        for left in range(width):
            for right in range(width):
                fit.hc3_meat[left][right] += (
                    scale * centered[left] * centered[right]
                )
    fit.cluster_scores.append(score)


def _streaming_clustered_summary(
    fit: _StreamingModelFit,
) -> ClusteredRegressionSummary:
    state = fit.state
    spec = state.spec
    n_trees = len(state.groups)
    width = 1 + len(spec.covariates)
    n_parameters = n_trees + width
    base = {
        "model_id": spec.clustered_model_id,
        "stratum": spec.stratum,
        "formula": _pooled_formula(spec.covariates),
        "covariates": spec.covariates,
        "n_trees": n_trees,
        "n_pairs": state.n_pairs,
        "n_parameters": n_parameters,
        "covariance_estimator": "CR1_tree_clustered_sandwich",
    }
    if (
        fit.coefficients is None
        or fit.inverse_gram is None
        or n_trees < 2
        or len(fit.cluster_scores) != n_trees
    ):
        return ClusteredRegressionSummary(
            **base,
            defined=False,
            direct_slope=None,
            tree_clustered_se=None,
            ci_low=None,
            ci_high=None,
            p_value_normal_approximation=None,
            r_squared=None,
            undefined_reason="rank_deficient_or_saturated_fixed_effect_design",
        )
    meat = [
        [
            math.fsum(score[left] * score[right] for score in fit.cluster_scores)
            for right in range(width)
        ]
        for left in range(width)
    ]
    covariance = _matmul(_matmul(fit.inverse_gram, meat), fit.inverse_gram)
    correction = (n_trees / (n_trees - 1.0)) * (
        (state.n_pairs - 1.0) / (state.n_pairs - n_parameters)
    )
    variance = covariance[0][0] * correction
    if not math.isfinite(variance) or variance < -1e-12:
        raise ValueError("streaming tree-clustered direct-slope variance is invalid")
    return _clustered_defined_summary(base, fit.coefficients[0], variance, fit.r_squared)


def _streaming_hc3_summary(
    fit: _StreamingModelFit,
) -> LegacyHC3RegressionSummary:
    state = fit.state
    spec = state.spec
    if spec.legacy_hc3_model_id is None:
        raise AssertionError("streaming HC3 requested for a nonlegacy model")
    n_trees = len(state.groups)
    width = 1 + len(spec.covariates)
    n_parameters = n_trees + width
    base = {
        "model_id": spec.legacy_hc3_model_id,
        "stratum": spec.stratum,
        "formula": _pooled_formula(spec.covariates),
        "covariates": spec.covariates,
        "n_trees": n_trees,
        "n_pairs": state.n_pairs,
        "n_parameters": n_parameters,
        "covariance_estimator": "HC3_pair_row_legacy_comparability",
    }
    if (
        fit.coefficients is None
        or fit.inverse_gram is None
        or fit.hc3_invalid_reason is not None
    ):
        return LegacyHC3RegressionSummary(
            **base,
            defined=False,
            direct_slope=None,
            hc3_standard_error=None,
            ci_low=None,
            ci_high=None,
            p_value_normal_approximation=None,
            r_squared=None,
            adjusted_r_squared=None,
            undefined_reason=(
                fit.hc3_invalid_reason
                or "rank_deficient_or_saturated_fixed_effect_design"
            ),
        )
    covariance = _matmul(
        _matmul(fit.inverse_gram, fit.hc3_meat), fit.inverse_gram
    )
    variance = covariance[0][0]
    if not math.isfinite(variance) or variance < -1e-12:
        raise ValueError("streaming legacy HC3 direct-slope variance is invalid")
    se, low, high, p_value = _normal_inference(fit.coefficients[0], variance)
    return LegacyHC3RegressionSummary(
        **base,
        defined=True,
        direct_slope=fit.coefficients[0],
        hc3_standard_error=se,
        ci_low=low,
        ci_high=high,
        p_value_normal_approximation=p_value,
        r_squared=fit.r_squared,
        adjusted_r_squared=fit.adjusted_r_squared,
        undefined_reason=None,
    )


def _clustered_defined_summary(
    base: Mapping[str, Any],
    slope: float,
    variance: float,
    r_squared: float | None,
) -> ClusteredRegressionSummary:
    se, low, high, p_value = _normal_inference(slope, variance)
    return ClusteredRegressionSummary(
        **base,
        defined=True,
        direct_slope=slope,
        tree_clustered_se=se,
        ci_low=low,
        ci_high=high,
        p_value_normal_approximation=p_value,
        r_squared=r_squared,
        undefined_reason=None,
    )


def _normal_inference(
    estimate: float, variance: float
) -> tuple[float, float, float, float]:
    se = math.sqrt(max(0.0, variance))
    critical = NormalDist().inv_cdf(0.975)
    p_value = (
        0.0
        if se == 0.0 and estimate != 0.0
        else 1.0
        if se == 0.0
        else math.erfc(abs(estimate / se) / math.sqrt(2.0))
    )
    return se, estimate - critical * se, estimate + critical * se, p_value


def _verified_scope(
    topology: TreeTopology,
    checkpoint_manifest: TreeCheckpointManifest,
) -> tuple[TreeTopology, TreeCheckpointManifest]:
    if not isinstance(topology, TreeTopology):
        raise ValueError("topology must be a TreeTopology")
    if not isinstance(checkpoint_manifest, TreeCheckpointManifest):
        raise ValueError("checkpoint_manifest must be a TreeCheckpointManifest")
    tree = read_tree_topology(topology.to_dict())
    manifest = read_tree_checkpoint_manifest(checkpoint_manifest.to_dict())
    if manifest.tree_id != tree.tree_id:
        raise ValueError("checkpoint manifest tree differs from topology")
    for field in (
        "topology_sha256",
        "pair_index_sha256",
        "leaf_subset_sha256",
        "nodes_sha256",
    ):
        if getattr(manifest, field) != getattr(tree, field):
            raise ValueError(f"checkpoint manifest {field} differs from topology")
    if tuple(binding.node_id for binding in manifest.bindings) != tree.ordered_node_ids:
        raise ValueError("checkpoint binding order differs from topology")
    if tree.root_id in tree.ordered_node_ids:
        raise ValueError("untrained root is present in the analysis population")
    return tree, manifest


def _verified_artifact(
    artifact: BoundDistanceArtifact,
    *,
    topology: TreeTopology,
    manifest: TreeCheckpointManifest,
    artifact_kind: str,
    endpoint_id: str,
    expected_metric: str | None,
) -> BoundDistanceArtifact:
    if not isinstance(artifact, BoundDistanceArtifact):
        raise ValueError("analysis distance input must be a BoundDistanceArtifact")
    if artifact.artifact_kind != artifact_kind:
        raise ValueError(f"distance artifact kind must be exactly {artifact_kind!r}")
    if artifact.endpoint_id != endpoint_id:
        raise ValueError("distance artifact endpoint differs from the analysis endpoint")
    _sha256(artifact.source_receipt_sha256, "source_receipt_sha256")
    if artifact.checkpoint_set_sha256 != manifest.checkpoint_set_sha256:
        raise ValueError("distance artifact checkpoint set differs from manifest")
    if not isinstance(artifact.matrix, ValidatedDistanceMatrix):
        raise ValueError("distance artifact matrix must be validated")
    rebuilt = validate_matrix_receipt(
        artifact.matrix.receipt.to_dict(),
        topology,
        metric=artifact.matrix.receipt.metric,
        ordered_node_ids=artifact.matrix.receipt.ordered_node_ids,
        matrix=artifact.matrix.matrix,
    )
    if rebuilt != artifact.matrix:
        raise ValueError("distance artifact differs from its recomputed matrix contract")
    if expected_metric is not None and rebuilt.receipt.metric != expected_metric:
        raise ValueError(f"distance metric must be exactly {expected_metric!r}")
    upper = 2.0
    if any(value > upper for row in rebuilt.matrix for value in row):
        raise ValueError("cosine-distance matrix contains a value above two")
    return artifact


def _verified_covariates(
    value: TreeNodeCovariates,
    *,
    topology: TreeTopology,
    manifest: TreeCheckpointManifest,
) -> TreeNodeCovariates:
    return validate_tree_node_covariates(value, topology, manifest)


def _pair_rows(
    topology: TreeTopology,
    covariates: TreeNodeCovariates,
    direct: ValidatedDistanceMatrix,
    semantic: ValidatedDistanceMatrix,
) -> tuple[SemanticPairRow, ...]:
    rows: list[SemanticPairRow] = []
    by_id = {node.node_id: node for node in topology.nodes}
    covariate_by_id = {row.node_id: row for row in covariates.nodes}
    for ordinal, pair in enumerate(topology.pair_index):
        if pair.pair_ordinal != ordinal:
            raise ValueError("topology pair ordinals are not contiguous")
        left, right = pair.left_ordinal, pair.right_ordinal
        if not 0 <= left < right < topology.n_nodes:
            raise ValueError("topology pair coordinate is invalid")
        left_role = topology.nodes[left].node_role
        right_role = topology.nodes[right].node_role
        if (left_role, right_role) != (pair.left_role, pair.right_role):
            raise ValueError("topology pair roles differ from node roles")
        if left_role == right_role == "leaf":
            role_class = "leaf_leaf"
        elif left_role == right_role == "internal":
            role_class = "internal_internal"
        else:
            role_class = "internal_leaf"
        left_covariate = covariate_by_id[pair.left_node_id]
        right_covariate = covariate_by_id[pair.right_node_id]
        lca_depth = 0 if pair.mrca_node_id == topology.root_id else by_id[pair.mrca_node_id].depth
        semantic_distance = semantic.matrix[left][right]
        rows.append(
            SemanticPairRow(
                pair_ordinal=ordinal,
                left_ordinal=left,
                right_ordinal=right,
                left_node_id=pair.left_node_id,
                right_node_id=pair.right_node_id,
                left_role=left_role,
                right_role=right_role,
                pair_role_class=role_class,
                pair_has_internal=role_class != "leaf_leaf",
                relationship=pair.relationship,
                is_parent_child=pair.is_parent_child,
                topological_distance=pair.topological_distance,
                depth_difference=pair.depth_difference,
                left_depth=topology.nodes[left].depth,
                right_depth=topology.nodes[right].depth,
                lca_depth=lca_depth,
                left_dataset_id=left_covariate.dataset_id,
                right_dataset_id=right_covariate.dataset_id,
                left_task_family=left_covariate.task_family,
                right_task_family=right_covariate.task_family,
                same_dataset=left_covariate.dataset_id == right_covariate.dataset_id,
                same_task_family=left_covariate.task_family == right_covariate.task_family,
                dataset_path_jaccard=task_path_jaccard(
                    covariates, pair.left_node_id, pair.right_node_id
                ),
                task_family_path_jaccard=task_path_jaccard(
                    covariates,
                    pair.left_node_id,
                    pair.right_node_id,
                    label="task_family",
                ),
                direct_distance=direct.matrix[left][right],
                semantic_distance=semantic_distance,
                semantic_similarity=1.0 - semantic_distance,
            )
        )
    return tuple(rows)


def _select_stratum(
    rows: Sequence[SemanticPairRow], stratum: str
) -> tuple[SemanticPairRow, ...]:
    if stratum == "all_trained_node_pairs":
        return tuple(rows)
    if stratum == "ancestor_descendant":
        return tuple(row for row in rows if row.relationship == "ancestor_descendant")
    if stratum == "parent_child":
        return tuple(row for row in rows if row.is_parent_child)
    if stratum == "collateral":
        return tuple(row for row in rows if row.relationship == "collateral")
    if stratum == "pairs_involving_internal":
        return tuple(row for row in rows if row.pair_has_internal)
    if stratum in PAIR_ROLE_CLASSES:
        return tuple(row for row in rows if row.pair_role_class == stratum)
    if stratum == "trained_leaf_pairs":
        return tuple(row for row in rows if row.pair_role_class == "leaf_leaf")
    raise ValueError(f"unknown semantic stratum {stratum!r}")


def _simple_regression(
    stratum: str, rows: Sequence[SemanticPairRow]
) -> TreeRegression:
    pairs = tuple(rows)
    nodes = sorted({ordinal for row in pairs for ordinal in (row.left_ordinal, row.right_ordinal)})
    formula = f"{OUTCOME} ~ {PREDICTOR}"
    result = _simple_values(
        [row.direct_distance for row in pairs],
        [row.semantic_similarity for row in pairs],
    )
    if result[0] is None:
        return TreeRegression(
            model_id=stratum,
            stratum=stratum,
            formula=formula,
            covariates=(),
            n_pairs=len(pairs),
            n_unique_nodes=len(nodes),
            defined=False,
            coefficients=(),
            direct_slope=None,
            standardized_direct_slope=None,
            pearson_r=None,
            r_squared=None,
            jackknife_variance_fisher_z=None,
            jackknife_n_nodes=len(nodes),
            undefined_reason=result[4],
        )
    slope, intercept, correlation, r_squared, _ = result
    jackknife_variance, jackknife_reason = _node_jackknife_variance(pairs, nodes)
    reason = None if jackknife_variance is not None else jackknife_reason
    # The regression is defined even when a small stratum cannot support its
    # secondary jackknife variance; that absence remains explicit.
    return TreeRegression(
        model_id=stratum,
        stratum=stratum,
        formula=formula,
        covariates=(),
        n_pairs=len(pairs),
        n_unique_nodes=len(nodes),
        defined=True,
        coefficients=(Coefficient("intercept", intercept), Coefficient(PREDICTOR, slope)),
        direct_slope=slope,
        standardized_direct_slope=correlation,
        pearson_r=correlation,
        r_squared=r_squared,
        jackknife_variance_fisher_z=jackknife_variance,
        jackknife_n_nodes=len(nodes),
        undefined_reason=reason,
    )


def _adjusted_regression(rows: Sequence[SemanticPairRow]) -> TreeRegression:
    pairs = tuple(rows)
    nodes = {ordinal for row in pairs for ordinal in (row.left_ordinal, row.right_ordinal)}
    formula = (
        f"{OUTCOME} ~ {PREDICTOR} + pair_has_internal + "
        "ancestor_descendant + parent_child"
    )
    design = [
        [
            1.0,
            row.direct_distance,
            float(row.pair_has_internal),
            float(row.relationship == "ancestor_descendant"),
            float(row.is_parent_child),
        ]
        for row in pairs
    ]
    outcome = [row.semantic_similarity for row in pairs]
    terms = ("intercept", PREDICTOR, *ADJUSTED_COVARIATES)
    coefficients, r_squared, reason = _ols(design, outcome)
    if coefficients is None:
        return TreeRegression(
            model_id=ADJUSTED_MODEL_ID,
            stratum=PRIMARY_STRATUM,
            formula=formula,
            covariates=ADJUSTED_COVARIATES,
            n_pairs=len(pairs),
            n_unique_nodes=len(nodes),
            defined=False,
            coefficients=(),
            direct_slope=None,
            standardized_direct_slope=None,
            pearson_r=None,
            r_squared=None,
            jackknife_variance_fisher_z=None,
            jackknife_n_nodes=0,
            undefined_reason=reason,
        )
    return TreeRegression(
        model_id=ADJUSTED_MODEL_ID,
        stratum=PRIMARY_STRATUM,
        formula=formula,
        covariates=ADJUSTED_COVARIATES,
        n_pairs=len(pairs),
        n_unique_nodes=len(nodes),
        defined=True,
        coefficients=tuple(Coefficient(term, value) for term, value in zip(terms, coefficients)),
        direct_slope=coefficients[1],
        standardized_direct_slope=None,
        pearson_r=None,
        r_squared=r_squared,
        jackknife_variance_fisher_z=None,
        jackknife_n_nodes=0,
        undefined_reason=None,
    )


def _controlled_regression(rows: Sequence[SemanticPairRow]) -> TreeRegression:
    """Inherited controlled formula, fit descriptively within one tree.

    The historical ``C(run)`` term reduces to this intercept in a per-tree fit.
    Runset inference additionally fits the exact pooled tree-fixed-effect form
    with a tree-clustered sandwich covariance.
    """

    pairs = tuple(rows)
    nodes = {ordinal for row in pairs for ordinal in (row.left_ordinal, row.right_ordinal)}
    formula = (
        f"{LEGACY_OUTCOME} ~ depth_i + depth_j + lca_depth + "
        f"task_jaccard + {LEGACY_PREDICTOR}"
    )
    design = [
        [
            1.0,
            row.direct_distance,
            float(row.left_depth),
            float(row.right_depth),
            float(row.lca_depth),
            row.dataset_path_jaccard,
        ]
        for row in pairs
    ]
    outcome = [row.semantic_similarity for row in pairs]
    terms = ("intercept", LEGACY_PREDICTOR, *CONTROLLED_COVARIATES)
    coefficients, r_squared, reason = _ols(design, outcome)
    if coefficients is None:
        return TreeRegression(
            model_id=CONTROLLED_MODEL_ID,
            stratum=PRIMARY_STRATUM,
            formula=formula,
            covariates=CONTROLLED_COVARIATES,
            n_pairs=len(pairs),
            n_unique_nodes=len(nodes),
            defined=False,
            coefficients=(),
            direct_slope=None,
            standardized_direct_slope=None,
            pearson_r=None,
            r_squared=None,
            jackknife_variance_fisher_z=None,
            jackknife_n_nodes=0,
            undefined_reason=reason,
        )
    return TreeRegression(
        model_id=CONTROLLED_MODEL_ID,
        stratum=PRIMARY_STRATUM,
        formula=formula,
        covariates=CONTROLLED_COVARIATES,
        n_pairs=len(pairs),
        n_unique_nodes=len(nodes),
        defined=True,
        coefficients=tuple(Coefficient(term, value) for term, value in zip(terms, coefficients)),
        direct_slope=coefficients[1],
        standardized_direct_slope=None,
        pearson_r=None,
        r_squared=r_squared,
        jackknife_variance_fisher_z=None,
        jackknife_n_nodes=0,
        undefined_reason=None,
    )


def _simple_values(
    x: Sequence[float], y: Sequence[float]
) -> tuple[float | None, float | None, float | None, float | None, str | None]:
    if len(x) != len(y):
        raise ValueError("regression vectors differ in length")
    if len(x) < 2:
        return None, None, None, None, "fewer_than_two_pairs"
    if any(not math.isfinite(value) for value in (*x, *y)):
        raise ValueError("regression vector contains a non-finite value")
    mean_x = math.fsum(x) / len(x)
    mean_y = math.fsum(y) / len(y)
    centered_x = [value - mean_x for value in x]
    centered_y = [value - mean_y for value in y]
    ss_x = math.fsum(value * value for value in centered_x)
    ss_y = math.fsum(value * value for value in centered_y)
    if ss_x == 0.0:
        return None, None, None, None, "constant_direct_distance"
    if ss_y == 0.0:
        return None, None, None, None, "constant_semantic_similarity"
    covariance = math.fsum(a * b for a, b in zip(centered_x, centered_y))
    slope = covariance / ss_x
    intercept = mean_y - slope * mean_x
    correlation = covariance / math.sqrt(ss_x * ss_y)
    correlation = min(1.0, max(-1.0, correlation))
    return slope, intercept, correlation, correlation * correlation, None


def _node_jackknife_variance(
    rows: Sequence[SemanticPairRow], node_ordinals: Sequence[int]
) -> tuple[float | None, str | None]:
    if len(node_ordinals) < 4:
        return None, "fewer_than_four_jackknife_nodes"
    effects: list[float] = []
    for omitted in node_ordinals:
        retained = [
            row
            for row in rows
            if row.left_ordinal != omitted and row.right_ordinal != omitted
        ]
        if len(retained) <= 3:
            return None, "jackknife_replicate_has_at_most_three_pairs"
        result = _simple_values(
            [row.direct_distance for row in retained],
            [row.semantic_similarity for row in retained],
        )
        if result[2] is None:
            return None, f"jackknife_regression_undefined:{result[4]}"
        effects.append(math.atanh(_clip_correlation(result[2])))
    mean = math.fsum(effects) / len(effects)
    variance = (len(effects) - 1.0) / len(effects) * math.fsum(
        (value - mean) ** 2 for value in effects
    )
    if not math.isfinite(variance) or variance <= 0.0:
        return None, "nonpositive_jackknife_variance"
    return variance, None


def _ols(
    design: Sequence[Sequence[float]], outcome: Sequence[float]
) -> tuple[tuple[float, ...] | None, float | None, str | None]:
    rows = [tuple(float(value) for value in row) for row in design]
    values = tuple(float(value) for value in outcome)
    if len(rows) != len(values):
        raise ValueError("OLS design and outcome lengths differ")
    if not rows:
        return None, None, "empty_design"
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError("OLS design is ragged")
    if len(rows) <= width:
        return None, None, "not_more_observations_than_parameters"
    if any(not math.isfinite(value) for row in rows for value in row) or any(
        not math.isfinite(value) for value in values
    ):
        raise ValueError("OLS design contains a non-finite value")
    gram = [
        [math.fsum(row[left] * row[right] for row in rows) for right in range(width)]
        for left in range(width)
    ]
    target = [math.fsum(row[column] * value for row, value in zip(rows, values)) for column in range(width)]
    coefficients = _solve(gram, target)
    if coefficients is None:
        return None, None, "rank_deficient_design"
    fitted = [math.fsum(coefficient * value for coefficient, value in zip(coefficients, row)) for row in rows]
    mean_y = math.fsum(values) / len(values)
    total = math.fsum((value - mean_y) ** 2 for value in values)
    residual = math.fsum((value - prediction) ** 2 for value, prediction in zip(values, fitted))
    if total == 0.0:
        return None, None, "constant_semantic_similarity"
    r_squared = 1.0 - residual / total
    return tuple(coefficients), r_squared, None


def _solve(matrix: Sequence[Sequence[float]], target: Sequence[float]) -> list[float] | None:
    size = len(matrix)
    augmented = [list(row) + [float(target[index])] for index, row in enumerate(matrix)]
    scale = max((abs(value) for row in matrix for value in row), default=0.0)
    tolerance = max(1e-14, scale * 1e-12)
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= tolerance:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        for index in range(column, size + 1):
            augmented[column][index] /= pivot_value
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            for index in range(column, size + 1):
                augmented[row][index] -= factor * augmented[column][index]
    result = [augmented[index][size] for index in range(size)]
    return result if all(math.isfinite(value) for value in result) else None


def _aggregate_stratum(
    stratum: str, rows: Sequence[TreeRegression]
) -> RunSetStratum:
    values = tuple(rows)
    correlations = [
        row.pearson_r
        for row in values
        if row.defined and row.pearson_r is not None
    ]
    z_values = [math.atanh(_clip_correlation(value)) for value in correlations]
    equal_r, empirical_se, ci_low, ci_high, empirical_p = _equal_tree_effect(z_values)
    dl_rows = [
        (math.atanh(_clip_correlation(row.pearson_r)), row.jackknife_variance_fisher_z)
        for row in values
        if row.defined
        and row.pearson_r is not None
        and row.jackknife_variance_fisher_z is not None
    ]
    legacy_rows = [
        (math.atanh(_clip_correlation(row.pearson_r)), 1.0 / (row.n_pairs - 3.0))
        for row in values
        if row.defined and row.pearson_r is not None and row.n_pairs > 3
    ]
    return RunSetStratum(
        stratum=stratum,
        n_trees=len(values),
        n_defined_trees=len(correlations),
        n_undefined_trees=len(values) - len(correlations),
        n_pairs_total_descriptive_only=sum(row.n_pairs for row in values),
        equal_tree_fisher_r=equal_r,
        empirical_tree_se_z=empirical_se,
        empirical_tree_ci_low_r=ci_low,
        empirical_tree_ci_high_r=ci_high,
        empirical_tree_p_value=empirical_p,
        dl=_dl(
            dl_rows,
            variance_estimator="delete_one_trained_node_jackknife_on_fisher_z",
            pair_count_variance_used=False,
        ),
        legacy_comparability_dl=_dl(
            legacy_rows,
            variance_estimator="legacy_1_over_n_pairs_minus_3",
            pair_count_variance_used=True,
        ),
    )


def _equal_tree_effect(
    z_values: Sequence[float],
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    if not z_values:
        return None, None, None, None, None
    mean = math.fsum(z_values) / len(z_values)
    if len(z_values) < 2:
        return math.tanh(mean), None, None, None, None
    variance = math.fsum((value - mean) ** 2 for value in z_values) / (len(z_values) - 1)
    se = math.sqrt(variance / len(z_values))
    if se == 0.0:
        p_value = 0.0 if mean != 0.0 else 1.0
        return math.tanh(mean), 0.0, math.tanh(mean), math.tanh(mean), p_value
    critical = NormalDist().inv_cdf(0.975)
    p_value = math.erfc(abs(mean / se) / math.sqrt(2.0))
    return (
        math.tanh(mean),
        se,
        math.tanh(mean - critical * se),
        math.tanh(mean + critical * se),
        p_value,
    )


def _dl(
    rows: Sequence[tuple[float, float]],
    *,
    variance_estimator: str,
    pair_count_variance_used: bool,
) -> DLSummary:
    values = tuple((float(effect), float(variance)) for effect, variance in rows)
    if len(values) < 3:
        return DLSummary(
            defined=False,
            n_trees=len(values),
            fisher_z=None,
            correlation=None,
            se_z=None,
            ci_low_r=None,
            ci_high_r=None,
            p_value=None,
            q=None,
            q_df=None,
            q_p_value=None,
            tau2=None,
            i2=None,
            undefined_reason=(
                "fewer_than_three_trees_with_positive_node_jackknife_variance"
                if not pair_count_variance_used
                else "fewer_than_three_trees_with_more_than_three_pairs"
            ),
            within_tree_variance=variance_estimator,
            pair_count_variance_used=pair_count_variance_used,
        )
    if any(not math.isfinite(effect) or not math.isfinite(variance) or variance <= 0 for effect, variance in values):
        raise ValueError("DL inputs require finite effects and positive jackknife variances")
    effects = [row[0] for row in values]
    variances = [row[1] for row in values]
    weights = [1.0 / variance for variance in variances]
    weight_sum = math.fsum(weights)
    fixed = math.fsum(weight * effect for weight, effect in zip(weights, effects)) / weight_sum
    q = math.fsum(weight * (effect - fixed) ** 2 for weight, effect in zip(weights, effects))
    df = len(values) - 1
    c_value = weight_sum - math.fsum(weight * weight for weight in weights) / weight_sum
    if c_value <= 0.0:
        raise ValueError("DL C is nonpositive")
    tau2 = max(0.0, (q - df) / c_value)
    random_weights = [1.0 / (variance + tau2) for variance in variances]
    random_sum = math.fsum(random_weights)
    pooled = math.fsum(weight * effect for weight, effect in zip(random_weights, effects)) / random_sum
    se = math.sqrt(1.0 / random_sum)
    critical = NormalDist().inv_cdf(0.975)
    statistic = pooled / se
    i2 = max(0.0, (q - df) / q) if q > 0.0 else 0.0
    return DLSummary(
        defined=True,
        n_trees=len(values),
        fisher_z=pooled,
        correlation=math.tanh(pooled),
        se_z=se,
        ci_low_r=math.tanh(pooled - critical * se),
        ci_high_r=math.tanh(pooled + critical * se),
        p_value=math.erfc(abs(statistic) / math.sqrt(2.0)),
        q=q,
        q_df=df,
        q_p_value=_chi_square_survival(q, df),
        tau2=tau2,
        i2=i2,
        undefined_reason=None,
        within_tree_variance=variance_estimator,
        pair_count_variance_used=pair_count_variance_used,
    )


def _slope_summary(model_id: str, rows: Sequence[TreeRegression]) -> SlopeSummary:
    slopes = [row.direct_slope for row in rows if row.defined and row.direct_slope is not None]
    if not slopes:
        return SlopeSummary(
            model_id=model_id,
            n_trees=len(rows),
            n_defined_trees=0,
            mean_direct_slope_equal_tree_weight=None,
            sample_sd_across_trees=None,
            standard_error_across_trees=None,
            ci_low=None,
            ci_high=None,
            p_value=None,
        )
    mean = math.fsum(slopes) / len(slopes)
    if len(slopes) < 2:
        return SlopeSummary(
            model_id=model_id,
            n_trees=len(rows),
            n_defined_trees=len(slopes),
            mean_direct_slope_equal_tree_weight=mean,
            sample_sd_across_trees=None,
            standard_error_across_trees=None,
            ci_low=None,
            ci_high=None,
            p_value=None,
        )
    variance = math.fsum((value - mean) ** 2 for value in slopes) / (len(slopes) - 1)
    sample_sd = math.sqrt(variance)
    se = sample_sd / math.sqrt(len(slopes))
    critical = NormalDist().inv_cdf(0.975)
    if se == 0.0:
        p_value = 0.0 if mean != 0.0 else 1.0
    else:
        p_value = math.erfc(abs(mean / se) / math.sqrt(2.0))
    return SlopeSummary(
        model_id=model_id,
        n_trees=len(rows),
        n_defined_trees=len(slopes),
        mean_direct_slope_equal_tree_weight=mean,
        sample_sd_across_trees=sample_sd,
        standard_error_across_trees=se,
        ci_low=mean - critical * se,
        ci_high=mean + critical * se,
        p_value=p_value,
    )


def _clustered_fixed_effect_regression(
    analyses: Sequence[TreeSemanticAnalysis],
    *,
    model_id: str,
    stratum: str,
    covariates: Sequence[str],
) -> ClusteredRegressionSummary:
    """Fit the inherited pooled OLS shape with tree-clustered covariance."""

    trees = tuple(analyses)
    covariate_names = tuple(covariates)
    allowed = set(CONTROLLED_COVARIATES) | set(ADJUSTED_COVARIATES)
    if any(name not in allowed for name in covariate_names):
        raise ValueError("clustered regression contains an unknown covariate")
    pair_groups = [tuple(_select_stratum(tree.pair_rows, stratum)) for tree in trees]
    formula = _pooled_formula(covariate_names)
    # Reference coding: intercept, IV, explicit numeric/binary controls, then
    # one dummy for every tree except the first.
    design: list[list[float]] = []
    outcome: list[float] = []
    clusters: list[int] = []
    for tree_index, pairs in enumerate(pair_groups):
        for pair in pairs:
            row = [1.0, pair.direct_distance]
            row.extend(_pair_covariate(pair, name) for name in covariate_names)
            row.extend(
                1.0 if tree_index == dummy_tree else 0.0
                for dummy_tree in range(1, len(trees))
            )
            design.append(row)
            outcome.append(pair.semantic_similarity)
            clusters.append(tree_index)
    n_pairs = len(design)
    n_parameters = len(design[0]) if design else 0
    base = {
        "model_id": model_id,
        "stratum": stratum,
        "formula": formula,
        "covariates": covariate_names,
        "n_trees": len(trees),
        "n_pairs": n_pairs,
        "n_parameters": n_parameters,
        "covariance_estimator": "CR1_tree_clustered_sandwich",
    }
    if len(trees) < 2:
        return ClusteredRegressionSummary(
            **base,
            defined=False,
            direct_slope=None,
            tree_clustered_se=None,
            ci_low=None,
            ci_high=None,
            p_value_normal_approximation=None,
            r_squared=None,
            undefined_reason="fewer_than_two_tree_clusters",
        )
    details = _ols_details(design, outcome)
    if details is None:
        return ClusteredRegressionSummary(
            **base,
            defined=False,
            direct_slope=None,
            tree_clustered_se=None,
            ci_low=None,
            ci_high=None,
            p_value_normal_approximation=None,
            r_squared=None,
            undefined_reason="rank_deficient_or_saturated_fixed_effect_design",
        )
    coefficients, inverse_gram, residuals, r_squared = details
    cluster_scores: list[list[float]] = []
    for cluster in range(len(trees)):
        indices = [index for index, value in enumerate(clusters) if value == cluster]
        cluster_scores.append(
            [
                math.fsum(design[index][column] * residuals[index] for index in indices)
                for column in range(n_parameters)
            ]
        )
    meat = [
        [
            math.fsum(score[left] * score[right] for score in cluster_scores)
            for right in range(n_parameters)
        ]
        for left in range(n_parameters)
    ]
    covariance = _matmul(_matmul(inverse_gram, meat), inverse_gram)
    if n_pairs <= n_parameters:
        raise AssertionError("OLS details returned a saturated design")
    correction = (len(trees) / (len(trees) - 1.0)) * (
        (n_pairs - 1.0) / (n_pairs - n_parameters)
    )
    variance = covariance[1][1] * correction
    if not math.isfinite(variance) or variance < -1e-12:
        raise ValueError("tree-clustered direct-slope variance is invalid")
    se = math.sqrt(max(0.0, variance))
    slope = coefficients[1]
    critical = NormalDist().inv_cdf(0.975)
    p_value = (
        0.0 if se == 0.0 and slope != 0.0 else
        1.0 if se == 0.0 else
        math.erfc(abs(slope / se) / math.sqrt(2.0))
    )
    return ClusteredRegressionSummary(
        **base,
        defined=True,
        direct_slope=slope,
        tree_clustered_se=se,
        ci_low=slope - critical * se,
        ci_high=slope + critical * se,
        p_value_normal_approximation=p_value,
        r_squared=r_squared,
        undefined_reason=None,
    )


def _legacy_hc3_fixed_effect_regression(
    analyses: Sequence[TreeSemanticAnalysis],
    *,
    model_id: str,
    stratum: str,
    covariates: Sequence[str],
) -> LegacyHC3RegressionSummary:
    """Reproduce the inherited pooled ``statsmodels`` OLS/HC3 estimator.

    This function deliberately retains pair-row HC3 for strict historical
    comparability.  Because incident dyads share trained nodes, it is never
    relabeled as the primary inference; the same coefficient is also emitted
    with CR1 tree-clustered inference by
    :func:`_clustered_fixed_effect_regression`.
    """

    trees = tuple(analyses)
    covariate_names = tuple(covariates)
    if any(name not in CONTROLLED_COVARIATES for name in covariate_names):
        raise ValueError("legacy HC3 regression contains an unknown covariate")
    pair_groups = [tuple(_select_stratum(tree.pair_rows, stratum)) for tree in trees]
    design: list[list[float]] = []
    outcome: list[float] = []
    for tree_index, pairs in enumerate(pair_groups):
        for pair in pairs:
            row = [1.0, pair.direct_distance]
            row.extend(_pair_covariate(pair, name) for name in covariate_names)
            row.extend(
                1.0 if tree_index == dummy_tree else 0.0
                for dummy_tree in range(1, len(trees))
            )
            design.append(row)
            outcome.append(pair.semantic_similarity)
    n_pairs = len(design)
    n_parameters = len(design[0]) if design else 0
    base = {
        "model_id": model_id,
        "stratum": stratum,
        "formula": _pooled_formula(covariate_names),
        "covariates": covariate_names,
        "n_trees": len(trees),
        "n_pairs": n_pairs,
        "n_parameters": n_parameters,
        "covariance_estimator": "HC3_pair_row_legacy_comparability",
    }
    details = _ols_details(design, outcome)
    if details is None:
        return LegacyHC3RegressionSummary(
            **base,
            defined=False,
            direct_slope=None,
            hc3_standard_error=None,
            ci_low=None,
            ci_high=None,
            p_value_normal_approximation=None,
            r_squared=None,
            adjusted_r_squared=None,
            undefined_reason="rank_deficient_or_saturated_fixed_effect_design",
        )
    coefficients, inverse_gram, residuals, r_squared = details
    width = len(inverse_gram)
    meat = [[0.0 for _ in range(width)] for _ in range(width)]
    for row, residual in zip(design, residuals):
        leverage = math.fsum(
            row[left] * inverse_gram[left][right] * row[right]
            for left in range(width)
            for right in range(width)
        )
        denominator = 1.0 - leverage
        if not math.isfinite(leverage) or denominator <= 1e-12:
            return LegacyHC3RegressionSummary(
                **base,
                defined=False,
                direct_slope=None,
                hc3_standard_error=None,
                ci_low=None,
                ci_high=None,
                p_value_normal_approximation=None,
                r_squared=None,
                adjusted_r_squared=None,
                undefined_reason="invalid_hc3_leverage",
            )
        scale = (residual / denominator) ** 2
        for left in range(width):
            for right in range(width):
                meat[left][right] += scale * row[left] * row[right]
    covariance = _matmul(_matmul(inverse_gram, meat), inverse_gram)
    variance = covariance[1][1]
    if not math.isfinite(variance) or variance < -1e-12:
        raise ValueError("legacy HC3 direct-slope variance is invalid")
    se = math.sqrt(max(0.0, variance))
    slope = coefficients[1]
    critical = NormalDist().inv_cdf(0.975)
    p_value = (
        0.0
        if se == 0.0 and slope != 0.0
        else 1.0
        if se == 0.0
        else math.erfc(abs(slope / se) / math.sqrt(2.0))
    )
    adjusted_r_squared = 1.0 - (1.0 - r_squared) * (
        (n_pairs - 1.0) / (n_pairs - n_parameters)
    )
    return LegacyHC3RegressionSummary(
        **base,
        defined=True,
        direct_slope=slope,
        hc3_standard_error=se,
        ci_low=slope - critical * se,
        ci_high=slope + critical * se,
        p_value_normal_approximation=p_value,
        r_squared=r_squared,
        adjusted_r_squared=adjusted_r_squared,
        undefined_reason=None,
    )


def _pooled_formula(covariates: Sequence[str]) -> str:
    names = tuple(covariates)
    if not names:
        return f"{LEGACY_OUTCOME} ~ {LEGACY_PREDICTOR} + C(run)"
    if names == CONTROLLED_COVARIATES:
        return (
            f"{LEGACY_OUTCOME} ~ depth_i + depth_j + lca_depth + "
            f"task_jaccard + {LEGACY_PREDICTOR} + C(run)"
        )
    return f"{OUTCOME} ~ {PREDICTOR} + " + " + ".join((*names, "C(run)"))


def _pair_covariate(pair: SemanticPairRow, name: str) -> float:
    legacy_mapping = {
        "depth_i": "left_depth",
        "depth_j": "right_depth",
        "lca_depth": "lca_depth",
        "task_jaccard": "dataset_path_jaccard",
    }
    if name in legacy_mapping:
        return float(getattr(pair, legacy_mapping[name]))
    if name == "pair_has_internal":
        return float(pair.pair_has_internal)
    if name == "ancestor_descendant":
        return float(pair.relationship == "ancestor_descendant")
    if name == "parent_child":
        return float(pair.is_parent_child)
    raise ValueError(f"unknown pair covariate {name!r}")


def _ols_details(
    design: Sequence[Sequence[float]], outcome: Sequence[float]
) -> tuple[list[float], list[list[float]], list[float], float] | None:
    rows = [list(map(float, row)) for row in design]
    values = list(map(float, outcome))
    if not rows or len(rows) != len(values) or any(len(row) != len(rows[0]) for row in rows):
        return None
    width = len(rows[0])
    if len(rows) <= width:
        return None
    gram = [
        [math.fsum(row[left] * row[right] for row in rows) for right in range(width)]
        for left in range(width)
    ]
    inverse = _inverse(gram)
    if inverse is None:
        return None
    target = [
        math.fsum(row[column] * value for row, value in zip(rows, values))
        for column in range(width)
    ]
    coefficients = [math.fsum(inverse[row][column] * target[column] for column in range(width)) for row in range(width)]
    fitted = [math.fsum(coefficient * value for coefficient, value in zip(coefficients, row)) for row in rows]
    residuals = [value - prediction for value, prediction in zip(values, fitted)]
    mean = math.fsum(values) / len(values)
    total = math.fsum((value - mean) ** 2 for value in values)
    if total == 0.0:
        return None
    residual_ss = math.fsum(value * value for value in residuals)
    return coefficients, inverse, residuals, 1.0 - residual_ss / total


def _inverse(matrix: Sequence[Sequence[float]]) -> list[list[float]] | None:
    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix):
        return None
    augmented = [
        [float(value) for value in row]
        + [1.0 if row_index == column else 0.0 for column in range(size)]
        for row_index, row in enumerate(matrix)
    ]
    scale = max(abs(value) for row in matrix for value in row)
    tolerance = max(1e-14, scale * 1e-12)
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= tolerance:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        for index in range(2 * size):
            augmented[column][index] /= divisor
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            for index in range(2 * size):
                augmented[row][index] -= factor * augmented[column][index]
    result = [row[size:] for row in augmented]
    return result if all(math.isfinite(value) for row in result for value in row) else None


def _matmul(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> list[list[float]]:
    if not left or not right or len(left[0]) != len(right):
        raise ValueError("matrix multiplication dimensions differ")
    return [
        [
            math.fsum(left_row[index] * right[index][column] for index in range(len(right)))
            for column in range(len(right[0]))
        ]
        for left_row in left
    ]


def _verified_tree_analysis(value: TreeSemanticAnalysis) -> TreeSemanticAnalysis:
    if not isinstance(value, TreeSemanticAnalysis):
        raise ValueError("runset entries must be TreeSemanticAnalysis objects")
    rebuilt = analyze_tree_semantic_bridge(
        value.topology,
        value.checkpoint_manifest,
        cohort_key=value.receipt.cohort_key,
        endpoint_id=value.receipt.endpoint_id,
        node_covariates=value.node_covariates,
        direct=value.direct,
        semantic=value.semantic,
    )
    if rebuilt != value:
        raise ValueError("tree semantic analysis differs from its recomputed contract")
    return rebuilt


def _clip_correlation(value: float) -> float:
    if not math.isfinite(value) or value < -1.0 or value > 1.0:
        raise ValueError("correlation is outside [-1,1]")
    return min(CORRELATION_CLIP, max(-CORRELATION_CLIP, value))


def _chi_square_survival(value: float, df: int) -> float:
    if value < 0.0 or df < 1:
        raise ValueError("chi-square inputs are invalid")
    return _regularized_gamma_q(df / 2.0, value / 2.0)


def _regularized_gamma_q(shape: float, value: float) -> float:
    if shape <= 0.0 or value < 0.0:
        raise ValueError("regularized gamma inputs are invalid")
    if value == 0.0:
        return 1.0
    epsilon = 1e-14
    maximum = 10_000
    log_prefactor = -value + shape * math.log(value) - math.lgamma(shape)
    if value < shape + 1.0:
        term = 1.0 / shape
        total = term
        cursor = shape
        for _ in range(maximum):
            cursor += 1.0
            term *= value / cursor
            total += term
            if abs(term) <= abs(total) * epsilon:
                return min(1.0, max(0.0, 1.0 - total * math.exp(log_prefactor)))
        raise RuntimeError("regularized gamma series did not converge")
    tiny = 1e-300
    b = value + 1.0 - shape
    if abs(b) < tiny:
        b = tiny
    c_value = 1.0 / tiny
    d_value = 1.0 / b
    fraction = d_value
    for iteration in range(1, maximum + 1):
        coefficient = -iteration * (iteration - shape)
        b += 2.0
        d_value = coefficient * d_value + b
        if abs(d_value) < tiny:
            d_value = tiny
        c_value = b + coefficient / c_value
        if abs(c_value) < tiny:
            c_value = tiny
        d_value = 1.0 / d_value
        delta = d_value * c_value
        fraction *= delta
        if abs(delta - 1.0) <= epsilon:
            return min(1.0, max(0.0, math.exp(log_prefactor) * fraction))
    raise RuntimeError("regularized gamma continued fraction did not converge")


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be nonempty text without surrounding whitespace")
    return value


def _sha256(value: object, label: str) -> str:
    text = _text(value, label)
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


def _integer(value: object, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value
