"""Ancestor-aware direct-versus-semantic distance analysis.

Pairs are joined only through the validated topology pair index.  Correlations
are computed within each tree; cross-tree inference gives every tree equal
weight and never treats model pairs as independent replicates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence

from weighttraits.behavior.allnode.manifest import (
    TreeTopology,
    canonical_sha256,
    read_tree_topology,
)
from weighttraits.behavior.allnode.matrices import (
    INTERNAL_NODE_TREATMENT,
    POPULATION,
    ValidatedDistanceMatrix,
    validate_matrix_receipt,
)


PRESPECIFIED_STRATA = (
    "all_trained_node_pairs",
    "ancestor_descendant",
    "parent_child",
    "collateral",
    "trained_leaf_pairs",
)
TREE_ANALYSIS_SCHEMA = "weighttraits.all_node_pair_analysis.tree.v1"
RUNSET_ANALYSIS_SCHEMA = "weighttraits.all_node_pair_analysis.runset.v1"
INFERENCE_UNIT = "tree"


@dataclass(frozen=True)
class TreeStratumResult:
    stratum: str
    n_pairs: int
    correlation_defined: bool
    pearson_r: float | None
    undefined_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TreePairAnalysisReceipt:
    schema: str
    valid: bool
    tree_id: str
    inference_unit: str
    population: str
    internal_node_treatment: str
    n_nodes: int
    n_internal: int
    n_leaves: int
    topology_sha256: str
    pair_index_sha256: str
    leaf_subset_sha256: str
    direct_metric: str
    direct_matrix_sha256: str
    semantic_metric: str
    semantic_matrix_sha256: str
    prespecified_strata: tuple[str, ...]
    strata: tuple[TreeStratumResult, ...]
    strata_sha256: str

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["prespecified_strata"] = list(self.prespecified_strata)
        row["strata"] = [result.to_dict() for result in self.strata]
        return row


@dataclass(frozen=True)
class TreePairAnalysis:
    receipt: TreePairAnalysisReceipt


@dataclass(frozen=True)
class RunSetStratumResult:
    stratum: str
    n_trees: int
    n_valid_trees: int
    n_undefined_trees: int
    n_pairs_total_descriptive_only: int
    mean_pearson_r: float | None
    sample_sd: float | None
    standard_error: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunSetPairAnalysisReceipt:
    schema: str
    valid: bool
    inference_unit: str
    aggregation: str
    pair_count_used_as_independent_sample_size: bool
    fisher_z_pair_level_standard_error: bool
    population: str
    internal_node_treatment: str
    direct_metric: str
    semantic_metric: str
    tree_ids: tuple[str, ...]
    tree_analysis_sha256: tuple[str, ...]
    prespecified_strata: tuple[str, ...]
    strata: tuple[RunSetStratumResult, ...]
    strata_sha256: str

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        for field in ("tree_ids", "tree_analysis_sha256", "prespecified_strata"):
            row[field] = list(row[field])
        row["strata"] = [result.to_dict() for result in self.strata]
        return row


def analyze_tree_pair_distances(
    topology: TreeTopology,
    direct: ValidatedDistanceMatrix,
    semantic: ValidatedDistanceMatrix,
) -> TreePairAnalysis:
    """Compute prespecified Pearson correlations for one observed tree."""

    if not isinstance(topology, TreeTopology):
        raise ValueError("topology must be a TreeTopology")
    if read_tree_topology(topology.to_dict()) != topology:
        raise ValueError("topology differs from its recomputed contract")
    direct_checked = _validate_artifact(topology, direct, label="direct")
    semantic_checked = _validate_artifact(topology, semantic, label="semantic")
    if direct_checked.receipt.ordered_node_ids != semantic_checked.receipt.ordered_node_ids:
        raise ValueError("direct and semantic ordered node IDs differ")

    expected_pairs = topology.n_nodes * (topology.n_nodes - 1) // 2
    if len(topology.pair_index) != expected_pairs:
        raise ValueError("topology pair index is incomplete")
    buckets: dict[str, list[tuple[float, float]]] = {
        stratum: [] for stratum in PRESPECIFIED_STRATA
    }
    leaf_indices = set(topology.leaf_sensitivity_indices)
    for ordinal, pair in enumerate(topology.pair_index):
        if pair.pair_ordinal != ordinal:
            raise ValueError("topology pair ordinals are not contiguous")
        left = pair.left_ordinal
        right = pair.right_ordinal
        if not (0 <= left < right < topology.n_nodes):
            raise ValueError("topology pair coordinates are invalid")
        if (
            pair.left_node_id != topology.ordered_node_ids[left]
            or pair.right_node_id != topology.ordered_node_ids[right]
        ):
            raise ValueError("topology pair IDs differ from matrix coordinates")
        value = (direct_checked.matrix[left][right], semantic_checked.matrix[left][right])
        buckets["all_trained_node_pairs"].append(value)
        if pair.relationship == "ancestor_descendant":
            buckets["ancestor_descendant"].append(value)
        elif pair.relationship == "collateral":
            buckets["collateral"].append(value)
        else:
            raise ValueError(f"unsupported topology relationship: {pair.relationship!r}")
        if pair.is_parent_child:
            if pair.relationship != "ancestor_descendant":
                raise ValueError("parent-child pair is not ancestor-descendant")
            buckets["parent_child"].append(value)
        if left in leaf_indices and right in leaf_indices:
            if pair.left_role != "leaf" or pair.right_role != "leaf":
                raise ValueError("leaf sensitivity pair has non-leaf topology roles")
            buckets["trained_leaf_pairs"].append(value)

    strata = tuple(_stratum_result(name, buckets[name]) for name in PRESPECIFIED_STRATA)
    strata_rows = [result.to_dict() for result in strata]
    receipt = TreePairAnalysisReceipt(
        schema=TREE_ANALYSIS_SCHEMA,
        valid=True,
        tree_id=topology.tree_id,
        inference_unit=INFERENCE_UNIT,
        population=POPULATION,
        internal_node_treatment=INTERNAL_NODE_TREATMENT,
        n_nodes=topology.n_nodes,
        n_internal=topology.n_internal,
        n_leaves=topology.n_leaves,
        topology_sha256=topology.topology_sha256,
        pair_index_sha256=topology.pair_index_sha256,
        leaf_subset_sha256=topology.leaf_subset_sha256,
        direct_metric=direct_checked.receipt.metric,
        direct_matrix_sha256=direct_checked.receipt.matrix_sha256,
        semantic_metric=semantic_checked.receipt.metric,
        semantic_matrix_sha256=semantic_checked.receipt.matrix_sha256,
        prespecified_strata=PRESPECIFIED_STRATA,
        strata=strata,
        strata_sha256=canonical_sha256(strata_rows),
    )
    return TreePairAnalysis(receipt=receipt)


def aggregate_tree_pair_analyses(
    analyses: Sequence[TreePairAnalysis],
) -> RunSetPairAnalysisReceipt:
    """Aggregate per-tree correlations with one equal weight per valid tree."""

    trees = tuple(analyses)
    if not trees:
        raise ValueError("at least one tree analysis is required")
    receipts = tuple(_validate_tree_analysis(item) for item in trees)
    tree_ids = tuple(receipt.tree_id for receipt in receipts)
    if len(tree_ids) != len(set(tree_ids)):
        raise ValueError("tree analyses contain duplicate tree IDs")
    direct_metrics = {receipt.direct_metric for receipt in receipts}
    semantic_metrics = {receipt.semantic_metric for receipt in receipts}
    if len(direct_metrics) != 1 or len(semantic_metrics) != 1:
        raise ValueError("tree analyses use inconsistent direct or semantic metrics")

    results: list[RunSetStratumResult] = []
    for stratum_index, stratum in enumerate(PRESPECIFIED_STRATA):
        rows = [receipt.strata[stratum_index] for receipt in receipts]
        correlations = [
            row.pearson_r
            for row in rows
            if row.correlation_defined and row.pearson_r is not None
        ]
        if any(not math.isfinite(value) for value in correlations):
            raise ValueError("tree correlation is non-finite")
        mean, sample_sd, standard_error = _equal_weight_summary(correlations)
        results.append(
            RunSetStratumResult(
                stratum=stratum,
                n_trees=len(rows),
                n_valid_trees=len(correlations),
                n_undefined_trees=len(rows) - len(correlations),
                n_pairs_total_descriptive_only=sum(row.n_pairs for row in rows),
                mean_pearson_r=mean,
                sample_sd=sample_sd,
                standard_error=standard_error,
            )
        )
    strata = tuple(results)
    tree_hashes = tuple(canonical_sha256(receipt.to_dict()) for receipt in receipts)
    return RunSetPairAnalysisReceipt(
        schema=RUNSET_ANALYSIS_SCHEMA,
        valid=True,
        inference_unit=INFERENCE_UNIT,
        aggregation="equal_weight_mean_of_per_tree_pearson_correlations",
        pair_count_used_as_independent_sample_size=False,
        fisher_z_pair_level_standard_error=False,
        population=POPULATION,
        internal_node_treatment=INTERNAL_NODE_TREATMENT,
        direct_metric=next(iter(direct_metrics)),
        semantic_metric=next(iter(semantic_metrics)),
        tree_ids=tree_ids,
        tree_analysis_sha256=tree_hashes,
        prespecified_strata=PRESPECIFIED_STRATA,
        strata=strata,
        strata_sha256=canonical_sha256([result.to_dict() for result in strata]),
    )


def _validate_artifact(
    topology: TreeTopology,
    artifact: ValidatedDistanceMatrix,
    *,
    label: str,
) -> ValidatedDistanceMatrix:
    if not isinstance(artifact, ValidatedDistanceMatrix):
        raise ValueError(f"{label} must be a ValidatedDistanceMatrix")
    rebuilt = validate_matrix_receipt(
        artifact.receipt.to_dict(),
        topology,
        metric=artifact.receipt.metric,
        ordered_node_ids=artifact.receipt.ordered_node_ids,
        matrix=artifact.matrix,
    )
    if rebuilt != artifact:
        raise ValueError(f"{label} matrix artifact differs from its recomputed contract")
    return rebuilt


def _stratum_result(
    stratum: str, pairs: Sequence[tuple[float, float]]
) -> TreeStratumResult:
    direct = [pair[0] for pair in pairs]
    semantic = [pair[1] for pair in pairs]
    correlation, reason = _pearson(direct, semantic)
    return TreeStratumResult(
        stratum=stratum,
        n_pairs=len(pairs),
        correlation_defined=correlation is not None,
        pearson_r=correlation,
        undefined_reason=reason,
    )


def _pearson(
    direct: Sequence[float], semantic: Sequence[float]
) -> tuple[float | None, str | None]:
    if len(direct) != len(semantic):
        raise ValueError("paired distance vectors have different lengths")
    if len(direct) < 2:
        return None, "fewer_than_two_pairs"
    if any(not math.isfinite(value) for value in (*direct, *semantic)):
        raise ValueError("paired distance vector contains non-finite values")
    direct_mean = math.fsum(direct) / len(direct)
    semantic_mean = math.fsum(semantic) / len(semantic)
    direct_centered = [value - direct_mean for value in direct]
    semantic_centered = [value - semantic_mean for value in semantic]
    direct_ss = math.fsum(value * value for value in direct_centered)
    semantic_ss = math.fsum(value * value for value in semantic_centered)
    if direct_ss == 0.0 or semantic_ss == 0.0:
        if direct_ss == 0.0 and semantic_ss == 0.0:
            reason = "constant_direct_and_semantic"
        elif direct_ss == 0.0:
            reason = "constant_direct"
        else:
            reason = "constant_semantic"
        return None, reason
    covariance = math.fsum(
        left * right for left, right in zip(direct_centered, semantic_centered)
    )
    correlation = covariance / math.sqrt(direct_ss * semantic_ss)
    if not math.isfinite(correlation):
        raise ValueError("Pearson correlation is non-finite")
    if correlation > 1.0:
        if correlation > 1.0 + 1e-12:
            raise ValueError("Pearson correlation exceeds one")
        correlation = 1.0
    elif correlation < -1.0:
        if correlation < -1.0 - 1e-12:
            raise ValueError("Pearson correlation is below negative one")
        correlation = -1.0
    return correlation, None


def _validate_tree_analysis(item: TreePairAnalysis) -> TreePairAnalysisReceipt:
    if not isinstance(item, TreePairAnalysis):
        raise ValueError("run-set inputs must be TreePairAnalysis objects")
    receipt = item.receipt
    if (
        receipt.schema != TREE_ANALYSIS_SCHEMA
        or receipt.valid is not True
        or receipt.inference_unit != INFERENCE_UNIT
        or receipt.population != POPULATION
        or receipt.internal_node_treatment != INTERNAL_NODE_TREATMENT
        or receipt.prespecified_strata != PRESPECIFIED_STRATA
        or tuple(result.stratum for result in receipt.strata) != PRESPECIFIED_STRATA
        or receipt.strata_sha256
        != canonical_sha256([result.to_dict() for result in receipt.strata])
    ):
        raise ValueError("invalid or drifted tree pair-analysis receipt")
    for result in receipt.strata:
        if result.n_pairs < 0:
            raise ValueError("tree stratum has a negative pair count")
        if result.correlation_defined:
            if (
                result.pearson_r is None
                or not math.isfinite(result.pearson_r)
                or result.undefined_reason is not None
            ):
                raise ValueError("defined tree correlation is invalid")
        elif result.pearson_r is not None or not result.undefined_reason:
            raise ValueError("undefined tree correlation lacks a reason")
    return receipt


def _equal_weight_summary(
    correlations: Sequence[float],
) -> tuple[float | None, float | None, float | None]:
    if not correlations:
        return None, None, None
    mean = math.fsum(correlations) / len(correlations)
    if len(correlations) == 1:
        return mean, None, None
    sample_variance = math.fsum((value - mean) ** 2 for value in correlations) / (
        len(correlations) - 1
    )
    sample_sd = math.sqrt(sample_variance)
    return mean, sample_sd, sample_sd / math.sqrt(len(correlations))
