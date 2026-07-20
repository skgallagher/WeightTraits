"""Independent whitebox analysis from training ledgers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np

from weighttraits.analysis.branch_ordering import branch_ordering_stats
from weighttraits.analysis.tree_diagnostics import TREE_DIAGNOSTICS, write_tree_diagnostics
from weighttraits.distances.readers import (
    CumulativeLoraReader,
    LoraFactorReader,
    TensorReader,
    reader_from_path,
)
from weighttraits.training.ledger import latest_status_by_node, load_ledger_events
from weighttraits.phylo.splits import splits_from_manifest_path


DIRECT_VECTOR_METRICS = {"cosine", "l1", "l2", "correlation", "threshold"}
DIRECT_MATRIX_METRICS = {"cka", "linear_cka"}
DIRECT_TERMINAL_STATUSES = {"completed", "skipped", "stopped_early"}


def analyze_training_ledger_direct(
    ledger: str | Path,
    *,
    truth_manifest: str | Path,
    out_dir: str | Path,
    artifact: str,
    metrics: list[str],
    node_ids: list[str] | None = None,
    path_base: str | Path = ".",
    chunk_size: int = 1_000_000,
    eps: float = 1e-3,
    layer: str | int | None = None,
    aggregate: str = "mean",
) -> dict[str, Any]:
    """Analyze one completed tree without using the legacy distance-cube workflow."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metric_names = _normalize_metrics(metrics)
    truth_path = _resolve_path(truth_manifest, Path(path_base))
    readers = _readers_from_ledger(
        ledger,
        truth_manifest=truth_path,
        artifact=artifact,
        node_ids=node_ids,
        path_base=path_base,
    )
    distances, layer_names, model_ids, distance_audit = _direct_distance_layers(
        readers,
        metrics=metric_names,
        chunk_size=chunk_size,
        eps=eps,
        artifact=artifact,
    )

    distance_path = out / "direct_distance_layers.npz"
    np.savez_compressed(distance_path, **distances)
    (out / "layers.json").write_text(json.dumps(layer_names, indent=2) + "\n")
    (out / "models.json").write_text(json.dumps(model_ids, indent=2) + "\n")
    (out / "distance_audit.json").write_text(
        json.dumps(distance_audit, indent=2, sort_keys=True) + "\n"
    )

    truth_newick = truth_newick_from_manifest(truth_path)
    truth_splits, _ = splits_from_manifest_path(str(truth_path))
    truth_newick_path = out / "truth_manifest.newick"
    truth_newick_path.write_text(truth_newick + "\n")

    score_records: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    for metric in sorted(distances):
        matrix, matrix_audit = _select_distance_matrix(
            distances[metric],
            layer_names=layer_names,
            model_ids=model_ids,
            metric=metric,
            layer=layer,
            aggregate=aggregate,
        )
        matrix_path = out / f"distance_matrix_{metric}.npy"
        tree_path = out / f"tree_{metric}.newick"
        audit_path = out / f"tree_{metric}.audit.json"
        score_path = out / f"score_{metric}.json"
        branch_ordering_path = out / f"branch_ordering_{metric}.json"

        np.save(matrix_path, matrix)
        newick = _neighbor_joining_newick(model_ids, matrix)
        tree_path.write_text(newick + "\n")
        audit = {
            "analysis_engine": "direct",
            "tree_builder": "Bio.Phylo.TreeConstruction.DistanceTreeConstructor.nj",
            "source_distance_layers": str(distance_path),
            "metric": metric,
            "n_models": len(model_ids),
            "model_ids": model_ids,
            **matrix_audit,
        }
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

        score = _score_newick_with_dendropy(
            truth_newick,
            newick,
            truth_manifest=truth_path,
            estimate=tree_path,
        )
        score["metric"] = metric
        score_path.write_text(json.dumps(score, indent=2, sort_keys=True) + "\n")
        diagnostics = write_tree_diagnostics(
            out,
            metric=metric,
            labels=model_ids,
            distances=matrix,
            truth_newick=truth_newick,
            truth_splits=truth_splits,
        )
        branch_ordering = branch_ordering_stats(
            truth_path,
            labels=model_ids,
            distances=matrix,
        )
        branch_ordering_path.write_text(
            json.dumps(branch_ordering, indent=2, sort_keys=True) + "\n"
        )
        score_records.append(score)
        result_rows.append(
            {
                "metric": metric,
                "tree": str(tree_path),
                "tree_audit": str(audit_path),
                "score": str(score_path),
                "rf": score["rf"],
                "normalized_rf": score["normalized_rf"],
                "exact_tree_recovery": score["exact_tree_recovery"],
                "polytomy_aware_exact_recovery": score[
                    "polytomy_aware_exact_recovery"
                ],
                "clade_recovery": score["clade_recovery"],
                "split_precision": score["split_precision"],
                "distance_min": matrix_audit["distance_min"],
                "distance_max": matrix_audit["distance_max"],
                "distance_mean": matrix_audit["distance_mean"],
                "branch_ordering": str(branch_ordering_path),
                **branch_ordering,
                **diagnostics,
            }
        )

    aggregate_recovery = _aggregate_score_records(score_records)
    aggregate_path = out / "aggregate_recovery.json"
    aggregate_path.write_text(json.dumps(aggregate_recovery, indent=2, sort_keys=True) + "\n")

    summary = {
        "analysis_engine": "direct",
        "distance_engine": "direct_streaming_sufficient_stats",
        "tree_builder": "biopython_neighbor_joining",
        "rf_engine": "dendropy_treecompare",
        "diagnostics": list(TREE_DIAGNOSTICS),
        "artifact": artifact,
        "representation": _direct_representation(artifact),
        "ledger": str(ledger),
        "truth_manifest": str(truth_path),
        "truth_newick": str(truth_newick_path),
        "distance_layers": str(distance_path),
        "n_models": len(model_ids),
        "n_layers": len(layer_names),
        "model_ids": model_ids,
        "metrics": sorted(distances),
        "results": result_rows,
        "aggregate_recovery": aggregate_recovery,
        "aggregate_recovery_path": str(aggregate_path),
    }
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _readers_from_ledger(
    ledger: str | Path,
    *,
    truth_manifest: Path,
    artifact: str,
    node_ids: list[str] | None,
    path_base: str | Path,
) -> list[TensorReader]:
    if artifact not in {"model", "merged", "adapter_chain"}:
        raise ValueError(f"unsupported ledger artifact mode: {artifact}")
    events = latest_status_by_node(load_ledger_events(ledger))
    targets = [str(node_id) for node_id in node_ids] if node_ids else _manifest_leaf_ids(truth_manifest)
    base = Path(path_base)
    if artifact == "adapter_chain":
        lineages = _manifest_lineages(truth_manifest)
        return [
            _cumulative_lora_reader(
                node_id,
                lineage=lineages[node_id],
                events=events,
                base=base,
            )
            for node_id in targets
        ]
    return [
        reader_from_path(
            _resolve_path(_event_artifact(events, node_id, artifact), base),
            model_id=node_id,
        )
        for node_id in targets
    ]


def _cumulative_lora_reader(
    node_id: str,
    *,
    lineage: Sequence[str],
    events: Mapping[str, Any],
    base: Path,
) -> CumulativeLoraReader:
    edge_readers = []
    for part in lineage:
        if part == "root":
            continue
        reader = reader_from_path(_resolve_path(_event_artifact(events, part, "adapter"), base))
        if not isinstance(reader, LoraFactorReader):
            raise ValueError(f"adapter_chain part {part!r} is not a PEFT adapter directory")
        edge_readers.append(reader)
    if not edge_readers:
        raise ValueError(f"node {node_id!r} has an empty adapter chain")
    return CumulativeLoraReader(edge_readers, model_id=node_id)


def _event_artifact(events: Mapping[str, Any], node_id: str, artifact: str) -> str:
    try:
        event = events[node_id]
    except KeyError as exc:
        raise ValueError(f"node {node_id!r} is missing from ledger") from exc
    if event.status not in DIRECT_TERMINAL_STATUSES:
        raise ValueError(f"node {node_id!r} is not analysis-ready in ledger: {event.status}")
    artifacts = event.extra.get("artifacts")
    if not isinstance(artifacts, Mapping) or artifact not in artifacts:
        raise ValueError(f"node {node_id!r} has no {artifact!r} artifact in ledger")
    return str(artifacts[artifact])


def _direct_distance_layers(
    readers: list[TensorReader],
    *,
    metrics: list[str],
    chunk_size: int,
    eps: float,
    artifact: str,
) -> tuple[dict[str, np.ndarray], list[str], list[str], dict[str, Any]]:
    if not readers:
        raise ValueError("at least one reader is required")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    keys = readers[0].keys()
    _validate_reader_keys(readers, keys)
    model_ids = [reader.model_id for reader in readers]

    layers_by_metric: dict[str, list[np.ndarray]] = {metric: [] for metric in metrics}
    layer_audit = []
    for key in keys:
        infos = [reader.tensor_info(key) for reader in readers]
        shapes = {info.shape for info in infos}
        if len(shapes) != 1:
            raise ValueError(f"shape mismatch for {key}: {sorted(shapes)}")
        layer_audit.append(
            {
                "name": key,
                "shape": list(infos[0].shape),
                "numel": infos[0].numel,
                "dtype_by_model": {
                    reader.model_id: info.dtype for reader, info in zip(readers, infos, strict=True)
                },
            }
        )

        vector_metrics = [metric for metric in metrics if metric in DIRECT_VECTOR_METRICS]
        if vector_metrics:
            for metric, matrix in _direct_vector_distances(
                readers,
                key,
                metrics=vector_metrics,
                chunk_size=chunk_size,
                eps=eps,
            ).items():
                layers_by_metric[metric].append(matrix)

        matrix_metrics = [metric for metric in metrics if metric in DIRECT_MATRIX_METRICS]
        if matrix_metrics:
            for metric, matrix in _direct_matrix_distances(
                readers,
                key,
                metrics=matrix_metrics,
            ).items():
                layers_by_metric[metric].append(matrix)

    for reader in readers:
        reader.close()

    distances = {
        metric: np.stack(matrices, axis=0)
        for metric, matrices in layers_by_metric.items()
        if matrices
    }
    audit = {
        "analysis_engine": "direct",
        "artifact": artifact,
        "metrics": sorted(distances),
        "chunk_size": chunk_size,
        "eps": eps,
        "n_models": len(model_ids),
        "n_layers": len(keys),
        "model_ids": model_ids,
        "reader_type_by_model": {reader.model_id: type(reader).__name__ for reader in readers},
        "layers": layer_audit,
    }
    return distances, keys, model_ids, audit


def _direct_vector_distances(
    readers: list[TensorReader],
    key: str,
    *,
    metrics: list[str],
    chunk_size: int,
    eps: float,
) -> dict[str, np.ndarray]:
    n_models = len(readers)
    dot = np.zeros((n_models, n_models), dtype=np.float64)
    sums = np.zeros(n_models, dtype=np.float64)
    sums_sq = np.zeros(n_models, dtype=np.float64)
    l1 = np.zeros((n_models, n_models), dtype=np.float64) if "l1" in metrics else None
    threshold = (
        np.zeros((n_models, n_models), dtype=np.float64) if "threshold" in metrics else None
    )
    total = 0
    chunk_iters = [reader.iter_flat_chunks(key, chunk_size) for reader in readers]
    for chunks in zip(*chunk_iters, strict=True):
        values = np.vstack([np.asarray(chunk, dtype=np.float64).reshape(1, -1) for chunk in chunks])
        total += values.shape[1]
        dot += values @ values.T
        sums += values.sum(axis=1)
        sums_sq += np.sum(values * values, axis=1)

        if l1 is not None or threshold is not None:
            for i in range(n_models):
                diff = np.abs(values - values[i])
                if l1 is not None:
                    l1[i, :] += diff.sum(axis=1)
                if threshold is not None:
                    threshold[i, :] += (diff > eps).sum(axis=1)
    if total == 0:
        raise ValueError(f"tensor {key!r} has no elements")
    return _metric_matrices_from_stats(
        metrics,
        dot=dot,
        sums=sums,
        sums_sq=sums_sq,
        total=total,
        l1=l1,
        threshold=threshold,
    )


def _metric_matrices_from_stats(
    metrics: list[str],
    *,
    dot: np.ndarray,
    sums: np.ndarray,
    sums_sq: np.ndarray,
    total: int,
    l1: np.ndarray | None,
    threshold: np.ndarray | None,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if "cosine" in metrics:
        norms = np.sqrt(np.maximum(np.diag(dot), 0.0))
        out["cosine"] = _one_minus_similarity(dot, np.outer(norms, norms))
    if "l2" in metrics:
        norm2 = np.diag(dot)
        squared = np.maximum(norm2[:, None] + norm2[None, :] - 2.0 * dot, 0.0)
        out["l2"] = np.sqrt(squared)
    if "correlation" in metrics:
        centered_dot = dot - np.outer(sums, sums) / total
        centered_norm2 = np.maximum(sums_sq - (sums * sums) / total, 0.0)
        out["correlation"] = _one_minus_similarity(
            centered_dot,
            np.outer(np.sqrt(centered_norm2), np.sqrt(centered_norm2)),
        )
    if l1 is not None:
        out["l1"] = l1
    if threshold is not None:
        out["threshold"] = threshold
    for matrix in out.values():
        matrix[:] = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        matrix[:] = (matrix + matrix.T) / 2.0
        np.fill_diagonal(matrix, 0.0)
    return out


def _direct_matrix_distances(
    readers: list[TensorReader],
    key: str,
    *,
    metrics: list[str],
) -> dict[str, np.ndarray]:
    tensors = [reader.read_tensor(key) for reader in readers]
    out: dict[str, np.ndarray] = {}
    if "cka" in metrics:
        out["cka"] = _pairwise_metric(tensors, _linear_cka_distance)
    return out


def _pairwise_metric(values: Sequence[np.ndarray], metric) -> np.ndarray:
    out = np.zeros((len(values), len(values)), dtype=np.float64)
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            value = float(metric(values[i], values[j]))
            out[i, j] = value
            out[j, i] = value
    return out


def _linear_cka_distance(left: np.ndarray, right: np.ndarray) -> float:
    x = _matrix_view(left)
    y = _matrix_view(right)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"CKA requires matching sample axes, got {x.shape[0]} and {y.shape[0]}")
    x_centered = x - x.mean(axis=0, keepdims=True)
    y_centered = y - y.mean(axis=0, keepdims=True)
    cross = x_centered.T @ y_centered
    x_auto = x_centered.T @ x_centered
    y_auto = y_centered.T @ y_centered
    denom = float(np.linalg.norm(x_auto, ord="fro") * np.linalg.norm(y_auto, ord="fro"))
    if denom < 1e-12:
        return 0.0
    similarity = float(np.sum(cross * cross) / denom)
    return 1.0 - float(np.clip(similarity, 0.0, 1.0))


def _matrix_view(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        raise ValueError("CKA needs an array with at least one dimension")
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr.reshape(arr.shape[0], -1)


def _one_minus_similarity(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    similarity = np.zeros_like(numerator, dtype=np.float64)
    mask = denominator > 1e-12
    similarity[mask] = numerator[mask] / denominator[mask]
    out = 1.0 - similarity
    out[~mask] = 0.0
    return out


def _select_distance_matrix(
    layers: np.ndarray,
    *,
    layer_names: Sequence[str],
    model_ids: Sequence[str],
    metric: str,
    layer: str | int | None,
    aggregate: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(layers, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"metric {metric!r} must have shape layer x model x model")
    if layer is None:
        if aggregate == "mean":
            matrix = np.mean(values, axis=0)
        elif aggregate == "median":
            matrix = np.median(values, axis=0)
        else:
            raise ValueError(f"unsupported layer aggregate: {aggregate}")
        selection = {"layer": None, "layer_index": None, "aggregate": aggregate}
    else:
        layer_index = _resolve_layer(layer, layer_names)
        matrix = values[layer_index]
        selection = {
            "layer": layer_names[layer_index],
            "layer_index": layer_index,
            "aggregate": None,
        }
    matrix = _validated_distance_matrix(matrix, model_ids)
    return matrix, {**selection, **_distance_summary(matrix)}


def _neighbor_joining_newick(labels: Sequence[str], distances: np.ndarray) -> str:
    phylo, distance_matrix, constructor = _biopython_tree_tools()
    matrix = _validated_distance_matrix(distances, labels)
    lower_triangle = [
        [float(matrix[row, col]) for col in range(row + 1)] for row in range(len(labels))
    ]
    tree = constructor().nj(distance_matrix(list(labels), lower_triangle))
    from io import StringIO

    out = StringIO()
    phylo.write(tree, out, "newick")
    return out.getvalue().strip()


def _score_newick_with_dendropy(
    truth_newick: str,
    estimate_newick: str,
    *,
    truth_manifest: Path,
    estimate: Path,
) -> dict[str, Any]:
    dendropy, treecompare = _dendropy_tools()
    taxon_namespace = dendropy.TaxonNamespace()
    truth = dendropy.Tree.get(
        data=truth_newick,
        schema="newick",
        taxon_namespace=taxon_namespace,
        rooting="force-unrooted",
    )
    estimated = dendropy.Tree.get(
        data=estimate_newick,
        schema="newick",
        taxon_namespace=taxon_namespace,
        rooting="force-unrooted",
    )
    truth.encode_bipartitions()
    estimated.encode_bipartitions()
    false_positive, false_negative = treecompare.false_positives_and_negatives(truth, estimated)
    rf = treecompare.symmetric_difference(truth, estimated)
    n_truth = _n_nontrivial_bipartitions(truth)
    n_estimate = _n_nontrivial_bipartitions(estimated)
    truth_leaves = _dendropy_leaf_labels(truth)
    estimate_leaves = _dendropy_leaf_labels(estimated)
    leaves_match = truth_leaves == estimate_leaves
    exact = false_positive == 0 and false_negative == 0 and leaves_match
    polytomy_aware_exact = false_negative == 0 and leaves_match
    return {
        "rf_engine": "dendropy.calculate.treecompare",
        "truth_manifest": str(truth_manifest),
        "estimate": str(estimate),
        "n_truth_leaves": len(truth_leaves),
        "n_estimate_leaves": len(estimate_leaves),
        "n_common_leaves": len(truth_leaves & estimate_leaves),
        "n_missing_truth_leaves": len(truth_leaves - estimate_leaves),
        "n_extra_estimate_leaves": len(estimate_leaves - truth_leaves),
        "n_truth_splits": n_truth,
        "n_estimate_splits": n_estimate,
        "true_positive": n_truth - false_negative,
        "false_negative": false_negative,
        "false_positive": false_positive,
        "rf": rf,
        "normalized_rf": _safe_divide(rf, n_truth + n_estimate, default=0.0),
        "clade_recovery": _safe_divide(n_truth - false_negative, n_truth, default=1.0),
        "false_negative_rate": _safe_divide(false_negative, n_truth, default=0.0),
        "split_precision": _safe_divide(
            n_truth - false_negative,
            n_estimate,
            default=1.0 if n_truth == 0 else 0.0,
        ),
        "false_discovery_rate": _safe_divide(false_positive, n_estimate, default=0.0),
        "exact_tree_recovery": exact,
        "exact_tree_recovery_numeric": 1.0 if exact else 0.0,
        "polytomy_aware_exact_recovery": polytomy_aware_exact,
        "polytomy_aware_exact_recovery_numeric": 1.0 if polytomy_aware_exact else 0.0,
    }


def _aggregate_score_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(record) for record in records]
    if not rows:
        raise ValueError("cannot aggregate zero recovery records")
    numeric_keys = sorted(
        key
        for key in set().union(*(row.keys() for row in rows))
        if all(_is_number(row.get(key)) for row in rows)
    )
    out: dict[str, Any] = {"n_records": len(rows)}
    for key in numeric_keys:
        values = [float(row[key]) for row in rows]
        out[f"{key}_mean"] = sum(values) / len(values)
        out[f"{key}_se"] = _sample_standard_error(values)

    tp = sum(int(row.get("true_positive", 0)) for row in rows)
    fn = sum(int(row.get("false_negative", 0)) for row in rows)
    fp = sum(int(row.get("false_positive", 0)) for row in rows)
    n_truth = sum(int(row.get("n_truth_splits", 0)) for row in rows)
    n_estimate = sum(int(row.get("n_estimate_splits", 0)) for row in rows)
    exact = sum(float(row.get("exact_tree_recovery_numeric", 0.0)) for row in rows)
    polytomy_aware_exact = sum(
        float(row.get("polytomy_aware_exact_recovery_numeric", 0.0)) for row in rows
    )
    out.update(
        {
            "pooled_true_positive": tp,
            "pooled_false_negative": fn,
            "pooled_false_positive": fp,
            "pooled_truth_splits": n_truth,
            "pooled_estimate_splits": n_estimate,
            "pooled_clade_recovery": _safe_divide(tp, n_truth, default=1.0),
            "pooled_clade_recovery_se": _binomial_standard_error(tp, n_truth),
            "pooled_false_negative_rate": _safe_divide(fn, n_truth, default=0.0),
            "pooled_false_negative_rate_se": _binomial_standard_error(fn, n_truth),
            "pooled_split_precision": _safe_divide(
                tp,
                n_estimate,
                default=1.0 if n_truth == 0 else 0.0,
            ),
            "pooled_split_precision_se": _binomial_standard_error(tp, n_estimate),
            "pooled_false_discovery_rate": _safe_divide(fp, n_estimate, default=0.0),
            "pooled_false_discovery_rate_se": _binomial_standard_error(fp, n_estimate),
            "exact_tree_recovery_rate": exact / len(rows),
            "exact_tree_recovery_rate_se": _binomial_standard_error(int(exact), len(rows)),
            "polytomy_aware_exact_recovery_rate": polytomy_aware_exact / len(rows),
            "polytomy_aware_exact_recovery_rate_se": _binomial_standard_error(
                int(polytomy_aware_exact), len(rows)
            ),
        }
    )
    return out


def truth_newick_from_manifest(path: str | Path) -> str:
    children = _manifest_children(path)

    def render(node_id: str) -> str:
        node_children = sorted(children.get(node_id, set()))
        if not node_children:
            return _newick_label(node_id)
        label = "" if node_id == "root" else _newick_label(node_id)
        return "(" + ",".join(render(child) for child in node_children) + ")" + label

    return render("root") + ";"


def _manifest_children(path: str | Path) -> dict[str, set[str]]:
    rows = _manifest_rows(path)
    trained_ids = {str(row["node_id"]) for row in rows if row.get("grow", "train") == "train"}
    children: dict[str, set[str]] = {}
    for row in rows:
        if row.get("grow", "train") != "train":
            continue
        node_id = str(row["node_id"])
        raw_lineage = [str(part) for part in row["path"]]
        lineage = [part for part in raw_lineage if part == "root" or part in trained_ids]
        if not lineage or lineage[-1] != node_id:
            lineage.append(node_id)
        for parent, child in zip(lineage, lineage[1:], strict=False):
            children.setdefault(parent, set()).add(child)
        children.setdefault(node_id, set())
    children.setdefault("root", set())
    return children


def _manifest_leaf_ids(path: str | Path) -> list[str]:
    children = _manifest_children(path)
    nodes = set(children)
    for node_children in children.values():
        nodes.update(node_children)
    return sorted(node for node in nodes if node != "root" and not children.get(node))


def _manifest_lineages(path: str | Path) -> dict[str, tuple[str, ...]]:
    rows = _manifest_rows(path)
    trained_ids = {str(row["node_id"]) for row in rows if row.get("grow", "train") == "train"}
    out = {}
    for row in rows:
        if row.get("grow", "train") != "train":
            continue
        node_id = str(row["node_id"])
        lineage = tuple(str(part) for part in row["path"] if part == "root" or part in trained_ids)
        if not lineage or lineage[-1] != node_id:
            lineage = (*lineage, node_id)
        out[node_id] = lineage
    return out


def _manifest_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "node_id" not in row or "path" not in row:
                raise ValueError(f"{path}:{line_number} needs node_id and path")
            rows.append(row)
    return rows


def _newick_label(label: str) -> str:
    if re.match(r"^[A-Za-z0-9_.:-]+$", label):
        return label
    return "'" + label.replace("'", "''") + "'"


def _validate_reader_keys(readers: list[TensorReader], keys: list[str]) -> None:
    ref = set(keys)
    for reader in readers[1:]:
        other = set(reader.keys())
        if other != ref:
            missing = sorted(ref - other)[:5]
            extra = sorted(other - ref)[:5]
            raise ValueError(
                f"reader {reader.model_id} key mismatch: missing={missing}, extra={extra}"
            )


def _validated_distance_matrix(matrix: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    out = np.asarray(matrix, dtype=np.float64)
    if out.shape != (len(labels), len(labels)):
        raise ValueError(f"distance matrix shape {out.shape} does not match {len(labels)} labels")
    if len(set(labels)) != len(labels):
        raise ValueError("distance matrix labels must be unique")
    if len(labels) < 2:
        raise ValueError("tree reconstruction requires at least two labels")
    if not np.all(np.isfinite(out)):
        raise ValueError("distance matrix contains non-finite values")
    out = (out + out.T) / 2.0
    out[np.abs(out) <= 1e-9] = 0.0
    np.fill_diagonal(out, 0.0)
    if np.min(out) < -1e-9:
        raise ValueError("distance matrix contains negative distances")
    return out


def _resolve_layer(layer: str | int, layer_names: Sequence[str]) -> int:
    if isinstance(layer, int):
        index = layer
    else:
        try:
            index = int(layer)
        except ValueError:
            if layer not in layer_names:
                raise ValueError(f"layer {layer!r} is not in distance layers") from None
            index = list(layer_names).index(layer)
    if index < 0 or index >= len(layer_names):
        raise ValueError(f"layer index {index} is outside 0..{len(layer_names) - 1}")
    return index


def _distance_summary(matrix: np.ndarray) -> dict[str, float]:
    off_diag = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    return {
        "distance_min": float(off_diag.min()) if off_diag.size else 0.0,
        "distance_max": float(off_diag.max()) if off_diag.size else 0.0,
        "distance_mean": float(off_diag.mean()) if off_diag.size else 0.0,
    }


def _normalize_metrics(metrics: list[str]) -> list[str]:
    if not metrics:
        raise ValueError("at least one metric is required")
    out = []
    for metric in metrics:
        normalized = "cka" if metric == "linear_cka" else metric
        if normalized not in DIRECT_VECTOR_METRICS | DIRECT_MATRIX_METRICS:
            raise ValueError(f"unsupported direct metric: {metric}")
        if normalized not in out:
            out.append(normalized)
    return out


def _direct_representation(artifact: str) -> str:
    if artifact == "adapter_chain":
        return "lora_cumulative_delta"
    if artifact in {"model", "merged"}:
        return "full_weight"
    raise ValueError(f"unsupported ledger artifact mode: {artifact}")


def _resolve_path(path: str | Path, base: Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return base / value


def _n_nontrivial_bipartitions(tree: Any) -> int:
    return sum(1 for bipartition in tree.bipartition_encoding if not bipartition.is_trivial())


def _dendropy_leaf_labels(tree: Any) -> set[str]:
    return {leaf.taxon.label for leaf in tree.leaf_node_iter() if leaf.taxon is not None}


def _safe_divide(numerator: float, denominator: float, *, default: float) -> float:
    if denominator == 0:
        return default
    return numerator / denominator


def _sample_standard_error(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def _binomial_standard_error(successes: int, trials: int) -> float:
    if trials <= 0:
        return 0.0
    p = successes / trials
    return math.sqrt(p * (1.0 - p) / trials)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _biopython_tree_tools():
    try:
        from Bio import Phylo
        from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
    except ImportError as exc:
        raise ImportError("Biopython is required for direct neighbor-joining analysis") from exc
    return Phylo, DistanceMatrix, DistanceTreeConstructor


def _dendropy_tools():
    try:
        import dendropy
        from dendropy.calculate import treecompare
    except ImportError as exc:
        raise ImportError("DendroPy is required for direct RF analysis") from exc
    return dendropy, treecompare
