"""Corrected, fail-closed LayerTrace subset scoring for paper Table 4.

The scorer operates only on pinned per-tree distance cubes.  It requires the
fixed 46 topology-eligible controlled trees, resolves the six documented Flan
subsets by exact tensor-name patterns, and checks that its all-tensor row is
numerically identical to the corresponding corrected Table 2 full-FT row.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml

from weighttraits.manifests.reference import manifest_leaf_ids
from weighttraits.paper.analysis_contracts import (
    ALL_TREE_IDS,
    TOPOLOGY_TREE_IDS,
    canonical_sha256,
    canonical_tree_id,
    load_truth_specs,
    pinned_file,
    required_text,
    sha256,
    truth_hashes_sha256,
    validate_strict_training_completion,
)
from weighttraits.paper.direct_weight_table import DIRECT_WEIGHT_TABLE_SCHEMA
from weighttraits.phylo.reconstruct import neighbor_joining_newick
from weighttraits.phylo.recovery import score_split_recovery
from weighttraits.phylo.splits import splits_from_manifest_path, splits_from_newick_text


LAYER_SUBSET_TABLE_SCHEMA = "weighttraits.layer_subset_table4.v1"
LAYER_SUBSET_INVENTORY_SCHEMA = "weighttraits.layer_subset_input_inventory.v1"
DIRECT_TREE_ANALYSIS_SCHEMA = "weighttraits.direct_tree_analysis.v1"
EXPECTED_LAYER_COUNT = 282
EXPECTED_SUBSET_COUNTS = {
    "full": 282,
    "high_signal": 48,
    "sak": 24,
    "encoder_h": 24,
    "enc_sak": 12,
    "low_signal": 36,
}
LAYER_SUBSET_TABLE_COLUMNS = (
    "subset_id",
    "description",
    "n_tensors",
    "tensor_fraction",
    "n_recovery",
    "clade_recovery",
    "clade_recovery_se",
    "paer",
    "paer_se",
    "rf",
    "rf_se",
    "false_negative",
    "false_negative_se",
)

_SELF_ATTENTION_K = re.compile(
    r"^(?:encoder|decoder)\.block\.\d+\.layer\.0\.SelfAttention\.k\.weight$"
)
_PRE_ATTENTION_NORM = re.compile(
    r"^(?:encoder|decoder)\.block\.\d+\.layer\.0\.layer_norm\.weight$"
)
_ENCODER_SELF_ATTENTION_K = re.compile(
    r"^encoder\.block\.\d+\.layer\.0\.SelfAttention\.k\.weight$"
)
_ENCODER_PRE_ATTENTION_NORM = re.compile(
    r"^encoder\.block\.\d+\.layer\.0\.layer_norm\.weight$"
)
_LOW_SIGNAL_NORM = re.compile(
    r"^(?:encoder\.block\.\d+\.layer\.1|"
    r"decoder\.block\.\d+\.layer\.[12])\.layer_norm\.weight$"
)


def _matches(pattern: re.Pattern[str], name: str) -> bool:
    return pattern.fullmatch(name) is not None


SUBSET_DEFINITIONS: tuple[tuple[str, str, Callable[[str], bool]], ...] = (
    ("full", "all trained tensors", lambda _name: True),
    (
        "high_signal",
        "self-attention k plus pre-attention LayerNorm, encoder and decoder",
        lambda name: _matches(_SELF_ATTENTION_K, name)
        or _matches(_PRE_ATTENTION_NORM, name),
    ),
    (
        "sak",
        "self-attention k, encoder and decoder",
        lambda name: _matches(_SELF_ATTENTION_K, name),
    ),
    (
        "encoder_h",
        "encoder self-attention k plus encoder pre-attention LayerNorm",
        lambda name: _matches(_ENCODER_SELF_ATTENTION_K, name)
        or _matches(_ENCODER_PRE_ATTENTION_NORM, name),
    ),
    (
        "enc_sak",
        "encoder self-attention k",
        lambda name: _matches(_ENCODER_SELF_ATTENTION_K, name),
    ),
    (
        "low_signal",
        "post-attention/cross-attention and post-FFN block LayerNorm control",
        lambda name: _matches(_LOW_SIGNAL_NORM, name),
    ),
)


def resolve_layer_subsets(layer_names: Sequence[str]) -> dict[str, tuple[int, ...]]:
    """Resolve the six fixed subsets and enforce their exact tensor counts."""

    names = [str(name) for name in layer_names]
    if len(names) != EXPECTED_LAYER_COUNT:
        raise ValueError(
            f"Flan full-FT layer list must contain exactly {EXPECTED_LAYER_COUNT} tensors, "
            f"got {len(names)}"
        )
    if len(set(names)) != len(names):
        raise ValueError("Flan full-FT layer list contains duplicate tensor names")

    resolved: dict[str, tuple[int, ...]] = {}
    for subset_id, _description, predicate in SUBSET_DEFINITIONS:
        indices = tuple(index for index, name in enumerate(names) if predicate(name))
        expected = EXPECTED_SUBSET_COUNTS[subset_id]
        if len(indices) != expected:
            selected = [names[index] for index in indices]
            raise ValueError(
                f"layer subset {subset_id!r} must resolve to exactly {expected} tensors, "
                f"got {len(indices)}: {selected}"
            )
        resolved[subset_id] = indices
    return resolved


def build_layer_subset_inventory(
    rollup_path: str | Path,
    completion_receipt_path: str | Path,
    *,
    cohort_id: str,
    base_dir: str | Path,
) -> dict[str, Any]:
    """Pin the exact direct-analysis files consumed by corrected Table 4.

    ``base_dir`` is deliberately required.  Direct-analysis rollups may contain
    paths relative to the frozen stage root, not relative to the rollup file.
    Requiring the caller to name that root prevents an apparently valid
    inventory from silently resolving against a different checkout.
    """

    if not isinstance(cohort_id, str) or not cohort_id.strip():
        raise ValueError("layer-subset inventory requires a nonempty cohort_id")
    cohort_id = cohort_id.strip()
    base = Path(base_dir).resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"layer-subset base directory is missing: {base}")
    rollup_file = Path(rollup_path).resolve()
    receipt_file = Path(completion_receipt_path).resolve()
    if not rollup_file.is_file():
        raise FileNotFoundError(f"direct-analysis rollup is missing: {rollup_file}")
    if not receipt_file.is_file():
        raise FileNotFoundError(
            f"training completion receipt is missing: {receipt_file}"
        )

    rollup = json.loads(rollup_file.read_text())
    if not isinstance(rollup, Mapping):
        raise ValueError("direct-analysis rollup must be a JSON object")
    _validate_direct_rollup(rollup)
    receipt = json.loads(receipt_file.read_text())
    _validate_completion_receipt(receipt, cohort_id=cohort_id)

    raw_rows = rollup.get("rows")
    assert isinstance(raw_rows, list)
    cosine_rows: dict[str, Mapping[str, Any]] = {}
    for row in raw_rows:
        if not isinstance(row, Mapping) or row.get("metric") != "cosine":
            continue
        tree_id = canonical_tree_id(row.get("tree_id"))
        if tree_id in cosine_rows:
            raise ValueError(f"direct-analysis rollup duplicates cosine row {tree_id}")
        cosine_rows[tree_id] = row
    if set(cosine_rows) != set(ALL_TREE_IDS):
        raise ValueError(
            "direct-analysis rollup must contain exactly one cosine row for all 50 "
            f"trees; missing={sorted(set(ALL_TREE_IDS) - set(cosine_rows))}, "
            f"unexpected={sorted(set(cosine_rows) - set(ALL_TREE_IDS))}"
        )

    trees: dict[str, dict[str, dict[str, str]]] = {}
    for tree_id in TOPOLOGY_TREE_IDS:
        row = cosine_rows[tree_id]
        _validate_rollup_row(row, tree_id=tree_id)
        summary_path = _resolve_contract_path(
            required_text(row, "summary", context=f"{tree_id} cosine row"),
            base=base,
        )
        if not summary_path.is_file():
            raise FileNotFoundError(f"{tree_id} summary is missing: {summary_path}")
        analysis_dir = summary_path.parent
        truth_path = _resolve_contract_path(
            required_text(row, "truth_manifest", context=f"{tree_id} cosine row"),
            base=base,
        )
        files = {
            "summary": summary_path,
            "layers": analysis_dir / "layers.json",
            "models": analysis_dir / "models.json",
            "distance_layers": analysis_dir / "direct_distance_layers.npz",
            "truth_manifest": truth_path,
        }
        missing = [str(path) for path in files.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{tree_id} inventory inputs are missing: {missing}")
        expected_truth = {"path": truth_path, "sha256": sha256(truth_path)}
        summary = json.loads(summary_path.read_text())
        _validate_analysis_summary(
            summary,
            tree_id=tree_id,
            metric="cosine",
            expected_truth=expected_truth,
            base=base,
            expected_files=files,
        )
        trees[tree_id] = {
            field: {"path": str(path), "sha256": sha256(path)}
            for field, path in files.items()
        }

    return {
        "schema": LAYER_SUBSET_INVENTORY_SCHEMA,
        "valid": True,
        "producer": "weighttraits",
        "cohort_id": cohort_id,
        "artifact": "model",
        "representation": "full_weight",
        "metric": "cosine",
        "base_dir": str(base),
        "source_rollup": {"path": str(rollup_file), "sha256": sha256(rollup_file)},
        "completion_receipt": {
            "path": str(receipt_file),
            "sha256": sha256(receipt_file),
        },
        "topology_tree_ids": list(TOPOLOGY_TREE_IDS),
        "n_topology_trees": len(TOPOLOGY_TREE_IDS),
        "trees": trees,
    }


def write_layer_subset_inventory_json(
    payload: Mapping[str, Any], path: str | Path
) -> None:
    """Atomically publish a complete, validated Table 4 input inventory."""

    if payload.get("schema") != LAYER_SUBSET_INVENTORY_SCHEMA:
        raise ValueError("refusing to write a non-Table-4 inventory payload")
    if payload.get("valid") is not True:
        raise ValueError("refusing to write an invalid Table-4 inventory payload")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_layer_subset_table(
    config_path: str | Path,
    *,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Score all corrected Table 4 subsets from pinned 46-tree inputs."""

    config_file = Path(config_path).resolve()
    config = yaml.safe_load(config_file.read_text())
    if not isinstance(config, Mapping):
        raise ValueError(f"layer-subset config must be a mapping: {config_file}")
    if config.get("version") != 1:
        raise ValueError("layer-subset config requires version: 1")
    base = Path(base_dir).resolve() if base_dir is not None else config_file.parent
    metric = required_text(config, "metric", context="layer-subset config")
    if metric != "cosine":
        raise ValueError("corrected Table 4 contract requires metric: cosine")

    truth_specs = load_truth_specs(config.get("truth_manifests"), base=base)
    truth_digest = truth_hashes_sha256(truth_specs)
    inventory_path, inventory_digest = pinned_file(
        config.get("analysis_inventory"),
        label="layer-subset analysis_inventory",
        base=base,
    )
    inventory = json.loads(inventory_path.read_text())
    if not isinstance(inventory, Mapping):
        raise ValueError("layer-subset analysis inventory must be a JSON object")
    if inventory.get("schema") != LAYER_SUBSET_INVENTORY_SCHEMA:
        raise ValueError(
            "layer-subset analysis inventory has wrong schema: "
            f"{inventory.get('schema')!r}"
        )
    cohort_id = required_text(inventory, "cohort_id", context="analysis inventory")
    if cohort_id != required_text(config, "cohort_id", context="layer-subset config"):
        raise ValueError("layer-subset cohort_id differs between config and inventory")
    if inventory.get("artifact") != "model":
        raise ValueError("corrected Table 4 inventory must declare artifact: model")
    if inventory.get("representation") != "full_weight":
        raise ValueError(
            "corrected Table 4 inventory must declare representation: full_weight"
        )
    if inventory.get("valid") is not True:
        raise ValueError("corrected Table 4 inventory must be explicitly valid")
    if inventory.get("metric") != metric:
        raise ValueError("corrected Table 4 inventory declares a different metric")
    source_rollup_path, source_rollup_digest = pinned_file(
        inventory.get("source_rollup"),
        label="layer-subset source_rollup",
        base=base,
    )
    source_rollup = json.loads(source_rollup_path.read_text())
    _validate_direct_rollup(source_rollup)
    completion_path, completion_digest = pinned_file(
        inventory.get("completion_receipt"),
        label="layer-subset completion_receipt",
        base=base,
    )
    completion = json.loads(completion_path.read_text())
    _validate_completion_receipt(completion, cohort_id=cohort_id)

    tree_specs = _exact_tree_mapping(inventory.get("trees"), context="analysis inventory")
    observations: list[dict[str, Any]] = []
    canonical_layer_names: list[str] | None = None
    canonical_subsets: dict[str, tuple[int, ...]] | None = None
    input_files: dict[str, dict[str, dict[str, str]]] = {}

    for tree_id in TOPOLOGY_TREE_IDS:
        tree_spec = tree_specs[tree_id]
        files: dict[str, tuple[Path, str]] = {}
        for field in (
            "summary",
            "layers",
            "models",
            "distance_layers",
            "truth_manifest",
        ):
            files[field] = pinned_file(
                tree_spec.get(field),
                label=f"{tree_id} {field}",
                base=base,
            )
        input_files[tree_id] = {
            field: {"path": str(path), "sha256": digest}
            for field, (path, digest) in files.items()
        }

        summary = json.loads(files["summary"][0].read_text())
        expected_truth_path = Path(truth_specs[tree_id]["path"]).resolve()
        if files["truth_manifest"][0] != expected_truth_path:
            raise ValueError(
                f"{tree_id} inventory truth path differs from the pinned Table 4 truth"
            )
        if files["truth_manifest"][1] != truth_specs[tree_id]["sha256"]:
            raise ValueError(
                f"{tree_id} inventory truth hash differs from the pinned Table 4 truth"
            )
        _validate_analysis_summary(
            summary,
            tree_id=tree_id,
            metric=metric,
            expected_truth=truth_specs[tree_id],
            base=base,
            expected_files={field: path for field, (path, _digest) in files.items()},
        )
        layer_names = _load_string_list(files["layers"][0], label=f"{tree_id} layers")
        model_ids = _load_string_list(files["models"][0], label=f"{tree_id} models")
        expected_models = manifest_leaf_ids(truth_specs[tree_id]["path"])
        if model_ids != expected_models:
            raise ValueError(
                f"{tree_id} models must exactly match sorted truth-manifest leaves; "
                f"expected={expected_models}, observed={model_ids}"
            )

        if canonical_layer_names is None:
            canonical_layer_names = layer_names
            canonical_subsets = resolve_layer_subsets(layer_names)
        elif layer_names != canonical_layer_names:
            raise ValueError(f"{tree_id} layer names/order differ from the first tree")
        assert canonical_subsets is not None

        with np.load(files["distance_layers"][0], allow_pickle=False) as archive:
            if metric not in archive.files:
                raise ValueError(
                    f"{tree_id} distance cube lacks metric {metric!r}; "
                    f"available={sorted(archive.files)}"
                )
            cube = np.asarray(archive[metric], dtype=np.float64)
        expected_shape = (EXPECTED_LAYER_COUNT, len(model_ids), len(model_ids))
        if cube.shape != expected_shape:
            raise ValueError(
                f"{tree_id} distance cube shape {cube.shape} differs from {expected_shape}"
            )
        _validate_distance_cube(cube, tree_id=tree_id)

        truth_splits, truth_leaves = splits_from_manifest_path(
            str(truth_specs[tree_id]["path"])
        )
        if not truth_splits:
            raise ValueError(f"{tree_id} is topology-eligible but has zero truth splits")
        for subset_id, description, _predicate in SUBSET_DEFINITIONS:
            indices = canonical_subsets[subset_id]
            matrix = np.mean(cube[np.asarray(indices)], axis=0)
            _validate_matrix_has_variation(matrix, tree_id=tree_id, subset_id=subset_id)
            estimated_newick = neighbor_joining_newick(model_ids, matrix)
            estimated_splits, estimated_leaves = splits_from_newick_text(estimated_newick)
            score = score_split_recovery(
                truth_splits,
                estimated_splits,
                truth_leaves,
                estimated_leaves,
            )
            observations.append(
                {
                    "tree_id": tree_id,
                    "subset_id": subset_id,
                    "description": description,
                    "n_tensors": len(indices),
                    "truth_manifest": str(truth_specs[tree_id]["path"]),
                    "truth_sha256": truth_specs[tree_id]["sha256"],
                    "n_truth_splits": score["n_truth_splits"],
                    "clade_recovery": score["clade_recovery"],
                    "polytomy_aware_exact_recovery": score[
                        "polytomy_aware_exact_recovery"
                    ],
                    "rf": score["rf"],
                    "false_negative": score["false_negative"],
                    "false_positive": score["false_positive"],
                }
            )

    assert canonical_layer_names is not None
    assert canonical_subsets is not None
    rows = _aggregate_observations(observations)
    table2_path, table2_digest = pinned_file(
        config.get("table2_receipt"),
        label="corrected Table 2 receipt",
        base=base,
    )
    table2 = json.loads(table2_path.read_text())
    table2_cohort_id = required_text(
        config,
        "table2_full_ft_cohort_id",
        context="layer-subset config",
    )
    _validate_full_row_against_table2(
        rows,
        table2=table2,
        cohort_id=table2_cohort_id,
        metric=metric,
        truth_digest=truth_digest,
    )

    selected_names = {
        subset_id: [canonical_layer_names[index] for index in indices]
        for subset_id, indices in canonical_subsets.items()
    }
    return {
        "schema": LAYER_SUBSET_TABLE_SCHEMA,
        "valid": True,
        "producer": "weighttraits",
        "config": str(config_file),
        "config_sha256": sha256(config_file),
        "cohort_id": cohort_id,
        "metric": metric,
        "artifact": "model",
        "representation": "full_weight",
        "analysis_inventory": str(inventory_path),
        "analysis_inventory_sha256": inventory_digest,
        "source_rollup": str(source_rollup_path),
        "source_rollup_sha256": source_rollup_digest,
        "completion_receipt": str(completion_path),
        "completion_receipt_sha256": completion_digest,
        "table2_receipt": str(table2_path),
        "table2_receipt_sha256": table2_digest,
        "table2_full_ft_cohort_id": table2_cohort_id,
        "topology_tree_ids": list(TOPOLOGY_TREE_IDS),
        "n_topology_trees": len(TOPOLOGY_TREE_IDS),
        "truth_manifests": {
            tree_id: {
                "path": str(truth_specs[tree_id]["path"]),
                "sha256": truth_specs[tree_id]["sha256"],
            }
            for tree_id in TOPOLOGY_TREE_IDS
        },
        "truth_hashes_sha256": truth_digest,
        "input_files": input_files,
        "layer_names_sha256": canonical_sha256(canonical_layer_names),
        "subset_contract": {
            "aggregation": "arithmetic mean of selected tensor distance matrices before NJ",
            "clade_recovery": "arithmetic mean with sample SE across 46 trees",
            "paer": "binary mean with binomial sqrt(p*(1-p)/n) SE",
            "rf": "arithmetic mean with sample SE across 46 trees",
            "false_negative": "arithmetic mean with sample SE across 46 trees",
            "expected_counts": EXPECTED_SUBSET_COUNTS,
            "selected_layer_names": selected_names,
            "selected_layer_names_sha256": {
                subset_id: canonical_sha256(names)
                for subset_id, names in selected_names.items()
            },
        },
        "table2_full_row_match": True,
        "n_rows": len(rows),
        "rows": rows,
        "n_observations": len(observations),
        "observations": observations,
    }


def write_layer_subset_table_json(payload: Mapping[str, Any], path: str | Path) -> None:
    if payload.get("schema") != LAYER_SUBSET_TABLE_SCHEMA or payload.get("valid") is not True:
        raise ValueError("refusing to write an invalid corrected Table 4 artifact")
    _write_text_atomic(
        Path(path), json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def write_layer_subset_table_csv(
    rows: Sequence[Mapping[str, Any]], path: str | Path
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            writer = csv.DictWriter(handle, fieldnames=list(LAYER_SUBSET_TABLE_COLUMNS))
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {column: row.get(column) for column in LAYER_SUBSET_TABLE_COLUMNS}
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_text_atomic(path: Path, text: str) -> None:
    output = path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def _exact_tree_mapping(raw: object, *, context: str) -> dict[str, Mapping[str, Any]]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{context} requires a trees mapping")
    normalized: dict[str, Mapping[str, Any]] = {}
    for raw_tree_id, value in raw.items():
        tree_id = canonical_tree_id(raw_tree_id)
        if tree_id in normalized:
            raise ValueError(f"{context} contains duplicate canonical tree ID {tree_id}")
        if not isinstance(value, Mapping):
            raise ValueError(f"{context} tree {tree_id} must be a mapping")
        normalized[tree_id] = value
    expected = set(TOPOLOGY_TREE_IDS)
    observed = set(normalized)
    if observed != expected:
        raise ValueError(
            f"{context} must contain exactly the canonical 46 topology tree IDs; "
            f"missing={sorted(expected - observed)}, unexpected={sorted(observed - expected)}"
        )
    return normalized


def _load_string_list(path: Path, *, label: str) -> list[str]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list) or not raw or not all(isinstance(item, str) for item in raw):
        raise ValueError(f"{label} must be a nonempty JSON string list: {path}")
    return [str(item) for item in raw]


def _validate_analysis_summary(
    summary: object,
    *,
    tree_id: str,
    metric: str,
    expected_truth: Mapping[str, Any],
    base: Path,
    expected_files: Mapping[str, Path],
) -> None:
    if not isinstance(summary, Mapping):
        raise ValueError(f"{tree_id} analysis summary must be a JSON object")
    if (
        summary.get("schema") != DIRECT_TREE_ANALYSIS_SCHEMA
        or summary.get("valid") is not True
        or summary.get("producer") != "weighttraits"
    ):
        raise ValueError(f"{tree_id} analysis summary is not a native WeightTraits receipt")
    if summary.get("analysis_engine") != "direct":
        raise ValueError(f"{tree_id} analysis summary must use analysis_engine: direct")
    for field, expected in (
        ("distance_engine", "direct_streaming_sufficient_stats"),
        ("tree_builder", "biopython_neighbor_joining"),
        ("rf_engine", "dendropy_treecompare"),
    ):
        if summary.get(field) != expected:
            raise ValueError(f"{tree_id} analysis summary {field} drift")
    if summary.get("artifact") != "model" or summary.get("representation") != "full_weight":
        raise ValueError(f"{tree_id} analysis summary is not model/full_weight")
    if summary.get("n_layers") != EXPECTED_LAYER_COUNT:
        raise ValueError(
            f"{tree_id} analysis summary n_layers must be {EXPECTED_LAYER_COUNT}, "
            f"got {summary.get('n_layers')!r}"
        )
    metrics = summary.get("metrics")
    if not isinstance(metrics, list) or metric not in metrics:
        raise ValueError(f"{tree_id} analysis summary lacks metric {metric!r}")
    raw_truth = summary.get("truth_manifest")
    if not isinstance(raw_truth, str):
        raise ValueError(
            f"{tree_id} analysis summary truth manifest does not identify the pinned truth"
        )
    referenced = Path(raw_truth)
    referenced = (
        referenced.resolve()
        if referenced.is_absolute()
        else (base / referenced).resolve()
    )
    expected_path = Path(expected_truth["path"]).resolve()
    if referenced != expected_path:
        raise ValueError(
            f"{tree_id} analysis summary truth path mismatch: "
            f"expected {expected_path}, got {referenced}"
        )
    if not referenced.is_file():
        raise FileNotFoundError(
            f"{tree_id} analysis summary truth manifest is unreadable: {referenced}"
        )
    if sha256(referenced) != expected_truth["sha256"]:
        raise ValueError(f"{tree_id} analysis summary references a different truth hash")
    if summary.get("truth_manifest_sha256") != expected_truth["sha256"]:
        raise ValueError(f"{tree_id} analysis summary truth digest drift")
    distance_path = _resolve_contract_path(
        required_text(summary, "distance_layers", context=f"{tree_id} analysis summary"),
        base=base,
    )
    if distance_path != expected_files["distance_layers"].resolve():
        raise ValueError(f"{tree_id} analysis summary distance-cube path drift")
    ledger_path = _resolve_contract_path(
        required_text(summary, "ledger", context=f"{tree_id} analysis summary"),
        base=base,
    )
    if not ledger_path.is_file():
        raise FileNotFoundError(f"{tree_id} analysis summary ledger is unreadable: {ledger_path}")
    if summary.get("ledger_sha256") != sha256(ledger_path):
        raise ValueError(f"{tree_id} analysis summary ledger digest drift")
    models = json.loads(expected_files["models"].read_text())
    if summary.get("model_ids") != models or summary.get("n_models") != len(models):
        raise ValueError(f"{tree_id} analysis summary model provenance drift")
    results = summary.get("results")
    matches = (
        [row for row in results if isinstance(row, Mapping) and row.get("metric") == metric]
        if isinstance(results, list)
        else []
    )
    if len(matches) != 1:
        raise ValueError(f"{tree_id} analysis summary requires one native {metric} result")
    outputs = summary.get("output_sha256")
    if not isinstance(outputs, Mapping):
        raise ValueError(f"{tree_id} analysis summary lacks native output hashes")
    for field in ("distance_layers", "layers", "models"):
        expected = expected_files[field].resolve()
        if outputs.get(field) != sha256(expected):
            raise ValueError(f"{tree_id} analysis summary {field} digest drift")
    for field in ("tree", "score", "tree_audit"):
        path = _resolve_contract_path(
            required_text(matches[0], field, context=f"{tree_id} {metric} result"),
            base=base,
        )
        if not path.is_file():
            raise FileNotFoundError(f"{tree_id} native {field} is unreadable: {path}")
        if outputs.get(f"{field}_{metric}") != sha256(path):
            raise ValueError(f"{tree_id} native {field} digest drift")
    matrix_path = expected_files["summary"].parent / f"distance_matrix_{metric}.npy"
    if not matrix_path.is_file() or outputs.get(f"distance_matrix_{metric}") != sha256(
        matrix_path
    ):
        raise ValueError(f"{tree_id} native distance-matrix digest drift")


def _resolve_contract_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _validate_direct_rollup(rollup: object) -> None:
    if not isinstance(rollup, Mapping):
        raise ValueError("direct-analysis rollup must be a JSON object")
    if rollup.get("valid") is not True:
        raise ValueError("direct-analysis rollup is not explicitly valid")
    if rollup.get("artifact") != "model":
        raise ValueError("corrected Table 4 requires a model-artifact rollup")
    metrics = rollup.get("metrics")
    if not isinstance(metrics, list) or set(metrics) != {"correlation", "cosine", "l2"}:
        raise ValueError(
            "corrected Table 4 rollup must contain exactly correlation/cosine/l2 metrics"
        )
    if rollup.get("missing_tree_ids") != []:
        raise ValueError("corrected Table 4 rollup contains missing trees")
    if rollup.get("n_tree_summaries") != 50:
        raise ValueError("corrected Table 4 rollup must contain 50 tree summaries")
    if rollup.get("n_rows") != 150:
        raise ValueError("corrected Table 4 rollup must contain 150 tree/metric rows")
    raw_tree_ids = rollup.get("tree_ids")
    if not isinstance(raw_tree_ids, list):
        raise ValueError("direct-analysis rollup tree_ids must be a list")
    normalized = [canonical_tree_id(value) for value in raw_tree_ids]
    if len(normalized) != len(set(normalized)) or set(normalized) != set(ALL_TREE_IDS):
        raise ValueError("corrected Table 4 rollup must declare the exact 50 tree IDs")
    rows = rollup.get("rows")
    if not isinstance(rows, list) or len(rows) != 150:
        raise ValueError("direct-analysis rollup rows must contain exactly 150 entries")


def _validate_rollup_row(row: Mapping[str, Any], *, tree_id: str) -> None:
    if canonical_tree_id(row.get("tree_id")) != tree_id:
        raise ValueError(f"{tree_id} rollup row has a different tree ID")
    if row.get("analysis_engine") != "direct":
        raise ValueError(f"{tree_id} rollup row is not from the direct engine")
    if row.get("metric") != "cosine":
        raise ValueError(f"{tree_id} Table 4 rollup row is not cosine")
    if row.get("artifact") != "model" or row.get("representation") != "full_weight":
        raise ValueError(f"{tree_id} Table 4 rollup row is not model/full_weight")
    if row.get("n_layers") != EXPECTED_LAYER_COUNT:
        raise ValueError(
            f"{tree_id} Table 4 rollup row must have {EXPECTED_LAYER_COUNT} layers"
        )
    if isinstance(row.get("n_models"), bool) or not isinstance(row.get("n_models"), int):
        raise ValueError(f"{tree_id} Table 4 rollup row has invalid n_models")
    if row["n_models"] < 4:
        raise ValueError(f"{tree_id} Table 4 rollup row has fewer than four models")


def _validate_completion_receipt(receipt: object, *, cohort_id: str) -> None:
    validate_strict_training_completion(receipt, cohort_id=cohort_id)


def _validate_matrix_has_variation(
    matrix: np.ndarray, *, tree_id: str, subset_id: str
) -> None:
    values = np.asarray(matrix, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{tree_id}/{subset_id} distance matrix contains non-finite values")
    if float(np.max(values)) <= 1e-14:
        raise ValueError(f"{tree_id}/{subset_id} distance matrix has no usable variation")


def _validate_distance_cube(cube: np.ndarray, *, tree_id: str) -> None:
    values = np.asarray(cube, dtype=np.float64)
    if (
        not np.isfinite(values).all()
        or not np.allclose(values, np.swapaxes(values, 1, 2), rtol=0.0, atol=1e-12)
        or not np.allclose(
            np.diagonal(values, axis1=1, axis2=2), 0.0, rtol=0.0, atol=1e-12
        )
        or np.any(values < -1e-12)
    ):
        raise ValueError(
            f"{tree_id} distance cube contains a non-finite, asymmetric, "
            "nonzero-diagonal, or negative layer matrix"
        )


def _aggregate_observations(observations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    descriptions = {subset_id: description for subset_id, description, _ in SUBSET_DEFINITIONS}
    for subset_id in EXPECTED_SUBSET_COUNTS:
        selected = [row for row in observations if row.get("subset_id") == subset_id]
        tree_ids = {str(row["tree_id"]) for row in selected}
        if len(selected) != len(TOPOLOGY_TREE_IDS) or tree_ids != set(TOPOLOGY_TREE_IDS):
            raise ValueError(f"subset {subset_id} does not contain the exact canonical 46 trees")
        clade = [_finite_float(row["clade_recovery"], label="clade_recovery") for row in selected]
        paer = [
            1.0
            if _strict_bool(
                row["polytomy_aware_exact_recovery"],
                label="polytomy_aware_exact_recovery",
            )
            else 0.0
            for row in selected
        ]
        rf = [_finite_float(row["rf"], label="rf") for row in selected]
        false_negative = [
            _finite_float(row["false_negative"], label="false_negative")
            for row in selected
        ]
        p = statistics.fmean(paer)
        rows.append(
            {
                "subset_id": subset_id,
                "description": descriptions[subset_id],
                "n_tensors": EXPECTED_SUBSET_COUNTS[subset_id],
                "tensor_fraction": EXPECTED_SUBSET_COUNTS[subset_id] / EXPECTED_LAYER_COUNT,
                "n_recovery": len(selected),
                "clade_recovery": statistics.fmean(clade),
                "clade_recovery_se": _sample_se(clade),
                "paer": p,
                "paer_se": math.sqrt(p * (1.0 - p) / len(paer)),
                "rf": statistics.fmean(rf),
                "rf_se": _sample_se(rf),
                "false_negative": statistics.fmean(false_negative),
                "false_negative_se": _sample_se(false_negative),
            }
        )
    return rows


def _validate_full_row_against_table2(
    rows: Sequence[Mapping[str, Any]],
    *,
    table2: object,
    cohort_id: str,
    metric: str,
    truth_digest: str,
) -> None:
    if not isinstance(table2, Mapping):
        raise ValueError("corrected Table 2 receipt must be a JSON object")
    if (
        table2.get("schema") != DIRECT_WEIGHT_TABLE_SCHEMA
        or table2.get("producer") != "weighttraits"
    ):
        raise ValueError("corrected Table 2 receipt is not a native WeightTraits Table 2 artifact")
    if table2.get("truth_hashes_sha256") != truth_digest:
        raise ValueError("corrected Table 2 receipt uses different truth-manifest hashes")
    if set(table2.get("topology_tree_ids", [])) != set(TOPOLOGY_TREE_IDS):
        raise ValueError("corrected Table 2 receipt does not use the canonical 46 tree IDs")
    table2_rows = table2.get("rows")
    if not isinstance(table2_rows, list):
        raise ValueError("corrected Table 2 receipt rows must be a list")
    matches = [row for row in table2_rows if isinstance(row, Mapping) and row.get("cohort_id") == cohort_id]
    if len(matches) != 1:
        raise ValueError(
            f"corrected Table 2 receipt must contain exactly one cohort {cohort_id!r}, "
            f"got {len(matches)}"
        )
    table2_row = matches[0]
    if table2_row.get("metric") != metric:
        raise ValueError("corrected Table 2 full-FT row uses a different metric")
    if table2_row.get("artifact") != "model" or table2_row.get("representation") != "full_weight":
        raise ValueError("corrected Table 2 full-FT row is not model/full_weight")
    if table2_row.get("n_recovery") != len(TOPOLOGY_TREE_IDS):
        raise ValueError("corrected Table 2 full-FT row does not use 46 recovery trees")
    full = next(row for row in rows if row["subset_id"] == "full")
    for field in (
        "clade_recovery",
        "clade_recovery_se",
        "paer",
        "paer_se",
        "rf",
        "rf_se",
        "false_negative",
        "false_negative_se",
    ):
        left = _finite_float(full.get(field), label=f"Table 4 full {field}")
        right = _finite_float(table2_row.get(field), label=f"Table 2 full-FT {field}")
        if not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                f"Table 4 full row does not match corrected Table 2 full-FT {field}: "
                f"{left} != {right}"
            )


def _sample_se(values: Sequence[float]) -> float:
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _strict_bool(value: object, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value
