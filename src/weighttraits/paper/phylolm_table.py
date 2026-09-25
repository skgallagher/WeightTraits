"""Corrected common-tree WeightTraits/PhyloLM comparison for paper Table 10."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from weighttraits.manifests.reference import manifest_leaf_ids
from weighttraits.paper.analysis_contracts import (
    ALL_TREE_IDS,
    TOPOLOGY_EXCLUDED_TREE_IDS,
    TOPOLOGY_TREE_IDS,
    canonical_tree_id,
    load_truth_specs,
    pinned_file,
    required_text,
    resolve_path,
    sha256,
    truth_hashes_sha256,
    validate_strict_training_completion,
)
from weighttraits.phylo.reconstruct import neighbor_joining_newick
from weighttraits.phylo.recovery import score_split_recovery
from weighttraits.phylo.splits import splits_from_manifest_path, splits_from_newick_text


PHYLOLM_TABLE_SCHEMA = "weighttraits.phylolm_table10.v1"
PHYLOLM_RECEIPT_SCHEMA = "weighttraits.phylolm_controlled_runset.v1"
PHYLOLM_TREE_RECEIPT_SCHEMA = "weighttraits.phylolm_controlled_tree.v1"
DIRECT_TREE_ANALYSIS_SCHEMA = "weighttraits.direct_tree_analysis.v1"
PHYLOLM_PROTOCOL_CONTRACT = {
    "upstream_commit": "8c70edf062a0adce2a3e6c8c79cd23a645fd0905",
    "n_genes": 128,
    "samples_per_gene": 32,
    "new_tokens": 4,
    "allele_characters": 4,
    "temperature": 1.0,
    "gene_seed": 0,
    "sampling_seed": 20260803,
    "batch_size": 64,
    "torch_dtype": "auto",
    "similarity_floor": 1e-3,
    "prompt_mode": "raw_continuation",
    "distance": "-log(max(nei_similarity,1e-3))",
    "tree_builder": "weighttraits_neighbor_joining",
}

SUITE_CONTRACTS: dict[str, dict[str, str]] = {
    "llama32_1b_lora_qkv_r8_legacy_causal_2000": {
        "label": "Llama-3.2-1B LoRA qkv (r=8)",
        "training": "lora_qkv_r8",
        "weight_artifact": "merged",
    },
    "llama32_1b_lora_qkv_r64_legacy_causal_2000": {
        "label": "Llama-3.2-1B LoRA qkv (r=64)",
        "training": "lora_qkv_r64",
        "weight_artifact": "merged",
    },
    "llama32_1b_full_finetune_legacy_causal_2000": {
        "label": "Llama-3.2-1B full fine-tuning",
        "training": "full_finetune",
        "weight_artifact": "model",
    },
}

PHYLOLM_TABLE_COLUMNS = (
    "suite_id",
    "label",
    "training",
    "n_trees",
    "weight_clade_recovery",
    "weight_clade_recovery_se",
    "phylolm_clade_recovery",
    "phylolm_clade_recovery_se",
    "clade_recovery_difference",
    "weight_paer",
    "weight_paer_se",
    "phylolm_paer",
    "phylolm_paer_se",
    "weight_rf",
    "weight_rf_se",
    "phylolm_rf",
    "phylolm_rf_se",
    "weight_false_negative",
    "weight_false_negative_se",
    "phylolm_false_negative",
    "phylolm_false_negative_se",
)


def resolve_legacy_causal_suite(name: object) -> dict[str, str]:
    """Resolve one exact corrected Llama legacy-causal-2000 suite name."""

    suite_id = str(name).strip()
    try:
        contract = SUITE_CONTRACTS[suite_id]
    except KeyError as exc:
        raise ValueError(
            f"unsupported Table 10 suite {suite_id!r}; "
            f"expected exactly one of {list(SUITE_CONTRACTS)}"
        ) from exc
    return {"suite_id": suite_id, **contract}


def build_phylolm_table(
    config_path: str | Path,
    *,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Combine corrected weight and PhyloLM rows on the exact canonical 46 trees."""

    config_file = Path(config_path).resolve()
    config = yaml.safe_load(config_file.read_text())
    if not isinstance(config, Mapping):
        raise ValueError(f"Table 10 config must be a mapping: {config_file}")
    if config.get("version") != 1:
        raise ValueError("Table 10 config requires version: 1")
    metric = required_text(config, "metric", context="Table 10 config")
    if metric != "cosine":
        raise ValueError("corrected Table 10 weight contract requires metric: cosine")
    base = Path(base_dir).resolve() if base_dir is not None else config_file.parent
    truth_specs = load_truth_specs(config.get("truth_manifests"), base=base)
    truth_digest = truth_hashes_sha256(truth_specs)

    raw_suites = config.get("suites")
    if not isinstance(raw_suites, list):
        raise ValueError("Table 10 config requires a suites list")
    suite_specs: dict[str, Mapping[str, Any]] = {}
    for index, raw_suite in enumerate(raw_suites, start=1):
        if not isinstance(raw_suite, Mapping):
            raise ValueError(f"Table 10 suite {index} must be a mapping")
        suite = resolve_legacy_causal_suite(
            required_text(raw_suite, "suite_id", context=f"Table 10 suite {index}")
        )
        suite_id = suite["suite_id"]
        if suite_id in suite_specs:
            raise ValueError(f"duplicate Table 10 suite {suite_id!r}")
        suite_specs[suite_id] = raw_suite
    expected_suites = set(SUITE_CONTRACTS)
    if set(suite_specs) != expected_suites:
        raise ValueError(
            "Table 10 requires exactly the three corrected Llama legacy-causal-2000 "
            f"suites; missing={sorted(expected_suites - set(suite_specs))}, "
            f"unexpected={sorted(set(suite_specs) - expected_suites)}"
        )

    rows: list[dict[str, Any]] = []
    paired_observations: list[dict[str, Any]] = []
    input_receipts: dict[str, dict[str, dict[str, str]]] = {}
    for suite_id, suite_contract in SUITE_CONTRACTS.items():
        raw_suite = suite_specs[suite_id]
        cohort_path, cohort_digest = pinned_file(
            raw_suite.get("cohort_contract"),
            label=f"{suite_id} cohort_contract",
            base=base,
        )
        completion_path, completion_digest = pinned_file(
            raw_suite.get("completion_receipt"),
            label=f"{suite_id} completion_receipt",
            base=base,
        )
        weight_path, weight_digest = pinned_file(
            raw_suite.get("weight_rollup"),
            label=f"{suite_id} weight_rollup",
            base=base,
        )
        phylolm_path, phylolm_digest = pinned_file(
            raw_suite.get("phylolm_receipt"),
            label=f"{suite_id} phylolm_receipt",
            base=base,
        )
        input_receipts[suite_id] = {
            "cohort_contract": {"path": str(cohort_path), "sha256": cohort_digest},
            "completion_receipt": {
                "path": str(completion_path),
                "sha256": completion_digest,
            },
            "weight_rollup": {"path": str(weight_path), "sha256": weight_digest},
            "phylolm_receipt": {"path": str(phylolm_path), "sha256": phylolm_digest},
        }

        cohort = json.loads(cohort_path.read_text())
        _validate_cohort_receipt(
            cohort,
            suite_id=suite_id,
            truth_specs=truth_specs,
        )
        completion = json.loads(completion_path.read_text())
        validate_strict_training_completion(completion, cohort_id=suite_id)
        weight = json.loads(weight_path.read_text())
        weight_rows = _load_weight_rows(
            weight,
            suite_id=suite_id,
            expected_artifact=suite_contract["weight_artifact"],
            metric=metric,
            truth_specs=truth_specs,
            base=base,
            source_dir=weight_path.parent,
        )
        phylolm = json.loads(phylolm_path.read_text())
        phylolm_rows = _load_phylolm_rows(
            phylolm,
            suite_id=suite_id,
            truth_specs=truth_specs,
            truth_digest=truth_digest,
        )

        if set(weight_rows) != set(TOPOLOGY_TREE_IDS):
            raise ValueError(f"{suite_id} weight rows are not the exact canonical 46 IDs")
        if set(phylolm_rows) != set(TOPOLOGY_TREE_IDS):
            raise ValueError(f"{suite_id} PhyloLM rows are not the exact canonical 46 IDs")
        if set(weight_rows) != set(phylolm_rows):
            raise ValueError(f"{suite_id} weight and PhyloLM tree IDs differ")

        for tree_id in TOPOLOGY_TREE_IDS:
            weight_row = weight_rows[tree_id]
            phylolm_row = phylolm_rows[tree_id]
            paired_observations.append(
                {
                    "suite_id": suite_id,
                    "tree_id": tree_id,
                    "truth_sha256": truth_specs[tree_id]["sha256"],
                    **{
                        f"weight_{field}": weight_row[field]
                        for field in (
                            "n_truth_splits",
                            "clade_recovery",
                            "polytomy_aware_exact_recovery",
                            "rf",
                            "false_negative",
                            "false_positive",
                        )
                    },
                    **{
                        f"phylolm_{field}": phylolm_row[field]
                        for field in (
                            "n_truth_splits",
                            "clade_recovery",
                            "polytomy_aware_exact_recovery",
                            "rf",
                            "false_negative",
                            "false_positive",
                        )
                    },
                }
            )
        rows.append(
            _aggregate_suite(
                suite_id=suite_id,
                suite_contract=suite_contract,
                weight_rows=weight_rows,
                phylolm_rows=phylolm_rows,
            )
        )

    observed_by_suite = {
        suite_id: {
            row["tree_id"]
            for row in paired_observations
            if row["suite_id"] == suite_id
        }
        for suite_id in SUITE_CONTRACTS
    }
    if any(tree_ids != set(TOPOLOGY_TREE_IDS) for tree_ids in observed_by_suite.values()):
        raise ValueError("Table 10 suites do not all use the exact canonical common-46 IDs")

    return {
        "schema": PHYLOLM_TABLE_SCHEMA,
        "valid": True,
        "producer": "weighttraits",
        "config": str(config_file),
        "config_sha256": sha256(config_file),
        "metric": metric,
        "suite_resolver": {
            suite_id: {
                **contract,
                "phylolm_group": _expected_phylolm_group(suite_id),
            }
            for suite_id, contract in SUITE_CONTRACTS.items()
        },
        "common_tree_ids": list(TOPOLOGY_TREE_IDS),
        "n_common_trees": len(TOPOLOGY_TREE_IDS),
        "truth_manifests": {
            tree_id: {
                "path": str(truth_specs[tree_id]["path"]),
                "sha256": truth_specs[tree_id]["sha256"],
            }
            for tree_id in TOPOLOGY_TREE_IDS
        },
        "truth_hashes_sha256": truth_digest,
        "input_receipts": input_receipts,
        "estimator_contract": {
            "clade_recovery": "arithmetic mean with sample SE across exact common-46 IDs",
            "paer": "binary mean with binomial sqrt(p*(1-p)/n) SE",
            "rf": "arithmetic mean with sample SE across exact common-46 IDs",
            "false_negative": "arithmetic mean with sample SE across exact common-46 IDs",
            "pairing": "tree-ID paired; no set-size-only or intersection-only acceptance",
        },
        "n_rows": len(rows),
        "rows": rows,
        "n_paired_observations": len(paired_observations),
        "paired_observations": paired_observations,
    }


def write_phylolm_table_json(payload: Mapping[str, Any], path: str | Path) -> None:
    if payload.get("schema") != PHYLOLM_TABLE_SCHEMA or payload.get("valid") is not True:
        raise ValueError("refusing to write an invalid corrected Table 10 artifact")
    _write_text_atomic(
        Path(path), json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def write_phylolm_table_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
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
            writer = csv.DictWriter(handle, fieldnames=list(PHYLOLM_TABLE_COLUMNS))
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {column: row.get(column) for column in PHYLOLM_TABLE_COLUMNS}
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


def _validate_cohort_receipt(
    receipt: object,
    *,
    suite_id: str,
    truth_specs: Mapping[str, Mapping[str, Any]],
) -> None:
    if not isinstance(receipt, Mapping):
        raise ValueError(f"{suite_id} cohort receipt must be a JSON object")
    if receipt.get("valid") is not True:
        raise ValueError(f"{suite_id} cohort receipt is not explicitly valid")
    for field, expected in (("n_trees", 50), ("n_errors", 0), ("n_warnings", 0)):
        if receipt.get(field) != expected:
            raise ValueError(
                f"{suite_id} cohort receipt {field} must be {expected}, "
                f"got {receipt.get(field)!r}"
            )
    if receipt.get("n_runs") != 641:
        raise ValueError(
            f"{suite_id} cohort receipt n_runs must be 641, "
            f"got {receipt.get('n_runs')!r}"
        )
    config_name = Path(str(receipt.get("config", ""))).name
    if config_name != f"{suite_id}.yaml":
        raise ValueError(
            f"{suite_id} cohort receipt config basename must be {suite_id}.yaml, "
            f"got {config_name!r}"
        )
    trees = receipt.get("trees")
    if not isinstance(trees, list):
        raise ValueError(f"{suite_id} cohort receipt trees must be a list")
    by_tree: dict[str, Mapping[str, Any]] = {}
    for row in trees:
        if not isinstance(row, Mapping):
            raise ValueError(f"{suite_id} cohort receipt tree rows must be mappings")
        tree_id = canonical_tree_id(row.get("tree_id"))
        if tree_id in by_tree:
            raise ValueError(f"{suite_id} cohort receipt duplicates {tree_id}")
        by_tree[tree_id] = row
    if set(by_tree) != set(ALL_TREE_IDS):
        raise ValueError(f"{suite_id} cohort receipt does not contain exact tree IDs 001..050")
    for tree_id in TOPOLOGY_TREE_IDS:
        row = by_tree[tree_id]
        if row.get("valid") is not True:
            raise ValueError(f"{suite_id}/{tree_id} cohort tree is not explicitly valid")
        if row.get("manifest_sha256") != truth_specs[tree_id]["sha256"]:
            raise ValueError(f"{suite_id}/{tree_id} cohort truth hash mismatch")


def _load_weight_rows(
    payload: object,
    *,
    suite_id: str,
    expected_artifact: str,
    metric: str,
    truth_specs: Mapping[str, Mapping[str, Any]],
    base: Path,
    source_dir: Path,
) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{suite_id} weight rollup must be a JSON object")
    if payload.get("valid") is not True:
        raise ValueError(f"{suite_id} weight rollup is not explicitly valid")
    if payload.get("artifact") != expected_artifact:
        raise ValueError(
            f"{suite_id} weight rollup artifact must be {expected_artifact!r}, "
            f"got {payload.get('artifact')!r}"
        )
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list):
        raise ValueError(f"{suite_id} weight rollup rows must be a list")
    metric_rows = [row for row in raw_rows if isinstance(row, Mapping) and row.get("metric") == metric]
    by_tree: dict[str, Mapping[str, Any]] = {}
    for row in metric_rows:
        tree_id = canonical_tree_id(row.get("tree_id"))
        if tree_id in by_tree:
            raise ValueError(f"{suite_id} weight rollup duplicates {tree_id}/{metric}")
        by_tree[tree_id] = row
    if set(by_tree) != set(ALL_TREE_IDS):
        raise ValueError(
            f"{suite_id} weight rollup must contain exactly one {metric} row for IDs 001..050"
        )

    selected: dict[str, dict[str, Any]] = {}
    for tree_id in ALL_TREE_IDS:
        row = by_tree[tree_id]
        if row.get("artifact") != expected_artifact:
            raise ValueError(f"{suite_id}/{tree_id} weight artifact mismatch")
        if row.get("representation") != "full_weight":
            raise ValueError(f"{suite_id}/{tree_id} weight representation is not full_weight")
        n_truth = _nonnegative_int(row.get("n_truth_splits"), label="n_truth_splits")
        if tree_id in TOPOLOGY_EXCLUDED_TREE_IDS:
            if n_truth != 0:
                raise ValueError(f"{suite_id}/{tree_id} is excluded but has truth splits")
            continue
        if n_truth <= 0:
            raise ValueError(f"{suite_id}/{tree_id} is eligible but has zero truth splits")
        _verify_row_truth_path(
            row,
            suite_id=suite_id,
            tree_id=tree_id,
            expected=truth_specs[tree_id],
            base=base,
            source_dir=source_dir,
        )
        _validate_native_direct_weight_row(
            row,
            suite_id=suite_id,
            tree_id=tree_id,
            expected_artifact=expected_artifact,
            metric=metric,
            truth_path=Path(truth_specs[tree_id]["path"]).resolve(),
            base=base,
            source_dir=source_dir,
        )
        selected[tree_id] = _validated_recovery_row(row, context=f"{suite_id}/{tree_id} weight")
    return selected


def _load_phylolm_rows(
    payload: object,
    *,
    suite_id: str,
    truth_specs: Mapping[str, Mapping[str, Any]],
    truth_digest: str,
) -> dict[str, dict[str, Any]]:
    # Local import avoids the constants-only import cycle: the native producer
    # imports this module's suite/schema contract.
    from weighttraits.behavior.phylolm import validate_phylolm_runset_receipt

    validate_phylolm_runset_receipt(payload, suite_id=suite_id)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{suite_id} PhyloLM receipt must be a JSON object")
    if payload.get("schema") != PHYLOLM_RECEIPT_SCHEMA:
        raise ValueError(f"{suite_id} PhyloLM receipt has wrong schema")
    if payload.get("valid") is not True:
        raise ValueError(f"{suite_id} PhyloLM receipt is not explicitly valid")
    if payload.get("suite_id") != suite_id:
        raise ValueError(f"{suite_id} PhyloLM receipt suite_id mismatch")
    if payload.get("group") != _expected_phylolm_group(suite_id):
        raise ValueError(
            f"{suite_id} PhyloLM group must be {_expected_phylolm_group(suite_id)!r}, "
            f"got {payload.get('group')!r}"
        )
    if payload.get("protocol") != PHYLOLM_PROTOCOL_CONTRACT:
        raise ValueError(f"{suite_id} PhyloLM protocol drift")
    if payload.get("n_trees") != len(TOPOLOGY_TREE_IDS):
        raise ValueError(f"{suite_id} PhyloLM receipt must declare n_trees=46")
    if payload.get("common_tree_ids") != list(TOPOLOGY_TREE_IDS):
        raise ValueError(f"{suite_id} PhyloLM common tree IDs/order differ")
    completion_path = _verified_payload_pin(
        payload.get("completion_receipt"),
        label=f"{suite_id} PhyloLM completion receipt",
    )
    validate_strict_training_completion(
        json.loads(completion_path.read_text()), cohort_id=suite_id
    )
    _verified_payload_pin(
        payload.get("stage_manifest"),
        label=f"{suite_id} PhyloLM stage manifest",
    )
    genome_path = Path(str(payload.get("genome_receipt", "")))
    if not genome_path.is_file() or sha256(genome_path) != payload.get(
        "genome_receipt_sha256"
    ):
        raise ValueError(f"{suite_id} PhyloLM sampled-genome pin does not verify")
    tree_receipts = payload.get("tree_receipts")
    if not isinstance(tree_receipts, Mapping) or set(tree_receipts) != set(
        TOPOLOGY_TREE_IDS
    ):
        raise ValueError(f"{suite_id} PhyloLM tree receipt pins are not exact common-46")
    if payload.get("truth_hashes_sha256") != truth_digest:
        raise ValueError(f"{suite_id} PhyloLM truth hash-set digest mismatch")
    declared_hashes = payload.get("truth_manifest_sha256")
    expected_hashes = {
        tree_id: truth_specs[tree_id]["sha256"] for tree_id in TOPOLOGY_TREE_IDS
    }
    if declared_hashes != expected_hashes:
        raise ValueError(f"{suite_id} PhyloLM truth hash map mismatch")
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list):
        raise ValueError(f"{suite_id} PhyloLM receipt rows must be a list")
    by_tree: dict[str, dict[str, Any]] = {}
    for raw_row in raw_rows:
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"{suite_id} PhyloLM rows must be mappings")
        tree_id = canonical_tree_id(raw_row.get("tree_id", raw_row.get("run")))
        if tree_id in by_tree:
            raise ValueError(f"{suite_id} PhyloLM receipt duplicates {tree_id}")
        if raw_row.get("truth_sha256") != truth_specs.get(tree_id, {}).get("sha256"):
            raise ValueError(f"{suite_id}/{tree_id} PhyloLM truth hash mismatch")
        tree_path = _verified_payload_pin(
            tree_receipts[tree_id],
            label=f"{suite_id}/{tree_id} PhyloLM tree receipt",
        )
        if raw_row.get("tree_receipt") != str(tree_path):
            raise ValueError(f"{suite_id}/{tree_id} PhyloLM row tree-receipt path drift")
        if raw_row.get("tree_receipt_sha256") != sha256(tree_path):
            raise ValueError(f"{suite_id}/{tree_id} PhyloLM row tree-receipt hash drift")
        tree_payload = json.loads(tree_path.read_text())
        _validate_tree_receipt_summary(
            tree_payload,
            suite_id=suite_id,
            tree_id=tree_id,
            truth_sha256=truth_specs[tree_id]["sha256"],
            row=raw_row,
        )
        by_tree[tree_id] = _validated_recovery_row(
            raw_row,
            context=f"{suite_id}/{tree_id} PhyloLM",
        )
    if set(by_tree) != set(TOPOLOGY_TREE_IDS):
        raise ValueError(
            f"{suite_id} PhyloLM receipt must contain the exact canonical 46 tree IDs; "
            f"missing={sorted(set(TOPOLOGY_TREE_IDS) - set(by_tree))}, "
            f"unexpected={sorted(set(by_tree) - set(TOPOLOGY_TREE_IDS))}"
        )
    return by_tree


def _verified_payload_pin(raw: object, *, label: str) -> Path:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must be a path+sha256 mapping")
    path = Path(str(raw.get("path", ""))).resolve()
    digest = raw.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"{label} has invalid sha256")
    if not path.is_file() or sha256(path) != digest:
        raise ValueError(f"{label} pin does not verify")
    return path


def _validate_tree_receipt_summary(
    payload: object,
    *,
    suite_id: str,
    tree_id: str,
    truth_sha256: str,
    row: Mapping[str, Any],
) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{suite_id}/{tree_id} tree receipt must be a JSON object")
    if (
        payload.get("schema") != PHYLOLM_TREE_RECEIPT_SCHEMA
        or payload.get("valid") is not True
        or payload.get("suite_id") != suite_id
        or canonical_tree_id(payload.get("tree_id")) != tree_id
        or payload.get("protocol") != PHYLOLM_PROTOCOL_CONTRACT
        or payload.get("truth_sha256") != truth_sha256
    ):
        raise ValueError(f"{suite_id}/{tree_id} tree receipt provenance drift")
    for field in (
        "n_truth_splits",
        "clade_recovery",
        "polytomy_aware_exact_recovery",
        "rf",
        "false_negative",
        "false_positive",
    ):
        if payload.get(field) != row.get(field):
            raise ValueError(
                f"{suite_id}/{tree_id} tree receipt {field} differs from runset row"
            )


def _verify_row_truth_path(
    row: Mapping[str, Any],
    *,
    suite_id: str,
    tree_id: str,
    expected: Mapping[str, Any],
    base: Path,
    source_dir: Path,
) -> None:
    raw = row.get("truth_manifest")
    if not isinstance(raw, str):
        raise ValueError(f"{suite_id}/{tree_id} weight row lacks truth_manifest")
    raw_path = Path(raw)
    candidates = (
        [raw_path]
        if raw_path.is_absolute()
        else [resolve_path(raw, base=base), (source_dir / raw_path).resolve()]
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"{suite_id}/{tree_id} truth_manifest is not readable: {raw}")
    expected_path = Path(expected["path"]).resolve()
    if Path(path).resolve() != expected_path:
        raise ValueError(
            f"{suite_id}/{tree_id} weight truth path mismatch: "
            f"expected {expected_path}, got {Path(path).resolve()}"
        )
    if sha256(path) != expected["sha256"]:
        raise ValueError(f"{suite_id}/{tree_id} weight truth hash mismatch")


def _validate_native_direct_weight_row(
    row: Mapping[str, Any],
    *,
    suite_id: str,
    tree_id: str,
    expected_artifact: str,
    metric: str,
    truth_path: Path,
    base: Path,
    source_dir: Path,
) -> None:
    """Replay the native direct-analysis chain instead of trusting rollup scores."""

    context = f"{suite_id}/{tree_id} weight"
    if (
        row.get("analysis_engine") != "direct"
        or row.get("tree_builder") != "biopython_neighbor_joining"
        or row.get("rf_engine") != "dendropy_treecompare"
    ):
        raise ValueError(f"{context} lacks native direct-analysis provenance")
    summary_path = _resolve_native_artifact_path(
        row.get("summary"),
        label=f"{context} summary",
        base=base,
        source_dir=source_dir,
    )
    summary = json.loads(summary_path.read_text())
    if not isinstance(summary, Mapping):
        raise ValueError(f"{context} summary must be a JSON object")
    if (
        summary.get("schema") != DIRECT_TREE_ANALYSIS_SCHEMA
        or summary.get("valid") is not True
        or summary.get("producer") != "weighttraits"
    ):
        raise ValueError(f"{context} summary is not a native WeightTraits receipt")
    expected_summary = {
        "analysis_engine": "direct",
        "distance_engine": "direct_streaming_sufficient_stats",
        "tree_builder": "biopython_neighbor_joining",
        "rf_engine": "dendropy_treecompare",
        "artifact": expected_artifact,
        "representation": "full_weight",
    }
    for field, expected in expected_summary.items():
        if summary.get(field) != expected:
            raise ValueError(f"{context} summary {field} drift")
    raw_metrics = summary.get("metrics")
    if not isinstance(raw_metrics, list) or metric not in raw_metrics:
        raise ValueError(f"{context} summary lacks metric {metric!r}")
    model_ids = summary.get("model_ids")
    if model_ids != manifest_leaf_ids(truth_path):
        raise ValueError(f"{context} model IDs differ from truth leaves")
    models_path = summary_path.parent / "models.json"
    layers_path = summary_path.parent / "layers.json"
    if (
        not models_path.is_file()
        or json.loads(models_path.read_text()) != model_ids
        or not layers_path.is_file()
    ):
        raise ValueError(f"{context} native models/layers artifacts do not verify")
    layer_names = json.loads(layers_path.read_text())
    if (
        not isinstance(layer_names, list)
        or len(layer_names) != summary.get("n_layers")
        or len(layer_names) != len(set(layer_names))
    ):
        raise ValueError(f"{context} native layer inventory is invalid")
    if summary.get("n_models") != len(model_ids):
        raise ValueError(f"{context} native model count drift")
    summary_truth = _resolve_native_artifact_path(
        summary.get("truth_manifest"),
        label=f"{context} summary truth",
        base=base,
        source_dir=source_dir,
        summary_dir=summary_path.parent,
    )
    if summary_truth != truth_path:
        raise ValueError(f"{context} summary truth path drift")
    if summary.get("truth_manifest_sha256") != sha256(truth_path):
        raise ValueError(f"{context} summary truth digest drift")
    ledger_path = _resolve_native_artifact_path(
        summary.get("ledger"),
        label=f"{context} ledger",
        base=base,
        source_dir=source_dir,
        summary_dir=summary_path.parent,
    )
    row_ledger = _resolve_native_artifact_path(
        row.get("ledger"),
        label=f"{context} rollup ledger",
        base=base,
        source_dir=source_dir,
        summary_dir=summary_path.parent,
    )
    if row_ledger != ledger_path:
        raise ValueError(f"{context} rollup/summary ledger path drift")
    if summary.get("ledger_sha256") != sha256(ledger_path):
        raise ValueError(f"{context} summary ledger digest drift")

    output_hashes = summary.get("output_sha256")
    if not isinstance(output_hashes, Mapping):
        raise ValueError(f"{context} summary lacks native output hashes")
    for field, path in (("models", models_path), ("layers", layers_path)):
        if output_hashes.get(field) != sha256(path):
            raise ValueError(f"{context} native {field} digest drift")

    cube_path = _resolve_native_artifact_path(
        summary.get("distance_layers"),
        label=f"{context} distance cube",
        base=base,
        source_dir=source_dir,
        summary_dir=summary_path.parent,
    )
    with np.load(cube_path, allow_pickle=False) as archive:
        if metric not in archive.files:
            raise ValueError(f"{context} distance cube lacks {metric!r}")
        cube = np.asarray(archive[metric], dtype=np.float64)
    if output_hashes.get("distance_layers") != sha256(cube_path):
        raise ValueError(f"{context} native distance-cube digest drift")
    expected_shape = (len(layer_names), len(model_ids), len(model_ids))
    if cube.shape != expected_shape:
        raise ValueError(f"{context} distance cube has shape {cube.shape}, expected {expected_shape}")
    _validate_native_distance_cube(cube, context=context)
    replay_matrix = np.mean(cube, axis=0)
    matrix_path = summary_path.parent / f"distance_matrix_{metric}.npy"
    if not matrix_path.is_file():
        raise ValueError(f"{context} native distance matrix is missing")
    if output_hashes.get(f"distance_matrix_{metric}") != sha256(matrix_path):
        raise ValueError(f"{context} native distance-matrix digest drift")
    matrix = np.asarray(np.load(matrix_path, allow_pickle=False), dtype=np.float64)
    if matrix.shape != replay_matrix.shape or not np.allclose(
        matrix, replay_matrix, rtol=0.0, atol=1e-12
    ):
        raise ValueError(f"{context} distance matrix differs from full-cube replay")

    results = summary.get("results")
    matches = (
        [result for result in results if isinstance(result, Mapping) and result.get("metric") == metric]
        if isinstance(results, list)
        else []
    )
    if len(matches) != 1:
        raise ValueError(f"{context} summary must contain one {metric!r} result")
    result = matches[0]
    tree_path = _resolve_native_artifact_path(
        result.get("tree"),
        label=f"{context} inferred tree",
        base=base,
        source_dir=source_dir,
        summary_dir=summary_path.parent,
    )
    if output_hashes.get(f"tree_{metric}") != sha256(tree_path):
        raise ValueError(f"{context} native inferred-tree digest drift")
    audit_path = _resolve_native_artifact_path(
        result.get("tree_audit"),
        label=f"{context} tree audit",
        base=base,
        source_dir=source_dir,
        summary_dir=summary_path.parent,
    )
    if output_hashes.get(f"tree_audit_{metric}") != sha256(audit_path):
        raise ValueError(f"{context} native tree-audit digest drift")
    observed_newick = tree_path.read_text().strip()
    replay_newick = neighbor_joining_newick(model_ids, replay_matrix)
    observed_splits, observed_leaves = splits_from_newick_text(observed_newick)
    replay_splits, replay_leaves = splits_from_newick_text(replay_newick)
    if observed_leaves != replay_leaves or observed_splits != replay_splits:
        raise ValueError(f"{context} inferred tree differs from distance replay")
    truth_splits, truth_leaves = splits_from_manifest_path(str(truth_path))
    replay_score = score_split_recovery(
        truth_splits,
        observed_splits,
        truth_leaves,
        observed_leaves,
    )
    score_path = _resolve_native_artifact_path(
        result.get("score"),
        label=f"{context} score",
        base=base,
        source_dir=source_dir,
        summary_dir=summary_path.parent,
    )
    row_score_path = _resolve_native_artifact_path(
        row.get("score"),
        label=f"{context} rollup score",
        base=base,
        source_dir=source_dir,
        summary_dir=summary_path.parent,
    )
    if row_score_path != score_path:
        raise ValueError(f"{context} rollup/summary score path drift")
    score = json.loads(score_path.read_text())
    if output_hashes.get(f"score_{metric}") != sha256(score_path):
        raise ValueError(f"{context} native score digest drift")
    if not isinstance(score, Mapping):
        raise ValueError(f"{context} score must be a JSON object")
    for field in (
        "n_truth_splits",
        "clade_recovery",
        "polytomy_aware_exact_recovery",
        "rf",
        "false_negative",
        "false_positive",
    ):
        expected = replay_score[field]
        if score.get(field) != expected or row.get(field) != expected:
            raise ValueError(f"{context} {field} differs from native replay")
        if field in result and result.get(field) != expected:
            raise ValueError(f"{context} summary result {field} differs from replay")


def _resolve_native_artifact_path(
    raw: object,
    *,
    label: str,
    base: Path,
    source_dir: Path,
    summary_dir: Path | None = None,
) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{label} path is missing")
    value = Path(raw)
    if value.is_absolute():
        candidates = [value]
    else:
        candidates = [(base / value).resolve(), (source_dir / value).resolve()]
        if summary_dir is not None:
            candidates.append((summary_dir / value).resolve())
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(f"{label} is not readable: {raw}")


def _validate_native_distance_cube(cube: np.ndarray, *, context: str) -> None:
    if (
        not np.isfinite(cube).all()
        or not np.allclose(cube, np.swapaxes(cube, 1, 2), rtol=0.0, atol=1e-12)
        or not np.allclose(
            np.diagonal(cube, axis1=1, axis2=2), 0.0, rtol=0.0, atol=1e-12
        )
        or np.any(cube < -1e-12)
    ):
        raise ValueError(f"{context} native distance cube violates metric invariants")


def _validated_recovery_row(row: Mapping[str, Any], *, context: str) -> dict[str, Any]:
    n_truth = _positive_int(row.get("n_truth_splits"), label=f"{context} n_truth_splits")
    false_negative = _nonnegative_int(
        row.get("false_negative"), label=f"{context} false_negative"
    )
    false_positive = _nonnegative_int(
        row.get("false_positive"), label=f"{context} false_positive"
    )
    rf = _nonnegative_int(row.get("rf"), label=f"{context} rf")
    clade = _bounded_float(
        row.get("clade_recovery"),
        label=f"{context} clade_recovery",
        minimum=0.0,
        maximum=1.0,
    )
    paer = row.get("polytomy_aware_exact_recovery")
    if not isinstance(paer, bool):
        raise ValueError(f"{context} polytomy_aware_exact_recovery must be boolean")
    expected_clade = (n_truth - false_negative) / n_truth
    if not math.isclose(clade, expected_clade, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(
            f"{context} clade_recovery is inconsistent with truth/FN counts: "
            f"{clade} != {expected_clade}"
        )
    if rf != false_negative + false_positive:
        raise ValueError(f"{context} RF is inconsistent with FN+FP")
    if paer != (false_negative == 0):
        raise ValueError(f"{context} PAER is inconsistent with false_negative")
    return {
        "n_truth_splits": n_truth,
        "clade_recovery": clade,
        "polytomy_aware_exact_recovery": paer,
        "rf": rf,
        "false_negative": false_negative,
        "false_positive": false_positive,
    }


def _aggregate_suite(
    *,
    suite_id: str,
    suite_contract: Mapping[str, str],
    weight_rows: Mapping[str, Mapping[str, Any]],
    phylolm_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    weight = _aggregate_method(weight_rows)
    phylolm = _aggregate_method(phylolm_rows)
    return {
        "suite_id": suite_id,
        "label": suite_contract["label"],
        "training": suite_contract["training"],
        "n_trees": len(TOPOLOGY_TREE_IDS),
        **{f"weight_{key}": value for key, value in weight.items()},
        **{f"phylolm_{key}": value for key, value in phylolm.items()},
        "clade_recovery_difference": (
            weight["clade_recovery"] - phylolm["clade_recovery"]
        ),
    }


def _aggregate_method(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    if set(rows) != set(TOPOLOGY_TREE_IDS):
        raise ValueError("method rows are not the exact canonical 46-tree set")
    ordered = [rows[tree_id] for tree_id in TOPOLOGY_TREE_IDS]
    clade = [float(row["clade_recovery"]) for row in ordered]
    paer = [1.0 if row["polytomy_aware_exact_recovery"] else 0.0 for row in ordered]
    rf = [float(row["rf"]) for row in ordered]
    false_negative = [float(row["false_negative"]) for row in ordered]
    p = statistics.fmean(paer)
    return {
        "clade_recovery": statistics.fmean(clade),
        "clade_recovery_se": _sample_se(clade),
        "paer": p,
        "paer_se": math.sqrt(p * (1.0 - p) / len(paer)),
        "rf": statistics.fmean(rf),
        "rf_se": _sample_se(rf),
        "false_negative": statistics.fmean(false_negative),
        "false_negative_se": _sample_se(false_negative),
    }


def _expected_phylolm_group(suite_id: str) -> str:
    return f"runs_weighttraits_{suite_id}_fresh_phylolm"


def _sample_se(values: Sequence[float]) -> float:
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def _nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _positive_int(value: object, *, label: str) -> int:
    result = _nonnegative_int(value, label=label)
    if result == 0:
        raise ValueError(f"{label} must be positive")
    return result


def _bounded_float(
    value: object, *, label: str, minimum: float, maximum: float
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum or result > maximum:
        raise ValueError(f"{label} must be finite and in [{minimum}, {maximum}]")
    return result
