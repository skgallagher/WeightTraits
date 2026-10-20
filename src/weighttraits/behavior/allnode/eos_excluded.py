"""App. F: the all-node semantic bridge after excluding blank response pairs.

A receipt-validating post-processor over the saved study (ported from the Wright-only
``run_table8_eos_sensitivity.py``). It reuses each tree's full-precision primary semantic matrix
and re-embeds only coordinates where at least one trained node emitted ``text.strip() == ""``;
for a node pair, a coordinate is excluded when either response is blank.

``python -m weighttraits.behavior.allnode.eos_excluded --config <semantic job config>
--endpoint <id> --output <path>``
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from weighttraits.behavior.allnode.checkpoints import ArtifactPin
from weighttraits.behavior.allnode.manifest import canonical_sha256
from weighttraits.behavior.allnode.matrices import validate_distance_matrix
from weighttraits.behavior.allnode.semantic_analysis import (
    DIRECT_ARTIFACT_KIND,
    SEMANTIC_ARTIFACT_KIND,
    SEMANTIC_METRIC,
    aggregate_production_semantic_runset_streaming,
    analyze_tree_semantic_bridge,
    bind_distance_artifact,
)
from weighttraits.behavior.allnode.semantic_runner import (
    NodeEndpointSource,
    _read_semantic_matrix_artifact,
    load_direct_cosine_source,
    load_pinned_minilm_encoder,
    load_semantic_job_config,
    topology_from_manifest,
    validate_direct_cohort_terminal,
    validate_encoder_receipt,
)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def load_pinned_sources(primary_payload, topology, endpoint):
    """Load text through the already validated primary artifact's exact source pins."""
    scope = primary_payload["response_scope"]
    if len(scope) != topology.n_nodes:
        raise ValueError("primary response scope node count drift")
    sources = []
    for ordinal, pin in enumerate(scope):
        node = topology.nodes[ordinal]
        if (pin["node_id"], pin["node_ordinal"], pin["node_role"]) != (
            node.node_id,
            node.node_ordinal,
            "leaf" if node.is_leaf else "internal",
        ):
            raise ValueError("primary response scope node identity/order drift")
        response_path = Path(pin["responses"]["path"])
        terminal_path = Path(pin["terminal_receipt"]["path"])
        response_bytes = response_path.read_bytes()
        terminal_bytes = terminal_path.read_bytes()
        if hashlib.sha256(response_bytes).hexdigest() != pin["responses"]["sha256"]:
            raise ValueError(f"response hash drift: {response_path}")
        if hashlib.sha256(terminal_bytes).hexdigest() != pin["terminal_receipt"]["sha256"]:
            raise ValueError(f"terminal hash drift: {terminal_path}")
        terminal = json.loads(terminal_bytes)
        if terminal.get("valid") is not True or terminal.get("status") != "completed":
            raise ValueError(f"invalid terminal receipt: {terminal_path}")
        if terminal.get("request_sha256") != pin["request_sha256"]:
            raise ValueError(f"terminal request hash drift: {terminal_path}")
        if terminal.get("responses") != pin["responses"]:
            raise ValueError(f"terminal response pin drift: {terminal_path}")
        selected = []
        endpoint_hash_rows = []
        for line in response_bytes.splitlines():
            row = json.loads(line)
            response_row = row["response_row"]
            if (
                response_row["protocol_id"] == endpoint.protocol_id
                and response_row["probe_id"] == endpoint.probe_id
            ):
                selected.append(row["text"])
                endpoint_hash_rows.append(
                    {
                        "protocol_id": response_row["protocol_id"],
                        "probe_id": response_row["probe_id"],
                        "prompt_id": response_row["prompt_id"],
                        "sample_id": response_row["sample_id"],
                        "response_sha256": response_row["response_sha256"],
                    }
                )
        if len(selected) != endpoint.n_paired_coordinates:
            raise ValueError(f"endpoint coordinate count drift: {response_path}")
        if canonical_sha256(endpoint_hash_rows) != pin["endpoint_rows_sha256"]:
            raise ValueError(f"endpoint row hash drift: {response_path}")
        sources.append(
            NodeEndpointSource(
                node_id=pin["node_id"],
                node_ordinal=pin["node_ordinal"],
                node_role=pin["node_role"],
                responses=ArtifactPin(**pin["responses"]),
                terminal_receipt=ArtifactPin(**pin["terminal_receipt"]),
                request_sha256=pin["request_sha256"],
                endpoint_rows_sha256=pin["endpoint_rows_sha256"],
                texts=tuple(selected),
            )
        )
    return sources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config, plan, inference, config_path, config_sha = load_semantic_job_config(args.config)
    endpoints = {value.endpoint_id: value for value in plan.endpoints}
    if args.endpoint not in endpoints:
        raise ValueError(f"unknown endpoint {args.endpoint!r}; choices={sorted(endpoints)}")
    endpoint = endpoints[args.endpoint]
    validate_direct_cohort_terminal(plan)
    snapshot = validate_encoder_receipt(plan)
    encoder = load_pinned_minilm_encoder(snapshot, batch_size=128)

    analyses = []
    tree_rows = []
    depth_counts: dict[int, list[int]] = {}
    total_leaf_empty = 0
    total_leaf_outputs = 0

    for tree_position, (manifest, covariates) in enumerate(
        zip(inference.checkpoint_manifests, inference.node_covariates)
    ):
        topology = topology_from_manifest(manifest)
        primary_path = (
            Path(plan.output_root)
            / endpoint.endpoint_id
            / topology.tree_id
            / "semantic_matrix.json"
        )
        primary_payload = json.loads(primary_path.read_text())
        sources = load_pinned_sources(primary_payload, topology, endpoint)
        response_scope = [source.scope_dict() for source in sources]
        response_scope_sha = canonical_sha256(response_scope)
        primary = _read_semantic_matrix_artifact(
            primary_path,
            topology=topology,
            endpoint=endpoint,
            checkpoint_set_sha256=manifest.checkpoint_set_sha256,
            node_covariates_sha256=covariates.covariates_sha256,
            response_scope_sha256=response_scope_sha,
            encoder_receipt_sha256=plan.encoder_receipt.sha256,
        )

        blank = np.asarray(
            [[not text.strip() for text in source.texts] for source in sources],
            dtype=bool,
        )
        union_coordinates = np.flatnonzero(np.any(blank, axis=0))
        n_coordinates = endpoint.n_paired_coordinates
        primary_sum = np.asarray(primary.matrix, dtype=np.float64) * float(n_coordinates)
        excluded_sum = np.zeros_like(primary_sum)
        excluded_count = np.zeros(primary_sum.shape, dtype=np.int64)

        if len(union_coordinates):
            embedded_nodes = []
            for source in sources:
                texts = [source.texts[index] for index in union_coordinates]
                values = np.asarray(encoder(texts), dtype=np.float64)
                if values.shape[0] != len(union_coordinates) or values.ndim != 2:
                    raise ValueError("encoder result shape drift")
                norms = np.linalg.norm(values, axis=1)
                if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
                    raise ValueError("nonfinite/zero embedding")
                embedded_nodes.append(values / norms[:, None])
            embeddings = np.stack(embedded_nodes, axis=0)
            for compact_index, coordinate in enumerate(union_coordinates):
                values = embeddings[:, compact_index, :]
                similarities = np.clip(values @ values.T, -1.0, 1.0)
                distances = np.clip(1.0 - similarities, 0.0, 2.0)
                np.fill_diagonal(distances, 0.0)
                excluded = blank[:, coordinate, None] | blank[None, :, coordinate]
                excluded_sum[excluded] += distances[excluded]
                excluded_count[excluded] += 1

        eligible_count = n_coordinates - excluded_count
        off_diagonal = ~np.eye(topology.n_nodes, dtype=bool)
        if np.any(eligible_count[off_diagonal] <= 0):
            bad = np.argwhere((eligible_count <= 0) & off_diagonal)
            raise ValueError(f"no nonempty paired coordinates for node pairs: {bad.tolist()}")
        eos_matrix_values = np.zeros_like(primary_sum)
        eos_matrix_values[off_diagonal] = (
            (primary_sum - excluded_sum)[off_diagonal] / eligible_count[off_diagonal]
        )
        eos_matrix_values = np.clip(
            (eos_matrix_values + eos_matrix_values.T) / 2.0, 0.0, 2.0
        )
        np.fill_diagonal(eos_matrix_values, 0.0)
        eos_matrix = validate_distance_matrix(
            topology,
            metric=SEMANTIC_METRIC,
            ordered_node_ids=topology.ordered_node_ids,
            matrix=eos_matrix_values.tolist(),
        )

        direct = load_direct_cosine_source(
            Path(plan.direct_output_root) / topology.tree_id,
            topology=topology,
            manifest=manifest,
            node_covariates_sha256=covariates.covariates_sha256,
        )
        method_receipt = {
            "schema": "weighttraits.table8_eos_excluded_source.v1",
            "tree_id": topology.tree_id,
            "endpoint_id": endpoint.endpoint_id,
            "primary_semantic_matrix": {
                "path": str(primary_path),
                "sha256": hashlib.sha256(primary_path.read_bytes()).hexdigest(),
            },
            "response_scope_sha256": response_scope_sha,
            "empty_definition": "text.strip() == empty string",
            "n_coordinates": n_coordinates,
            "n_union_blank_coordinates": int(len(union_coordinates)),
        }
        semantic_bound = bind_distance_artifact(
            eos_matrix,
            artifact_kind=SEMANTIC_ARTIFACT_KIND,
            endpoint_id=endpoint.endpoint_id,
            checkpoint_set_sha256=manifest.checkpoint_set_sha256,
            source_receipt_sha256=hashlib.sha256(canonical_bytes(method_receipt)).hexdigest(),
        )
        direct_bound = bind_distance_artifact(
            direct.matrix,
            artifact_kind=DIRECT_ARTIFACT_KIND,
            endpoint_id="weight_cosine",
            checkpoint_set_sha256=manifest.checkpoint_set_sha256,
            source_receipt_sha256=direct.artifact_receipt.sha256,
        )
        analysis = analyze_tree_semantic_bridge(
            topology,
            manifest,
            cohort_key=plan.cohort_key,
            endpoint_id=endpoint.endpoint_id,
            node_covariates=covariates,
            direct=direct_bound,
            semantic=semantic_bound,
        )
        analyses.append(analysis)

        tree_leaf_empty = 0
        tree_leaf_outputs = 0
        for ordinal, node in enumerate(topology.nodes):
            if not node.is_leaf:
                continue
            n_empty = int(np.sum(blank[ordinal]))
            tree_leaf_empty += n_empty
            tree_leaf_outputs += n_coordinates
            bucket = depth_counts.setdefault(node.depth, [0, 0])
            bucket[0] += n_empty
            bucket[1] += n_coordinates
        total_leaf_empty += tree_leaf_empty
        total_leaf_outputs += tree_leaf_outputs
        tree_rows.append(
            {
                "tree_id": topology.tree_id,
                "n_nodes": topology.n_nodes,
                "n_leaves": topology.n_leaves,
                "n_union_blank_coordinates": int(len(union_coordinates)),
                "leaf_empty": tree_leaf_empty,
                "leaf_outputs": tree_leaf_outputs,
            }
        )
        print(
            json.dumps(
                {
                    "tree": topology.tree_id,
                    "endpoint": endpoint.endpoint_id,
                    "union_blank_coordinates": int(len(union_coordinates)),
                    "leaf_empty": tree_leaf_empty,
                    "leaf_outputs": tree_leaf_outputs,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    runset = aggregate_production_semantic_runset_streaming(
        lambda: iter(analyses),
        expected_nodes_per_tree=[analysis.topology.n_nodes for analysis in analyses],
    )
    runset_payload = runset.to_dict()
    primary_stratum = next(
        row for row in runset_payload["strata"] if row["stratum"] == "all_trained_node_pairs"
    )
    result = {
        "schema": "weighttraits.table8_eos_excluded_endpoint.v1",
        "valid": True,
        "cohort_key": plan.cohort_key,
        "endpoint_id": endpoint.endpoint_id,
        "empty_definition": "text.strip() == empty string",
        "semantic_policy": "exclude each prompt/draw coordinate for a node pair when either decoded response is empty",
        "config": {"path": str(config_path), "sha256": config_sha},
        "semantic_plan_sha256": plan.semantic_plan_sha256,
        "encoder_receipt": plan.encoder_receipt.to_dict(),
        "counts": {
            "n_trees": len(tree_rows),
            "n_nodes": sum(row["n_nodes"] for row in tree_rows),
            "n_leaves": sum(row["n_leaves"] for row in tree_rows),
            "leaf_empty": total_leaf_empty,
            "leaf_outputs": total_leaf_outputs,
            "pooled_leaf_empty_percent": 100.0 * total_leaf_empty / total_leaf_outputs,
        },
        "depth": [
            {
                "depth": depth,
                "n_empty": values[0],
                "n_outputs": values[1],
                "pooled_empty_percent": 100.0 * values[0] / values[1],
            }
            for depth, values in sorted(depth_counts.items())
        ],
        "eos_excluded_all_node_dl": primary_stratum["dl"],
        "tree_rows": tree_rows,
        "runset_analysis": runset_payload,
    }
    result["artifact_sha256"] = canonical_sha256(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_bytes(canonical_bytes(result))
    temporary.replace(args.output)
    print(json.dumps({"status": "completed", "output": str(args.output), "summary": result["eos_excluded_all_node_dl"], "counts": result["counts"], "depth": result["depth"]}, sort_keys=True))


if __name__ == "__main__":
    main()
