"""Read-only replay of the all-node semantic study from its saved artifacts.

For each endpoint of a cohort's semantic job config, this follows the production runner's cache
path (``semantic_runner.execute_tree_endpoint_streaming``) without writing anything:

* re-read every node's pinned responses to rebuild the response scope;
* load the saved semantic matrix and the direct cosine matrix;
* recompute the tree analysis and compare it with the saved ``tree_analysis.json``;
* re-aggregate the 50 trees and compare with the saved run-set analysis.

``python -m weighttraits.behavior.allnode.replay --config <semantic job config>`` prints one row
per endpoint and exits nonzero on any mismatch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from weighttraits.behavior.allnode.manifest import canonical_sha256
from weighttraits.behavior.allnode.semantic_analysis import (
    DIRECT_ARTIFACT_KIND,
    SEMANTIC_ARTIFACT_KIND,
    aggregate_production_semantic_runset_streaming,
    analyze_tree_semantic_bridge,
    bind_distance_artifact,
)
from weighttraits.behavior.allnode.semantic_runner import (
    _read_semantic_matrix_artifact,
    _sha256_file,
    _validate_node_source,
    load_direct_cosine_source,
    load_node_endpoint_source,
    load_semantic_job_config,
    topology_from_manifest,
    validate_direct_cohort_terminal,
)


def replay_semantic_cohort(config_path: str | Path, *, endpoints=None) -> dict[str, Any]:
    _, plan, inference, _, config_sha = load_semantic_job_config(config_path)
    validate_direct_cohort_terminal(plan)
    rows = []
    for endpoint in plan.endpoints:
        if endpoints and endpoint.endpoint_id not in endpoints:
            continue
        root = Path(plan.output_root) / endpoint.endpoint_id
        mismatched: list[str] = []

        def analyses(endpoint=endpoint, root=root, mismatched=mismatched):
            pairs = zip(inference.checkpoint_manifests, inference.node_covariates)
            for position, (manifest, covariates) in enumerate(pairs):
                topology = topology_from_manifest(manifest)
                direct = load_direct_cosine_source(
                    Path(plan.direct_output_root) / topology.tree_id,
                    topology=topology,
                    manifest=manifest,
                    node_covariates_sha256=covariates.covariates_sha256,
                )
                scope = []
                for ordinal in range(topology.n_nodes):
                    source = load_node_endpoint_source(
                        inference, stage_root=plan.stage_root, tree_position=position,
                        node_ordinal=ordinal, endpoint=endpoint,
                    )
                    _validate_node_source(source, topology=topology, endpoint=endpoint,
                                          ordinal=ordinal)
                    scope.append(source.scope_dict())
                tree_dir = root / topology.tree_id
                semantic = _read_semantic_matrix_artifact(
                    tree_dir / "semantic_matrix.json",
                    topology=topology,
                    endpoint=endpoint,
                    checkpoint_set_sha256=manifest.checkpoint_set_sha256,
                    node_covariates_sha256=covariates.covariates_sha256,
                    response_scope_sha256=canonical_sha256(scope),
                    encoder_receipt_sha256=plan.encoder_receipt.sha256,
                )
                analysis = analyze_tree_semantic_bridge(
                    topology,
                    manifest,
                    cohort_key=plan.cohort_key,
                    endpoint_id=endpoint.endpoint_id,
                    node_covariates=covariates,
                    direct=bind_distance_artifact(
                        direct.matrix,
                        artifact_kind=DIRECT_ARTIFACT_KIND,
                        endpoint_id="weight_cosine",
                        checkpoint_set_sha256=manifest.checkpoint_set_sha256,
                        source_receipt_sha256=direct.artifact_receipt.sha256,
                    ),
                    semantic=bind_distance_artifact(
                        semantic,
                        artifact_kind=SEMANTIC_ARTIFACT_KIND,
                        endpoint_id=endpoint.endpoint_id,
                        checkpoint_set_sha256=manifest.checkpoint_set_sha256,
                        source_receipt_sha256=_sha256_file(tree_dir / "semantic_matrix.json"),
                    ),
                )
                saved = json.loads((tree_dir / "tree_analysis.json").read_text())
                if analysis.receipt.to_dict() != saved:
                    mismatched.append(topology.tree_id)
                yield analysis

        runset = aggregate_production_semantic_runset_streaming(
            analyses,
            expected_nodes_per_tree=tuple(m.n_nodes for m in inference.checkpoint_manifests),
        )
        saved = json.loads((root / "runset_receipt.json").read_text())
        primary = next(r for r in runset.to_dict()["strata"]
                       if r["stratum"] == "all_trained_node_pairs")
        rows.append(dict(
            endpoint=endpoint.endpoint_id,
            n_trees=len(inference.checkpoint_manifests),
            tree_analyses_mismatched=mismatched,
            runset_matches=runset.to_dict() == saved["analysis"],
            analysis_sha256=runset.analysis_sha256,
            saved_analysis_sha256=saved["analysis_sha256"],
            primary_dl=primary["dl"],
        ))
    return dict(cohort=plan.cohort_key, config=str(config_path), config_sha256=config_sha,
                endpoints=rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--endpoint", action="append")
    args = parser.parse_args(argv)
    result = replay_semantic_cohort(args.config, endpoints=args.endpoint)
    print(json.dumps(result, indent=2, sort_keys=True))
    ok = all(r["runset_matches"] and not r["tree_analyses_mismatched"] for r in result["endpoints"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
