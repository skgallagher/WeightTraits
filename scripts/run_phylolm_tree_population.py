#!/usr/bin/env python3
"""Collect or resume all leaf populations for one corrected PhyloLM tree."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from weighttraits.behavior.phylolm import load_population_receipt
from weighttraits.behavior.phylolm_inference import collect_phylolm_leaf_population
from weighttraits.behavior.probe_inference import (
    resolve_leaf_checkpoints,
    sha256_path,
)
from weighttraits.paper.analysis_contracts import (
    sha256,
    validate_strict_training_completion,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--completion-receipt", type=Path, required=True)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--genome-receipt", type=Path, required=True)
    parser.add_argument("--cohort-id", required=True)
    parser.add_argument("--tree-id", required=True)
    parser.add_argument("--base-model-id", required=True)
    parser.add_argument("--base-model-revision", required=True)
    parser.add_argument("--path-base", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.batch_size != 64:
        parser.error("corrected PhyloLM requires --batch-size 64")
    if args.torch_dtype != "auto":
        parser.error("corrected PhyloLM requires --torch-dtype auto")

    completion_path = args.completion_receipt.resolve()
    completion = json.loads(completion_path.read_text())
    validate_strict_training_completion(completion, cohort_id=args.cohort_id)
    tree = resolve_leaf_checkpoints(
        args.training_summary,
        cohort_id=args.cohort_id,
        tree_id=args.tree_id,
        model_task="causal_lm",
        base_model_id=args.base_model_id,
        base_model_revision=args.base_model_revision,
        path_base=args.path_base,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "valid": True,
                    "cohort_id": tree.cohort_id,
                    "tree_id": tree.tree_id,
                    "method": tree.method,
                    "artifact_name": tree.artifact_name,
                    "n_leaves": len(tree.leaves),
                    "leaf_ids": [leaf.leaf_id for leaf in tree.leaves],
                    "checkpoints": [str(leaf.checkpoint) for leaf in tree.leaves],
                    "truth_manifest": str(tree.leaves[0].truth_manifest),
                    "truth_manifest_sha256": tree.leaves[0].truth_manifest_sha256,
                    "completion_receipt": str(completion_path),
                    "completion_receipt_sha256": sha256(completion_path),
                    "stage_manifest": str(args.stage_manifest.resolve()),
                    "stage_manifest_sha256": sha256(args.stage_manifest.resolve()),
                    "genome_receipt": str(args.genome_receipt.resolve()),
                    "genome_receipt_sha256": sha256(args.genome_receipt.resolve()),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    population_specs = {}
    for leaf in tree.leaves:
        output = out_dir / f"{leaf.leaf_id}.population.json"
        checkpoint_digest = sha256_path(leaf.checkpoint)
        if output.exists():
            load_population_receipt(
                output,
                suite_id=tree.cohort_id,
                tree_id=tree.tree_id,
                model_id=leaf.leaf_id,
                genome_receipt_path=args.genome_receipt,
                checkpoint_path=leaf.checkpoint,
                checkpoint_sha256=checkpoint_digest,
                completion_receipt_path=completion_path,
                stage_manifest_path=args.stage_manifest,
                checkpoint_provenance=leaf.to_dict(),
            )
        else:
            collect_phylolm_leaf_population(
                leaf,
                genome_receipt_path=args.genome_receipt,
                completion_receipt_path=completion_path,
                stage_manifest_path=args.stage_manifest,
                output_path=output,
                batch_size=args.batch_size,
                torch_dtype=args.torch_dtype,
            )
        population_specs[leaf.leaf_id] = {
            "path": str(output),
            "sha256": sha256(output),
        }
    specs_path = out_dir / "population_specs.json"
    payload = json.dumps(population_specs, indent=2, sort_keys=True) + "\n"
    if specs_path.exists():
        if specs_path.read_text() != payload:
            raise ValueError(f"existing population specs drift: {specs_path}")
    else:
        _write_text_atomic(specs_path, payload)
    print(specs_path)
    return 0


def _write_text_atomic(path: Path, text: str) -> None:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
