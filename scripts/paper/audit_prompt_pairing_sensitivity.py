#!/usr/bin/env python3
"""Audit prompt-paired semantic Table 3 inputs and resampling stability.

The input layout is compatible with the historical ELLMTrees result roots:

  GROUP/run_001/weight/{avg_distance_cosine.npy,distance_metadata.json}
  GROUP/run_001/behavioral_TASK/{predictions.json,behavioral_metadata.json,
                                dist_semantic_paired.npy}

No inference is performed. Saved text outputs are embedded, the paired matrix is
rebuilt and checked, and prompt/run-cluster bootstrap diagnostics are written as
JSON. The primary endpoint remains the saved prompt-paired semantic endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from weighttraits.analysis.prompt_pairing import (
    PromptPairingRun,
    flatten_prompt_outputs,
    prompt_pairing_sensitivity,
    validate_saved_matrix_rebuild_error,
)


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DEFAULT_SENTENCE_TRANSFORMERS_VERSION = "5.4.1"
DEFAULT_REBUILD_MAX_ABS_ERROR = 1e-6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_specs(
    group: Path, probe_subdir: str, expected_prompts: int | None
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    missing: list[str] = []
    for run_dir in sorted(group.glob("run_*")):
        if not run_dir.is_dir():
            continue
        behavior_dir = run_dir / probe_subdir
        paths = {
            "predictions": behavior_dir / "predictions.json",
            "behavior_metadata": behavior_dir / "behavioral_metadata.json",
            "paired_matrix": behavior_dir / "dist_semantic_paired.npy",
            "weight_matrix": run_dir / "weight" / "avg_distance_cosine.npy",
            "weight_metadata": run_dir / "weight" / "distance_metadata.json",
        }
        absent = [name for name, path in paths.items() if not path.exists()]
        if absent:
            missing.append(f"{run_dir.name}: {', '.join(absent)}")
            continue

        predictions = json.loads(paths["predictions"].read_text())
        behavior_nodes = json.loads(paths["behavior_metadata"].read_text())["node_ids"]
        weight_nodes = json.loads(paths["weight_metadata"].read_text())["model_names"]
        leaves = [node for node in weight_nodes if node in behavior_nodes and node in predictions]
        if len(leaves) < 3:
            raise ValueError(f"{run_dir.name}: fewer than three common leaf models")

        prompt_counts = set()
        for node in leaves:
            outputs = predictions[node]
            if not isinstance(outputs, list) or not all(isinstance(text, str) for text in outputs):
                raise ValueError(
                    f"{run_dir.name}/{node}: predictions must be one flat list of strings"
                )
            prompt_counts.add(len(outputs))
        if len(prompt_counts) != 1:
            raise ValueError(f"{run_dir.name}: models do not share one prompt count")
        n_prompts = prompt_counts.pop()
        if expected_prompts is not None and n_prompts != expected_prompts:
            raise ValueError(
                f"{run_dir.name}: expected {expected_prompts} prompts, found {n_prompts}"
            )

        specs.append(
            {
                "run_dir": run_dir,
                "behavior_dir": behavior_dir,
                "paths": paths,
                "predictions": predictions,
                "behavior_nodes": behavior_nodes,
                "weight_nodes": weight_nodes,
                "leaves": leaves,
                "n_prompts": n_prompts,
            }
        )
    if not specs:
        detail = "; ".join(missing[:5])
        raise ValueError(f"no complete run inputs under {group}: {detail}")
    return specs


def _flatten_prediction_outputs(specs: list[dict[str, Any]]) -> list[str]:
    """Preserve the old builder's exact leaf-major, prompt-minor text sequence."""

    flat_outputs: list[str] = []
    for spec in specs:
        flat_outputs.extend(flatten_prompt_outputs(spec["predictions"], spec["leaves"]))
    return flat_outputs


def _embed_runs(specs: list[dict[str, Any]], model: Any, batch_size: int) -> tuple[list, list]:
    flat_outputs = _flatten_prediction_outputs(specs)
    slices: list[tuple[int, int]] = []
    offset = 0
    for spec in specs:
        start = offset
        offset += len(spec["leaves"]) * spec["n_prompts"]
        slices.append((start, offset))

    embeddings = model.encode(
        flat_outputs,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)

    runs: list[PromptPairingRun] = []
    receipts: list[dict[str, Any]] = []
    for spec, (start, stop) in zip(specs, slices, strict=True):
        leaves = spec["leaves"]
        n_prompts = spec["n_prompts"]
        run_embeddings = embeddings[start:stop].reshape(len(leaves), n_prompts, -1)
        leaf_pairs = list(combinations(range(len(leaves)), 2))

        prompt_similarity = np.empty((len(leaf_pairs), n_prompts), dtype=np.float32)
        weight_distance = np.empty(len(leaf_pairs), dtype=np.float64)
        rebuilt_distance = np.empty(len(leaf_pairs), dtype=np.float64)
        saved_distance = np.empty(len(leaf_pairs), dtype=np.float64)

        weight_matrix = np.load(spec["paths"]["weight_matrix"])
        saved_matrix = np.load(spec["paths"]["paired_matrix"])
        weight_index = {node: index for index, node in enumerate(spec["weight_nodes"])}
        behavior_index = {node: index for index, node in enumerate(spec["behavior_nodes"])}

        for pair_index, (left, right) in enumerate(leaf_pairs):
            left_node, right_node = leaves[left], leaves[right]
            cosine_similarity = np.sum(
                run_embeddings[left] * run_embeddings[right],
                axis=1,
            )
            cosine_distance = np.clip(1.0 - cosine_similarity, 0.0, 2.0)
            prompt_similarity[pair_index] = 1.0 - cosine_distance
            rebuilt_distance[pair_index] = float(np.mean(cosine_distance))
            saved_distance[pair_index] = saved_matrix[
                behavior_index[left_node], behavior_index[right_node]
            ]
            weight_distance[pair_index] = weight_matrix[
                weight_index[left_node], weight_index[right_node]
            ]

        runs.append(
            PromptPairingRun(
                tree_id=spec["run_dir"].name,
                weight_distance=weight_distance,
                prompt_similarity=prompt_similarity,
            )
        )
        receipts.append(
            {
                "tree_id": spec["run_dir"].name,
                "n_leaf_models": len(leaves),
                "n_leaf_pairs": len(leaf_pairs),
                "n_prompts": n_prompts,
                "predictions_sha256": _sha256(spec["paths"]["predictions"]),
                "paired_matrix_sha256": _sha256(spec["paths"]["paired_matrix"]),
                "saved_matrix_rebuild_max_abs": float(
                    np.max(np.abs(saved_distance - rebuilt_distance))
                ),
            }
        )
    return runs, receipts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True, type=Path)
    parser.add_argument("--probe-subdir", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-revision", default=DEFAULT_EMBEDDING_REVISION)
    parser.add_argument(
        "--expected-sentence-transformers-version",
        default=DEFAULT_SENTENCE_TRANSFORMERS_VERSION,
    )
    parser.add_argument(
        "--max-rebuild-abs-error",
        type=float,
        default=DEFAULT_REBUILD_MAX_ABS_ERROR,
        help="Fail when any saved paired distance differs more than this absolute tolerance.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20_260_810)
    parser.add_argument("--expected-prompts", type=int, default=200)
    parser.add_argument(
        "--allow-online-model",
        action="store_true",
        help="Allow sentence-transformers to fetch a missing embedding model.",
    )
    args = parser.parse_args()

    specs = _load_specs(args.group, args.probe_subdir, args.expected_prompts)
    sentence_transformers_version = importlib.metadata.version("sentence-transformers")
    if sentence_transformers_version != args.expected_sentence_transformers_version:
        raise RuntimeError(
            "sentence-transformers runtime differs from the frozen semantic audit: "
            f"{sentence_transformers_version!r} != "
            f"{args.expected_sentence_transformers_version!r}"
        )
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        args.embedding_model,
        revision=args.embedding_revision,
        local_files_only=not args.allow_online_model,
    )
    runs, receipts = _embed_runs(specs, model, args.batch_size)
    rebuild_max_abs = validate_saved_matrix_rebuild_error(
        [receipt["saved_matrix_rebuild_max_abs"] for receipt in receipts],
        max_abs_error=args.max_rebuild_abs_error,
    )
    sensitivity = prompt_pairing_sensitivity(
        runs,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    report = {
        "schema": "weighttraits.prompt-pairing-sensitivity.v1",
        "group": str(args.group.resolve()),
        "probe_subdir": args.probe_subdir,
        "embedding_model": args.embedding_model,
        "embedding_revision": args.embedding_revision,
        "sentence_transformers_version": sentence_transformers_version,
        "primary_endpoint": "dist_semantic_paired.npy",
        "draw_axis_present": False,
        "outputs_per_model": args.expected_prompts,
        "saved_matrix_rebuild_max_abs": rebuild_max_abs,
        "saved_matrix_rebuild_tolerance": args.max_rebuild_abs_error,
        "saved_matrix_rebuild_valid": True,
        **sensitivity,
        "run_receipts": receipts,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    concise = {key: value for key, value in report.items() if key not in {"run_r", "run_receipts"}}
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
