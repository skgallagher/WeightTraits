#!/usr/bin/env python3
"""Rematerialize cumulative LoRA leaves and analyze their merged full weights.

The materialization stage may use training provenance to apply each ordered
root-to-leaf adapter chain.  The whitebox estimator is then run only on the
standalone merged leaf checkpoints recorded in a synthetic leaf-only ledger.
Existing delta-space analyses are never overwritten.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from weighttraits.analysis.direct import analyze_training_ledger_direct
from weighttraits.training.ledger import TrainingLedgerEvent, append_ledger_event


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty run list: {path}")
    node_ids = [str(row["node_id"]) for row in rows]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError(f"duplicate node ids in {path}")
    return rows


def _leaf_chains(
    rows: list[dict[str, Any]],
    repo_root: Path,
    *,
    revision_override: str | None = None,
) -> tuple[dict[str, list[Path]], str, str]:
    by_node = {str(row["node_id"]): row for row in rows}
    parents = {str(row["node_id"]): str(row.get("parent_id") or "root") for row in rows}
    children = {node_id: [] for node_id in by_node}
    for node_id, parent_id in parents.items():
        if parent_id != "root":
            if parent_id not in by_node:
                raise ValueError(f"node {node_id} has missing parent {parent_id}")
            children[parent_id].append(node_id)
    leaves = sorted(node_id for node_id, child_ids in children.items() if not child_ids)

    base_models = {str(row["job"]["base_model"]) for row in rows}
    recorded_revisions = {
        str(row["job"].get("base_model_revision") or "") for row in rows
    }
    if len(base_models) != 1:
        raise ValueError("run list must name exactly one base model")
    if revision_override:
        nonempty = recorded_revisions - {""}
        if nonempty and nonempty != {revision_override}:
            raise ValueError(
                "base revision override conflicts with run-list revision: "
                f"override={revision_override!r}, recorded={sorted(nonempty)!r}"
            )
        revision = revision_override
    elif len(recorded_revisions) == 1 and "" not in recorded_revisions:
        revision = next(iter(recorded_revisions))
    else:
        raise ValueError(
            "run list must pin exactly one base revision or "
            "--base-revision-override must be provided"
        )

    chains: dict[str, list[Path]] = {}
    for leaf in leaves:
        lineage: list[str] = []
        cursor = leaf
        while cursor != "root":
            lineage.append(cursor)
            cursor = parents[cursor]
        lineage.reverse()
        adapters = [repo_root / str(by_node[node]["expected_artifacts"]["adapter"]) for node in lineage]
        for adapter in adapters:
            for filename in ("adapter_config.json", "adapter_model.safetensors"):
                if not (adapter / filename).is_file():
                    raise FileNotFoundError(adapter / filename)
        chains[leaf] = adapters
    return chains, next(iter(base_models)), revision


def _fingerprint(
    *, run_list: Path, leaf: str, adapters: list[Path], base_model: str, revision: str
) -> dict[str, Any]:
    return {
        "representation": "sequentially_merged_full_weight",
        "leaf": leaf,
        "base_model": base_model,
        "base_revision": revision,
        "run_list": str(run_list.resolve()),
        "run_list_sha256": _sha256(run_list),
        "adapter_chain": [str(path.resolve()) for path in adapters],
        "adapter_sha256": [_sha256(path / "adapter_model.safetensors") for path in adapters],
        "materializer": str(Path(__file__).resolve()),
        "materializer_sha256": _sha256(Path(__file__).resolve()),
    }


def _checkpoint_exists(path: Path) -> bool:
    return (path / "model.safetensors").is_file() or (
        path / "model.safetensors.index.json"
    ).is_file()


def _materialize_leaf(
    *,
    leaf: str,
    adapters: list[Path],
    base_model: str,
    revision: str,
    out_dir: Path,
    provenance: dict[str, Any],
    local_files_only: bool,
) -> None:
    provenance_path = out_dir / "merged_weight_provenance.json"
    if _checkpoint_exists(out_dir) and provenance_path.is_file():
        if json.loads(provenance_path.read_text()) == provenance:
            print(f"[skip] {leaf}: matching merged checkpoint exists", flush=True)
            return
        raise RuntimeError(f"refusing incompatible merged checkpoint: {out_dir}")
    if out_dir.exists():
        raise RuntimeError(f"partial merged checkpoint directory requires inspection: {out_dir}")

    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM

    print(f"[merge] {leaf}: {len(adapters)} adapters -> {out_dir}", flush=True)
    config = AutoConfig.from_pretrained(
        base_model,
        revision=revision,
        local_files_only=local_files_only,
    )
    model_class = AutoModelForSeq2SeqLM if config.is_encoder_decoder else AutoModelForCausalLM
    model = model_class.from_pretrained(
        base_model,
        revision=revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
    )
    for adapter in adapters:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
        model = model.merge_and_unload(safe_merge=True)

    tmp_dir = out_dir.with_name(f"{out_dir.name}.tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    model.save_pretrained(tmp_dir, safe_serialization=True, max_shard_size="5GB")
    (tmp_dir / "merged_weight_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    tmp_dir.replace(out_dir)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-list", type=Path, required=True)
    parser.add_argument("--truth-manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metric", action="append", default=[])
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument(
        "--base-revision-override",
        help=(
            "Explicit immutable base revision for historical run lists that did not record one; "
            "the override is included in every leaf's provenance"
        ),
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    run_list = args.run_list.resolve()
    truth_manifest = args.truth_manifest.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    out_dir = args.out.resolve()
    metrics = args.metric or ["cosine"]

    rows = _load_rows(run_list)
    chains, base_model, revision = _leaf_chains(
        rows,
        repo_root,
        revision_override=args.base_revision_override,
    )
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    leaf_paths: dict[str, Path] = {}
    for leaf, adapters in chains.items():
        leaf_dir = checkpoint_root / leaf
        provenance = _fingerprint(
            run_list=run_list,
            leaf=leaf,
            adapters=adapters,
            base_model=base_model,
            revision=revision,
        )
        _materialize_leaf(
            leaf=leaf,
            adapters=adapters,
            base_model=base_model,
            revision=revision,
            out_dir=leaf_dir,
            provenance=provenance,
            local_files_only=args.local_files_only,
        )
        leaf_paths[leaf] = leaf_dir

    ledger = checkpoint_root / "merged_leaf_ledger.jsonl"
    if ledger.exists():
        ledger.unlink()
    for leaf, checkpoint in sorted(leaf_paths.items()):
        append_ledger_event(
            ledger,
            TrainingLedgerEvent(
                node_id=leaf,
                status="completed",
                extra={"artifacts": {"merged": str(checkpoint)}},
            ),
        )

    summary = analyze_training_ledger_direct(
        ledger,
        truth_manifest=truth_manifest,
        out_dir=out_dir,
        artifact="merged",
        metrics=metrics,
        node_ids=sorted(leaf_paths),
        path_base=repo_root,
        chunk_size=args.chunk_size,
    )
    audit = {
        "status": "completed",
        "representation": "sequentially_merged_full_weight",
        "run_list": str(run_list),
        "run_list_sha256": _sha256(run_list),
        "truth_manifest": str(truth_manifest),
        "base_model": base_model,
        "base_revision": revision,
        "materializer": str(Path(__file__).resolve()),
        "materializer_sha256": _sha256(Path(__file__).resolve()),
        "n_leaves": len(leaf_paths),
        "leaf_checkpoints": {leaf: str(path) for leaf, path in leaf_paths.items()},
        "analysis_summary": str(out_dir / "summary.json"),
        "metrics": summary["metrics"],
    }
    (out_dir / "merged_weight_analysis_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
