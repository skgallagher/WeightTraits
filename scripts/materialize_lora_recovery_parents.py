#!/usr/bin/env python3
"""Rebuild pruned LoRA parent models needed by a scoped tree retry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _local_path(value: str, root: Path) -> Path | None:
    path = Path(value)
    if path.is_absolute():
        return path
    if value.startswith("outputs/"):
        return root / path
    return None


def _materialize(row: dict[str, object], root: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

    artifacts = row["expected_artifacts"]
    assert isinstance(artifacts, dict)
    adapter = root / str(artifacts["adapter"])
    merged = root / str(artifacts["merged"])
    init_from = str(row["init_from"])
    local_init = _local_path(init_from, root)
    model_source = str(local_init if local_init is not None else init_from)
    job = row["job"]
    assert isinstance(job, dict)
    trainer = job.get("trainer") or {}
    assert isinstance(trainer, dict)
    model_cls = (
        AutoModelForSeq2SeqLM
        if trainer.get("model_task") == "seq2seq"
        else AutoModelForCausalLM
    )
    kwargs: dict[str, object] = {
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "device_map": "auto",
    }
    if local_init is None and job.get("base_model_revision"):
        kwargs["revision"] = job["base_model_revision"]
    print(f"materializing {row['node_id']}: {adapter} -> {merged}", flush=True)
    base = model_cls.from_pretrained(model_source, **kwargs)
    model = PeftModel.from_pretrained(base, str(adapter))
    merged_model = model.merge_and_unload()
    merged.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(merged)
    AutoTokenizer.from_pretrained(str(adapter)).save_pretrained(merged)
    del merged_model, model, base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def materialize_recovery_parents(run_list: Path, start_index: int, root: Path) -> list[str]:
    rows = _load_rows(run_list)
    by_node = {str(row["node_id"]): row for row in rows}
    index_by_node = {str(row["node_id"]): int(row["array_index"]) for row in rows}
    needed: set[str] = set()

    def require_parent(row: dict[str, object]) -> None:
        parent_id = str(row["parent_id"])
        if parent_id == "root" or index_by_node[parent_id] >= start_index:
            return
        parent = by_node[parent_id]
        artifacts = parent["expected_artifacts"]
        assert isinstance(artifacts, dict)
        merged = root / str(artifacts["merged"])
        if merged.is_dir():
            return
        adapter = root / str(artifacts["adapter"])
        if not adapter.is_dir():
            raise FileNotFoundError(f"missing retained adapter for {parent_id}: {adapter}")
        require_parent(parent)
        needed.add(parent_id)

    for row in rows[start_index:]:
        require_parent(row)
    ordered = sorted(needed, key=index_by_node.__getitem__)
    for node_id in ordered:
        _materialize(by_node[node_id], root)
    return ordered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-list", type=Path, required=True)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    nodes = materialize_recovery_parents(
        args.run_list, args.start_index, args.repo_root.resolve()
    )
    print(json.dumps({"materialized_node_ids": nodes}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
