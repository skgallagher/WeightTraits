"""Lineage-aware retention for large training artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Sequence

from weighttraits.training.ledger import TrainingLedgerEvent, latest_status_by_node
from weighttraits.training.runlist import TrainingRunSpec


SUCCESS_STATUSES = {"completed", "skipped", "stopped_early"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ParentArtifactPruneResult:
    selected_node_id: str
    parent_node_id: str | None
    action: str
    reason: str
    method: str | None = None
    artifact_kind: str | None = None
    artifact_path: str | None = None
    direct_child_ids: tuple[str, ...] = ()
    incomplete_child_ids: tuple[str, ...] = ()
    bytes_removed: int = 0
    dry_run: bool = False
    timestamp: str = field(default_factory=_utc_now)
    schema_version: int = 1

    def to_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["direct_child_ids"] = list(self.direct_child_ids)
        out["incomplete_child_ids"] = list(self.incomplete_child_ids)
        return out


@dataclass(frozen=True)
class LoraTreeMaterializationPruneResult:
    action: str
    reason: str
    n_runs: int
    incomplete_node_ids: tuple[str, ...] = ()
    missing_adapter_node_ids: tuple[str, ...] = ()
    retained_adapter_paths: tuple[str, ...] = ()
    pruned_merged_paths: tuple[str, ...] = ()
    already_absent_merged_paths: tuple[str, ...] = ()
    bytes_removed: int = 0
    dry_run: bool = False
    timestamp: str = field(default_factory=_utc_now)
    schema_version: int = 1

    def to_dict(self) -> dict[str, object]:
        out = asdict(self)
        for key in (
            "incomplete_node_ids",
            "missing_adapter_node_ids",
            "retained_adapter_paths",
            "pruned_merged_paths",
            "already_absent_merged_paths",
        ):
            out[key] = list(out[key])
        return out


def prune_completed_parent_artifact(
    runs: Sequence[TrainingRunSpec],
    *,
    selected_node_id: str,
    ledger_events: Sequence[TrainingLedgerEvent],
    success_not_before: str | None = None,
    dry_run: bool = False,
) -> ParentArtifactPruneResult:
    """Prune an internal parent's lineage artifact after all direct children succeed."""

    by_node = {run.node_id: run for run in runs}
    if len(by_node) != len(runs):
        raise ValueError("training run list contains duplicate node IDs")
    if selected_node_id not in by_node:
        raise ValueError(f"selected node is absent from training run list: {selected_node_id}")
    selected = by_node[selected_node_id]
    if selected.parent_id == "root":
        return ParentArtifactPruneResult(
            selected_node_id=selected_node_id,
            parent_node_id=None,
            action="not_applicable",
            reason="selected node has no trained parent",
            dry_run=dry_run,
        )

    parent = by_node.get(selected.parent_id)
    if parent is None:
        raise ValueError(
            f"selected node {selected_node_id} references missing parent {selected.parent_id}"
        )
    children = tuple(run.node_id for run in runs if run.parent_id == parent.node_id)
    latest = latest_status_by_node(list(ledger_events))
    incomplete = tuple(
        child_id
        for child_id in children
        if child_id not in latest
        or latest[child_id].status not in SUCCESS_STATUSES
        or (
            success_not_before is not None
            and latest[child_id].timestamp < success_not_before
        )
    )
    artifact_kind = "model" if parent.method == "full" else "merged"
    artifact_path_text = parent.expected_artifacts.get(artifact_kind)
    if not artifact_path_text:
        raise ValueError(
            f"parent {parent.node_id} has no expected {artifact_kind} artifact to prune"
        )
    common = {
        "selected_node_id": selected_node_id,
        "parent_node_id": parent.node_id,
        "method": parent.method,
        "artifact_kind": artifact_kind,
        "artifact_path": artifact_path_text,
        "direct_child_ids": children,
        "incomplete_child_ids": incomplete,
        "dry_run": dry_run,
    }
    if incomplete:
        return ParentArtifactPruneResult(
            action="not_ready",
            reason=(
                "one or more direct children lack a successful terminal ledger status "
                "in the current attempt"
                if success_not_before is not None
                else "one or more direct children lack a successful terminal ledger status"
            ),
            **common,
        )

    artifact_path = Path(artifact_path_text)
    output_dir = Path(parent.output_dir)
    _validate_prunable_path(artifact_path, output_dir, artifact_kind)
    if not artifact_path.exists():
        return ParentArtifactPruneResult(
            action="already_absent",
            reason="all direct children succeeded and the parent artifact is already absent",
            **common,
        )
    bytes_removed = _path_size_bytes(artifact_path)
    if dry_run:
        return ParentArtifactPruneResult(
            action="would_prune",
            reason="all direct children succeeded",
            bytes_removed=bytes_removed,
            **common,
        )
    shutil.rmtree(artifact_path)
    return ParentArtifactPruneResult(
        action="pruned",
        reason="all direct children succeeded",
        bytes_removed=bytes_removed,
        **common,
    )


def prune_completed_lora_tree_materializations(
    runs: Sequence[TrainingRunSpec],
    *,
    ledger_events: Sequence[TrainingLedgerEvent],
    dry_run: bool = False,
) -> LoraTreeMaterializationPruneResult:
    """Remove every merged LoRA model after a whole tree is safely reconstructable."""

    if not runs:
        raise ValueError("training run list is empty")
    by_node = {run.node_id: run for run in runs}
    if len(by_node) != len(runs):
        raise ValueError("training run list contains duplicate node IDs")
    non_lora = tuple(run.node_id for run in runs if run.method != "lora")
    if non_lora:
        raise ValueError(
            "LoRA materialization pruning requires every run to use method=lora; "
            f"non-LoRA nodes: {', '.join(non_lora)}"
        )

    latest = latest_status_by_node(list(ledger_events))
    incomplete = tuple(
        run.node_id
        for run in runs
        if run.node_id not in latest or latest[run.node_id].status not in SUCCESS_STATUSES
    )

    adapter_paths: list[Path] = []
    merged_paths: list[Path] = []
    missing_adapters: list[str] = []
    for run in runs:
        adapter_path_text = run.expected_artifacts.get("adapter")
        merged_path_text = run.expected_artifacts.get("merged")
        if not adapter_path_text or not merged_path_text:
            raise ValueError(
                f"LoRA run {run.node_id} must declare adapter and merged artifacts"
            )
        adapter_path = Path(adapter_path_text)
        merged_path = Path(merged_path_text)
        _validate_artifact_path(adapter_path, Path(run.output_dir), "adapter")
        _validate_prunable_path(merged_path, Path(run.output_dir), "merged")
        adapter_paths.append(adapter_path)
        merged_paths.append(merged_path)
        if not adapter_path.is_dir():
            missing_adapters.append(run.node_id)

    common = {
        "n_runs": len(runs),
        "incomplete_node_ids": incomplete,
        "missing_adapter_node_ids": tuple(missing_adapters),
        "retained_adapter_paths": tuple(str(path) for path in adapter_paths if path.is_dir()),
        "dry_run": dry_run,
    }
    if incomplete:
        return LoraTreeMaterializationPruneResult(
            action="not_ready",
            reason="one or more tree nodes lack a successful terminal ledger status",
            **common,
        )
    if missing_adapters:
        return LoraTreeMaterializationPruneResult(
            action="not_ready",
            reason="one or more completed tree nodes lack a retained adapter",
            **common,
        )

    present = tuple(path for path in merged_paths if path.exists())
    absent = tuple(path for path in merged_paths if not path.exists())
    bytes_removed = sum(_path_size_bytes(path) for path in present)
    if not present:
        return LoraTreeMaterializationPruneResult(
            action="already_absent",
            reason="all tree nodes succeeded, all adapters are retained, and no merged models remain",
            already_absent_merged_paths=tuple(str(path) for path in absent),
            **common,
        )
    if dry_run:
        return LoraTreeMaterializationPruneResult(
            action="would_prune",
            reason="all tree nodes succeeded and all adapters are retained",
            pruned_merged_paths=tuple(str(path) for path in present),
            already_absent_merged_paths=tuple(str(path) for path in absent),
            bytes_removed=bytes_removed,
            **common,
        )
    for path in present:
        shutil.rmtree(path)
    return LoraTreeMaterializationPruneResult(
        action="pruned",
        reason="all tree nodes succeeded and all adapters are retained",
        pruned_merged_paths=tuple(str(path) for path in present),
        already_absent_merged_paths=tuple(str(path) for path in absent),
        bytes_removed=bytes_removed,
        **common,
    )


def append_retention_audit(
    path: str | Path,
    result: ParentArtifactPruneResult | LoraTreeMaterializationPruneResult,
) -> None:
    audit_path = Path(path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a") as handle:
        handle.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")


def default_retention_audit_path(ledger_path: str | Path) -> Path:
    ledger = Path(ledger_path)
    return ledger.parent / f"{ledger.stem}.retention_audit.jsonl"


def _validate_prunable_path(artifact_path: Path, output_dir: Path, artifact_kind: str) -> None:
    _validate_artifact_path(artifact_path, output_dir, artifact_kind)
    if artifact_path.is_symlink():
        raise ValueError(f"refusing to prune symlinked parent artifact: {artifact_path}")


def _validate_artifact_path(artifact_path: Path, output_dir: Path, artifact_kind: str) -> None:
    if artifact_path.name != artifact_kind:
        raise ValueError(
            f"unexpected {artifact_kind} path name: {artifact_path}"
        )
    resolved_artifact = artifact_path.resolve()
    resolved_output = output_dir.resolve()
    if not resolved_artifact.is_relative_to(resolved_output):
        raise ValueError(
            f"artifact is outside its output directory: {artifact_path}"
        )


def _path_size_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
