"""Completion audits for planned training trees."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from weighttraits.training.ledger import (
    TERMINAL_STATUSES,
    TrainingLedgerEvent,
    latest_status_by_node,
    load_ledger_events,
)
from weighttraits.training.runlist import TrainingRunSpec, load_training_run_specs


OK_TERMINAL_STATUSES = {"completed", "skipped", "stopped_early"}


@dataclass(frozen=True)
class TreeCompletionIssue:
    severity: str
    issue: str
    message: str
    node_id: str | None = None
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArtifactCheck:
    name: str
    path: str
    exists: bool
    kind: str | None = None
    n_records: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NodeCompletionReport:
    node_id: str
    parent_id: str
    array_index: int
    dataset_id: str | None
    task_family: str | None
    method: str
    status: str | None
    terminal: bool
    step: int | None = None
    train_loss: float | None = None
    eval_loss: float | None = None
    warnings: tuple[str, ...] = ()
    stop_reasons: tuple[str, ...] = ()
    artifacts: tuple[ArtifactCheck, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["warnings"] = list(self.warnings)
        out["stop_reasons"] = list(self.stop_reasons)
        out["artifacts"] = [artifact.to_dict() for artifact in self.artifacts]
        return out


@dataclass(frozen=True)
class TreeCompletionReport:
    run_list: str
    ledger: str
    path_base: str
    valid: bool
    n_runs: int
    n_events: int
    n_ledger_nodes: int
    n_terminal_nodes: int
    n_ok_nodes: int
    n_failed_nodes: int
    n_missing_nodes: int
    n_expected_artifacts: int
    n_existing_artifacts: int
    status_counts: dict[str, int]
    issues: tuple[TreeCompletionIssue, ...] = ()
    nodes: tuple[NodeCompletionReport, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["issues"] = [issue.to_dict() for issue in self.issues]
        out["nodes"] = [node.to_dict() for node in self.nodes]
        return out


@dataclass(frozen=True)
class RunSetTreeCompletionRow:
    tree_id: str
    run_list: str
    ledger: str
    manifest: str | None
    output_root: str | None
    valid: bool
    ready_for_analysis: bool
    n_runs: int
    n_events: int
    n_ledger_nodes: int
    n_terminal_nodes: int
    n_ok_nodes: int
    n_failed_nodes: int
    n_missing_nodes: int
    n_expected_artifacts: int
    n_existing_artifacts: int
    n_errors: int
    n_warnings: int
    status_counts: dict[str, int]
    issue_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunSetCompletionReport:
    summary: str
    path_base: str
    valid: bool
    n_trees: int
    n_ready: int
    n_failed: int
    n_in_progress: int
    n_not_started: int
    n_total_runs: int
    n_terminal_nodes: int
    n_ok_nodes: int
    n_failed_nodes: int
    n_missing_nodes: int
    n_errors: int
    n_warnings: int
    ready_tree_ids: tuple[str, ...]
    ready_run_lists: tuple[str, ...]
    rows: tuple[RunSetTreeCompletionRow, ...]

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["ready_tree_ids"] = list(self.ready_tree_ids)
        out["ready_run_lists"] = list(self.ready_run_lists)
        out["rows"] = [row.to_dict() for row in self.rows]
        return out


def audit_training_tree_completion(
    run_list: str | Path,
    *,
    ledger: str | Path | None = None,
    path_base: str | Path = ".",
    ok_statuses: set[str] | None = None,
    optional_artifacts: set[str] | None = None,
    require_artifacts: bool = True,
    require_parent_order: bool = True,
) -> TreeCompletionReport:
    """Audit one planned training tree against its ledger and expected artifacts."""
    base = Path(path_base)
    run_list_path = _resolve(str(run_list), base)
    runs = load_training_run_specs(run_list_path)
    inferred_ledger = _infer_ledger_path(runs)
    ledger_path = Path(ledger) if ledger is not None else Path(inferred_ledger)
    if not ledger_path.is_absolute():
        ledger_path = base / ledger_path
    events = load_ledger_events(ledger_path)
    latest = latest_status_by_node(events)
    event_positions = _event_positions(events)
    acceptable = ok_statuses or OK_TERMINAL_STATUSES
    optional = optional_artifacts or set()

    issues: list[TreeCompletionIssue] = []
    node_reports: list[NodeCompletionReport] = []
    expected_artifact_count = 0
    existing_artifact_count = 0
    run_by_node = {run.node_id: run for run in runs}

    for run in runs:
        latest_event = latest.get(run.node_id)
        artifacts: list[ArtifactCheck] = []
        for name, raw_path in sorted(run.expected_artifacts.items()):
            expected_artifact_count += 1
            check = _check_artifact(name, raw_path, base)
            artifacts.append(check)
            if check.exists:
                existing_artifact_count += 1
            elif name in optional:
                issues.append(
                    TreeCompletionIssue(
                        severity="warning",
                        issue="missing_optional_artifact",
                        node_id=run.node_id,
                        path=check.path,
                        message=f"optional artifact is missing: {name}",
                    )
                )
            elif require_artifacts:
                issues.append(
                    TreeCompletionIssue(
                        severity="error",
                        issue="missing_artifact",
                        node_id=run.node_id,
                        path=check.path,
                        message=f"expected artifact is missing: {name}",
                    )
                )
        if latest_event is None:
            issues.append(
                TreeCompletionIssue(
                    severity="error",
                    issue="missing_ledger_node",
                    node_id=run.node_id,
                    message="planned node has no ledger events",
                )
            )
        else:
            _check_status(run, latest_event, acceptable, issues)
            _check_ledger_artifacts(run, latest_event, issues)
            if require_parent_order:
                _check_parent_order(run, run_by_node, latest, event_positions, issues)

        node_reports.append(_node_report(run, latest_event, artifacts))

    status_counts = _status_counts(latest.values())
    valid = not any(issue.severity == "error" for issue in issues)
    return TreeCompletionReport(
        run_list=str(run_list_path),
        ledger=str(ledger_path),
        path_base=str(path_base),
        valid=valid,
        n_runs=len(runs),
        n_events=len(events),
        n_ledger_nodes=len(latest),
        n_terminal_nodes=sum(1 for event in latest.values() if event.status in TERMINAL_STATUSES),
        n_ok_nodes=sum(1 for run in runs if (latest.get(run.node_id) or _empty_event()).status in acceptable),
        n_failed_nodes=sum(1 for run in runs if (latest.get(run.node_id) or _empty_event()).status == "failed"),
        n_missing_nodes=sum(1 for run in runs if run.node_id not in latest),
        n_expected_artifacts=expected_artifact_count,
        n_existing_artifacts=existing_artifact_count,
        status_counts=status_counts,
        issues=tuple(issues),
        nodes=tuple(node_reports),
    )


def audit_training_run_set_completion(
    summary: str | Path,
    *,
    path_base: str | Path = ".",
    optional_artifacts: set[str] | None = None,
    require_artifacts: bool = True,
    require_parent_order: bool = True,
    only_ready: bool = False,
) -> RunSetCompletionReport:
    """Audit every tree declared in a generated training run-list-set summary."""
    summary_path = Path(summary)
    raw = json.loads(summary_path.read_text())
    trees = raw.get("trees")
    if not isinstance(trees, list):
        raise ValueError(f"run-list-set summary has no trees list: {summary}")
    rows: list[RunSetTreeCompletionRow] = []
    for index, tree in enumerate(trees):
        if not isinstance(tree, Mapping):
            raise ValueError(f"tree entry {index} is not a mapping: {summary}")
        run_list = tree.get("run_list")
        if not run_list:
            raise ValueError(f"tree entry {index} has no run_list: {summary}")
        tree_id = str(tree.get("tree_id") or Path(str(run_list)).stem.removesuffix(".runs"))
        tree_report = audit_training_tree_completion(
            str(run_list),
            ledger=_optional_tree_path(tree.get("ledger")),
            path_base=path_base,
            optional_artifacts=optional_artifacts,
            require_artifacts=require_artifacts,
            require_parent_order=require_parent_order,
        )
        row = _run_set_row(
            tree_id=tree_id,
            tree=tree,
            tree_report=tree_report,
        )
        if not only_ready or row.ready_for_analysis:
            rows.append(row)

    return _run_set_report(
        summary=summary_path,
        path_base=path_base,
        rows=rows,
    )


def write_run_set_completion_csv(report: RunSetCompletionReport, path: str | Path) -> None:
    """Write a compact per-tree CSV view of a run-set completion report."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "tree_id",
        "valid",
        "ready_for_analysis",
        "n_runs",
        "n_events",
        "n_ledger_nodes",
        "n_terminal_nodes",
        "n_ok_nodes",
        "n_failed_nodes",
        "n_missing_nodes",
        "n_expected_artifacts",
        "n_existing_artifacts",
        "n_errors",
        "n_warnings",
        "status_counts",
        "issue_counts",
        "run_list",
        "ledger",
        "manifest",
        "output_root",
    ]
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report.rows:
            item = row.to_dict()
            item["status_counts"] = json.dumps(item["status_counts"], sort_keys=True)
            item["issue_counts"] = json.dumps(item["issue_counts"], sort_keys=True)
            writer.writerow({field: item[field] for field in fields})


def _infer_ledger_path(runs: Sequence[TrainingRunSpec]) -> str:
    if not runs:
        raise ValueError("cannot infer ledger path from an empty run list")
    ledgers = {run.ledger_path for run in runs}
    if len(ledgers) != 1:
        raise ValueError("run list contains multiple ledger paths; pass --ledger explicitly")
    return next(iter(ledgers))


def _run_set_row(
    *,
    tree_id: str,
    tree: Mapping[str, Any],
    tree_report: TreeCompletionReport,
) -> RunSetTreeCompletionRow:
    issue_counts = _issue_counts(tree_report.issues)
    n_errors = sum(1 for issue in tree_report.issues if issue.severity == "error")
    n_warnings = sum(1 for issue in tree_report.issues if issue.severity == "warning")
    return RunSetTreeCompletionRow(
        tree_id=tree_id,
        run_list=tree_report.run_list,
        ledger=tree_report.ledger,
        manifest=_optional_tree_path(tree.get("manifest")),
        output_root=_optional_tree_path(tree.get("output_root")),
        valid=tree_report.valid,
        ready_for_analysis=tree_report.valid,
        n_runs=tree_report.n_runs,
        n_events=tree_report.n_events,
        n_ledger_nodes=tree_report.n_ledger_nodes,
        n_terminal_nodes=tree_report.n_terminal_nodes,
        n_ok_nodes=tree_report.n_ok_nodes,
        n_failed_nodes=tree_report.n_failed_nodes,
        n_missing_nodes=tree_report.n_missing_nodes,
        n_expected_artifacts=tree_report.n_expected_artifacts,
        n_existing_artifacts=tree_report.n_existing_artifacts,
        n_errors=n_errors,
        n_warnings=n_warnings,
        status_counts=tree_report.status_counts,
        issue_counts=issue_counts,
    )


def _run_set_report(
    *,
    summary: Path,
    path_base: str | Path,
    rows: Sequence[RunSetTreeCompletionRow],
) -> RunSetCompletionReport:
    n_errors = sum(row.n_errors for row in rows)
    n_warnings = sum(row.n_warnings for row in rows)
    ready_rows = [row for row in rows if row.ready_for_analysis]
    return RunSetCompletionReport(
        summary=str(summary),
        path_base=str(path_base),
        valid=n_errors == 0,
        n_trees=len(rows),
        n_ready=len(ready_rows),
        n_failed=sum(1 for row in rows if row.n_failed_nodes),
        n_in_progress=sum(
            1
            for row in rows
            if row.n_terminal_nodes < row.n_runs and row.n_ledger_nodes > 0 and row.n_failed_nodes == 0
        ),
        n_not_started=sum(1 for row in rows if row.n_ledger_nodes == 0),
        n_total_runs=sum(row.n_runs for row in rows),
        n_terminal_nodes=sum(row.n_terminal_nodes for row in rows),
        n_ok_nodes=sum(row.n_ok_nodes for row in rows),
        n_failed_nodes=sum(row.n_failed_nodes for row in rows),
        n_missing_nodes=sum(row.n_missing_nodes for row in rows),
        n_errors=n_errors,
        n_warnings=n_warnings,
        ready_tree_ids=tuple(row.tree_id for row in ready_rows),
        ready_run_lists=tuple(row.run_list for row in ready_rows),
        rows=tuple(rows),
    )


def _issue_counts(issues: Sequence[TreeCompletionIssue]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for issue in issues:
        counts[issue.issue] = counts.get(issue.issue, 0) + 1
    return dict(sorted(counts.items()))


def _optional_tree_path(value: object) -> str | None:
    return None if value is None else str(value)


def _check_artifact(name: str, raw_path: str, base: Path) -> ArtifactCheck:
    path = _resolve(raw_path, base)
    exists = path.exists()
    n_records = _jsonl_record_count(path) if exists and path.is_file() and path.suffix == ".jsonl" else None
    kind = "dir" if exists and path.is_dir() else "file" if exists and path.is_file() else None
    return ArtifactCheck(
        name=name,
        path=str(path),
        exists=exists,
        kind=kind,
        n_records=n_records,
    )


def _check_status(
    run: TrainingRunSpec,
    latest_event: TrainingLedgerEvent,
    acceptable: set[str],
    issues: list[TreeCompletionIssue],
) -> None:
    if latest_event.status == "failed":
        issues.append(
            TreeCompletionIssue(
                severity="error",
                issue="failed_node",
                node_id=run.node_id,
                message=latest_event.message or "latest ledger status is failed",
            )
        )
    elif latest_event.status not in acceptable:
        issues.append(
            TreeCompletionIssue(
                severity="error",
                issue="non_terminal_node",
                node_id=run.node_id,
                message=f"latest ledger status is {latest_event.status}",
            )
        )


def _check_ledger_artifacts(
    run: TrainingRunSpec,
    latest_event: TrainingLedgerEvent,
    issues: list[TreeCompletionIssue],
) -> None:
    ledger_artifacts = latest_event.extra.get("artifacts")
    if not isinstance(ledger_artifacts, Mapping):
        return
    for name, expected_path in sorted(run.expected_artifacts.items()):
        if name == "training_log" or name not in ledger_artifacts:
            continue
        observed_path = str(ledger_artifacts[name])
        if observed_path != expected_path:
            issues.append(
                TreeCompletionIssue(
                    severity="warning",
                    issue="ledger_artifact_path_mismatch",
                    node_id=run.node_id,
                    path=observed_path,
                    message=f"ledger artifact {name} does not match run-list expected path",
                )
            )


def _check_parent_order(
    run: TrainingRunSpec,
    run_by_node: Mapping[str, TrainingRunSpec],
    latest: Mapping[str, TrainingLedgerEvent],
    event_positions: Mapping[tuple[str, str], int],
    issues: list[TreeCompletionIssue],
) -> None:
    if run.parent_id in {"", "root"} or run.parent_id not in run_by_node:
        return
    parent_event = latest.get(run.parent_id)
    if parent_event is None or parent_event.status not in OK_TERMINAL_STATUSES:
        issues.append(
            TreeCompletionIssue(
                severity="error",
                issue="parent_not_complete",
                node_id=run.node_id,
                message=f"parent {run.parent_id} is not complete before child audit",
            )
        )
        return
    parent_terminal_pos = event_positions.get((run.parent_id, parent_event.status))
    child_started_pos = event_positions.get((run.node_id, "started"))
    if (
        parent_terminal_pos is not None
        and child_started_pos is not None
        and parent_terminal_pos > child_started_pos
    ):
        issues.append(
            TreeCompletionIssue(
                severity="error",
                issue="child_started_before_parent_finished",
                node_id=run.node_id,
                message=f"child started before parent {run.parent_id} reached terminal status",
            )
        )


def _node_report(
    run: TrainingRunSpec,
    latest_event: TrainingLedgerEvent | None,
    artifacts: Sequence[ArtifactCheck],
) -> NodeCompletionReport:
    return NodeCompletionReport(
        node_id=run.node_id,
        parent_id=run.parent_id,
        array_index=run.array_index,
        dataset_id=run.dataset_id,
        task_family=run.task_family,
        method=run.method,
        status=None if latest_event is None else latest_event.status,
        terminal=False if latest_event is None else latest_event.status in TERMINAL_STATUSES,
        step=None if latest_event is None else latest_event.step,
        train_loss=None if latest_event is None else latest_event.train_loss,
        eval_loss=None if latest_event is None else latest_event.eval_loss,
        warnings=() if latest_event is None else latest_event.warnings,
        stop_reasons=() if latest_event is None else latest_event.stop_reasons,
        artifacts=tuple(artifacts),
    )


def _status_counts(events: Sequence[TrainingLedgerEvent]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        counts[event.status] = counts.get(event.status, 0) + 1
    return dict(sorted(counts.items()))


def _event_positions(events: Sequence[TrainingLedgerEvent]) -> dict[tuple[str, str], int]:
    positions: dict[tuple[str, str], int] = {}
    for index, event in enumerate(events):
        positions[(event.node_id, event.status)] = index
    return positions


def _jsonl_record_count(path: Path) -> int:
    count = 0
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            json.loads(line)
            count += 1
    return count


def _resolve(path: str, base: Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return base / value


def _empty_event() -> TrainingLedgerEvent:
    return TrainingLedgerEvent(node_id="", status="")
