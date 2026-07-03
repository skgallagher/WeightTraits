"""JSONL training ledger helpers for resumable node jobs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


TERMINAL_STATUSES = {"completed", "failed", "skipped", "stopped_early"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class TrainingLedgerEvent:
    node_id: str
    status: str
    step: int | None = None
    train_loss: float | None = None
    eval_loss: float | None = None
    warnings: tuple[str, ...] = ()
    stop_reasons: tuple[str, ...] = ()
    message: str | None = None
    timestamp: str = field(default_factory=_utc_now)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["warnings"] = list(self.warnings)
        out["stop_reasons"] = list(self.stop_reasons)
        return out


def append_ledger_event(path: str | Path, event: TrainingLedgerEvent) -> None:
    ledger = Path(path)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a") as handle:
        handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")


def load_ledger_events(path: str | Path) -> list[TrainingLedgerEvent]:
    ledger = Path(path)
    if not ledger.exists():
        return []
    events = []
    with ledger.open() as handle:
        for line in handle:
            if line.strip():
                events.append(_event_from_dict(json.loads(line)))
    return events


def latest_status_by_node(events: list[TrainingLedgerEvent]) -> dict[str, TrainingLedgerEvent]:
    latest: dict[str, TrainingLedgerEvent] = {}
    for event in events:
        latest[event.node_id] = event
    return latest


def completed_nodes(events: list[TrainingLedgerEvent]) -> set[str]:
    latest = latest_status_by_node(events)
    return {
        node_id
        for node_id, event in latest.items()
        if event.status in {"completed", "skipped", "stopped_early"}
    }


def failed_nodes(events: list[TrainingLedgerEvent]) -> set[str]:
    latest = latest_status_by_node(events)
    return {node_id for node_id, event in latest.items() if event.status == "failed"}


def should_skip_node(
    node_id: str,
    events: list[TrainingLedgerEvent],
    *,
    rerun_stopped_early: bool = False,
) -> bool:
    latest = latest_status_by_node(events).get(node_id)
    if latest is None:
        return False
    if latest.status in {"completed", "skipped"}:
        return True
    if latest.status == "stopped_early":
        return not rerun_stopped_early
    return False


def ledger_summary(events: list[TrainingLedgerEvent]) -> dict[str, Any]:
    latest = latest_status_by_node(events)
    counts: dict[str, int] = {}
    for event in latest.values():
        counts[event.status] = counts.get(event.status, 0) + 1
    return {
        "n_events": len(events),
        "n_nodes": len(latest),
        "status_counts": dict(sorted(counts.items())),
        "completed_nodes": sorted(completed_nodes(events)),
        "failed_nodes": sorted(failed_nodes(events)),
    }


def _event_from_dict(row: dict[str, Any]) -> TrainingLedgerEvent:
    return TrainingLedgerEvent(
        node_id=str(row["node_id"]),
        status=str(row["status"]),
        step=row.get("step"),
        train_loss=row.get("train_loss"),
        eval_loss=row.get("eval_loss"),
        warnings=tuple(row.get("warnings", [])),
        stop_reasons=tuple(row.get("stop_reasons", [])),
        message=row.get("message"),
        timestamp=str(row.get("timestamp", _utc_now())),
        extra=dict(row.get("extra", {})),
    )
