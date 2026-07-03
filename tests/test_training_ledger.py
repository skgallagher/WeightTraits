from pathlib import Path

from weighttraits.cli import build_parser
from weighttraits.training.ledger import (
    TrainingLedgerEvent,
    append_ledger_event,
    completed_nodes,
    failed_nodes,
    ledger_summary,
    load_ledger_events,
    should_skip_node,
)


def test_training_ledger_roundtrip_and_summary(tmp_path):
    ledger = tmp_path / "training.jsonl"
    append_ledger_event(ledger, TrainingLedgerEvent(node_id="n0", status="started", step=0))
    append_ledger_event(
        ledger,
        TrainingLedgerEvent(
            node_id="n0",
            status="stopped_early",
            step=12,
            eval_loss=0.42,
            warnings=("loss rose",),
            stop_reasons=("plateau",),
        ),
    )
    append_ledger_event(ledger, TrainingLedgerEvent(node_id="n1", status="failed"))

    events = load_ledger_events(ledger)
    summary = ledger_summary(events)

    assert len(events) == 3
    assert completed_nodes(events) == {"n0"}
    assert failed_nodes(events) == {"n1"}
    assert summary["status_counts"] == {"failed": 1, "stopped_early": 1}


def test_should_skip_node_respects_terminal_statuses():
    events = [
        TrainingLedgerEvent(node_id="n0", status="completed"),
        TrainingLedgerEvent(node_id="n1", status="stopped_early"),
        TrainingLedgerEvent(node_id="n2", status="failed"),
    ]

    assert should_skip_node("n0", events)
    assert should_skip_node("n1", events)
    assert not should_skip_node("n1", events, rerun_stopped_early=True)
    assert not should_skip_node("n2", events)
    assert not should_skip_node("missing", events)


def test_training_ledger_summary_parser():
    args = build_parser().parse_args(
        [
            "training-ledger-summary",
            "--ledger",
            "/tmp/training.jsonl",
        ]
    )

    assert args.ledger == Path("/tmp/training.jsonl")
