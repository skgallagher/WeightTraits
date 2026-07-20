import json
from pathlib import Path

from weighttraits.cli import build_parser
from weighttraits.training.ledger import TrainingLedgerEvent
from weighttraits.training.retention import (
    append_retention_audit,
    default_retention_audit_path,
    prune_completed_parent_artifact,
)
from weighttraits.training.runlist import TrainingRunSpec


def _run(
    root: Path,
    *,
    index: int,
    node_id: str,
    parent_id: str,
    method: str = "full",
) -> TrainingRunSpec:
    output_dir = root / node_id
    expected_artifacts = {
        "training_log": str(output_dir / "training_log.jsonl"),
    }
    if method == "full":
        expected_artifacts["model"] = str(output_dir / "model")
    else:
        expected_artifacts["adapter"] = str(output_dir / "adapter")
        expected_artifacts["merged"] = str(output_dir / "merged")
    return TrainingRunSpec(
        array_index=index,
        run_id=f"run-{node_id}",
        node_id=node_id,
        parent_id=parent_id,
        depth=0 if parent_id == "root" else 1,
        method=method,
        dataset_id="rte",
        task_family="classification",
        init_from="base" if parent_id == "root" else str(root / parent_id / "model"),
        output_dir=str(output_dir),
        expected_artifacts=expected_artifacts,
        ledger_path=str(root / "training_ledger.jsonl"),
        runner={},
        job={},
    )


def _branching_runs(root: Path, method: str = "full") -> list[TrainingRunSpec]:
    return [
        _run(root, index=0, node_id="n0", parent_id="root", method=method),
        _run(root, index=1, node_id="n1", parent_id="n0", method=method),
        _run(root, index=2, node_id="n2", parent_id="n0", method=method),
    ]


def test_parent_artifact_waits_for_every_direct_child(tmp_path: Path) -> None:
    runs = _branching_runs(tmp_path)
    parent_model = Path(runs[0].expected_artifacts["model"])
    parent_model.mkdir(parents=True)
    (parent_model / "model.safetensors").write_bytes(b"weights")

    result = prune_completed_parent_artifact(
        runs,
        selected_node_id="n1",
        ledger_events=[TrainingLedgerEvent(node_id="n1", status="completed")],
    )

    assert result.action == "not_ready"
    assert result.incomplete_child_ids == ("n2",)
    assert parent_model.exists()


def test_parent_model_is_pruned_after_all_children_succeed(tmp_path: Path) -> None:
    runs = _branching_runs(tmp_path)
    parent_model = Path(runs[0].expected_artifacts["model"])
    parent_model.mkdir(parents=True)
    (parent_model / "model.safetensors").write_bytes(b"weights")
    child_model = Path(runs[2].expected_artifacts["model"])
    child_model.mkdir(parents=True)
    events = [
        TrainingLedgerEvent(node_id="n1", status="completed"),
        TrainingLedgerEvent(node_id="n2", status="stopped_early"),
    ]

    result = prune_completed_parent_artifact(
        runs,
        selected_node_id="n2",
        ledger_events=events,
    )

    assert result.action == "pruned"
    assert result.artifact_kind == "model"
    assert result.bytes_removed == len(b"weights")
    assert not parent_model.exists()
    assert child_model.exists()

    repeated = prune_completed_parent_artifact(
        runs,
        selected_node_id="n2",
        ledger_events=events,
    )
    assert repeated.action == "already_absent"


def test_parent_artifact_ignores_stale_successes_from_an_earlier_attempt(
    tmp_path: Path,
) -> None:
    runs = _branching_runs(tmp_path)
    parent_model = Path(runs[0].expected_artifacts["model"])
    parent_model.mkdir(parents=True)
    events = [
        TrainingLedgerEvent(
            node_id="n1", status="completed", timestamp="2026-07-16T10:00:00+00:00"
        ),
        TrainingLedgerEvent(
            node_id="n2", status="completed", timestamp="2026-07-20T10:01:00+00:00"
        ),
    ]

    result = prune_completed_parent_artifact(
        runs,
        selected_node_id="n2",
        ledger_events=events,
        success_not_before="2026-07-20T10:00:00+00:00",
    )

    assert result.action == "not_ready"
    assert result.incomplete_child_ids == ("n1",)
    assert "current attempt" in result.reason
    assert parent_model.exists()


def test_lora_pruning_retains_parent_adapter(tmp_path: Path) -> None:
    runs = _branching_runs(tmp_path, method="lora")
    parent_adapter = Path(runs[0].expected_artifacts["adapter"])
    parent_merged = Path(runs[0].expected_artifacts["merged"])
    parent_adapter.mkdir(parents=True)
    parent_merged.mkdir(parents=True)
    (parent_adapter / "adapter.safetensors").write_bytes(b"adapter")
    (parent_merged / "model.safetensors").write_bytes(b"merged")
    events = [
        TrainingLedgerEvent(node_id="n1", status="completed"),
        TrainingLedgerEvent(node_id="n2", status="completed"),
    ]

    result = prune_completed_parent_artifact(
        runs,
        selected_node_id="n2",
        ledger_events=events,
    )

    assert result.action == "pruned"
    assert result.artifact_kind == "merged"
    assert parent_adapter.exists()
    assert not parent_merged.exists()


def test_retention_dry_run_and_audit_are_non_destructive(tmp_path: Path) -> None:
    runs = _branching_runs(tmp_path)
    parent_model = Path(runs[0].expected_artifacts["model"])
    parent_model.mkdir(parents=True)
    (parent_model / "model.safetensors").write_bytes(b"weights")
    events = [
        TrainingLedgerEvent(node_id="n1", status="completed"),
        TrainingLedgerEvent(node_id="n2", status="completed"),
    ]

    result = prune_completed_parent_artifact(
        runs,
        selected_node_id="n2",
        ledger_events=events,
        dry_run=True,
    )
    audit_path = default_retention_audit_path(runs[0].ledger_path)
    append_retention_audit(audit_path, result)

    assert result.action == "would_prune"
    assert parent_model.exists()
    assert json.loads(audit_path.read_text())["action"] == "would_prune"


def test_retention_refuses_artifact_outside_parent_output(tmp_path: Path) -> None:
    runs = _branching_runs(tmp_path)
    outside = tmp_path / "outside" / "model"
    outside.mkdir(parents=True)
    parent = runs[0]
    parent.expected_artifacts["model"] = str(outside)
    events = [
        TrainingLedgerEvent(node_id="n1", status="completed"),
        TrainingLedgerEvent(node_id="n2", status="completed"),
    ]

    try:
        prune_completed_parent_artifact(
            runs,
            selected_node_id="n2",
            ledger_events=events,
        )
    except ValueError as exc:
        assert "outside parent output directory" in str(exc)
    else:
        raise AssertionError("expected unsafe retention path to be rejected")


def test_prune_parent_cli_parser_requires_a_row_selector() -> None:
    args = build_parser().parse_args(
        [
            "prune-training-parent-artifact",
            "--run-list",
            "/tmp/runs.jsonl",
            "--node-id",
            "n2",
            "--dry-run",
        ]
    )

    assert args.node_id == "n2"
    assert args.dry_run


def test_prune_parent_cli_accepts_attempt_boundary() -> None:
    args = build_parser().parse_args(
        [
            "prune-training-parent-artifact",
            "--run-list",
            "/tmp/runs.jsonl",
            "--index",
            "2",
            "--success-not-before",
            "2026-07-20T10:00:00+00:00",
        ]
    )

    assert args.success_not_before == "2026-07-20T10:00:00+00:00"
