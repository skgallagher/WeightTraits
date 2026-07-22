import json
from pathlib import Path

from weighttraits.analysis.completion import (
    audit_training_run_set_completion,
    audit_training_tree_completion,
    write_run_set_completion_csv,
)
from weighttraits.cli import build_parser, main
from weighttraits.training.ledger import TrainingLedgerEvent, append_ledger_event
from weighttraits.training.planner import build_training_jobs
from weighttraits.training.runlist import build_training_run_list, write_training_run_list


def test_audit_training_tree_completion_accepts_complete_tree(tmp_path):
    run_list = _write_run_list(tmp_path)
    ledger = tmp_path / "ledgers/tree.training_ledger.jsonl"
    _write_artifact_tree(tmp_path, "n0")
    _write_artifact_tree(tmp_path, "n1")
    append_ledger_event(ledger, TrainingLedgerEvent(node_id="n0", status="started"))
    append_ledger_event(
        ledger,
        TrainingLedgerEvent(
            node_id="n0",
            status="completed",
            step=20,
            train_loss=0.3,
            eval_loss=0.4,
            extra={"artifacts": {"model": "outputs/tree/n0/model"}},
        ),
    )
    append_ledger_event(ledger, TrainingLedgerEvent(node_id="n1", status="started"))
    append_ledger_event(
        ledger,
        TrainingLedgerEvent(
            node_id="n1",
            status="completed",
            step=20,
            extra={"artifacts": {"model": "outputs/tree/n1/model"}},
        ),
    )

    report = audit_training_tree_completion(run_list, path_base=tmp_path)

    assert report.valid
    assert report.n_runs == 2
    assert report.n_events == 4
    assert report.n_terminal_nodes == 2
    assert report.n_ok_nodes == 2
    assert report.n_expected_artifacts == 4
    assert report.n_existing_artifacts == 4
    assert report.status_counts == {"completed": 2}
    assert report.nodes[0].artifacts[1].name == "training_log"
    assert report.nodes[0].artifacts[1].n_records == 1


def test_audit_training_tree_completion_reports_failures(tmp_path):
    run_list = _write_run_list(tmp_path)
    ledger = tmp_path / "ledgers/tree.training_ledger.jsonl"
    _write_artifact_tree(tmp_path, "n0")
    append_ledger_event(ledger, TrainingLedgerEvent(node_id="n0", status="completed"))
    append_ledger_event(ledger, TrainingLedgerEvent(node_id="n1", status="failed", message="boom"))

    report = audit_training_tree_completion(run_list, path_base=tmp_path)
    issues = [issue.issue for issue in report.issues]

    assert not report.valid
    assert report.n_failed_nodes == 1
    assert "failed_node" in issues
    assert issues.count("missing_artifact") == 2


def test_audit_training_tree_completion_can_warn_on_optional_artifacts(tmp_path):
    run_list = _write_run_list(tmp_path)
    ledger = tmp_path / "ledgers/tree.training_ledger.jsonl"
    model_dir = tmp_path / "outputs/tree/n0/model"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}\n")
    append_ledger_event(ledger, TrainingLedgerEvent(node_id="n0", status="completed"))

    report = audit_training_tree_completion(
        run_list,
        path_base=tmp_path,
        optional_artifacts={"training_log"},
    )
    issues = [issue.issue for issue in report.issues]

    assert not report.valid
    assert "missing_optional_artifact" in issues
    assert issues.count("missing_artifact") == 1


def test_audit_training_tree_completion_allows_pruned_internal_model(tmp_path):
    run_list = _write_run_list(tmp_path)
    ledger = tmp_path / "ledgers/tree.training_ledger.jsonl"
    _write_artifact_tree(tmp_path, "n1")
    append_ledger_event(ledger, TrainingLedgerEvent(node_id="n0", status="completed"))
    append_ledger_event(ledger, TrainingLedgerEvent(node_id="n1", status="completed"))

    report = audit_training_tree_completion(
        run_list,
        path_base=tmp_path,
        optional_artifacts={"training_log"},
        required_artifact_nodes_by_name={"model": {"n1"}},
    )
    issues = [issue.issue for issue in report.issues]

    assert report.valid
    assert issues.count("missing_unrequired_artifact") == 1
    assert "missing_artifact" not in issues


def test_audit_training_run_set_completion_rolls_up_tree_status(tmp_path):
    ready = _write_named_run_list(tmp_path, "tree_001")
    partial = _write_named_run_list(tmp_path, "tree_002")
    summary = tmp_path / "summary.json"
    csv_out = tmp_path / "dashboard.csv"
    _write_named_artifact_tree(tmp_path, "tree_001", "n0")
    _write_named_artifact_tree(tmp_path, "tree_001", "n1")
    append_ledger_event(
        tmp_path / "ledgers/tree_001.training_ledger.jsonl",
        TrainingLedgerEvent(node_id="n0", status="completed"),
    )
    append_ledger_event(
        tmp_path / "ledgers/tree_001.training_ledger.jsonl",
        TrainingLedgerEvent(node_id="n1", status="completed"),
    )
    append_ledger_event(
        tmp_path / "ledgers/tree_002.training_ledger.jsonl",
        TrainingLedgerEvent(node_id="n0", status="running"),
    )
    summary.write_text(
        json.dumps(
            {
                "trees": [
                    {
                        "tree_id": "tree_001",
                        "run_list": str(ready.relative_to(tmp_path)),
                        "ledger": "ledgers/tree_001.training_ledger.jsonl",
                        "manifest": "assigned/tree_001.manifest.jsonl",
                        "output_root": "outputs/tree_001",
                    },
                    {
                        "tree_id": "tree_002",
                        "run_list": str(partial.relative_to(tmp_path)),
                        "ledger": "ledgers/tree_002.training_ledger.jsonl",
                        "manifest": "assigned/tree_002.manifest.jsonl",
                        "output_root": "outputs/tree_002",
                    },
                ]
            }
        )
        + "\n"
    )

    report = audit_training_run_set_completion(summary, path_base=tmp_path)
    ready_only = audit_training_run_set_completion(summary, path_base=tmp_path, only_ready=True)
    write_run_set_completion_csv(report, csv_out)

    assert not report.valid
    assert report.n_trees == 2
    assert report.n_ready == 1
    assert report.n_in_progress == 1
    assert report.n_not_started == 0
    assert report.ready_tree_ids == ("tree_001",)
    assert report.rows[0].ready_for_analysis
    assert report.rows[1].issue_counts["non_terminal_node"] == 1
    assert ready_only.n_trees == 1
    assert ready_only.rows[0].tree_id == "tree_001"
    assert csv_out.read_text().splitlines()[0].startswith("tree_id,valid,ready_for_analysis")


def test_audit_training_tree_cli_writes_report_and_sets_exit_code(tmp_path):
    run_list = _write_run_list(tmp_path)
    ledger = tmp_path / "ledgers/tree.training_ledger.jsonl"
    out = tmp_path / "report.json"
    _write_artifact_tree(tmp_path, "n0")
    append_ledger_event(ledger, TrainingLedgerEvent(node_id="n0", status="completed"))

    code = main(
        [
            "audit-training-tree",
            "--run-list",
            str(run_list),
            "--path-base",
            str(tmp_path),
            "--out",
            str(out),
            "--optional-artifact",
            "training_log",
            "--allow-issues",
        ]
    )

    report = json.loads(out.read_text())
    assert code == 0
    assert report["valid"] is False
    assert report["n_missing_nodes"] == 1


def test_audit_training_run_set_cli_writes_json_and_csv(tmp_path):
    run_list = _write_named_run_list(tmp_path, "tree_001")
    summary = tmp_path / "summary.json"
    out = tmp_path / "run_set.json"
    csv_out = tmp_path / "run_set.csv"
    _write_named_artifact_tree(tmp_path, "tree_001", "n0")
    _write_named_artifact_tree(tmp_path, "tree_001", "n1")
    append_ledger_event(
        tmp_path / "ledgers/tree_001.training_ledger.jsonl",
        TrainingLedgerEvent(node_id="n0", status="completed"),
    )
    append_ledger_event(
        tmp_path / "ledgers/tree_001.training_ledger.jsonl",
        TrainingLedgerEvent(node_id="n1", status="completed"),
    )
    summary.write_text(
        json.dumps(
            {
                "trees": [
                    {
                        "tree_id": "tree_001",
                        "run_list": str(run_list.relative_to(tmp_path)),
                        "ledger": "ledgers/tree_001.training_ledger.jsonl",
                    }
                ]
            }
        )
        + "\n"
    )

    code = main(
        [
            "audit-training-run-set",
            "--summary",
            str(summary),
            "--path-base",
            str(tmp_path),
            "--out",
            str(out),
            "--csv-out",
            str(csv_out),
        ]
    )

    payload = json.loads(out.read_text())
    assert code == 0
    assert payload["valid"] is True
    assert payload["n_ready"] == 1
    assert "tree_001" in csv_out.read_text()


def test_audit_training_tree_parser_accepts_options():
    args = build_parser().parse_args(
        [
            "audit-training-tree",
            "--run-list",
            "/tmp/tree.runs.jsonl",
            "--ledger",
            "/tmp/tree.training_ledger.jsonl",
            "--path-base",
            "/tmp/base",
            "--out",
            "/tmp/report.json",
            "--optional-artifact",
            "training_log",
            "--skip-artifact-check",
            "--skip-parent-order-check",
            "--allow-issues",
        ]
    )

    assert args.run_list == Path("/tmp/tree.runs.jsonl")
    assert args.ledger == Path("/tmp/tree.training_ledger.jsonl")
    assert args.path_base == Path("/tmp/base")
    assert args.out == Path("/tmp/report.json")
    assert args.optional_artifact == ["training_log"]
    assert args.skip_artifact_check
    assert args.skip_parent_order_check
    assert args.allow_issues


def test_audit_training_run_set_parser_accepts_options():
    args = build_parser().parse_args(
        [
            "audit-training-run-set",
            "--summary",
            "/tmp/summary.json",
            "--path-base",
            "/tmp/base",
            "--out",
            "/tmp/report.json",
            "--csv-out",
            "/tmp/report.csv",
            "--optional-artifact",
            "training_log",
            "--only-ready",
            "--skip-artifact-check",
            "--skip-parent-order-check",
            "--allow-issues",
        ]
    )

    assert args.summary == Path("/tmp/summary.json")
    assert args.path_base == Path("/tmp/base")
    assert args.out == Path("/tmp/report.json")
    assert args.csv_out == Path("/tmp/report.csv")
    assert args.optional_artifact == ["training_log"]
    assert args.only_ready
    assert args.skip_artifact_check
    assert args.skip_parent_order_check
    assert args.allow_issues


def _write_run_list(tmp_path: Path) -> Path:
    jobs = build_training_jobs(
        [
            {
                "node_id": "n0",
                "parent_id": "root",
                "depth": 1,
                "path": ["root", "n0"],
                "grow": "train",
                "task_family": "qa",
                "dataset_id": "squad",
            },
            {
                "node_id": "n1",
                "parent_id": "n0",
                "depth": 2,
                "path": ["root", "n0", "n1"],
                "grow": "train",
                "task_family": "qa",
                "dataset_id": "squad",
            },
        ],
        {
            "base_model": "google/flan-t5-small",
            "method": "full",
            "output_root": "outputs/tree",
            "trainer": {"max_steps": 20},
            "prompt": {"default_template": "{question}\n{answer}"},
        },
    )
    run_list = build_training_run_list(
        jobs,
        run_list_path="run_lists/tree.runs.jsonl",
        ledger_path="ledgers/tree.training_ledger.jsonl",
        check_filesystem=False,
    )
    out = tmp_path / "run_lists/tree.runs.jsonl"
    write_training_run_list(run_list, out)
    return out


def _write_named_run_list(tmp_path: Path, tree_id: str) -> Path:
    jobs = build_training_jobs(
        [
            {
                "node_id": "n0",
                "parent_id": "root",
                "depth": 1,
                "path": ["root", "n0"],
                "grow": "train",
                "task_family": "qa",
                "dataset_id": "squad",
            },
            {
                "node_id": "n1",
                "parent_id": "n0",
                "depth": 2,
                "path": ["root", "n0", "n1"],
                "grow": "train",
                "task_family": "qa",
                "dataset_id": "squad",
            },
        ],
        {
            "base_model": "google/flan-t5-small",
            "method": "full",
            "output_root": f"outputs/{tree_id}",
            "trainer": {"max_steps": 20},
            "prompt": {"default_template": "{question}\n{answer}"},
        },
    )
    run_list = build_training_run_list(
        jobs,
        run_list_path=f"run_lists/{tree_id}.runs.jsonl",
        ledger_path=f"ledgers/{tree_id}.training_ledger.jsonl",
        check_filesystem=False,
    )
    out = tmp_path / f"run_lists/{tree_id}.runs.jsonl"
    write_training_run_list(run_list, out)
    return out


def _write_artifact_tree(tmp_path: Path, node_id: str) -> None:
    model_dir = tmp_path / f"outputs/tree/{node_id}/model"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}\n")
    training_log = tmp_path / f"outputs/tree/{node_id}/training_log.jsonl"
    training_log.write_text(json.dumps({"step": 20, "train_loss": 0.3}) + "\n")


def _write_named_artifact_tree(tmp_path: Path, tree_id: str, node_id: str) -> None:
    model_dir = tmp_path / f"outputs/{tree_id}/{node_id}/model"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}\n")
    training_log = tmp_path / f"outputs/{tree_id}/{node_id}/training_log.jsonl"
    training_log.write_text(json.dumps({"step": 20, "train_loss": 0.3}) + "\n")
