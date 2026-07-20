import csv
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("Bio")
safetensors_np = pytest.importorskip("safetensors.numpy")

from weighttraits.cli import build_parser, main  # noqa: E402
from weighttraits.training.ledger import TrainingLedgerEvent, append_ledger_event  # noqa: E402


def test_analyze_training_ledger_cli_writes_summary_scores_and_aggregate(tmp_path):
    ledger = tmp_path / "training_ledger.jsonl"
    truth = tmp_path / "truth_manifest.jsonl"
    checkpoints = tmp_path / "checkpoints"
    out = tmp_path / "whitebox"

    for node_id, value in {"n2": 0.0, "n3": 1.0, "n4": 10.0, "n5": 11.0}.items():
        model_dir = checkpoints / node_id / "model"
        model_dir.mkdir(parents=True)
        safetensors_np.save_file(
            {"layer.weight": np.array([value], dtype=np.float32)},
            str(model_dir / "model.safetensors"),
        )
        append_ledger_event(
            ledger,
            TrainingLedgerEvent(
                node_id=node_id,
                status="completed",
                extra={"artifacts": {"model": str(model_dir)}},
            ),
        )
    _write_truth_manifest(truth)

    assert main(
        [
            "analyze-training-ledger",
            "--ledger",
            str(ledger),
            "--truth-manifest",
            str(truth),
            "--artifact",
            "model",
            "--metric",
            "l2",
            "--out",
            str(out),
        ]
    ) == 0

    summary = json.loads((out / "summary.json").read_text())
    score = json.loads((out / "score_l2.json").read_text())
    aggregate = json.loads((out / "aggregate_recovery.json").read_text())

    assert (out / "model_distance_inputs.yaml").exists()
    assert (out / "distance_cube" / "distance_cube.npz").exists()
    assert (out / "tree_l2.newick").exists()
    assert (out / "tree_l2.audit.json").exists()
    assert (out / "four_point_additivity_l2.json").exists()
    assert (out / "atteson_margin_l2.json").exists()
    assert (out / "branch_ordering_l2.json").exists()
    assert summary["artifact"] == "model"
    assert summary["representation"] == "full_weight"
    assert summary["n_models"] == 4
    assert summary["metrics"] == ["l2"]
    assert summary["diagnostics"] == ["four_point_additivity", "atteson_margin"]
    assert summary["results"][0]["four_point_mean_additivity"] is not None
    assert summary["results"][0]["atteson_bottleneck_margin"] is not None
    assert summary["results"][0]["branch_ordering_valid"] is True
    assert summary["results"][0]["branch_rank_biserial"] == 1.0
    assert summary["results"][0]["exact_tree_recovery"] is True
    assert summary["results"][0]["rf"] == 0
    assert summary["aggregate_recovery"]["exact_tree_recovery_rate"] == 1.0
    assert score["clade_recovery"] == 1.0
    assert aggregate["n_records"] == 1


def test_analyze_training_run_set_cli_dry_run_selects_ready_trees(tmp_path):
    ready = _write_ready_run_set_tree(tmp_path, "tree_001")
    partial = _write_leaf_run_list(tmp_path, "tree_002")
    summary = tmp_path / "summary.json"
    report = tmp_path / "analysis_plan.json"
    summary.write_text(
        json.dumps(
            {
                "trees": [
                    {
                        "tree_id": "tree_001",
                        "run_list": str(ready["run_list"].relative_to(tmp_path)),
                        "ledger": "ledgers/tree_001.training_ledger.jsonl",
                        "manifest": "truth/tree_001.manifest.jsonl",
                    },
                    {
                        "tree_id": "tree_002",
                        "run_list": str(partial.relative_to(tmp_path)),
                        "ledger": "ledgers/tree_002.training_ledger.jsonl",
                        "manifest": "truth/tree_002.manifest.jsonl",
                    },
                ]
            }
        )
        + "\n"
    )

    assert main(
        [
            "analyze-training-run-set",
            "--summary",
            str(summary),
            "--path-base",
            str(tmp_path),
            "--artifact",
            "model",
            "--metric",
            "l2",
            "--metric",
            "cosine",
            "--out",
            str(tmp_path / "analyses"),
            "--report-out",
            str(report),
            "--dry-run",
        ]
    ) == 0

    payload = json.loads(report.read_text())
    assert payload["status"] == "planned"
    assert payload["dry_run"] is True
    assert payload["analysis_engine"] == "direct"
    assert payload["tree_builder"] == "biopython_neighbor_joining"
    assert payload["rf_engine"] == "dendropy_treecompare"
    assert payload["completion"]["n_ready"] == 1
    assert payload["selected_tree_ids"] == ["tree_001"]
    assert payload["n_selected"] == 1
    assert payload["n_analyzed"] == 0
    assert payload["analyses"][0]["status"] == "planned"
    assert payload["analyses"][0]["out"].endswith("analyses/tree_001/model_leaf_analysis")


def test_analyze_training_run_set_cli_writes_per_tree_analysis(tmp_path):
    ready = _write_ready_run_set_tree(tmp_path, "tree_001")
    summary = tmp_path / "summary.json"
    report = tmp_path / "analysis_report.json"
    out = tmp_path / "analyses"
    summary.write_text(
        json.dumps(
            {
                "trees": [
                    {
                        "tree_id": "tree_001",
                        "run_list": str(ready["run_list"].relative_to(tmp_path)),
                        "ledger": "ledgers/tree_001.training_ledger.jsonl",
                        "manifest": "truth/tree_001.manifest.jsonl",
                    }
                ]
            }
        )
        + "\n"
    )

    assert main(
        [
            "analyze-training-run-set",
            "--summary",
            str(summary),
            "--path-base",
            str(tmp_path),
            "--artifact",
            "model",
            "--metric",
            "l2",
            "--out",
            str(out),
            "--report-out",
            str(report),
        ]
    ) == 0

    payload = json.loads(report.read_text())
    tree_summary = json.loads((out / "tree_001/model_leaf_analysis/summary.json").read_text())
    assert payload["status"] == "completed"
    assert payload["analysis_engine"] == "direct"
    assert payload["n_selected"] == 1
    assert payload["n_analyzed"] == 1
    assert payload["analyses"][0]["aggregate_recovery"]["exact_tree_recovery_rate"] == 1.0
    assert (
        payload["analyses"][0]["aggregate_recovery"][
            "polytomy_aware_exact_recovery_rate"
        ]
        == 1.0
    )
    assert tree_summary["analysis_engine"] == "direct"
    assert tree_summary["tree_builder"] == "biopython_neighbor_joining"
    assert tree_summary["rf_engine"] == "dendropy_treecompare"
    assert tree_summary["aggregate_recovery"]["exact_tree_recovery_rate"] == 1.0
    assert tree_summary["aggregate_recovery"]["polytomy_aware_exact_recovery_rate"] == 1.0

    second_report = tmp_path / "analysis_report_second.json"
    assert main(
        [
            "analyze-training-run-set",
            "--summary",
            str(summary),
            "--path-base",
            str(tmp_path),
            "--artifact",
            "model",
            "--metric",
            "l2",
            "--out",
            str(out),
            "--report-out",
            str(second_report),
            "--skip-existing",
        ]
    ) == 0
    second_payload = json.loads(second_report.read_text())
    assert second_payload["status"] == "up_to_date"
    assert second_payload["skip_existing"] is True
    assert second_payload["n_existing"] == 1
    assert second_payload["n_analyzed"] == 0
    assert second_payload["analyses"][0]["status"] == "skipped_existing"

    rollup = tmp_path / "run_set_rollup.json"
    rollup_csv = tmp_path / "run_set_rollup.csv"
    assert main(
        [
            "summarize-training-run-set-analysis",
            "--analysis-root",
            str(out),
            "--artifact",
            "model",
            "--out",
            str(rollup),
            "--csv-out",
            str(rollup_csv),
        ]
    ) == 0
    rollup_payload = json.loads(rollup.read_text())
    assert rollup_payload["n_tree_summaries"] == 1
    assert rollup_payload["n_rows"] == 1
    assert rollup_payload["tree_ids"] == ["tree_001"]
    assert rollup_payload["aggregate_by_metric"]["l2"]["n_trees"] == 1
    assert rollup_payload["aggregate_by_metric"]["l2"]["exact_tree_recovery_rate"] == 1.0
    assert (
        rollup_payload["aggregate_by_metric"]["l2"][
            "polytomy_aware_exact_recovery_rate"
        ]
        == 1.0
    )
    assert rollup_payload["aggregate_by_metric"]["l2"]["pooled_clade_recovery"] == 1.0
    assert rollup_payload["aggregate_by_metric"]["l2"]["n_ordering_trees"] == 1
    assert rollup_payload["aggregate_by_metric"]["l2"]["n_branch_pairs"] == 6
    assert rollup_payload["aggregate_by_metric"]["l2"][
        "branch_rank_biserial_mean"
    ] == 1.0
    assert rollup_payload["aggregate_by_metric"]["l2"][
        "atteson_theorem_certified_rate"
    ] in {0.0, 1.0}
    assert rollup_payload["aggregate_by_metric"]["l2"][
        "atteson_theorem_certified_rate_se"
    ] == 0.0
    with rollup_csv.open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["tree_id"] == "tree_001"
    assert rows[0]["rf_engine"] == "dendropy_treecompare"
    assert rows[0]["polytomy_aware_exact_recovery"] == "true"
    assert rows[0]["branch_ordering_valid"] == "true"
    assert rows[0]["branch_ordering_status"] == "ok"


def test_analyze_training_run_set_parser_accepts_options():
    args = build_parser().parse_args(
        [
            "analyze-training-run-set",
            "--summary",
            "/tmp/summary.json",
            "--artifact",
            "adapter_chain",
            "--metric",
            "l2",
            "--metric",
            "cosine",
            "--out",
            "/tmp/analyses",
            "--report-out",
            "/tmp/report.json",
            "--tree-id",
            "tree_001",
            "--node-id",
            "n2",
            "--path-base",
            "/tmp/base",
            "--optional-artifact",
            "training_log",
            "--dry-run",
            "--skip-existing",
            "--chunk-size",
            "100",
            "--eps",
            "0.2",
            "--representation",
            "lora_cumulative_delta",
            "--layer",
            "0",
            "--aggregate",
            "median",
        ]
    )

    assert args.summary == Path("/tmp/summary.json")
    assert args.artifact == "adapter_chain"
    assert args.metric == ["l2", "cosine"]
    assert args.out == Path("/tmp/analyses")
    assert args.report_out == Path("/tmp/report.json")
    assert args.tree_id == ["tree_001"]
    assert args.node_id == ["n2"]
    assert args.path_base == Path("/tmp/base")
    assert args.optional_artifact == ["training_log"]
    assert args.dry_run
    assert args.skip_existing
    assert args.chunk_size == 100
    assert args.eps == 0.2
    assert args.representation == "lora_cumulative_delta"
    assert args.layer == "0"
    assert args.aggregate == "median"


def test_summarize_training_run_set_analysis_parser_accepts_options():
    args = build_parser().parse_args(
        [
            "summarize-training-run-set-analysis",
            "--analysis-root",
            "/tmp/analyses",
            "--artifact",
            "merged",
            "--path-base",
            "/tmp/base",
            "--truth-manifest-root",
            "/tmp/manifests",
            "--tree-id",
            "tree_001",
            "--out",
            "/tmp/rollup.json",
            "--csv-out",
            "/tmp/rollup.csv",
            "--allow-empty",
        ]
    )

    assert args.analysis_root == Path("/tmp/analyses")
    assert args.artifact == "merged"
    assert args.path_base == Path("/tmp/base")
    assert args.truth_manifest_root == Path("/tmp/manifests")
    assert args.tree_id == ["tree_001"]
    assert args.out == Path("/tmp/rollup.json")
    assert args.csv_out == Path("/tmp/rollup.csv")
    assert args.allow_empty


def test_run_set_rollup_relocates_cluster_truth_manifest(tmp_path):
    analysis = tmp_path / "analysis/tree_001/model_leaf_analysis"
    manifests = tmp_path / "manifests"
    analysis.mkdir(parents=True)
    manifests.mkdir()
    truth = manifests / "tree_001.manifest.jsonl"
    _write_truth_manifest(truth)
    labels = ["n2", "n3", "n4", "n5"]
    distances = np.array(
        [
            [0.0, 1.0, 10.0, 11.0],
            [1.0, 0.0, 9.0, 10.0],
            [10.0, 9.0, 0.0, 1.0],
            [11.0, 10.0, 1.0, 0.0],
        ]
    )
    np.save(analysis / "distance_matrix_cosine.npy", distances)
    (analysis / "summary.json").write_text(
        json.dumps(
            {
                "artifact": "model",
                "representation": "full_weight",
                "truth_manifest": "/remote/checkout/manifests/tree_001.manifest.jsonl",
                "model_ids": labels,
                "n_models": 4,
                "results": [{"metric": "cosine"}],
            }
        )
        + "\n"
    )
    out = tmp_path / "rollup.json"

    assert main(
        [
            "summarize-training-run-set-analysis",
            "--analysis-root",
            str(tmp_path / "analysis"),
            "--artifact",
            "model",
            "--truth-manifest-root",
            str(manifests),
            "--out",
            str(out),
        ]
    ) == 0

    payload = json.loads(out.read_text())
    aggregate = payload["aggregate_by_metric"]["cosine"]
    assert payload["truth_manifest_root"] == str(manifests)
    assert aggregate["n_ordering_trees"] == 1
    assert aggregate["branch_rank_biserial_mean"] == 1.0


def _write_truth_manifest(path) -> None:
    rows = [
        {"node_id": "n0", "parent_id": "root", "path": ["root", "n0"], "grow": "train"},
        {"node_id": "n1", "parent_id": "root", "path": ["root", "n1"], "grow": "train"},
        {"node_id": "n2", "parent_id": "n0", "path": ["root", "n0", "n2"], "grow": "train"},
        {"node_id": "n3", "parent_id": "n0", "path": ["root", "n0", "n3"], "grow": "train"},
        {"node_id": "n4", "parent_id": "n1", "path": ["root", "n1", "n4"], "grow": "train"},
        {"node_id": "n5", "parent_id": "n1", "path": ["root", "n1", "n5"], "grow": "train"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _write_ready_run_set_tree(tmp_path: Path, tree_id: str) -> dict[str, Path]:
    run_list = _write_leaf_run_list(tmp_path, tree_id)
    ledger = tmp_path / f"ledgers/{tree_id}.training_ledger.jsonl"
    truth = tmp_path / f"truth/{tree_id}.manifest.jsonl"
    truth.parent.mkdir(parents=True, exist_ok=True)
    _write_truth_manifest(truth)
    values = {"n2": 0.0, "n3": 1.0, "n4": 10.0, "n5": 11.0}
    for node_id, value in values.items():
        model_dir = tmp_path / f"checkpoints/{tree_id}/{node_id}/model"
        model_dir.mkdir(parents=True, exist_ok=True)
        safetensors_np.save_file(
            {"layer.weight": np.array([value], dtype=np.float32)},
            str(model_dir / "model.safetensors"),
        )
        (tmp_path / f"checkpoints/{tree_id}/{node_id}/training_log.jsonl").write_text(
            json.dumps({"step": 1, "train_loss": value}) + "\n"
        )
        append_ledger_event(
            ledger,
            TrainingLedgerEvent(
                node_id=node_id,
                status="completed",
                extra={"artifacts": {"model": f"checkpoints/{tree_id}/{node_id}/model"}},
            ),
        )
    return {"run_list": run_list, "ledger": ledger, "truth": truth}


def _write_leaf_run_list(tmp_path: Path, tree_id: str) -> Path:
    rows = []
    parents = {"n2": "n0", "n3": "n0", "n4": "n1", "n5": "n1"}
    for index, (node_id, parent_id) in enumerate(parents.items()):
        rows.append(
            {
                "array_index": index,
                "run_id": f"{tree_id}-{node_id}",
                "node_id": node_id,
                "parent_id": parent_id,
                "depth": 2,
                "method": "full",
                "dataset_id": "squad",
                "task_family": "qa",
                "init_from": "base",
                "output_dir": f"checkpoints/{tree_id}/{node_id}",
                "expected_artifacts": {
                    "model": f"checkpoints/{tree_id}/{node_id}/model",
                    "training_log": f"checkpoints/{tree_id}/{node_id}/training_log.jsonl",
                },
                "ledger_path": f"ledgers/{tree_id}.training_ledger.jsonl",
                "runner": {},
                "job": {},
            }
        )
    run_list = tmp_path / f"run_lists/{tree_id}.runs.jsonl"
    run_list.parent.mkdir(parents=True, exist_ok=True)
    run_list.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    return run_list
