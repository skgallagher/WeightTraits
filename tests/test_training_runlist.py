import json
from pathlib import Path

from weighttraits.cli import build_parser
from weighttraits.training.planner import build_training_jobs
from weighttraits.training.runlist import (
    build_training_run_list,
    load_execution_profile,
    load_training_run_specs,
    render_slurm_array_script,
    select_training_run,
    write_training_run_list,
)


def _rows():
    return [
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
    ]


def _config(tmp_path: Path, method: str = "full"):
    return {
        "base_model": "google/flan-t5-small",
        "method": method,
        "output_root": str(tmp_path / "outputs"),
        "trainer": {"max_steps": 20},
        "prompt": {"default_template": "Question: {question}\nAnswer: {answer}"},
        "stopping": {
            "early_stopping": {"patience": 3, "min_delta": 0.01},
            "plateau": {"window": 4, "min_delta": 0.0005},
            "warnings": {"loss_increase_relative": 0.05, "loss_increase_patience": 2},
        },
    }


def test_build_training_run_list_records_stable_runner_contract(tmp_path):
    jobs = build_training_jobs(_rows(), _config(tmp_path))

    report = build_training_run_list(
        jobs,
        run_list_path=tmp_path / "runs.jsonl",
        ledger_path=tmp_path / "ledger.jsonl",
    )
    run = report.runs[1]

    assert report.valid
    assert report.n_errors == 0
    assert run.array_index == 1
    assert run.node_id == "n1"
    assert run.parent_id == "n0"
    assert run.init_from == str(tmp_path / "outputs/n0/model")
    assert run.ledger_path == str(tmp_path / "ledger.jsonl")
    assert run.runner == {
        "entrypoint": "pending_hf_peft_executor",
        "options": {},
        "run_list_path": str(tmp_path / "runs.jsonl"),
        "status": "planned",
    }


def test_training_run_list_roundtrips_jsonl_and_selects_rows(tmp_path):
    jobs = build_training_jobs(_rows(), _config(tmp_path))
    report = build_training_run_list(jobs)
    out = tmp_path / "runs.jsonl"

    write_training_run_list(report, out)
    loaded = load_training_run_specs(out)

    assert len(loaded) == 2
    assert select_training_run(loaded, index=0).node_id == "n0"
    assert select_training_run(loaded, node_id="n1").array_index == 1


def test_training_run_list_reports_artifact_collisions(tmp_path):
    jobs = build_training_jobs(_rows(), _config(tmp_path))
    collision_path = str(tmp_path / "outputs/n0/model")

    report = build_training_run_list(
        jobs,
        path_exists=lambda value: value == collision_path,
    )
    issues = [issue.to_dict() for issue in report.issues]

    assert not report.valid
    assert issues == [
        {
            "severity": "error",
            "issue": "artifact_exists",
            "node_id": "n0",
            "message": "expected artifact already exists: model",
            "path": collision_path,
        }
    ]


def test_training_run_list_reports_parent_order_errors(tmp_path):
    rows = list(reversed(_rows()))
    jobs = build_training_jobs(rows, _config(tmp_path))

    report = build_training_run_list(jobs, check_filesystem=False)

    assert not report.valid
    assert report.issues[0].issue == "parent_after_child"
    assert report.issues[0].node_id == "n1"


def test_training_run_list_warns_when_stopping_guards_are_missing(tmp_path):
    config = _config(tmp_path)
    config["stopping"] = {}
    jobs = build_training_jobs(_rows()[:1], config)

    report = build_training_run_list(jobs)

    assert report.valid
    assert [issue.issue for issue in report.issues] == [
        "no_early_stopping_patience",
        "no_plateau_stopping",
        "no_loss_increase_warning",
    ]


def test_training_run_list_requires_lora_merge_for_lineage_children(tmp_path):
    config = _config(tmp_path, method="lora")
    config["lora"] = {"merge_after_train": False}
    jobs = build_training_jobs(_rows(), config)

    report = build_training_run_list(jobs)

    assert not report.valid
    assert any(issue.issue == "lora_merge_disabled" for issue in report.issues)


def test_load_execution_profile_and_render_slurm_dry_run_script(tmp_path):
    profile = load_execution_profile("configs/cluster/wright.yaml")
    jobs = build_training_jobs(_rows(), _config(tmp_path))
    report = build_training_run_list(jobs, profile=profile)

    script = render_slurm_array_script(
        report,
        run_list_path="/scratch/runs.jsonl",
        profile=profile,
        max_concurrent=2,
    )

    assert profile.name == "wright"
    assert profile.scheduler == "slurm"
    assert "#SBATCH --array=0-1%2" in script
    assert "#SBATCH --account=statds" in script
    assert "describe-training-run" in script
    assert "RUN_LIST=\"${RUN_LIST:-/scratch/runs.jsonl}\"" in script


def test_render_slurm_script_calls_training_runner_when_configured(tmp_path):
    profile = load_execution_profile("configs/cluster/wright.yaml")
    jobs = build_training_jobs(_rows(), _config(tmp_path))
    report = build_training_run_list(
        jobs,
        profile=profile,
        runner_entrypoint="weighttraits.cli run-training-row",
        runner_options={
            "registry_path": "configs/task_data_candidates.yaml",
            "formats_path": "examples/training/dataset_formats_smoke.yaml",
            "max_train_samples": 2,
            "allow_missing_eval": True,
        },
    )

    script = render_slurm_array_script(
        report,
        run_list_path="/scratch/runs.jsonl",
        profile=profile,
    )

    assert "run-training-row" in script
    assert "--registry configs/task_data_candidates.yaml" in script
    assert "--formats examples/training/dataset_formats_smoke.yaml" in script
    assert "--max-train-samples 2" in script
    assert "--allow-missing-eval" in script


def test_describe_training_run_parser_accepts_index_or_node_id():
    by_index = build_parser().parse_args(
        ["describe-training-run", "--run-list", "/tmp/runs.jsonl", "--index", "2"]
    )
    by_node = build_parser().parse_args(
        ["describe-training-run", "--run-list", "/tmp/runs.jsonl", "--node-id", "n2"]
    )

    assert by_index.run_list == Path("/tmp/runs.jsonl")
    assert by_index.index == 2
    assert by_node.node_id == "n2"


def test_make_training_run_list_parser_accepts_outputs():
    args = build_parser().parse_args(
        [
            "make-training-run-list",
            "--manifest",
            "/tmp/manifest.jsonl",
            "--config",
            "/tmp/training.yaml",
            "--out",
            "/tmp/runs.jsonl",
            "--profile",
            "/tmp/wright.yaml",
            "--ledger",
            "/tmp/ledger.jsonl",
            "--report",
            "/tmp/report.json",
            "--slurm-out",
            "/tmp/train.sbatch",
            "--registry",
            "/tmp/task_data.yaml",
            "--formats",
            "/tmp/formats.yaml",
            "--max-concurrent",
            "3",
            "--runner-dry-run",
            "--allow-existing-artifacts",
            "--allow-issues",
        ]
    )

    assert args.manifest == Path("/tmp/manifest.jsonl")
    assert args.config == Path("/tmp/training.yaml")
    assert args.out == Path("/tmp/runs.jsonl")
    assert args.profile == Path("/tmp/wright.yaml")
    assert args.ledger == Path("/tmp/ledger.jsonl")
    assert args.report == Path("/tmp/report.json")
    assert args.slurm_out == Path("/tmp/train.sbatch")
    assert args.registry == Path("/tmp/task_data.yaml")
    assert args.formats == Path("/tmp/formats.yaml")
    assert args.max_concurrent == 3
    assert args.runner_dry_run
    assert args.allow_existing_artifacts
    assert args.allow_issues


def test_make_training_run_list_set_writes_per_tree_run_lists(tmp_path):
    manifests = []
    for tree_id in ("tree_a", "tree_b"):
        manifest = tmp_path / f"{tree_id}.manifest.jsonl"
        manifest.write_text(
            "\n".join(
                json.dumps(dict(row, tree_id=tree_id))
                for row in _rows()
            )
            + "\n"
        )
        manifests.append({"tree_id": tree_id, "assigned_manifest": str(manifest)})
    assignment_summary = tmp_path / "assignment_summary.json"
    assignment_summary.write_text(json.dumps({"assignments": manifests}) + "\n")
    config = tmp_path / "training.yaml"
    config.write_text(
        f"""
training:
  base_model: google/flan-t5-small
  method: full
  output_root: {tmp_path / "outputs"}
  trainer:
    max_steps: 2
  prompt:
    default_template: "Question: {{question}}\\nAnswer: {{answer}}"
  stopping:
    early_stopping:
      patience: 1
    plateau:
      window: 2
    warnings:
      loss_increase_relative: 0.1
"""
    )
    args = build_parser().parse_args(
        [
            "make-training-run-list-set",
            "--assignment-summary",
            str(assignment_summary),
            "--config",
            str(config),
            "--out-dir",
            str(tmp_path / "runlists"),
            "--runner-dry-run",
        ]
    )

    assert args.func(args) == 0
    summary = json.loads((tmp_path / "runlists/run_list_summary.json").read_text())
    tree_a_runs = load_training_run_specs(tmp_path / "runlists/run_lists/tree_a.runs.jsonl")
    tree_b_runs = load_training_run_specs(tmp_path / "runlists/run_lists/tree_b.runs.jsonl")

    assert summary["valid"]
    assert summary["n_trees"] == 2
    assert summary["n_runs"] == 4
    assert summary["runner_entrypoint"] == "weighttraits.cli run-training-row"
    assert summary["runner_options"] == {"dry_run": True}
    assert tree_a_runs[0].output_dir == str(tmp_path / "outputs/tree_a/n0")
    assert tree_b_runs[0].output_dir == str(tmp_path / "outputs/tree_b/n0")
    assert tree_a_runs[0].ledger_path.endswith("ledgers/tree_a.training_ledger.jsonl")
    assert tree_b_runs[0].runner["run_list_path"].endswith("run_lists/tree_b.runs.jsonl")


def test_make_training_run_list_set_parser_accepts_batch_options():
    args = build_parser().parse_args(
        [
            "make-training-run-list-set",
            "--assignment-summary",
            "/tmp/assignment_summary.json",
            "--config",
            "/tmp/training.yaml",
            "--out-dir",
            "/tmp/runlists",
            "--summary-out",
            "/tmp/summary.json",
            "--registry",
            "/tmp/task_data.yaml",
            "--formats",
            "/tmp/formats.yaml",
            "--max-train-samples",
            "16",
            "--runner-dry-run",
            "--allow-existing-artifacts",
            "--allow-issues",
        ]
    )

    assert args.assignment_summary == Path("/tmp/assignment_summary.json")
    assert args.config == Path("/tmp/training.yaml")
    assert args.out_dir == Path("/tmp/runlists")
    assert args.summary_out == Path("/tmp/summary.json")
    assert args.registry == Path("/tmp/task_data.yaml")
    assert args.formats == Path("/tmp/formats.yaml")
    assert args.max_train_samples == 16
    assert args.runner_dry_run
    assert args.allow_existing_artifacts
    assert args.allow_issues
