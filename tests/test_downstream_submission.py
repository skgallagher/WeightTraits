from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from weighttraits.downstream import jobs as downstream_jobs
from weighttraits.downstream.submission import (
    BEHAVIOR_KIND,
    DIRECT_KIND,
    PHYLOLM_KIND,
    sha256_file,
    submit_downstream_plan,
)
from weighttraits.paper.analysis_contracts import ALL_TREE_IDS


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _json(path: Path, payload: object) -> Path:
    return _write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _pin(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _gate_fixture(tmp_path: Path, kinds: tuple[str, ...]) -> tuple[Path, dict[str, Path]]:
    stage = tmp_path / "downstream"
    frozen = tmp_path / "frozen"
    code = stage / "code"
    stage.mkdir()
    frozen.mkdir()
    wrapper_names = {
        DIRECT_KIND: "downstream_direct_runset_cpu.sbatch",
        BEHAVIOR_KIND: "downstream_behavior_smoke.sbatch",
        PHYLOLM_KIND: "downstream_phylolm_smoke.sbatch",
    }
    wrappers: dict[str, Path] = {}
    manifest_rows = []
    for kind in kinds:
        wrapper = _write(code / "scripts" / "slurm" / wrapper_names[kind], "#!/bin/bash\n")
        wrappers[kind] = wrapper
        manifest_rows.append(f"{sha256_file(wrapper)}  {wrapper.relative_to(code)}")
    for relative in (
        "scripts/run_downstream_job.py",
        "scripts/submit_downstream_jobs.py",
        "src/weighttraits/downstream/jobs.py",
        "src/weighttraits/downstream/submission.py",
    ):
        source = _write(code / relative, "VALUE = 1\n")
        manifest_rows.append(f"{sha256_file(source)}  {source.relative_to(code)}")
    manifest = _write(stage / "runtime_receipts" / "DEPLOYED_CODE_SHA256", "\n".join(manifest_rows) + "\n")
    jobs = []
    outputs: dict[str, Path] = {}
    for index, kind in enumerate(kinds):
        output = stage / "outputs" / f"job-{index}"
        outputs[kind + str(index)] = output
        sources = [
            _write(frozen / f"input-{index}-{source_index}.json", "{}\n")
            for source_index in range(5)
        ]
        if kind == DIRECT_KIND:
            job_payload = {
                "schema": f"weighttraits.downstream_job.{kind}.v1",
                "kind": kind,
                "stage_root": str(stage),
                "output_dir": str(output),
                "cohort_id": "cohort",
                "training_summary": _pin(sources[0]),
                "completion_receipt": _pin(sources[1]),
                "stage_manifest": _pin(sources[2]),
                "path_base": str(frozen),
                "artifact_kind": "model",
                "metrics": ["l2", "cosine", "correlation"],
                "chunk_size": 1000000,
                "eps": 0.001,
                "aggregate": "mean",
                "expected_n_trees": 50,
                "expected_n_rows": 150,
            }
            pinned_sources = sources[:3]
        elif kind == BEHAVIOR_KIND:
            job_payload = {
                "schema": f"weighttraits.downstream_job.{kind}.v1",
                "kind": kind,
                "stage_root": str(stage),
                "output_dir": str(output),
                "cohort_id": "cohort",
                "tree_id": "confirm_paper_tree_001",
                "model_task": "causal_lm",
                "path_base": str(frozen),
                "registry": _pin(sources[0]),
                "training_summary": _pin(sources[1]),
                "completion_receipt": _pin(sources[2]),
                "prompt_artifacts": {"protocol": _pin(sources[3])},
                "expected_leaf_ids": list(downstream_jobs.TREE001_SMOKE_LEAF_IDS),
                "leaf_id": "n3",
                "device": "cuda",
                "torch_dtype": "auto",
            }
            pinned_sources = sources[:4]
        else:
            job_payload = {
                "schema": f"weighttraits.downstream_job.{kind}.v1",
                "kind": kind,
                "stage_root": str(stage),
                "output_dir": str(output),
                "cohort_id": "llama32_1b_full_finetune_legacy_causal_2000",
                "tree_id": "confirm_paper_tree_001",
                "training_summary": _pin(sources[0]),
                "completion_receipt": _pin(sources[1]),
                "stage_manifest": _pin(sources[2]),
                "genome_receipt": _pin(sources[3]),
                "truth_manifest": _pin(sources[4]),
                "base_model_id": "meta-llama/Llama-3.2-1B",
                "base_model_revision": "a" * 40,
                "path_base": str(frozen),
                "batch_size": 64,
                "torch_dtype": "auto",
            }
            pinned_sources = sources
        job_config = _json(
            stage / "configs" / f"job-{index}.json",
            job_payload,
        )
        jobs.append(
            {
                "kind": kind,
                "name": f"wt-{kind}-{index}",
                "config": _pin(job_config),
                "wrapper": _pin(wrappers[kind]),
                "output_dir": str(output),
                "input_pins": [_pin(source) for source in pinned_sources],
            }
        )
    config = _json(
        stage / "configs" / "submission.json",
        {
            "schema": "weighttraits.downstream_submission.v1",
            "submission_id": "native-smokes-20260831",
            "stage_root": str(stage),
            "frozen_stage": str(frozen),
            "code_manifest": {"root": str(code), **_pin(manifest)},
            "intent_path": str(stage / "runtime_receipts" / "SUBMIT_INTENT.json"),
            "jobs_receipt_path": str(stage / "runtime_receipts" / "SUBMIT_JOBS.json"),
            "log_dir": str(stage / "slurm_logs"),
            "jobs": jobs,
        },
    )
    return config, outputs


class _Slurm:
    def __init__(self, intent: Path, fail_sbatch: int | None = None) -> None:
        self.intent = intent
        self.fail_sbatch = fail_sbatch
        self.sbatch_calls: list[list[str]] = []

    def __call__(self, command, **kwargs):
        if command[0] in {"squeue", "sacct"}:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        assert command[0] == "sbatch"
        assert self.intent.is_file(), "intent must be published before the first sbatch"
        self.sbatch_calls.append(list(command))
        if self.fail_sbatch == len(self.sbatch_calls):
            raise subprocess.CalledProcessError(1, command, stderr="submission rejected")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=str(9000 + len(self.sbatch_calls)) + "\n",
            stderr="",
        )


def test_gate_is_one_shot_exact_route_and_serializes_gpu_smokes(tmp_path: Path) -> None:
    kinds = (DIRECT_KIND, BEHAVIOR_KIND, PHYLOLM_KIND)
    config, outputs = _gate_fixture(tmp_path, kinds)
    stage = config.parents[1]
    intent = stage / "runtime_receipts" / "SUBMIT_INTENT.json"
    scheduler = _Slurm(intent)
    receipt = submit_downstream_plan(
        config,
        expected_sha256=sha256_file(config),
        runner=scheduler,
    )

    assert receipt["valid"] is True
    assert receipt["n_submitted"] == 3
    assert intent.stat().st_mode & 0o222 == 0
    assert not any(path.exists() for path in outputs.values())
    for command in scheduler.sbatch_calls:
        for exact in (
            "--account=statds",
            "--partition=all",
            "--qos=normal",
            "--nodelist=n03",
            "--exclude=n01",
            "--no-requeue",
        ):
            assert exact in command
    assert "--gres=gpu:nvidia-l40:1" not in scheduler.sbatch_calls[0]
    assert "--array=0-0%1" not in scheduler.sbatch_calls[0]
    assert "--gres=gpu:nvidia-l40:1" in scheduler.sbatch_calls[1]
    assert "--array=0-0%1" in scheduler.sbatch_calls[1]
    assert not any(value.startswith("--dependency=") for value in scheduler.sbatch_calls[1])
    assert "--dependency=afterok:9002" in scheduler.sbatch_calls[2]

    with pytest.raises(FileExistsError, match="already exists"):
        submit_downstream_plan(
            config,
            expected_sha256=sha256_file(config),
            runner=scheduler,
        )


def test_gate_duplicate_scan_prevents_intent(tmp_path: Path) -> None:
    config, _ = _gate_fixture(tmp_path, (BEHAVIOR_KIND,))
    stage = config.parents[1]

    def duplicate(command, **kwargs):
        output = "123|wt-behavior_smoke-0|RUNNING\n" if command[0] == "squeue" else ""
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    with pytest.raises(FileExistsError, match="duplicate Slurm job name"):
        submit_downstream_plan(
            config,
            expected_sha256=sha256_file(config),
            runner=duplicate,
        )
    assert not (stage / "runtime_receipts" / "SUBMIT_INTENT.json").exists()


def test_partial_submission_writes_terminal_invalid_receipt_and_does_not_retry(
    tmp_path: Path,
) -> None:
    config, _ = _gate_fixture(tmp_path, (BEHAVIOR_KIND, PHYLOLM_KIND))
    stage = config.parents[1]
    intent = stage / "runtime_receipts" / "SUBMIT_INTENT.json"
    scheduler = _Slurm(intent, fail_sbatch=2)
    with pytest.raises(RuntimeError, match="fail-closed"):
        submit_downstream_plan(
            config,
            expected_sha256=sha256_file(config),
            runner=scheduler,
        )
    jobs_path = stage / "runtime_receipts" / "SUBMIT_JOBS.json"
    payload = json.loads(jobs_path.read_text())
    assert payload["valid"] is False
    assert payload["partial"] is True
    assert payload["n_planned"] == 2
    assert payload["n_submitted"] == 1
    assert payload["no_automatic_recovery"] is True
    assert len(scheduler.sbatch_calls) == 2
    with pytest.raises(FileExistsError):
        submit_downstream_plan(
            config,
            expected_sha256=sha256_file(config),
            runner=scheduler,
        )


def _direct_raw(tmp_path: Path) -> tuple[dict[str, object], Path, Path]:
    stage = tmp_path / "downstream"
    frozen = tmp_path / "frozen"
    stage.mkdir()
    frozen.mkdir()
    summary = _write(frozen / "summary.json", "{}\n")
    completion = _write(frozen / "completion.json", "{}\n")
    manifest = _write(frozen / "manifest.sha256", "frozen\n")
    output = stage / "outputs" / "direct"
    config = stage / "configs" / "direct.json"
    raw: dict[str, object] = {
        "schema": "weighttraits.downstream_job.direct_runset_cpu.v1",
        "kind": DIRECT_KIND,
        "stage_root": str(stage),
        "output_dir": str(output),
        "cohort_id": "cohort",
        "training_summary": _pin(summary),
        "completion_receipt": _pin(completion),
        "stage_manifest": _pin(manifest),
        "path_base": str(frozen),
        "artifact_kind": "merged",
        "metrics": ["l2", "cosine", "correlation"],
        "chunk_size": 1000000,
        "eps": 0.001,
        "aggregate": "mean",
        "expected_n_trees": 50,
        "expected_n_rows": 150,
    }
    _json(config, raw)
    return raw, config, output


def test_direct_runner_enforces_exact_50_tree_150_row_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw, config, output = _direct_raw(tmp_path)
    monkeypatch.setattr(downstream_jobs, "validate_strict_training_completion", lambda *a, **k: None)
    commands: list[list[str]] = []

    def runner(command, **kwargs):
        commands.append(list(command))
        if "analyze-training-run-set" in command:
            report = Path(command[command.index("--report-out") + 1])
            _json(
                report,
                {
                    "valid": True,
                    "status": "completed",
                    "analysis_engine": "direct",
                    "metrics": ["l2", "cosine", "correlation"],
                    "n_selected": 50,
                    "n_analyzed": 50,
                    "n_existing": 0,
                    "n_skipped": 0,
                    "selected_tree_ids": list(ALL_TREE_IDS),
                },
            )
        else:
            out = Path(command[command.index("--out") + 1])
            csv_out = Path(command[command.index("--csv-out") + 1])
            rows = [
                {"tree_id": tree_id, "metric": metric}
                for tree_id in ALL_TREE_IDS
                for metric in ("l2", "cosine", "correlation")
            ]
            _json(
                out,
                {
                    "valid": True,
                    "n_tree_summaries": 50,
                    "n_rows": 150,
                    "tree_ids": list(ALL_TREE_IDS),
                    "metrics": ["correlation", "cosine", "l2"],
                    "rows": rows,
                },
            )
            csv_out.parent.mkdir(parents=True, exist_ok=True)
            with csv_out.open("w") as handle:
                handle.write("tree_id,metric\n")
                for row in rows:
                    handle.write(f"{row['tree_id']},{row['metric']}\n")
        return subprocess.CompletedProcess(command, 0)

    receipt = downstream_jobs._run_direct(
        raw,
        config=config,
        code_root=tmp_path,
        runner=runner,
    )
    assert receipt["valid"] is True
    assert receipt["artifact"] == "merged"
    assert receipt["n_selected"] == 50
    assert receipt["n_tree_summaries"] == 50
    assert receipt["n_rows"] == 150
    assert (output / "receipt.json").is_file()
    analysis_command = commands[0]
    assert "--skip-existing" not in analysis_command
    assert "--optional-artifact" not in analysis_command
    assert "--allow-empty" not in commands[1]
    metric_values = [
        analysis_command[index + 1]
        for index, value in enumerate(analysis_command)
        if value == "--metric"
    ]
    assert metric_values == ["l2", "cosine", "correlation"]


def test_behavior_smoke_live_resolves_tree001_but_runs_only_n3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = tmp_path / "downstream"
    frozen = tmp_path / "frozen"
    stage.mkdir()
    frozen.mkdir()
    registry = _write(frozen / "registry.json", "{}\n")
    summary = _write(frozen / "summary.json", "{}\n")
    completion = _write(frozen / "completion.json", "{}\n")
    prompts = _write(frozen / "prompts.jsonl", "{}\n")
    output = stage / "outputs" / "behavior-smoke"
    config = stage / "configs" / "behavior.json"
    expected = list(downstream_jobs.TREE001_SMOKE_LEAF_IDS)
    raw = {
        "schema": "weighttraits.downstream_job.behavior_smoke.v1",
        "kind": BEHAVIOR_KIND,
        "stage_root": str(stage),
        "output_dir": str(output),
        "cohort_id": "cohort",
        "tree_id": "confirm_paper_tree_001",
        "model_task": "causal_lm",
        "path_base": str(frozen),
        "registry": _pin(registry),
        "training_summary": _pin(summary),
        "completion_receipt": _pin(completion),
        "prompt_artifacts": {"protocol": _pin(prompts)},
        "expected_leaf_ids": expected,
        "leaf_id": "n3",
        "device": "cuda",
        "torch_dtype": "auto",
    }
    _json(config, raw)
    monkeypatch.setattr(downstream_jobs, "validate_strict_training_completion", lambda *a, **k: None)
    model_pin = SimpleNamespace(model_id="model", revision="a" * 40)
    monkeypatch.setattr(
        downstream_jobs,
        "load_behavior_protocol_registry",
        lambda path: SimpleNamespace(model_pins={"causal_lm": model_pin}),
    )

    class Resolved:
        leaves = tuple(SimpleNamespace(leaf_id=value) for value in expected)

        def leaf(self, leaf_id):
            assert leaf_id == "n3"
            return self.leaves[0]

    monkeypatch.setattr(downstream_jobs, "resolve_leaf_checkpoints", lambda *a, **k: Resolved())
    commands: list[list[str]] = []

    def runner(command, **kwargs):
        commands.append(list(command))
        response = Path(command[command.index("--out") + 1])
        inference = Path(command[command.index("--receipt") + 1])
        _write(response, '{"response":"ok"}\n')
        _json(
            inference,
            {
                "valid": True,
                "status": "completed",
                "request": {
                    "cohort_id": "cohort",
                    "tree_id": "confirm_paper_tree_001",
                    "model_id": "n3",
                },
            },
        )
        return subprocess.CompletedProcess(command, 0)

    receipt = downstream_jobs._run_behavior(
        raw,
        config=config,
        code_root=tmp_path,
        runner=runner,
    )
    assert receipt["valid"] is True
    assert receipt["n_tree_leaves"] == 8
    assert receipt["leaf_id"] == "n3"
    assert len(commands) == 1
    assert commands[0][commands[0].index("--leaf-id") + 1] == "n3"
    assert (output / "receipt.json").is_file()
