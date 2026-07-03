import json
from pathlib import Path

from weighttraits.cli import build_parser
from weighttraits.training.planner import (
    build_training_jobs,
    build_training_jobs_from_files,
    load_training_config,
    write_training_plan,
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
            "task_family": "classification",
            "dataset_id": "ag_news",
        },
    ]


def _config(tmp_path: Path, method: str = "full"):
    return {
        "base_model": "google/flan-t5-small",
        "model_family": "flan_t5",
        "method": method,
        "output_root": str(tmp_path / "outputs"),
        "trainer": {"max_steps": 20, "per_device_train_batch_size": 2},
        "prompt": {
            "default_template": "Instruction: {instruction}\nAnswer: {target}",
            "task_templates": {
                "classification": "Classify: {text}\nLabel: {label}",
            },
            "dataset_templates": {
                "squad": "Question: {question}\nContext: {context}\nAnswer: {answers}",
            },
            "model_overrides": {
                "google/flan-t5-small": {
                    "dataset_templates": {
                        "squad": "flan qa: {question}\n{context}\n{answers}",
                    }
                }
            },
        },
        "stopping": {
            "early_stopping": {"patience": 3, "min_delta": 0.01},
            "plateau": {"window": 4, "min_delta": 0.0005},
            "warnings": {"loss_increase_relative": 0.05, "loss_increase_patience": 2},
        },
    }


def test_build_training_jobs_resolves_prompt_overrides_and_parent_sources(tmp_path):
    jobs = build_training_jobs(_rows(), _config(tmp_path))

    assert [job.node_id for job in jobs] == ["n0", "n1"]
    assert jobs[0].prompt_source == "prompt.model[google/flan-t5-small].dataset"
    assert jobs[0].prompt_template.startswith("flan qa")
    assert jobs[0].prompt_fields == ("answers", "context", "question")
    assert jobs[1].prompt_source == "prompt.task"
    assert jobs[0].init_from == "google/flan-t5-small"
    assert jobs[1].init_from == str(tmp_path / "outputs/n0/model")
    assert jobs[1].expected_artifacts["model"] == str(tmp_path / "outputs/n1/model")
    assert jobs[0].stopping["patience"] == 3


def test_prompt_overrides_can_key_on_model_family(tmp_path):
    config = _config(tmp_path)
    config["prompt"]["model_overrides"] = {
        "flan_t5": {
            "task_templates": {
                "classification": "family classify: {text} -> {label}",
            }
        }
    }

    jobs = build_training_jobs(_rows(), config)

    assert jobs[1].prompt_source == "prompt.model[flan_t5].task"
    assert jobs[1].prompt_template.startswith("family classify")


def test_lora_training_jobs_use_merged_parent_and_adapter_artifacts(tmp_path):
    config = _config(tmp_path, method="lora")
    config["lora"] = {
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
        "merge_after_train": True,
    }

    jobs = build_training_jobs(_rows(), config)

    assert jobs[0].method == "lora"
    assert jobs[0].lora["r"] == 16
    assert jobs[1].init_from == str(tmp_path / "outputs/n0/merged")
    assert jobs[1].expected_artifacts["adapter"] == str(tmp_path / "outputs/n1/adapter")
    assert jobs[1].expected_artifacts["merged"] == str(tmp_path / "outputs/n1/merged")


def test_training_plan_roundtrip_from_files(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in _rows()) + "\n")
    config = tmp_path / "training.yaml"
    config.write_text(
        """
training:
  base_model: google/flan-t5-small
  model_family: flan_t5
  method: full
  output_root: outputs
  max_steps: 5
  prompt:
    default_template: "{instruction}\\n{target}"
"""
    )
    out = tmp_path / "plan.jsonl"

    jobs = build_training_jobs_from_files(manifest, config)
    write_training_plan(jobs, out)
    rows = [json.loads(line) for line in out.read_text().splitlines()]

    assert len(rows) == 2
    assert rows[0]["trainer"]["max_steps"] == 5
    assert rows[0]["prompt_fields"] == ["instruction", "target"]


def test_plan_training_parser_accepts_manifest_config_and_out():
    args = build_parser().parse_args(
        [
            "plan-training",
            "--manifest",
            "/tmp/manifest.jsonl",
            "--config",
            "/tmp/training.yaml",
            "--out",
            "/tmp/jobs.jsonl",
        ]
    )

    assert args.manifest == Path("/tmp/manifest.jsonl")
    assert args.config == Path("/tmp/training.yaml")
    assert args.out == Path("/tmp/jobs.jsonl")


def test_load_training_config_accepts_top_level_training_key(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
training:
  base_model: base
  output_root: outputs
  prompt:
    default_template: "{text}"
"""
    )

    assert load_training_config(path)["base_model"] == "base"
