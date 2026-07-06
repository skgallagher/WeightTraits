from pathlib import Path

from weighttraits.training.data_formats import (
    load_dataset_format_specs,
    validate_training_jobs_against_formats,
)
from weighttraits.training.datasets import audit_dataset_registry, load_dataset_registry
from weighttraits.training.executor import prepare_training_data
from weighttraits.manifests.reference import manifest_leaf_ids
from weighttraits.training.planner import build_training_jobs_from_files
from weighttraits.training.runlist import build_training_run_list


EXAMPLES = Path("examples/training")


def _loader(*args, **kwargs):
    assert args == ("json",)
    assert kwargs == {
        "data_files": {
            "train": "examples/training/tiny_train.jsonl",
            "validation": "examples/training/tiny_validation.jsonl",
        }
    }
    return {
        "train": [
            {
                "question": "Is water wet?",
                "passage": "Water is a liquid that makes surfaces wet.",
                "answer": "yes",
            },
            {
                "question": "Is fire cold?",
                "passage": "Fire is hot and gives off heat.",
                "answer": "no",
            },
        ],
        "validation": [
            {
                "question": "Is snow usually cold?",
                "passage": "Snow forms from frozen water and is usually cold.",
                "answer": "yes",
            }
        ],
    }


def test_tiny_training_examples_validate_and_render_targets():
    registry = load_dataset_registry(EXAMPLES / "tiny_dataset_registry.yaml")
    formats = load_dataset_format_specs(EXAMPLES / "tiny_dataset_formats.yaml")
    jobs = build_training_jobs_from_files(
        EXAMPLES / "tiny_manifest.jsonl",
        EXAMPLES / "tiny_full_smoke.yaml",
    )

    validation = validate_training_jobs_against_formats(jobs, formats)
    audit = audit_dataset_registry(registry, format_specs=formats, loader=_loader)
    run = build_training_run_list(jobs).runs[0]
    data = prepare_training_data(run, registry, formats, loader=_loader)

    assert validation.valid
    assert audit.valid
    assert run.method == "full"
    assert run.job["trainer"]["target_field"] == "answer"
    assert data.valid
    assert data.train_records[0].target == "yes"
    assert data.train_records[0].text == (
        "Question: Is water wet?\n"
        "Context: Water is a liquid that makes surfaces wet.\n"
        "Answer yes or no:"
    )
    assert data.summary()["n_train_records"] == 2
    assert data.summary()["n_eval_records"] == 1


def test_tiny_lora_example_plans_adapter_and_merged_artifacts():
    jobs = build_training_jobs_from_files(
        EXAMPLES / "tiny_manifest.jsonl",
        EXAMPLES / "tiny_lora_smoke.yaml",
    )
    run = build_training_run_list(jobs).runs[0]

    assert run.method == "lora"
    assert run.job["lora"]["merge_after_train"] is True
    assert run.expected_artifacts["adapter"] == "outputs/tiny_lora_smoke/n0/adapter"
    assert run.expected_artifacts["merged"] == "outputs/tiny_lora_smoke/n0/merged"


def test_tiny_full_lineage_example_uses_parent_model_artifact():
    jobs = build_training_jobs_from_files(
        EXAMPLES / "tiny_lineage_manifest.jsonl",
        EXAMPLES / "tiny_full_lineage_smoke.yaml",
    )
    runs = build_training_run_list(jobs).runs

    assert [run.node_id for run in runs] == ["n0", "n1"]
    assert runs[0].init_from == "hf-internal-testing/tiny-random-t5"
    assert runs[1].parent_id == "n0"
    assert runs[1].init_from == "outputs/tiny_full_lineage_smoke/n0/model"
    assert runs[1].expected_artifacts["model"] == "outputs/tiny_full_lineage_smoke/n1/model"
    assert runs[0].ledger_path == "outputs/tiny_full_lineage_smoke/training_ledger.jsonl"
    assert runs[1].ledger_path == "outputs/tiny_full_lineage_smoke/training_ledger.jsonl"


def test_tiny_lora_lineage_example_uses_parent_merged_artifact():
    jobs = build_training_jobs_from_files(
        EXAMPLES / "tiny_lineage_manifest.jsonl",
        EXAMPLES / "tiny_lora_lineage_smoke.yaml",
    )
    runs = build_training_run_list(jobs).runs

    assert [run.node_id for run in runs] == ["n0", "n1"]
    assert runs[0].init_from == "hf-internal-testing/tiny-random-t5"
    assert runs[1].parent_id == "n0"
    assert runs[1].init_from == "outputs/tiny_lora_lineage_smoke/n0/merged"
    assert runs[1].expected_artifacts["adapter"] == "outputs/tiny_lora_lineage_smoke/n1/adapter"
    assert runs[1].expected_artifacts["merged"] == "outputs/tiny_lora_lineage_smoke/n1/merged"
    assert runs[0].ledger_path == "outputs/tiny_lora_lineage_smoke/training_ledger.jsonl"
    assert runs[1].ledger_path == "outputs/tiny_lora_lineage_smoke/training_ledger.jsonl"


def test_tiny_full_branching_example_plans_four_leaf_topology():
    jobs = build_training_jobs_from_files(
        EXAMPLES / "tiny_branching_manifest.jsonl",
        EXAMPLES / "tiny_full_branching_smoke.yaml",
    )
    runs = build_training_run_list(jobs).runs

    assert manifest_leaf_ids(EXAMPLES / "tiny_branching_manifest.jsonl") == [
        "n2",
        "n3",
        "n4",
        "n5",
    ]
    assert [run.node_id for run in runs] == ["n0", "n1", "n2", "n3", "n4", "n5"]
    assert runs[0].init_from == "hf-internal-testing/tiny-random-t5"
    assert runs[1].init_from == "hf-internal-testing/tiny-random-t5"
    assert runs[2].init_from == "outputs/tiny_full_branching_smoke/n0/model"
    assert runs[3].init_from == "outputs/tiny_full_branching_smoke/n0/model"
    assert runs[4].init_from == "outputs/tiny_full_branching_smoke/n1/model"
    assert runs[5].init_from == "outputs/tiny_full_branching_smoke/n1/model"
    assert runs[5].expected_artifacts["model"] == "outputs/tiny_full_branching_smoke/n5/model"


def test_tiny_lora_branching_example_plans_parent_merged_artifacts():
    jobs = build_training_jobs_from_files(
        EXAMPLES / "tiny_branching_manifest.jsonl",
        EXAMPLES / "tiny_lora_branching_smoke.yaml",
    )
    runs = build_training_run_list(jobs).runs

    assert [run.node_id for run in runs] == ["n0", "n1", "n2", "n3", "n4", "n5"]
    assert runs[0].init_from == "hf-internal-testing/tiny-random-t5"
    assert runs[1].init_from == "hf-internal-testing/tiny-random-t5"
    assert runs[2].init_from == "outputs/tiny_lora_branching_smoke/n0/merged"
    assert runs[3].init_from == "outputs/tiny_lora_branching_smoke/n0/merged"
    assert runs[4].init_from == "outputs/tiny_lora_branching_smoke/n1/merged"
    assert runs[5].init_from == "outputs/tiny_lora_branching_smoke/n1/merged"
    assert runs[5].expected_artifacts["adapter"] == "outputs/tiny_lora_branching_smoke/n5/adapter"
    assert runs[5].expected_artifacts["merged"] == "outputs/tiny_lora_branching_smoke/n5/merged"
