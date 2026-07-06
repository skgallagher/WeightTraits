import json
from pathlib import Path

from weighttraits.training.data_formats import (
    load_dataset_format_specs,
    validate_training_jobs_against_formats,
)
from weighttraits.training.datasets import audit_dataset_registry, load_dataset_registry
from weighttraits.training.executor import prepare_training_data
from weighttraits.manifests.reference import (
    load_manifest,
    manifest_leaf_ids,
    nontrivial_reference_splits,
)
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


def _jsonl_loader(*args, **kwargs):
    assert args == ("json",)
    data_files = kwargs["data_files"]
    return {
        split: [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        for split, path in data_files.items()
    }


def test_tiny_training_examples_validate_and_render_targets():
    registry = load_dataset_registry(EXAMPLES / "tiny_dataset_registry.yaml")
    formats = load_dataset_format_specs(EXAMPLES / "tiny_dataset_formats.yaml")
    jobs = build_training_jobs_from_files(
        EXAMPLES / "tiny_manifest.jsonl",
        EXAMPLES / "tiny_full_smoke.yaml",
    )

    validation = validate_training_jobs_against_formats(jobs, formats)
    audit = audit_dataset_registry(
        registry,
        dataset_ids=["tiny_boolq_json"],
        format_specs=formats,
        loader=_loader,
    )
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


def test_tiny_branching_contrast_example_renders_distinguishable_node_targets():
    registry = load_dataset_registry(EXAMPLES / "tiny_dataset_registry.yaml")
    formats = load_dataset_format_specs(EXAMPLES / "tiny_dataset_formats.yaml")
    jobs = build_training_jobs_from_files(
        EXAMPLES / "tiny_branching_contrast_manifest.jsonl",
        EXAMPLES / "tiny_full_branching_contrast_smoke.yaml",
    )
    runs = build_training_run_list(jobs).runs

    validation = validate_training_jobs_against_formats(jobs, formats)
    rendered_by_node = {
        run.node_id: prepare_training_data(
            run,
            registry,
            formats,
            loader=_jsonl_loader,
            max_train_samples=2,
            max_eval_samples=1,
        )
        for run in runs
    }
    targets_by_node = {
        node_id: {record.target for record in data.train_records}
        for node_id, data in rendered_by_node.items()
    }

    assert validation.valid
    assert manifest_leaf_ids(EXAMPLES / "tiny_branching_contrast_manifest.jsonl") == [
        "n2",
        "n3",
        "n4",
        "n5",
    ]
    assert [job.dataset_id for job in jobs] == [
        "tiny_branch_left_root_json",
        "tiny_branch_right_root_json",
        "tiny_branch_left_alpha_json",
        "tiny_branch_left_beta_json",
        "tiny_branch_right_alpha_json",
        "tiny_branch_right_beta_json",
    ]
    assert runs[2].init_from == "outputs/tiny_full_branching_contrast_smoke/n0/model"
    assert runs[4].init_from == "outputs/tiny_full_branching_contrast_smoke/n1/model"
    assert all(data.valid for data in rendered_by_node.values())
    assert targets_by_node == {
        "n0": {"left-root"},
        "n1": {"right-root"},
        "n2": {"left-alpha"},
        "n3": {"left-beta"},
        "n4": {"right-alpha"},
        "n5": {"right-beta"},
    }
    assert rendered_by_node["n2"].train_records[0].text.endswith("Reply with the code:")
    assert "branch left sibling alpha" in rendered_by_node["n2"].train_records[0].text
    assert "branch left sibling beta" in rendered_by_node["n3"].train_records[0].text
    assert "branch right sibling alpha" in rendered_by_node["n4"].train_records[0].text


def test_tiny_lora_branching_contrast_example_plans_parent_merged_artifacts():
    jobs = build_training_jobs_from_files(
        EXAMPLES / "tiny_branching_contrast_manifest.jsonl",
        EXAMPLES / "tiny_lora_branching_contrast_smoke.yaml",
    )
    runs = build_training_run_list(jobs).runs

    assert [run.node_id for run in runs] == ["n0", "n1", "n2", "n3", "n4", "n5"]
    assert runs[2].init_from == "outputs/tiny_lora_branching_contrast_smoke/n0/merged"
    assert runs[3].init_from == "outputs/tiny_lora_branching_contrast_smoke/n0/merged"
    assert runs[4].init_from == "outputs/tiny_lora_branching_contrast_smoke/n1/merged"
    assert runs[5].init_from == "outputs/tiny_lora_branching_contrast_smoke/n1/merged"
    assert runs[5].expected_artifacts["adapter"] == (
        "outputs/tiny_lora_branching_contrast_smoke/n5/adapter"
    )
    assert runs[5].expected_artifacts["merged"] == (
        "outputs/tiny_lora_branching_contrast_smoke/n5/merged"
    )


def test_tiny_mid_branching_contrast_example_renders_nested_seven_leaf_topology():
    registry = load_dataset_registry(EXAMPLES / "tiny_dataset_registry.yaml")
    formats = load_dataset_format_specs(EXAMPLES / "tiny_dataset_formats.yaml")
    manifest = EXAMPLES / "tiny_mid_branching_contrast_manifest.jsonl"
    jobs = build_training_jobs_from_files(
        manifest,
        EXAMPLES / "tiny_full_mid_branching_contrast_smoke.yaml",
    )
    runs = build_training_run_list(jobs).runs

    validation = validate_training_jobs_against_formats(jobs, formats)
    rendered_by_node = {
        run.node_id: prepare_training_data(
            run,
            registry,
            formats,
            loader=_jsonl_loader,
            max_train_samples=2,
            max_eval_samples=1,
        )
        for run in runs
    }
    targets_by_node = {
        node_id: {record.target for record in data.train_records}
        for node_id, data in rendered_by_node.items()
    }

    assert validation.valid
    assert manifest_leaf_ids(manifest) == ["n03", "n04", "n05", "n06", "n08", "n09", "n10"]
    assert nontrivial_reference_splits(load_manifest(manifest)) == {
        frozenset({"n03", "n04"}),
        frozenset({"n05", "n06"}),
        frozenset({"n08", "n09", "n10"}),
        frozenset({"n09", "n10"}),
    }
    assert [run.node_id for run in runs] == [
        "n00",
        "n01",
        "n02",
        "n03",
        "n04",
        "n05",
        "n06",
        "n07",
        "n08",
        "n09",
        "n10",
    ]
    assert [job.dataset_id for job in jobs] == [
        "tiny_mid_branch_north_root_json",
        "tiny_mid_branch_south_root_json",
        "tiny_mid_branch_east_root_json",
        "tiny_mid_branch_north_alpha_json",
        "tiny_mid_branch_north_beta_json",
        "tiny_mid_branch_south_alpha_json",
        "tiny_mid_branch_south_beta_json",
        "tiny_mid_branch_east_inner_json",
        "tiny_mid_branch_east_gamma_json",
        "tiny_mid_branch_east_inner_alpha_json",
        "tiny_mid_branch_east_inner_beta_json",
    ]
    assert runs[0].init_from == "hf-internal-testing/tiny-random-t5"
    assert runs[1].init_from == "hf-internal-testing/tiny-random-t5"
    assert runs[2].init_from == "hf-internal-testing/tiny-random-t5"
    assert runs[3].init_from == "outputs/tiny_full_mid_branching_contrast_smoke/n00/model"
    assert runs[4].init_from == "outputs/tiny_full_mid_branching_contrast_smoke/n00/model"
    assert runs[5].init_from == "outputs/tiny_full_mid_branching_contrast_smoke/n01/model"
    assert runs[6].init_from == "outputs/tiny_full_mid_branching_contrast_smoke/n01/model"
    assert runs[7].init_from == "outputs/tiny_full_mid_branching_contrast_smoke/n02/model"
    assert runs[8].init_from == "outputs/tiny_full_mid_branching_contrast_smoke/n02/model"
    assert runs[9].init_from == "outputs/tiny_full_mid_branching_contrast_smoke/n07/model"
    assert runs[10].init_from == "outputs/tiny_full_mid_branching_contrast_smoke/n07/model"
    assert all(data.valid for data in rendered_by_node.values())
    assert targets_by_node == {
        "n00": {"north-root"},
        "n01": {"south-root"},
        "n02": {"east-root"},
        "n03": {"north-alpha"},
        "n04": {"north-beta"},
        "n05": {"south-alpha"},
        "n06": {"south-beta"},
        "n07": {"east-inner"},
        "n08": {"east-gamma"},
        "n09": {"east-inner-alpha"},
        "n10": {"east-inner-beta"},
    }
    assert rendered_by_node["n03"].train_records[0].text.endswith("Reply with the code:")
    assert "mid branch north leaf alpha" in rendered_by_node["n03"].train_records[0].text
    assert "mid branch east inner root" in rendered_by_node["n07"].train_records[0].text
    assert "mid branch east inner leaf beta" in rendered_by_node["n10"].train_records[0].text


def test_tiny_lora_mid_branching_contrast_example_plans_nested_parent_merges():
    jobs = build_training_jobs_from_files(
        EXAMPLES / "tiny_mid_branching_contrast_manifest.jsonl",
        EXAMPLES / "tiny_lora_mid_branching_contrast_smoke.yaml",
    )
    runs = build_training_run_list(jobs).runs

    assert [run.node_id for run in runs] == [
        "n00",
        "n01",
        "n02",
        "n03",
        "n04",
        "n05",
        "n06",
        "n07",
        "n08",
        "n09",
        "n10",
    ]
    assert runs[3].init_from == "outputs/tiny_lora_mid_branching_contrast_smoke/n00/merged"
    assert runs[5].init_from == "outputs/tiny_lora_mid_branching_contrast_smoke/n01/merged"
    assert runs[7].init_from == "outputs/tiny_lora_mid_branching_contrast_smoke/n02/merged"
    assert runs[9].init_from == "outputs/tiny_lora_mid_branching_contrast_smoke/n07/merged"
    assert runs[10].expected_artifacts["adapter"] == (
        "outputs/tiny_lora_mid_branching_contrast_smoke/n10/adapter"
    )
    assert runs[10].expected_artifacts["merged"] == (
        "outputs/tiny_lora_mid_branching_contrast_smoke/n10/merged"
    )
