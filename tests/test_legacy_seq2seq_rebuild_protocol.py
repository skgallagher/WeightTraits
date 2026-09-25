import json
from pathlib import Path

import pytest

from weighttraits.taskdata.assignment import load_manifest_rows
from weighttraits.training.planner import (
    build_training_jobs,
    load_training_config,
    resolve_prompt_template,
)
from weighttraits.training.prompts import render_prompt


ROOT = Path(__file__).parents[1]
ASSETS = ROOT / "examples" / "training" / "confirm_paper_numbers"
ASSIGNMENT_SUMMARY = ASSETS / "assignment_summary.json"
BASE_CONFIG = ASSETS / "flan_t5_legacy_seq2seq_2000_base.yaml"
FULL_CONFIG = ASSETS / "flan_t5_full_finetune_legacy_seq2seq_2000.yaml"
MATCHED_ASSETS = ROOT / "examples" / "training" / "translation_holdout_20260803"
MATCHED_ASSIGNMENT_SUMMARY = MATCHED_ASSETS / "matched_assignment_summary.json"
MATCHED_FULL_CONFIG = MATCHED_ASSETS / "flan_t5_full_finetune_legacy_seq2seq_2000.yaml"
LORA_CONFIGS = {
    ASSETS / "flan_t5_lora_qv_r8_legacy_seq2seq_2000.yaml": (["q", "v"], 72),
    ASSETS / "flan_t5_lora_k_r8_legacy_seq2seq_2000.yaml": (["k"], 36),
    ASSETS / "flan_t5_lora_qkv_r8_legacy_seq2seq_2000.yaml": (["q", "k", "v"], 108),
    ASSETS / "flan_t5_lora_qkvo_r8_legacy_seq2seq_2000.yaml": (
        ["q", "k", "v", "o"],
        144,
    ),
    ASSETS / "flan_t5_lora_full_ft_approx_actual_r8_legacy_seq2seq_2000.yaml": (
        ["q", "k", "v", "o", "wo"],
        168,
    ),
    ASSETS / "flan_t5_lora_all_projections_corrected_r8_legacy_seq2seq_2000.yaml": (
        ["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
        216,
    ),
}
COHORT_CONFIGS = (FULL_CONFIG, *LORA_CONFIGS)


def test_shared_flan_protocol_matches_legacy_seq2seq_contract():
    base = load_training_config(BASE_CONFIG)

    assert base["base_model"] == "google/flan-t5-base"
    assert base["base_model_revision"] == "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2"
    assert "model_dtype" not in base["trainer"]
    assert base["trainer"] == {
        "model_task": "seq2seq",
        "padding_side": "right",
        "pad_to_multiple_of": 8,
        "require_max_steps": True,
        "seed": 42,
        "max_steps": 2000,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "learning_rate": 0.0003,
        "warmup_steps": 200,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "eval_steps": 250,
        "logging_steps": 10,
        "save_strategy": "no",
        "bf16": True,
        "target_field": "target",
        "max_source_length": 512,
        "max_target_length": 128,
        "predict_with_generate": True,
        "generation_max_length": 128,
        "report_to": [],
    }
    assert base["stopping"] == {"enabled": False}


def test_full_and_lora_overlays_only_change_cohort_specific_training_fields():
    base = load_training_config(BASE_CONFIG)
    full = load_training_config(FULL_CONFIG)

    assert full["method"] == "full"
    assert full["trainer"] == {**base["trainer"], "gradient_checkpointing": True}
    assert "lora" not in full

    for config_path in LORA_CONFIGS:
        config = load_training_config(config_path)
        assert config["base_model"] == base["base_model"]
        assert config["base_model_revision"] == base["base_model_revision"]
        assert config["prompt"] == base["prompt"]
        assert config["stopping"] == base["stopping"]
        assert config["method"] == "lora"
        assert config["trainer"] == {
            **base["trainer"],
            "gradient_checkpointing": False,
        }
        assert config["protocol_id"]
        assert config["model_family"]
        assert config["output_root"]


@pytest.mark.parametrize("config_path", LORA_CONFIGS)
def test_lora_overlays_pin_exact_executable_projection_scopes(config_path):
    expected_targets, expected_module_count = LORA_CONFIGS[config_path]
    lora = load_training_config(config_path)["lora"]

    assert lora == {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": expected_targets,
        "merge_after_train": True,
    }
    # Flan-T5-base has 36 q/k/v/o projections, 24 FFN projections of each
    # gated type, and 24 wo projections. This documents the strict PEFT audit
    # count expected at launch for each explicit scope.
    assert expected_module_count in {36, 72, 108, 144, 168, 216}


def test_legacy_full_ft_approx_and_corrected_all_projection_scopes_are_distinct():
    approx_path = (
        ASSETS / "flan_t5_lora_full_ft_approx_actual_r8_legacy_seq2seq_2000.yaml"
    )
    corrected_path = (
        ASSETS / "flan_t5_lora_all_projections_corrected_r8_legacy_seq2seq_2000.yaml"
    )
    approx = load_training_config(approx_path)["lora"]["target_modules"]
    corrected = load_training_config(corrected_path)["lora"]["target_modules"]

    assert approx == ["q", "k", "v", "o", "wo"]
    assert "wi" not in approx
    assert corrected == ["q", "k", "v", "o", "wi_0", "wi_1", "wo"]


@pytest.mark.parametrize(
    ("task_family", "dataset_id", "fields", "expected"),
    [
        ("classification", "sst2", {"text": "good"}, "classify: good"),
        (
            "classification",
            "mnli",
            {"text": "premise", "hypothesis": "claim"},
            "classify: premise hypothesis: claim",
        ),
        ("summarization", "xsum", {"document": "doc"}, "summarize: doc"),
        (
            "qa",
            "quartz",
            {"qa_input": "question: q choices: A) x B) y context: c"},
            "question: q choices: A) x B) y context: c",
        ),
        (
            "translation",
            "opus_en_ja",
            {"source_text": "hello"},
            "translate English to Japanese: hello",
        ),
    ],
)
def test_representative_prompts_match_legacy_flan_strings(
    task_family,
    dataset_id,
    fields,
    expected,
):
    config = load_training_config(FULL_CONFIG)
    resolution = resolve_prompt_template(
        {"task_family": task_family, "dataset_id": dataset_id},
        config,
    )

    assert render_prompt(resolution.template, fields) == expected


def test_all_translation_prefixes_match_legacy_flan_strings():
    config = load_training_config(FULL_CONFIG)
    expected_languages = {
        "wmt14_en_fr": "French",
        "wmt14_en_de": "German",
        "opus_en_de": "German",
        "opus_en_fr": "French",
        "opus_en_es": "Spanish",
        "opus_en_it": "Italian",
        "opus_en_ru": "Russian",
        "opus_en_zh": "Chinese",
        "opus_en_ja": "Japanese",
    }

    for dataset_id, language in expected_languages.items():
        resolution = resolve_prompt_template(
            {"task_family": "translation", "dataset_id": dataset_id},
            config,
        )
        assert render_prompt(resolution.template, {"source_text": "text"}) == (
            f"translate English to {language}: text"
        )


def test_seven_cohort_configs_plan_all_50_assigned_trees_and_641_nodes():
    summary = json.loads(ASSIGNMENT_SUMMARY.read_text())
    assignments = summary["assignments"]

    assert len(COHORT_CONFIGS) == 7
    assert len(assignments) == 50
    assert sum(int(assignment["n_rows"]) for assignment in assignments) == 641

    for config_path in COHORT_CONFIGS:
        config = load_training_config(config_path)
        n_jobs = 0
        for assignment in assignments:
            manifest = ROOT / assignment["assigned_manifest"]
            jobs = build_training_jobs(load_manifest_rows(manifest), config)
            assert len(jobs) == int(assignment["n_rows"])
            assert all(job.protocol_id == config["protocol_id"] for job in jobs)
            n_jobs += len(jobs)
        assert n_jobs == 641


def test_matched_flan_full_ft_only_changes_identity_output_and_assignments():
    ordinary = load_training_config(FULL_CONFIG)
    matched = load_training_config(MATCHED_FULL_CONFIG)

    assert matched["model_family"] == (
        "flan_t5_full_finetune_no_translation_legacy_seq2seq_2000"
    )
    assert matched["output_root"] == (
        "outputs/translation_holdout_20260803/"
        "flan_t5_full_finetune_legacy_seq2seq_2000"
    )
    for field in (
        "base_model",
        "base_model_revision",
        "method",
        "protocol_id",
        "trainer",
        "prompt",
        "stopping",
    ):
        assert matched[field] == ordinary[field]
    assert "lora" not in matched

    summary = json.loads(MATCHED_ASSIGNMENT_SUMMARY.read_text())
    assignments = summary["assignments"]
    assert len(assignments) == 50
    assert sum(int(assignment["n_rows"]) for assignment in assignments) == 641

    n_jobs = 0
    for assignment in assignments:
        manifest = ROOT / assignment["assigned_manifest"]
        jobs = build_training_jobs(load_manifest_rows(manifest), matched)
        assert len(jobs) == int(assignment["n_rows"])
        assert all(job.task_family != "translation" for job in jobs)
        assert all(job.protocol_id == ordinary["protocol_id"] for job in jobs)
        n_jobs += len(jobs)
    assert n_jobs == 641


def test_full_and_lora_lineage_use_their_required_parent_artifacts():
    summary = json.loads(ASSIGNMENT_SUMMARY.read_text())
    manifest = ROOT / summary["assignments"][0]["assigned_manifest"]
    rows = load_manifest_rows(manifest)
    full_jobs = build_training_jobs(rows, load_training_config(FULL_CONFIG))
    lora_jobs = build_training_jobs(
        rows,
        load_training_config(ASSETS / "flan_t5_lora_qv_r8_legacy_seq2seq_2000.yaml"),
    )
    full_child = next(job for job in full_jobs if job.parent_id != "root")
    lora_child = next(job for job in lora_jobs if job.parent_id != "root")

    assert full_child.init_from.endswith(f"/{full_child.parent_id}/model")
    assert set(full_child.expected_artifacts) == {"model", "training_log"}
    assert lora_child.init_from.endswith(f"/{lora_child.parent_id}/merged")
    assert set(lora_child.expected_artifacts) == {
        "adapter",
        "merged",
        "lora_target_audit",
        "training_log",
    }
