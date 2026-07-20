from pathlib import Path

from weighttraits.training.planner import build_training_jobs_from_files, load_training_config


ROOT = Path("examples/training/confirm_paper_numbers")
MANIFEST = ROOT / "assigned_manifests/confirm_paper_tree_001.manifest.jsonl"
CONFIGS = {
    "full": ROOT / "llama32_1b_full_finetune.yaml",
    "r8": ROOT / "llama32_1b_lora_qkv_r8.yaml",
    "r64": ROOT / "llama32_1b_lora_qkv_r64.yaml",
}
PINNED_REVISION = "4e20de362430cd3b72f300e6b0f18e50e7166e08"


def test_llama32_configs_share_causal_training_contract() -> None:
    configs = {name: load_training_config(path) for name, path in CONFIGS.items()}

    for config in configs.values():
        trainer = config["trainer"]
        assert config["base_model"] == "meta-llama/Llama-3.2-1B"
        assert config["base_model_revision"] == PINNED_REVISION
        assert trainer["model_task"] == "causal_lm"
        assert trainer["causal_loss_scope"] == "completion"
        assert trainer["seed"] == 20260713
        assert trainer["max_seq_length"] == 640
        assert trainer["per_device_train_batch_size"] == 1
        assert trainer["gradient_accumulation_steps"] == 32
        assert trainer["cleanup_checkpoints_on_success"] is True

    assert configs["full"]["method"] == "full"
    assert configs["full"]["trainer"]["learning_rate"] == 2e-5
    for name, rank in (("r8", 8), ("r64", 64)):
        assert configs[name]["method"] == "lora"
        assert configs[name]["lora"]["r"] == rank
        assert configs[name]["lora"]["lora_alpha"] == 2 * rank
        assert configs[name]["lora"]["target_modules"] == [
            "q_proj",
            "k_proj",
            "v_proj",
        ]


def test_llama32_configs_preserve_parent_lineage_artifacts() -> None:
    full_jobs = build_training_jobs_from_files(MANIFEST, CONFIGS["full"])
    lora_jobs = build_training_jobs_from_files(MANIFEST, CONFIGS["r8"])

    assert len(full_jobs) == len(lora_jobs) == 14
    assert [(job.node_id, job.dataset_id) for job in full_jobs] == [
        (job.node_id, job.dataset_id) for job in lora_jobs
    ]
    assert full_jobs[0].init_from == "meta-llama/Llama-3.2-1B"
    assert lora_jobs[0].init_from == "meta-llama/Llama-3.2-1B"
    child_index = next(index for index, job in enumerate(full_jobs) if job.parent_id != "root")
    parent_id = full_jobs[child_index].parent_id
    assert full_jobs[child_index].init_from.endswith(f"/{parent_id}/model")
    assert lora_jobs[child_index].init_from.endswith(f"/{parent_id}/merged")
    assert lora_jobs[child_index].expected_artifacts["adapter"].endswith(
        f"/{lora_jobs[child_index].node_id}/adapter"
    )
