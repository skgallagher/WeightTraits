from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from weighttraits.behavior.phylolm import (
    build_sampled_genome,
    write_genome_receipt,
)
from weighttraits.behavior.phylolm_inference import (
    collect_phylolm_leaf_population,
    extract_raw_continuation_alleles,
    generate_phylolm_alleles,
)
from weighttraits.behavior.probe_inference import ResolvedLeafCheckpoint
from weighttraits.paper.analysis_contracts import ALL_TREE_IDS


class _Tokenizer:
    pad_token_id = None
    eos_token_id = 2
    eos_token = "<eos>"
    pad_token = None
    padding_side = "right"


def _genome(tmp_path: Path) -> Path:
    pool = tmp_path / "genes.json"
    pool.write_text(json.dumps([f"gene-{index}" for index in range(256)]) + "\n")
    genome = tmp_path / "genome.json"
    write_genome_receipt(build_sampled_genome(pool), genome)
    return genome


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_completion(path: Path, cohort_id: str) -> None:
    rows = []
    for index, tree_id in enumerate(ALL_TREE_IDS):
        n_runs = 53 if index == len(ALL_TREE_IDS) - 1 else 12
        rows.append(
            {
                "tree_id": tree_id,
                "valid": True,
                "ready_for_analysis": True,
                "n_runs": n_runs,
                "n_ledger_nodes": n_runs,
                "n_terminal_nodes": n_runs,
                "n_ok_nodes": n_runs,
                "n_failed_nodes": 0,
                "n_missing_nodes": 0,
                "n_errors": 0,
                "n_expected_artifacts": 2 * n_runs,
                "n_existing_artifacts": 2 * n_runs,
                "status_counts": {"completed": n_runs},
            }
        )
    path.write_text(
        json.dumps(
            {
                "cohort_id": cohort_id,
                "valid": True,
                "n_trees": 50,
                "n_ready": 50,
                "n_failed": 0,
                "n_in_progress": 0,
                "n_not_started": 0,
                "n_total_runs": 641,
                "n_terminal_nodes": 641,
                "n_ok_nodes": 641,
                "n_failed_nodes": 0,
                "n_missing_nodes": 0,
                "n_errors": 0,
                "n_warnings": 0,
                "rows": rows,
            }
        )
        + "\n"
    )


def test_generate_phylolm_alleles_uses_exact_sampling_grid() -> None:
    genes = [f"g-{index}" for index in range(128)]
    calls = {}
    seeds = []

    def factory(task: str, *, model: object, tokenizer: object):
        calls["factory"] = (task, model, tokenizer)

        def generate(prompts: list[str], **kwargs: object):
            calls["prompts"] = prompts
            calls["kwargs"] = kwargs
            return [[{"generated_text": prompt + "ABCDextra"}] for prompt in prompts]

        return generate

    tokenizer = _Tokenizer()
    alleles = generate_phylolm_alleles(
        object(),
        tokenizer,
        genes,
        pipeline_factory=factory,
        set_seed=seeds.append,
    )
    assert alleles.shape == (128, 32)
    assert np.all(alleles == "ABCD")
    assert seeds == [20260803]
    assert len(calls["prompts"]) == 4096
    assert calls["kwargs"] == {
        "do_sample": True,
        "min_new_tokens": 4,
        "max_new_tokens": 4,
        "temperature": 1.0,
        "return_full_text": True,
        "pad_token_id": 2,
        "batch_size": 64,
    }
    assert tokenizer.padding_side == "left"


def test_extract_raw_continuation_refuses_prompt_rewrite() -> None:
    with pytest.raises(ValueError, match="does not preserve"):
        extract_raw_continuation_alleles(["prompt"], [[{"generated_text": "other"}]])


def test_collect_leaf_population_binds_checkpoint_and_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    genome = _genome(tmp_path)
    completion = tmp_path / "completion.json"
    stage = tmp_path / "stage.sha256"
    cohort_id = "llama32_1b_full_finetune_legacy_causal_2000"
    _write_completion(completion, cohort_id)
    stage.write_text("stage\n")
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    (checkpoint_path / "config.json").write_text("{}\n")
    (checkpoint_path / "model.safetensors").write_text("weights\n")
    summary = tmp_path / "summary.json"
    run_list = tmp_path / "runs.jsonl"
    ledger = tmp_path / "ledger.jsonl"
    truth = tmp_path / "truth.jsonl"
    for path in (summary, run_list, ledger):
        path.write_text("source\n")
    truth.write_text(
        json.dumps(
            {
                "tree_id": "confirm_paper_tree_001",
                "node_id": "n1",
                "path": ["root", "n1"],
            }
        )
        + "\n"
    )
    leaf = ResolvedLeafCheckpoint(
        cohort_id=cohort_id,
        tree_id="confirm_paper_tree_001",
        leaf_id="n1",
        leaf_ordinal=0,
        method="full",
        artifact_name="model",
        checkpoint=checkpoint_path,
        base_model_id="meta-llama/Llama-3.2-1B",
        base_model_revision="4e20de36255befb501235df90d39eebbac5bd237",
        model_task="causal_lm",
        training_summary=summary,
        training_summary_sha256=_sha(summary),
        run_list=run_list,
        run_list_sha256=_sha(run_list),
        ledger=ledger,
        ledger_sha256=_sha(ledger),
        truth_manifest=truth,
        truth_manifest_sha256=_sha(truth),
    )
    monkeypatch.setattr(
        "weighttraits.behavior.phylolm_inference.generate_phylolm_alleles",
        lambda *_args, **_kwargs: np.full((128, 32), "ABCD", dtype=object),
    )

    def loader(_checkpoint: ResolvedLeafCheckpoint, _dtype: str):
        return object(), object(), {
            "backend": "huggingface_text_generation_pipeline",
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_artifact": "model",
            "tokenizer_model_id": "meta-llama/Llama-3.2-1B",
            "tokenizer_revision": "4e20de36255befb501235df90d39eebbac5bd237",
            "local_files_only": True,
            "torch_dtype": "auto",
            "torch_version": "test",
            "transformers_version": "test",
        }

    output = tmp_path / "population.json"
    payload = collect_phylolm_leaf_population(
        leaf,
        genome_receipt_path=genome,
        completion_receipt_path=completion,
        stage_manifest_path=stage,
        output_path=output,
        model_loader=loader,
    )
    assert output.is_file()
    assert payload["valid"] is True
    assert payload["model_id"] == "n1"
    assert payload["generation_receipt"]["completion_receipt"] == str(completion)
    assert payload["generation_receipt"]["stage_manifest"] == str(stage)
    assert payload["population"][0] == {"ABCD": 1.0}
