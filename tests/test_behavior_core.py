import json
from pathlib import Path

import numpy as np

from weighttraits.behavior.distances import paired_cosine_distances
from weighttraits.behavior.meta import CorrelationEffect, dersimonian_laird_correlations
from weighttraits.behavior.phylolm import (
    compute_population,
    nei_distance_matrix,
    nei_similarity,
    sample_genes,
    save_population,
    write_phylolm_analysis,
)
from weighttraits.behavior.phylolm_inference import extract_pipeline_alleles
from weighttraits.behavior.probe_inference import (
    build_embedding_tensor,
    extract_behavior_pipeline_outputs,
    generate_causal_responses,
    generate_seq2seq_responses,
)
from weighttraits.behavior.probes import (
    heldout_probe_prompts_from_rows,
    hellaswag_prompts_from_rows,
)
from weighttraits.behavior.regression import paired_distance_rows
from weighttraits.behavior.responses import (
    BehaviorResponse,
    audit_behavior_responses,
    write_behavior_responses,
)
from weighttraits.behavior.surface import (
    classify_response_text,
    paired_surface_distances,
    response_surface_features,
)
from weighttraits.cli import build_parser
from weighttraits.distances.streaming import DistanceCube, write_distance_cube
from weighttraits.phylo.newick import leaf_names, parse_newick


def _response(model_id: str, prompt_id: str, sample_id: int, response: str = "ok"):
    return BehaviorResponse(
        run_id="tree-001",
        model_id=model_id,
        probe_id="hellaswag",
        prompt_id=prompt_id,
        sample_id=sample_id,
        prompt=f"prompt {prompt_id}",
        response=response,
    )


def test_behavior_response_audit_checks_rectangular_samples_and_empty_text() -> None:
    valid = [
        _response(model_id, prompt_id, sample_id)
        for model_id in ("n1", "n2")
        for prompt_id in ("p1", "p2")
        for sample_id in (0, 1)
    ]
    report = audit_behavior_responses(valid, expected_samples_per_prompt=2)
    assert report["valid"]
    assert report["n_records"] == 8
    assert report["n_models"] == 2
    assert report["quality"]["unique_response_fraction"] == 0.125
    assert report["quality"]["mean_response_characters"] == 2.0
    assert report["quality"]["dominant_response_fraction_by_model"] == {
        "n1": 1.0,
        "n2": 1.0,
    }
    assert report["quality"]["mean_unique_draws_per_prompt_by_model"] == {
        "n1": 1.0,
        "n2": 1.0,
    }
    assert report["quality"]["varying_prompt_fraction_by_model"] == {
        "n1": 0.0,
        "n2": 0.0,
    }

    broken = valid[:-1] + [_response("n1", "p1", 0), _response("n2", "p1", 0, "")]
    report = audit_behavior_responses(broken, expected_samples_per_prompt=2)
    assert not report["valid"]
    assert "duplicate response keys" in report["issues"]
    assert "completed responses with empty text" in report["issues"]
    assert "incomplete prompt/sample grids" in report["issues"]

    dropped = valid[:-1] + [
        BehaviorResponse(
            run_id="tree-001",
            model_id="n2",
            probe_id="hellaswag",
            prompt_id="p2",
            sample_id=1,
            prompt="prompt p2",
            response=None,
            status="dropped",
            reason="empty_generation",
        )
    ]
    report = audit_behavior_responses(dropped, expected_samples_per_prompt=2)
    assert not report["valid"]
    assert "non-completed response records" in report["issues"]


def test_behavior_quality_audit_scores_reference_overlap() -> None:
    record = BehaviorResponse(
        run_id="tree-001",
        model_id="n1",
        probe_id="hellaswag",
        prompt_id="p1",
        sample_id=0,
        prompt="continue",
        response="the dog runs",
        reference="the dog sleeps",
    )
    quality = audit_behavior_responses([record])["quality"]
    assert quality["n_reference_scored"] == 1
    assert abs(quality["mean_reference_rouge_l_f1"] - 2 / 3) < 1e-12


def test_surface_phenotypes_preserve_empty_and_collapsed_output() -> None:
    assert classify_response_text("") == "empty"
    assert classify_response_text("0.0.0.0") == "label_only"
    assert classify_response_text("True True True True True") == "label_only"
    assert classify_response_text("word word word word other") == "repetition_loop"
    assert classify_response_text("The model gives a complete natural language response.") == (
        "natural_language"
    )
    assert classify_response_text("perhaps") == "fragment"
    assert response_surface_features("").shape == (10,)

    records = [
        _response("n1", "p1", 0, "A"),
        BehaviorResponse(
            run_id="tree-001",
            model_id="n1",
            probe_id="hellaswag",
            prompt_id="p2",
            sample_id=0,
            prompt="prompt p2",
            response=None,
            status="dropped",
            reason="empty_generation",
        ),
        _response("n2", "p1", 0, "This is a complete natural language answer."),
        _response("n2", "p2", 0, "Another complete answer written using natural language."),
    ]
    result = paired_surface_distances(records)
    assert result.observation_counts[0, 1] == 2
    assert np.isfinite(result.distances[0, 1]) and result.distances[0, 1] > 0
    assert result.traits_by_model["n1"]["class_counts"]["empty"] == 1
    assert result.traits_by_model["n1"]["semantic_coverage"] == 0.0
    assert result.traits_by_model["n2"]["semantic_coverage"] == 1.0


def test_paired_cosine_distances_use_only_shared_observations() -> None:
    embeddings = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        ]
    )
    valid = np.asarray([[True, True, False], [True, True, True]])
    result = paired_cosine_distances(
        embeddings,
        model_ids=["n1", "n2"],
        observation_ids=["p1:0", "p2:0", "p3:0"],
        valid_mask=valid,
    )

    assert result.observation_counts[0, 1] == 2
    assert result.distances[0, 1] == 0.5
    assert result.audit["usable_observations_by_model"] == {"n1": 2, "n2": 3}


def test_dersimonian_laird_pooling_preserves_identical_correlations() -> None:
    result = dersimonian_laird_correlations(
        [
            CorrelationEffect(run_id="tree-001", r=-0.25, n_pairs=20),
            CorrelationEffect(run_id="tree-002", r=-0.25, n_pairs=30),
            CorrelationEffect(run_id="tree-003", r=-0.25, n_pairs=40),
        ]
    )

    assert abs(result.r + 0.25) < 1e-12
    assert result.tau_squared == 0.0
    assert result.ci_low < result.r < result.ci_high
    assert result.n_runs == 3
    assert result.total_pairs == 90


def test_phylolm_population_nei_distance_and_analysis_outputs(tmp_path: Path) -> None:
    left = compute_population([["aa", "aa", "bb", "bb"], ["cc", "cc", "cc", "cc"]])
    same = compute_population([["aa", "aa", "bb", "bb"], ["cc", "cc", "cc", "cc"]])
    other = compute_population([["xx", "xx", "yy", "yy"], ["zz", "zz", "zz", "zz"]])
    assert nei_similarity(left, same) == 1.0
    assert nei_similarity(left, other) == 0.0

    ids, similarity, distance = nei_distance_matrix(
        {"n1": left, "n2": same, "n3": other}, eps=1e-3
    )
    assert ids == ["n1", "n2", "n3"]
    assert similarity[0, 1] == 1.0
    assert distance[0, 1] == 0.0
    assert abs(distance[0, 2] + np.log(1e-3)) < 1e-12

    populations = tmp_path / "populations"
    save_population(populations / "n1.json", model_id="n1", population=left)
    save_population(populations / "n2.json", model_id="n2", population=same)
    save_population(populations / "n3.json", model_id="n3", population=other)
    out = tmp_path / "analysis"
    audit = write_phylolm_analysis(populations, out)

    assert audit["n_models"] == 3
    assert (out / "distance_cube" / "distance_cube.npz").exists()
    assert leaf_names(parse_newick((out / "phylolm_nj.newick").read_text())) == ids
    assert json.loads((out / "phylolm_audit.json").read_text())["n_genes"] == 2


def test_phylolm_gene_sampling_is_deterministic() -> None:
    pool = [f"gene-{index}" for index in range(20)]
    assert sample_genes(pool, n_genes=5, seed=7) == sample_genes(
        pool, n_genes=5, seed=7
    )
    assert len(set(sample_genes(pool, n_genes=5, seed=7))) == 5


def test_phylolm_pipeline_alleles_use_raw_prompt_suffix_characters() -> None:
    prompts = ["gene one", "gene two"]
    outputs = [
        [{"generated_text": "gene oneABCD extra"}],
        {"generated_text": "gene twoé12345"},
    ]
    assert extract_pipeline_alleles(prompts, outputs, allele_characters=4) == ["ABCD", "é123"]


def test_hellaswag_prompt_materialization_is_deterministic_and_task_specific() -> None:
    rows = [
        {
            "activity_label": f"activity {index}",
            "ctx": f"context {index}",
            "endings": [f"ending {index}-{choice}" for choice in range(4)],
            "label": str(index % 4),
        }
        for index in range(10)
    ]
    seq = hellaswag_prompts_from_rows(rows, model_task="seq2seq", n_prompts=4, seed=9)
    causal = hellaswag_prompts_from_rows(rows, model_task="causal_lm", n_prompts=4, seed=9)
    assert [prompt.metadata["source_index"] for prompt in seq] == [
        prompt.metadata["source_index"] for prompt in causal
    ]
    assert seq[0].prompt.startswith("complete:")
    assert causal[0].prompt.startswith("Complete the following sentence.")
    source = seq[0].metadata["source_index"]
    assert seq[0].reference == rows[source]["endings"][int(rows[source]["label"])]


def test_additional_heldout_probe_prompt_contracts() -> None:
    mmlu = heldout_probe_prompts_from_rows(
        [{"question": "Q?", "choices": ["one", "two", "three", "four"], "answer": 2, "subject": "x"}],
        probe_id="mmlu",
        model_task="causal_lm",
        n_prompts=1,
        seed=42,
    )[0]
    assert "A) one" in mmlu.prompt and "D) four" in mmlu.prompt
    assert mmlu.reference == "three"
    assert mmlu.metadata["subject"] == "x"

    arc = heldout_probe_prompts_from_rows(
        [{"question": "Science?", "choices": {"label": ["1", "2"], "text": ["no", "yes"]}, "answerKey": "2"}],
        probe_id="arc_challenge",
        model_task="seq2seq",
        n_prompts=1,
        seed=42,
    )[0]
    assert arc.prompt == "answer: Science?\n1) no\n2) yes"
    assert arc.reference == "yes"

    truthful = heldout_probe_prompts_from_rows(
        [{"question": "What is true?", "best_answer": "This."}],
        probe_id="truthfulqa",
        model_task="causal_lm",
        n_prompts=1,
        seed=42,
    )[0]
    assert truthful.prompt.endswith("Question: What is true?\n\nAnswer:\n")
    assert truthful.reference == "This."

    rescue = heldout_probe_prompts_from_rows(
        [{"question": "Q?", "choices": ["one", "two"], "answer": 1}],
        probe_id="mmlu",
        model_task="causal_lm",
        n_prompts=1,
        seed=42,
        prompt_style="explanation_first",
    )[0]
    assert "Do not reply with only a letter" in rescue.prompt
    assert rescue.metadata["prompt_style"] == "explanation_first"

    dolly = heldout_probe_prompts_from_rows(
        [
            {
                "instruction": "Write a short scene about rain.",
                "context": "",
                "response": "Rain tapped against the window.",
                "category": "creative_writing",
            },
            {
                "instruction": "Classify this.",
                "context": "x",
                "response": "y",
                "category": "classification",
            },
        ],
        probe_id="dolly_open_ended",
        model_task="causal_lm",
        n_prompts=1,
        seed=42,
    )[0]
    assert "Request: Write a short scene about rain." in dolly.prompt
    assert "A)" not in dolly.prompt and "Answer:" not in dolly.prompt
    assert dolly.metadata["category"] == "creative_writing"
    assert dolly.metadata["source_index"] == 0


def test_behavior_output_extraction_and_embedding_alignment() -> None:
    assert extract_behavior_pipeline_outputs(
        ["prompt"], [[{"generated_text": "prompt answer"}]], model_task="causal_lm"
    ) == ["answer"]
    assert extract_behavior_pipeline_outputs(
        ["prompt"], [{"generated_text": "summary"}], model_task="seq2seq"
    ) == ["summary"]

    records = [
        _response(model_id, prompt_id, 0, response)
        for model_id, responses in (("n1", ("a", "b")), ("n2", ("a", "c")))
        for prompt_id, response in zip(("p1", "p2"), responses)
    ]
    vectors = {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [-1.0, 0.0]}
    embeddings, model_ids, observation_ids, valid, audit = build_embedding_tensor(
        records,
        encode=lambda texts: np.asarray([vectors[text] for text in texts]),
    )
    assert model_ids == ["n1", "n2"]
    assert observation_ids == ["p1:0", "p2:0"]
    assert embeddings.shape == (2, 2, 2)
    assert valid.all()
    assert audit["n_valid_embeddings"] == 4

    natural_records = [
        _response("n1", "p1", 0, "This is one complete natural language response."),
        _response("n1", "p2", 0, "A"),
        _response("n2", "p1", 0, "This is another complete natural language response."),
        _response("n2", "p2", 0, "0"),
    ]
    natural_embeddings, _, _, natural_valid, natural_audit = build_embedding_tensor(
        natural_records,
        encode=lambda texts: np.ones((len(texts), 2)),
        natural_language_only=True,
    )
    assert natural_embeddings.shape == (2, 2, 2)
    assert natural_valid.sum() == 2
    assert natural_audit["semantic_coverage_by_model"] == {"n1": 0.5, "n2": 0.5}


def test_seq2seq_generation_uses_direct_batched_generate() -> None:
    import torch

    class Tokenizer:
        pad_token_id = 0

        def __call__(self, texts, *, padding, return_tensors):
            assert padding and return_tensors == "pt"
            return {"input_ids": torch.ones((len(texts), 2), dtype=torch.long)}

        def batch_decode(self, values, *, skip_special_tokens):
            assert skip_special_tokens
            return [f" response-{int(row[0])} " for row in values]

    class Model:
        device = torch.device("cpu")

        def generate(self, input_ids, **kwargs):
            assert kwargs["do_sample"]
            assert kwargs["top_p"] == 0.9
            return torch.arange(1, len(input_ids) + 1).reshape(-1, 1)

    assert generate_seq2seq_responses(
        Model(),
        Tokenizer(),
        ["a", "b", "c"],
        samples_per_batch=2,
        min_new_tokens=1,
        max_new_tokens=4,
        do_sample=True,
        temperature=1.0,
        top_p=0.9,
    ) == ["response-1", "response-2", "response-1"]


def test_causal_generation_decodes_only_new_tokens() -> None:
    import torch

    class Tokenizer:
        pad_token_id = 0

        def __call__(self, texts, **kwargs):
            assert kwargs["truncation"] and kwargs["max_length"] == 512
            return {"input_ids": torch.ones((len(texts), 2), dtype=torch.long)}

        def batch_decode(self, values, **kwargs):
            assert kwargs["skip_special_tokens"]
            assert not kwargs["clean_up_tokenization_spaces"]
            return [f" answer-{int(row[0])} " for row in values]

    class Model:
        device = torch.device("cpu")

        def generate(self, input_ids, **kwargs):
            assert "top_p" not in kwargs
            suffix = torch.arange(1, len(input_ids) + 1).reshape(-1, 1)
            return torch.cat([input_ids, suffix], dim=1)

    assert generate_causal_responses(
        Model(),
        Tokenizer(),
        ["a", "b"],
        samples_per_batch=2,
        min_new_tokens=1,
        max_new_tokens=4,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
    ) == ["answer-1", "answer-2"]


def test_behavior_and_phylolm_cli_parsers(tmp_path: Path) -> None:
    parser = build_parser()
    audit = parser.parse_args(
        [
            "audit-behavior-responses",
            "--responses",
            str(tmp_path / "responses.jsonl"),
            "--expected-samples-per-prompt",
            "3",
        ]
    )
    assert audit.expected_samples_per_prompt == 3

    genome = parser.parse_args(
        [
            "make-phylolm-genome",
            "--gene-pool",
            str(tmp_path / "genes.json"),
            "--out",
            str(tmp_path / "genome.json"),
        ]
    )
    assert genome.n_genes == 128

    analysis = parser.parse_args(
        [
            "build-phylolm-analysis",
            "--populations",
            str(tmp_path / "populations"),
            "--out",
            str(tmp_path / "analysis"),
        ]
    )
    assert analysis.eps == 1e-3

    collect = parser.parse_args(
        [
            "collect-phylolm-population",
            "--input-manifest",
            str(tmp_path / "leaves.yaml"),
            "--model-index",
            "2",
            "--genome",
            str(tmp_path / "genome.json"),
            "--out-dir",
            str(tmp_path / "populations"),
            "--base-model",
            "meta-llama/Llama-3.2-1B",
        ]
    )
    assert collect.model_index == 2
    assert collect.samples_per_gene == 32
    assert collect.local_files_only

    prepare = parser.parse_args(
        [
            "prepare-hellaswag-probe",
            "--model-task",
            "seq2seq",
            "--out",
            str(tmp_path / "prompts.jsonl"),
        ]
    )
    assert prepare.n_prompts == 100
    assert prepare.local_files_only

    prepare_panel = parser.parse_args(
        [
            "prepare-behavior-probe",
            "--probe",
            "arc_challenge",
            "--model-task",
            "causal_lm",
            "--out",
            str(tmp_path / "arc.jsonl"),
        ]
    )
    assert prepare_panel.probe == "arc_challenge"
    assert prepare_panel.n_prompts == 100
    assert prepare_panel.prompt_style == "standard"

    behavior_collect = parser.parse_args(
        [
            "collect-behavior-responses",
            "--input-manifest",
            str(tmp_path / "leaves.yaml"),
            "--model-id",
            "n3",
            "--run-id",
            "tree-001",
            "--prompts",
            str(tmp_path / "prompts.jsonl"),
            "--out-dir",
            str(tmp_path / "responses"),
            "--model-task",
            "seq2seq",
        ]
    )
    assert behavior_collect.samples_per_prompt == 3
    assert behavior_collect.min_new_tokens == 0
    assert behavior_collect.do_sample
    assert behavior_collect.temperature == 1.0
    assert behavior_collect.top_p == 1.0
    assert behavior_collect.empty_policy == "drop"

    behavior_embed = parser.parse_args(
        [
            "embed-behavior-responses",
            "--responses-dir",
            str(tmp_path / "responses"),
            "--out",
            str(tmp_path / "embeddings.npz"),
        ]
    )
    assert behavior_embed.local_files_only
    assert not behavior_embed.natural_language_only
    assert behavior_embed.probe_id is None

    surface = parser.parse_args(
        [
            "build-behavior-surface-distances",
            "--responses-dir",
            str(tmp_path / "responses"),
            "--out",
            str(tmp_path / "surface"),
        ]
    )
    assert not surface.allow_incomplete_pairs
    assert surface.probe_id is None


def test_behavior_surface_cli_selects_one_probe(tmp_path: Path) -> None:
    responses = tmp_path / "responses"
    rows = []
    for model_id, text in (("n1", "A"), ("n2", "A complete answer.")):
        first = _response(model_id, "p1", 0, response=text)
        second = BehaviorResponse(
            run_id=first.run_id,
            model_id=first.model_id,
            probe_id="second_probe",
            prompt_id="q1",
            sample_id=0,
            prompt="Other prompt",
            response="Other response",
            reference=None,
            status="completed",
            reason=None,
            seed=first.seed,
            metadata=first.metadata,
        )
        write_behavior_responses([first, second], responses / f"{model_id}.jsonl")
        rows.extend((first, second))

    output = tmp_path / "surface"
    args = build_parser().parse_args(
        [
            "build-behavior-surface-distances",
            "--responses-dir",
            str(responses),
            "--probe-id",
            rows[0].probe_id,
            "--out",
            str(output),
        ]
    )
    assert args.func(args) == 0
    audit = json.loads((output / "audit.json").read_text())
    assert audit["probe_id"] == rows[0].probe_id
    assert audit["n_observations"] == 1


def test_behavior_cli_audits_a_response_directory(tmp_path: Path) -> None:
    responses = tmp_path / "responses"
    write_behavior_responses([_response("n1", "p1", 0)], responses / "n1.jsonl")
    write_behavior_responses([_response("n2", "p1", 0)], responses / "n2.jsonl")
    output = tmp_path / "audit.json"
    args = build_parser().parse_args(
        [
            "audit-behavior-responses",
            "--responses",
            str(responses),
            "--expected-samples-per-prompt",
            "1",
            "--out",
            str(output),
        ]
    )
    assert args.func(args) == 0
    report = json.loads(output.read_text())
    assert report["n_models"] == 2
    assert len(report["source_files"]) == 2


def test_behavior_regression_pairs_align_model_ids(tmp_path: Path) -> None:
    weight_dir = tmp_path / "weight"
    behavior_dir = tmp_path / "behavior"
    write_distance_cube(
        DistanceCube(
            distances={"cosine": np.asarray([[[0, 0.2, 0.4], [0.2, 0, 0.6], [0.4, 0.6, 0]]])},
            layer_names=["mean"],
            model_ids=["n1", "n2", "n3"],
            audit={},
        ),
        weight_dir,
    )
    write_distance_cube(
        DistanceCube(
            distances={
                "semantic_paired": np.asarray(
                    [[[0, 0.7, 0.8], [0.7, 0, 0.9], [0.8, 0.9, 0]]]
                )
            },
            layer_names=["paired"],
            model_ids=["n3", "n1", "n4"],
            audit={},
        ),
        behavior_dir,
    )
    rows, audit = paired_distance_rows(
        run_id="tree-001",
        weight_cube=weight_dir,
        behavior_cube=behavior_dir,
        weight_metric="cosine",
    )
    assert rows == [
        {
            "run_id": "tree-001",
            "pair_id": "n1::n3",
            "model_a": "n1",
            "model_b": "n3",
            "weight_distance": 0.4,
            "behavior_distance": 0.7,
            "behavior_similarity": 0.30000000000000004,
        }
    ]
    assert audit["n_common_models"] == 2
    assert audit["weight_only_model_ids"] == ["n2"]
    assert audit["behavior_only_model_ids"] == ["n4"]
