from pathlib import Path

import pytest

from weighttraits.training.data_formats import load_dataset_format_specs
from weighttraits.training.datasets import canonical_dataset_example, load_dataset_registry
from weighttraits.training.planner import load_training_config, resolve_prompt_template
from weighttraits.training.prompts import render_prompt


ROOT = Path(__file__).parents[1]
ASSETS = ROOT / "examples" / "training" / "confirm_paper_numbers"
HOLDOUT_ASSETS = ROOT / "examples" / "training" / "translation_holdout_20260803"
CONFIG = ASSETS / "llama32_1b_full_finetune_legacy_causal_2000.yaml"
LORA_CONFIGS = {
    8: ASSETS / "llama32_1b_lora_qkv_r8_legacy_causal_2000.yaml",
    64: ASSETS / "llama32_1b_lora_qkv_r64_legacy_causal_2000.yaml",
}
HOLDOUT_CONFIG = HOLDOUT_ASSETS / "llama32_1b_full_finetune_legacy_causal_2000.yaml"
FORMATS = ASSETS / "dataset_formats_legacy_causal.yaml"
REGISTRY = ASSETS / "dataset_registry_legacy_causal.yaml"


def test_fixed_step_protocol_matches_legacy_llama_optimizer_contract():
    config = load_training_config(CONFIG)
    trainer = config["trainer"]

    assert config["protocol_id"] == "legacy_causal_llama32_1b_full_ft_2000_seed42_v1"
    assert config["base_model"] == "meta-llama/Llama-3.2-1B"
    assert config["method"] == "full"
    assert trainer == {
        "model_task": "causal_lm",
        "model_dtype": "bfloat16",
        "padding_side": "right",
        "pad_to_multiple_of": 8,
        "causal_loss_scope": "completion",
        "min_prompt_tokens": 8,
        "require_max_steps": True,
        "seed": 42,
        "max_steps": 2000,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "learning_rate": 0.00002,
        "warmup_steps": 200,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "eval_steps": 250,
        "logging_steps": 10,
        "save_strategy": "no",
        "gradient_checkpointing": False,
        "ddp_find_unused_parameters": False,
        "bf16": True,
        "target_field": "target",
        "max_seq_length": 1024,
        "report_to": [],
    }
    assert config["stopping"] == {"enabled": False}


@pytest.mark.parametrize(("rank", "alpha"), [(8, 16), (64, 128)])
def test_llama_qkv_lora_overlays_change_only_the_historical_method_delta(rank, alpha):
    full = load_training_config(CONFIG)
    lora = load_training_config(LORA_CONFIGS[rank])
    expected_trainer = dict(full["trainer"], learning_rate=0.0003)

    assert lora["protocol_id"] == (
        f"legacy_causal_llama32_1b_lora_qkv_r{rank}_2000_seed42_v1"
    )
    assert lora["model_family"] == f"llama32_1b_lora_qkv_r{rank}_legacy_causal_2000"
    assert lora["method"] == "lora"
    assert lora["base_model"] == full["base_model"]
    assert lora["base_model_revision"] == full["base_model_revision"]
    assert lora["trainer"] == expected_trainer
    assert lora["prompt"] == full["prompt"]
    assert lora["stopping"] == full["stopping"]
    assert lora["lora"] == {
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj"],
        "merge_after_train": True,
    }
    assert lora["output_root"] == (
        f"outputs/confirm_paper_numbers/llama32_1b_lora_qkv_r{rank}_legacy_causal_2000"
    )


def test_no_translation_config_inherits_the_identical_training_protocol():
    ordinary = load_training_config(CONFIG)
    holdout = load_training_config(HOLDOUT_CONFIG)

    assert holdout["protocol_id"] == ordinary["protocol_id"]
    assert holdout["base_model"] == ordinary["base_model"]
    assert holdout["base_model_revision"] == ordinary["base_model_revision"]
    assert holdout["method"] == ordinary["method"]
    assert holdout["trainer"] == ordinary["trainer"]
    assert holdout["prompt"] == ordinary["prompt"]
    assert holdout["stopping"] == ordinary["stopping"]
    assert holdout["output_root"] != ordinary["output_root"]


def test_protocol_assets_cover_the_same_36_dataset_ids_without_char_filters():
    registry = load_dataset_registry(REGISTRY)
    formats = load_dataset_format_specs(FORMATS)
    qa_dataset_ids = (
        "squad",
        "squad_v2",
        "trivia_qa",
        "boolq",
        "quartz",
        "arc_easy",
        "arc_challenge",
        "sciq",
    )

    assert set(registry) == set(formats)
    assert len(registry) == 36
    assert all(entry.filter is None for entry in registry.values())
    assert {formats[dataset_id].prompt_fields for dataset_id in qa_dataset_ids} == {("qa_input",)}
    assert {
        dataset_id: formats[dataset_id].eval_split
        for dataset_id in ("quartz", "arc_easy", "arc_challenge", "sciq")
    } == {
        "quartz": "test",
        "arc_easy": "test",
        "arc_challenge": "test",
        "sciq": "test",
    }


@pytest.mark.parametrize(
    ("dataset_id", "raw", "expected_target"),
    [
        ("sst2", {"sentence": "good", "label": 1}, "positive"),
        ("ag_news", {"text": "markets", "label": 2}, "business"),
        (
            "mnli",
            {"premise": "p", "hypothesis": "h", "label": 2},
            "contradiction",
        ),
        ("imdb", {"text": "bad", "label": 0}, "negative"),
        (
            "rte",
            {"sentence1": "p", "sentence2": "h", "label": 1},
            "not_entailment",
        ),
        ("cola", {"sentence": "text", "label": 0}, "unacceptable"),
        (
            "qnli",
            {"question": "q", "sentence": "s", "label": 0},
            "entailment",
        ),
        ("yelp_polarity", {"text": "nice", "label": 1}, "positive"),
        ("dbpedia_14", {"content": "film", "label": 12}, "film"),
        (
            "trec",
            {"text": "How heavy?", "label_text": "weight"},
            "weight",
        ),
    ],
)
def test_classification_targets_are_human_readable_labels(dataset_id, raw, expected_target):
    spec = load_dataset_format_specs(FORMATS)[dataset_id]

    assert canonical_dataset_example(raw, spec)["target"] == expected_target


@pytest.mark.parametrize(
    (
        "dataset_id",
        "raw",
        "expected_target",
        "expected_choices",
        "expected_qa_input",
    ),
    [
        (
            "squad",
            {"question": "q", "context": "c", "answers": {"text": ["Ada"]}},
            "Ada",
            None,
            "question: q context: c",
        ),
        (
            "squad_v2",
            {"question": "q", "context": "c", "answers": {"text": []}},
            "unanswerable",
            None,
            "question: q context: c",
        ),
        (
            "trivia_qa",
            {"question": "q", "answer": {"value": "Ada"}},
            "Ada",
            None,
            "question: q",
        ),
        (
            "boolq",
            {"question": "q", "passage": "c", "answer": True},
            "yes",
            None,
            "question: q context: c",
        ),
        (
            "quartz",
            {
                "question": "q",
                "para": "c",
                "choices": {"label": ["A", "B"], "text": ["wrong", "right"]},
                "answerKey": "B",
            },
            "right",
            "A) wrong B) right",
            "question: q choices: A) wrong B) right context: c",
        ),
        (
            "arc_easy",
            {
                "question": "q",
                "choices": {"label": ["A", "B"], "text": ["wrong", "right"]},
                "answerKey": "B",
            },
            "right",
            "A) wrong B) right",
            "question: q choices: A) wrong B) right",
        ),
        (
            "arc_challenge",
            {
                "question": "q",
                "choices": {"label": ["A", "B"], "text": ["wrong", "right"]},
                "answerKey": "B",
            },
            "right",
            "A) wrong B) right",
            "question: q choices: A) wrong B) right",
        ),
        (
            "sciq",
            {"question": "q", "support": "c", "correct_answer": "right"},
            "right",
            None,
            "question: q context: c",
        ),
    ],
)
def test_qa_targets_and_choices_match_legacy_semantics(
    dataset_id,
    raw,
    expected_target,
    expected_choices,
    expected_qa_input,
):
    spec = load_dataset_format_specs(FORMATS)[dataset_id]

    canonical = canonical_dataset_example(raw, spec)

    assert canonical["target"] == expected_target
    assert canonical["qa_input"] == expected_qa_input
    if expected_choices is not None:
        assert canonical["choices"] == expected_choices
    assert canonical_dataset_example(canonical, spec)["qa_input"] == expected_qa_input


@pytest.mark.parametrize(
    ("support", "expected_qa_input"),
    [
        ("", "question: q"),
        ("supporting text", "question: q context: supporting text"),
        (" ", "question: q context:  "),
    ],
)
def test_sciq_context_matches_legacy_raw_truthiness(support, expected_qa_input):
    spec = load_dataset_format_specs(FORMATS)["sciq"]

    canonical = canonical_dataset_example(
        {"question": "q", "support": support, "correct_answer": "right"},
        spec,
    )

    assert canonical["qa_input"] == expected_qa_input


@pytest.mark.parametrize(
    ("context", "expected_qa_input"),
    [
        ("", "question: q choices: A) wrong B) right"),
        ("c", "question: q choices: A) wrong B) right context: c"),
    ],
)
def test_multiple_choice_qa_always_includes_choices_and_only_truthy_context(
    context,
    expected_qa_input,
):
    spec = load_dataset_format_specs(FORMATS)["quartz"]

    canonical = canonical_dataset_example(
        {
            "question": "q",
            "para": context,
            "choices": {"label": ["A", "B"], "text": ["wrong", "right"]},
            "answerKey": "B",
        },
        spec,
    )

    assert canonical["qa_input"] == expected_qa_input


def test_all_legacy_qa_datasets_resolve_one_shared_task_template():
    config = load_training_config(CONFIG)
    qa_dataset_ids = (
        "squad",
        "squad_v2",
        "trivia_qa",
        "boolq",
        "quartz",
        "arc_easy",
        "arc_challenge",
        "sciq",
    )
    expected = "Answer the following question.\n\n{qa_input}\n\nAnswer:"

    assert config["prompt"]["task_templates"]["qa"] == expected
    assert not set(qa_dataset_ids) & set(config["prompt"]["dataset_templates"])
    for dataset_id in qa_dataset_ids:
        resolution = resolve_prompt_template(
            {"task_family": "qa", "dataset_id": dataset_id},
            config,
        )
        assert resolution.template == expected
        assert resolution.source == "prompt.task"


def test_representative_prompts_match_legacy_causal_strings_before_tokenizer_newline():
    config = load_training_config(CONFIG)
    cases = [
        (
            "summarization",
            "xsum",
            {"document": "doc"},
            "Summarize the following document.\n\nDocument:\ndoc\n\nSummary:",
        ),
        (
            "classification",
            "sst2",
            {"text": "good"},
            "Classify the following text into one of: negative, positive.\n\nText: good\n\nLabel:",
        ),
        (
            "classification",
            "mnli",
            {"text": "p", "hypothesis": "h"},
            "Classify the following text pair into one of: entailment, neutral, "
            "contradiction.\n\nText: p\nHypothesis: h\n\nLabel:",
        ),
        (
            "qa",
            "quartz",
            {"qa_input": "question: q choices: A) x B) y context: c"},
            "Answer the following question.\n\nquestion: q choices: A) x B) y context: c\n\n"
            "Answer:",
        ),
        (
            "translation",
            "opus_en_ja",
            {"source_text": "hello"},
            "Translate the following English text to Japanese.\n\nText:\nhello\n\nTranslation:",
        ),
    ]

    for task_family, dataset_id, example, expected in cases:
        resolution = resolve_prompt_template(
            {"task_family": task_family, "dataset_id": dataset_id},
            config,
        )
        assert render_prompt(resolution.template, example) == expected
