from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import weighttraits.training.executor as executor_module
from weighttraits.training.data_formats import DatasetFormatSpec
from weighttraits.training.datasets import DatasetRegistryEntry
from weighttraits.training.executor import (
    BackendTrainResult,
    HfPeftTrainingBackend,
    PreparedTrainingData,
    RenderedTrainingRecord,
    _apply_tokenizer_padding_side,
    _CausalDataCollator,
    _data_collator,
    _hf_dataset_from_records,
    _model_pretrained_kwargs,
    _pretrained_kwargs,
    _tokenize_causal_batch,
    _training_arguments,
    run_training_run,
)
from weighttraits.training.ledger import load_ledger_events
from weighttraits.training.planner import build_training_jobs
from weighttraits.training.runlist import build_training_run_list


class _CharacterTokenizer:
    eos_token_id = 99

    def __call__(
        self,
        value,
        *,
        add_special_tokens=True,
        truncation=False,
        max_length=None,
    ):
        def encode(text):
            ids = ([1] if add_special_tokens else []) + [ord(char) % 50 + 2 for char in text]
            return ids[:max_length] if truncation and max_length is not None else ids

        if isinstance(value, list):
            return {"input_ids": [encode(text) for text in value]}
        return {"input_ids": encode(value)}


def test_min_prompt_tokens_reserves_exact_budget_for_long_completion():
    tokenizer = _CharacterTokenizer()
    target = "abcdefghijklmnopqrst"

    encoded = _tokenize_causal_batch(
        tokenizer,
        ["a prompt that is longer than eight tokens"],
        [target],
        max_length=16,
        loss_scope="completion",
        min_prompt_tokens=8,
    )
    expected_completion = tokenizer(target, add_special_tokens=False)["input_ids"][:8]

    assert len(encoded["input_ids"][0]) == 16
    assert encoded["labels"][0][:8] == [-100] * 8
    assert encoded["labels"][0][8:] == expected_completion


@pytest.mark.parametrize("min_prompt_tokens", [-1, 16, True, 1.5])
def test_min_prompt_tokens_rejects_invalid_values(min_prompt_tokens):
    with pytest.raises(ValueError, match="min_prompt_tokens"):
        _tokenize_causal_batch(
            _CharacterTokenizer(),
            ["prompt"],
            ["target"],
            max_length=16,
            loss_scope="completion",
            min_prompt_tokens=min_prompt_tokens,
        )


class _MappedDataset:
    def __init__(self, rows):
        self.rows = rows

    @classmethod
    def from_list(cls, rows):
        return cls(rows)

    def map(self, function, *, batched, remove_columns):
        assert batched is True
        assert remove_columns == ["text", "target"]
        return function(
            {
                "text": [row["text"] for row in self.rows],
                "target": [row["target"] for row in self.rows],
            }
        )


def test_hf_dataset_plumbing_passes_min_prompt_tokens():
    encoded = _hf_dataset_from_records(
        {"Dataset": _MappedDataset},
        _CharacterTokenizer(),
        "causal_lm",
        (RenderedTrainingRecord(text="a sufficiently long prompt", target="x" * 30),),
        {
            "trainer": {
                "max_seq_length": 16,
                "causal_loss_scope": "completion",
                "min_prompt_tokens": 8,
            }
        },
    )

    assert encoded["labels"][0][:8] == [-100] * 8
    assert len(encoded["labels"][0]) == 16


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_tokenizer_padding_side_is_applied_when_requested(padding_side):
    tokenizer = SimpleNamespace(padding_side="right")

    _apply_tokenizer_padding_side(
        tokenizer,
        {"trainer": {"padding_side": padding_side}},
    )

    assert tokenizer.padding_side == padding_side


@pytest.mark.parametrize("padding_side", ["middle", "", None, 1])
def test_tokenizer_padding_side_rejects_invalid_values(padding_side):
    with pytest.raises(ValueError, match="padding_side"):
        _apply_tokenizer_padding_side(
            SimpleNamespace(padding_side="right"),
            {"trainer": {"padding_side": padding_side}},
        )


class _MultiplePadTokenizer:
    padding_side = "right"

    def __init__(self):
        self.seen_pad_to_multiple_of = None

    def pad(self, features, *, padding, return_tensors, pad_to_multiple_of):
        import torch

        assert padding is True
        assert return_tensors == "pt"
        self.seen_pad_to_multiple_of = pad_to_multiple_of
        width = max(len(feature["input_ids"]) for feature in features)
        width = ((width + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

        def right_pad(row, value):
            return row + [value] * (width - len(row))

        return {
            "input_ids": torch.tensor([right_pad(feature["input_ids"], 0) for feature in features]),
            "attention_mask": torch.tensor(
                [right_pad(feature["attention_mask"], 0) for feature in features]
            ),
        }


def test_causal_collator_right_pads_inputs_and_labels_to_multiple_of_eight():
    pytest.importorskip("torch", reason="the causal collator requires torch")
    tokenizer = _MultiplePadTokenizer()
    collator = _CausalDataCollator(tokenizer, pad_to_multiple_of=8)

    batch = collator(
        [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [11, 12, 13]},
            {
                "input_ids": [1, 2, 3, 4, 5],
                "attention_mask": [1, 1, 1, 1, 1],
                "labels": [21, 22, 23, 24, 25],
            },
        ]
    )

    assert tokenizer.seen_pad_to_multiple_of == 8
    assert tuple(batch["input_ids"].shape) == (2, 8)
    assert batch["labels"][0].tolist() == [11, 12, 13] + [-100] * 5
    assert batch["labels"][1].tolist() == [21, 22, 23, 24, 25] + [-100] * 3


@pytest.mark.parametrize("pad_to_multiple_of", [0, -1, True, 1.5, "8"])
def test_causal_collator_rejects_invalid_padding_multiple(pad_to_multiple_of):
    with pytest.raises(ValueError, match="pad_to_multiple_of"):
        _CausalDataCollator(
            _MultiplePadTokenizer(),
            pad_to_multiple_of=pad_to_multiple_of,
        )


class _RecordingSeq2SeqCollator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_seq2seq_collator_honors_legacy_padding_multiple_and_label_mask():
    tokenizer = object()
    model = object()

    collator = _data_collator(
        {"DataCollatorForSeq2Seq": _RecordingSeq2SeqCollator},
        tokenizer,
        model,
        "seq2seq",
        {"trainer": {"pad_to_multiple_of": 8}},
    )

    assert collator.kwargs == {
        "tokenizer": tokenizer,
        "model": model,
        "label_pad_token_id": -100,
        "pad_to_multiple_of": 8,
    }


def test_seq2seq_collator_omits_unconfigured_padding_multiple():
    collator = _data_collator(
        {"DataCollatorForSeq2Seq": _RecordingSeq2SeqCollator},
        object(),
        object(),
        "seq2seq",
        {"trainer": {}},
    )

    assert collator.kwargs["label_pad_token_id"] == -100
    assert "pad_to_multiple_of" not in collator.kwargs


@pytest.mark.parametrize("pad_to_multiple_of", [0, -1, True, 1.5, "8"])
def test_seq2seq_collator_rejects_invalid_padding_multiple(pad_to_multiple_of):
    with pytest.raises(ValueError, match="pad_to_multiple_of"):
        _data_collator(
            {"DataCollatorForSeq2Seq": _RecordingSeq2SeqCollator},
            object(),
            object(),
            "seq2seq",
            {"trainer": {"pad_to_multiple_of": pad_to_multiple_of}},
        )


class _StepBackend:
    def __init__(self, step, *, status="completed"):
        self.step = step
        self.status = status

    def train(self, run, data, event_callback):
        return BackendTrainResult(
            status=self.status,
            step=self.step,
            artifacts=dict(run.expected_artifacts),
        )


def _guarded_run(tmp_path: Path, *, require_max_steps=True, max_steps=4):
    rows = [
        {
            "node_id": "n0",
            "parent_id": "root",
            "depth": 1,
            "path": ["root", "n0"],
            "grow": "train",
            "task_family": "qa",
            "dataset_id": "tiny_qa",
        }
    ]
    jobs = build_training_jobs(
        rows,
        {
            "base_model": "base",
            "method": "full",
            "output_root": str(tmp_path / "outputs"),
            "trainer": {
                "max_steps": max_steps,
                "require_max_steps": require_max_steps,
                "min_prompt_tokens": 8,
                "target_field": "target",
            },
            "prompt": {"default_template": "Question: {question}\nAnswer:"},
        },
    )
    return build_training_run_list(
        jobs,
        ledger_path=tmp_path / "training_ledger.jsonl",
    ).runs[0]


def _registry():
    return {
        "tiny_qa": DatasetRegistryEntry(
            dataset_id="tiny_qa",
            task_family="qa",
            hf_args=("tiny_qa",),
            train_split="train",
        )
    }


def _formats():
    return {
        "tiny_qa": DatasetFormatSpec(
            dataset_id="tiny_qa",
            task_family="qa",
            prompt_fields=("question",),
            field_map={"target": "answer"},
            train_split="train",
        )
    }


def _loader(*args):
    assert args == ("tiny_qa",)
    return {"train": [{"question": "Q", "answer": "A"}]}


@pytest.mark.parametrize(
    ("model_dtype", "expected_attr"),
    [
        ("bfloat16", "bfloat16"),
        ("float16", "float16"),
        ("float32", "float32"),
    ],
)
def test_model_dtype_is_applied_to_root_and_local_lineage_loads(
    tmp_path,
    model_dtype,
    expected_attr,
):
    run = _guarded_run(tmp_path)
    trainer = dict(run.job["trainer"])
    trainer["model_dtype"] = model_dtype
    job = {**run.job, "base_model_revision": "pinned-revision", "trainer": trainer}
    root_run = replace(run, job=job)
    torch_module = SimpleNamespace(
        bfloat16=object(),
        float16=object(),
        float32=object(),
    )

    root_kwargs = _model_pretrained_kwargs(root_run, torch_module)
    local_kwargs = _model_pretrained_kwargs(
        replace(root_run, init_from=str(tmp_path / "parent-model")),
        torch_module,
    )

    assert root_kwargs == {
        "revision": "pinned-revision",
        "torch_dtype": getattr(torch_module, expected_attr),
    }
    assert _pretrained_kwargs(root_run) == {"revision": "pinned-revision"}
    assert local_kwargs == {"torch_dtype": getattr(torch_module, expected_attr)}


def test_model_dtype_auto_is_forwarded_as_auto(tmp_path):
    run = _guarded_run(tmp_path)
    job = {
        **run.job,
        "trainer": {**run.job["trainer"], "model_dtype": "auto"},
    }

    assert _model_pretrained_kwargs(replace(run, job=job), SimpleNamespace()) == {
        "torch_dtype": "auto"
    }


@pytest.mark.parametrize("model_dtype", ["bf16", "float64", "", None, 1])
def test_model_dtype_rejects_invalid_values(tmp_path, model_dtype):
    run = _guarded_run(tmp_path)
    job = {
        **run.job,
        "trainer": {**run.job["trainer"], "model_dtype": model_dtype},
    }

    with pytest.raises(ValueError, match="model_dtype"):
        _model_pretrained_kwargs(replace(run, job=job), SimpleNamespace())


def test_require_max_steps_accepts_exact_backend_step(tmp_path):
    run = _guarded_run(tmp_path)
    result = run_training_run(
        run,
        _registry(),
        _formats(),
        loader=_loader,
        backend=_StepBackend(4),
    )

    assert result.status == "completed"
    assert result.step == 4
    assert len(result.provenance["sha256"]) == 64
    receipt_path = Path(result.provenance["path"])
    receipt = json.loads(receipt_path.read_text())
    assert receipt["planned_training_config_sha256"] == run.job["training_config_sha256"]
    assert receipt["effective_job_sha256"]
    assert receipt["run_spec_sha256"]
    assert receipt["dataset_registry_entry_sha256"]
    assert receipt["dataset_format_spec_sha256"]
    assert receipt["runtime_versions"]["python"]


def test_provenance_receipt_refuses_to_mix_changed_job_in_existing_output(tmp_path):
    run = _guarded_run(tmp_path)
    run_training_run(
        run,
        _registry(),
        _formats(),
        loader=_loader,
        backend=_StepBackend(4),
    )
    changed = replace(
        run,
        job={
            **run.job,
            "trainer": {**run.job["trainer"], "learning_rate": 9e-5},
        },
    )

    with pytest.raises(RuntimeError, match="refusing to mix a different training protocol"):
        run_training_run(
            changed,
            _registry(),
            _formats(),
            loader=_loader,
            backend=_StepBackend(4),
        )


def test_require_max_steps_rejects_early_stop_and_writes_failed_ledger(tmp_path):
    run = _guarded_run(tmp_path)

    with pytest.raises(RuntimeError, match="requires exactly 4 training steps"):
        run_training_run(
            run,
            _registry(),
            _formats(),
            loader=_loader,
            backend=_StepBackend(2, status="stopped_early"),
        )

    events = load_ledger_events(run.ledger_path)
    assert events[-1].status == "failed"
    assert "backend reported 2" in events[-1].message


def test_hf_backend_rejects_short_run_before_saving_final_artifacts(tmp_path, monkeypatch):
    run = _guarded_run(tmp_path, max_steps=4)

    class FakeTokenizer:
        pad_token = None
        eos_token = "</s>"
        padding_side = "left"

    class FakeModel:
        pass

    class FakeFactory:
        value = None

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls.value

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.state = SimpleNamespace(global_step=2, log_history=[])

        def train(self):
            return SimpleNamespace(metrics={})

    tokenizer = FakeTokenizer()
    FakeFactory.value = tokenizer
    model_factory = type(
        "FakeModelFactory",
        (),
        {"from_pretrained": classmethod(lambda cls, *args, **kwargs: FakeModel())},
    )
    deps = {
        "AutoTokenizer": FakeFactory,
        "AutoModelForCausalLM": model_factory,
        "AutoModelForSeq2SeqLM": model_factory,
        "Trainer": FakeTrainer,
        "Seq2SeqTrainer": FakeTrainer,
        "TrainerCallback": object,
        "torch": SimpleNamespace(),
        "set_seed": lambda seed: None,
    }
    monkeypatch.setattr(executor_module, "_load_hf_deps", lambda: deps)
    monkeypatch.setattr(executor_module, "_hf_dataset_from_records", lambda *args: object())
    monkeypatch.setattr(executor_module, "_training_arguments", lambda *args, **kwargs: object())
    monkeypatch.setattr(executor_module, "_data_collator", lambda *args: object())

    backend = HfPeftTrainingBackend()
    save_calls = []
    monkeypatch.setattr(
        backend,
        "_save_artifacts",
        lambda *args: save_calls.append(args) or {},
    )

    with pytest.raises(RuntimeError, match="requires exactly 4 training steps"):
        backend.train(
            run,
            PreparedTrainingData(
                train_records=(RenderedTrainingRecord(text="Question: Q", target="A"),)
            ),
            lambda event: None,
        )

    assert save_calls == []


def test_protocol_only_trainer_keys_are_not_forwarded_to_hf_arguments(tmp_path):
    class FakeArguments:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    args = _training_arguments(
        {"TrainingArguments": FakeArguments},
        _guarded_run(tmp_path),
        has_eval=False,
        model_task="causal_lm",
    )

    assert "min_prompt_tokens" not in args.kwargs
    assert "require_max_steps" not in args.kwargs
    assert "padding_side" not in args.kwargs
    assert "model_dtype" not in args.kwargs
    assert "pad_to_multiple_of" not in args.kwargs
    assert args.kwargs["max_steps"] == 4
