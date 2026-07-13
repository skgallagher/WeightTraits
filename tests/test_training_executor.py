import json
from dataclasses import replace
from pathlib import Path

from weighttraits.cli import build_parser
from weighttraits.training.data_formats import DatasetFormatSpec
from weighttraits.training.datasets import DatasetRegistryEntry, dataset_cache_split_path
from weighttraits.training.executor import (
    BackendTrainResult,
    _CausalDataCollator,
    _audit_wrapped_lora_model,
    _cleanup_trainer_checkpoints,
    _pretrained_kwargs,
    _requested_lora_targets,
    _resolve_lora_target_modules,
    _set_training_seed,
    _tokenize_causal_batch,
    _training_arguments,
    _trainer_tokenizer_kwargs,
    dry_run_training_row,
    prepare_training_data,
    run_training_run,
)
from weighttraits.training.ledger import load_ledger_events
from weighttraits.training.monitor import TrainingEvent
from weighttraits.training.planner import build_training_jobs
from weighttraits.training.runlist import build_training_run_list


class FakeBackend:
    def __init__(self, events, status="completed", metadata=None):
        self.events = events
        self.status = status
        self.metadata = metadata or {}
        self.seen_data = None
        self.seen_run = None

    def train(self, run, data, event_callback):
        self.seen_run = run
        self.seen_data = data
        stopped = False
        last_event = None
        for event in self.events:
            last_event = event
            decision = event_callback(event)
            if decision.should_stop:
                stopped = True
                break
        status = "stopped_early" if stopped else self.status
        return BackendTrainResult(
            status=status,
            step=None if last_event is None else last_event.step,
            train_loss=None if last_event is None else last_event.train_loss,
            eval_loss=None if last_event is None else last_event.eval_loss,
            artifacts=dict(run.expected_artifacts),
            metadata=self.metadata,
            message="fake backend finished",
        )


def _rows():
    return [
        {
            "node_id": "n0",
            "parent_id": "root",
            "depth": 1,
            "path": ["root", "n0"],
            "grow": "train",
            "task_family": "qa_reasoning",
            "dataset_id": "boolq",
        }
    ]


def _config(tmp_path: Path):
    return {
        "base_model": "google/flan-t5-small",
        "method": "full",
        "output_root": str(tmp_path / "outputs"),
        "trainer": {"max_steps": 4, "target_field": "answer"},
        "prompt": {"default_template": "Question: {question}\nContext: {context}"},
        "stopping": {
            "early_stopping": {"patience": 1, "min_delta": 0.0},
            "warnings": {"loss_increase_relative": 0.05, "loss_increase_patience": 1},
        },
    }


def _run(tmp_path: Path):
    jobs = build_training_jobs(_rows(), _config(tmp_path))
    return build_training_run_list(
        jobs,
        ledger_path=tmp_path / "training_ledger.jsonl",
    ).runs[0]


def test_pretrained_kwargs_pins_only_remote_root_initialization(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["base_model_revision"] = "0123456789abcdef"
    run = build_training_run_list(
        build_training_jobs(_rows(), config),
        ledger_path=tmp_path / "training_ledger.jsonl",
    ).runs[0]

    assert _pretrained_kwargs(run) == {"revision": "0123456789abcdef"}
    assert _pretrained_kwargs(replace(run, init_from=str(tmp_path / "parent/model"))) == {}


def test_cleanup_trainer_checkpoints_is_opt_in_and_directory_scoped(tmp_path: Path) -> None:
    run = _run(tmp_path)
    checkpoint = Path(run.output_dir) / "checkpoint-2"
    checkpoint.mkdir(parents=True)
    (checkpoint / "optimizer.pt").write_text("resume state")
    similarly_named_file = Path(run.output_dir) / "checkpoint-note"
    similarly_named_file.write_text("keep")

    assert _cleanup_trainer_checkpoints(run) == []
    assert checkpoint.exists()

    job = dict(run.job)
    trainer = dict(job["trainer"])
    trainer["cleanup_checkpoints_on_success"] = True
    job["trainer"] = trainer
    cleanup_run = replace(run, job=job)
    assert _cleanup_trainer_checkpoints(cleanup_run) == [str(checkpoint)]
    assert not checkpoint.exists()
    assert similarly_named_file.exists()


def _registry():
    return {
        "boolq": DatasetRegistryEntry(
            dataset_id="boolq",
            task_family="qa_reasoning",
            hf_args=("google/boolq",),
            train_split="train",
            eval_split="validation",
        )
    }


def _formats():
    return {
        "boolq": DatasetFormatSpec(
            dataset_id="boolq",
            task_family="qa_reasoning",
            prompt_fields=("question", "context", "answer"),
            field_map={"question": "question", "context": "passage", "answer": "answer"},
            train_split="train",
            eval_split="validation",
        )
    }


def _loader(*args):
    assert args == ("google/boolq",)
    return {
        "train": [
            {"question": "Q1", "passage": "P1", "answer": "yes"},
            {"question": "Q2", "passage": "P2", "answer": "no"},
        ],
        "validation": [{"question": "QV", "passage": "PV", "answer": "yes"}],
    }


def test_prepare_training_data_applies_field_map_and_target_field(tmp_path):
    data = prepare_training_data(
        _run(tmp_path),
        _registry(),
        _formats(),
        loader=_loader,
        max_train_samples=1,
    )

    assert data.valid
    assert data.train_records[0].text == "Question: Q1\nContext: P1"
    assert data.train_records[0].target == "yes"
    assert data.eval_records[0].text == "Question: QV\nContext: PV"
    assert data.summary()["n_train_records"] == 1


def test_prepare_training_data_applies_nested_field_map(tmp_path):
    run = _run(tmp_path)
    run = type(run)(
        **{
            **run.to_dict(),
            "dataset_id": "toy_translation",
            "task_family": "translation",
            "job": {
                **run.job,
                "dataset_id": "toy_translation",
                "task_family": "translation",
                "prompt_template": "Translate: {source_text}",
                "trainer": {**run.job["trainer"], "target_field": "target"},
            },
        }
    )
    registry = {
        "toy_translation": DatasetRegistryEntry(
            dataset_id="toy_translation",
            task_family="translation",
            hf_args=("toy/translation",),
            train_split="train",
        )
    }
    formats = {
        "toy_translation": DatasetFormatSpec(
            dataset_id="toy_translation",
            task_family="translation",
            prompt_fields=("source_text",),
            field_map={"source_text": "translation.en", "target": "translation.fr"},
            train_split="train",
        )
    }

    data = prepare_training_data(
        run,
        registry,
        formats,
        loader=lambda *args: {"train": [{"translation": {"en": "hello", "fr": "bonjour"}}]},
    )

    assert data.valid
    assert data.train_records[0].text == "Translate: hello"
    assert data.train_records[0].target == "bonjour"


def test_prepare_training_data_filters_before_sample_cap(tmp_path):
    registry = {
        "boolq": DatasetRegistryEntry(
            dataset_id="boolq",
            task_family="qa_reasoning",
            hf_args=("google/boolq",),
            filter={"max_chars": {"context": 2}},
            train_split="train",
        )
    }

    data = prepare_training_data(
        _run(tmp_path),
        registry,
        _formats(),
        loader=lambda *args: {
            "train": [
                {"question": "Q long", "passage": "long", "answer": "bad"},
                {"question": "Q short", "passage": "ok", "answer": "good"},
            ]
        },
        max_train_samples=1,
    )

    assert data.valid
    assert data.train_records[0].text == "Question: Q short\nContext: ok"
    assert data.train_records[0].target == "good"


def test_prepare_training_data_can_require_cached_rows(tmp_path):
    cache_root = tmp_path / "cache"
    train_path = dataset_cache_split_path(cache_root, "boolq", "train")
    validation_path = dataset_cache_split_path(cache_root, "boolq", "validation")
    train_path.parent.mkdir(parents=True)
    train_path.write_text('{"answer": "yes", "context": "P cache", "question": "Q cache"}\n')
    validation_path.write_text(
        '{"answer": "yes", "context": "PV cache", "question": "QV cache"}\n'
    )

    def forbidden_loader(*args):
        raise AssertionError("loader should not be called when cache is required")

    data = prepare_training_data(
        _run(tmp_path),
        _registry(),
        _formats(),
        loader=forbidden_loader,
        data_cache_root=cache_root,
        require_data_cache=True,
        max_train_samples=1,
    )

    assert data.valid
    assert data.train_records[0].text == "Question: Q cache\nContext: P cache"
    assert data.eval_records[0].text == "Question: QV cache\nContext: PV cache"


def test_run_training_run_writes_monitor_and_stopped_early_ledger(tmp_path):
    run = _run(tmp_path)
    backend = FakeBackend(
        [
            TrainingEvent(step=1, eval_loss=1.0),
            TrainingEvent(step=2, eval_loss=1.2),
        ]
    )

    result = run_training_run(
        run,
        _registry(),
        _formats(),
        loader=_loader,
        backend=backend,
        max_train_samples=2,
    )
    events = load_ledger_events(run.ledger_path)
    training_log = [json.loads(line) for line in Path(run.expected_artifacts["training_log"]).read_text().splitlines()]

    assert result.status == "stopped_early"
    assert result.step == 2
    assert any("increased relative to best" in warning for warning in result.warnings)
    assert any("no eval_loss improvement" in reason for reason in result.stop_reasons)
    assert [event.status for event in events] == [
        "started",
        "running",
        "running",
        "running",
        "stopped_early",
    ]
    assert events[-1].extra["data"]["n_train_records"] == 2
    assert [event["status"] for event in training_log] == [
        "started",
        "running",
        "running",
        "running",
        "stopped_early",
    ]
    assert backend.seen_data.train_records[1].target == "no"


def test_run_training_run_records_execution_overrides(tmp_path):
    run = _run(tmp_path)
    run = type(run)(
        **{
            **run.to_dict(),
            "job": {
                **run.job,
                "trainer": {
                    **run.job["trainer"],
                    "max_steps": 2,
                    "report_to": ["wandb"],
                    "run_name": "confirm-paper-tree-001-n0-smoke",
                },
            },
        }
    )
    backend = FakeBackend([], status="completed")

    result = run_training_run(
        run,
        _registry(),
        _formats(),
        loader=_loader,
        backend=backend,
        max_train_samples=1,
        execution_overrides={
            "trainer": {
                "max_steps": 2,
                "report_to": ["wandb"],
                "run_name": "confirm-paper-tree-001-n0-smoke",
            }
        },
    )
    events = load_ledger_events(run.ledger_path)
    training_log = [json.loads(line) for line in Path(run.expected_artifacts["training_log"]).read_text().splitlines()]

    overrides = {
        "trainer": {
            "max_steps": 2,
            "report_to": ["wandb"],
            "run_name": "confirm-paper-tree-001-n0-smoke",
        }
    }
    assert result.status == "completed"
    assert result.execution_overrides == overrides
    assert backend.seen_run.job["trainer"]["max_steps"] == 2
    assert backend.seen_run.job["trainer"]["report_to"] == ["wandb"]
    assert backend.seen_run.job["trainer"]["run_name"] == "confirm-paper-tree-001-n0-smoke"
    assert events[0].extra["execution_overrides"] == overrides
    assert events[-1].extra["execution_overrides"] == overrides
    assert training_log[0]["extra"]["execution_overrides"] == overrides
    assert training_log[-1]["extra"]["execution_overrides"] == overrides


def test_run_training_run_records_backend_metadata(tmp_path):
    run = _run(tmp_path)
    metadata = {
        "lora_target_audit": {
            "requested_target_modules": ["q"],
            "resolved_module_names": ["encoder.block.0.SelfAttention.q"],
        }
    }

    result = run_training_run(
        run,
        _registry(),
        _formats(),
        loader=_loader,
        backend=FakeBackend([], metadata=metadata),
        max_train_samples=1,
    )
    events = load_ledger_events(run.ledger_path)

    assert result.backend_metadata == metadata
    assert events[-1].extra["backend_metadata"] == metadata


def test_run_training_run_writes_failed_ledger_on_bad_data(tmp_path):
    run = _run(tmp_path)

    try:
        run_training_run(
            run,
            _registry(),
            _formats(),
            loader=lambda *args: {"train": [{"question": "Q only"}]},
            backend=FakeBackend([]),
            allow_missing_eval=True,
        )
    except ValueError as exc:
        assert "training data is not valid" in str(exc)
    else:
        raise AssertionError("expected bad training data to fail")

    events = load_ledger_events(run.ledger_path)
    assert events[-1].status == "failed"
    training_log = [json.loads(line) for line in Path(run.expected_artifacts["training_log"]).read_text().splitlines()]
    assert training_log[-1]["status"] == "failed"


def test_dry_run_training_row_does_not_require_dataset_loading(tmp_path):
    result = dry_run_training_row(_run(tmp_path))

    assert result.status == "dry_run"
    assert result.data == {}
    assert result.artifacts["model"].endswith("/outputs/n0/model")


def test_run_training_row_parser_accepts_real_and_dry_run_modes():
    args = build_parser().parse_args(
        [
            "run-training-row",
            "--run-list",
            "/tmp/runs.jsonl",
            "--index",
            "0",
            "--registry",
            "/tmp/task_data.yaml",
            "--formats",
            "/tmp/formats.yaml",
            "--max-train-samples",
            "8",
            "--data-cache-root",
            "/tmp/cache",
            "--require-data-cache",
            "--allow-missing-eval",
            "--override-max-steps",
            "2",
            "--report-to",
            "wandb,tensorboard",
            "--run-name",
            "confirm-paper-smoke",
        ]
    )
    dry = build_parser().parse_args(
        ["run-training-row", "--run-list", "/tmp/runs.jsonl", "--node-id", "n0", "--dry-run"]
    )

    assert args.run_list == Path("/tmp/runs.jsonl")
    assert args.registry == Path("/tmp/task_data.yaml")
    assert args.formats == Path("/tmp/formats.yaml")
    assert args.max_train_samples == 8
    assert args.data_cache_root == Path("/tmp/cache")
    assert args.require_data_cache
    assert args.allow_missing_eval
    assert args.override_max_steps == 2
    assert args.report_to == ["wandb,tensorboard"]
    assert args.run_name == "confirm-paper-smoke"
    assert dry.node_id == "n0"
    assert dry.dry_run


def test_run_training_row_dry_run_reports_runtime_trainer_override(tmp_path, capsys):
    run_list_path = tmp_path / "runs.jsonl"
    run_list_path.write_text(json.dumps(_run(tmp_path).to_dict()) + "\n")
    args = build_parser().parse_args(
        [
            "run-training-row",
            "--run-list",
            str(run_list_path),
            "--node-id",
            "n0",
            "--override-max-steps",
            "2",
            "--report-to",
            "wandb",
            "--run-name",
            "confirm-paper-tree-001-n0-smoke",
            "--dry-run",
        ]
    )

    assert args.func(args) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "dry_run"
    assert output["execution_overrides"] == {
        "trainer": {
            "max_steps": 2,
            "report_to": ["wandb"],
            "run_name": "confirm-paper-tree-001-n0-smoke",
        }
    }


def test_run_training_row_report_to_none_disables_runtime_tracking(tmp_path, capsys):
    run_list_path = tmp_path / "runs.jsonl"
    run_list_path.write_text(json.dumps(_run(tmp_path).to_dict()) + "\n")
    args = build_parser().parse_args(
        [
            "run-training-row",
            "--run-list",
            str(run_list_path),
            "--node-id",
            "n0",
            "--report-to",
            "none",
            "--dry-run",
        ]
    )

    assert args.func(args) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["execution_overrides"] == {"trainer": {"report_to": []}}


def test_audit_training_row_data_loads_required_cached_rows(tmp_path, capsys):
    cache_root = tmp_path / "cache"
    train_path = dataset_cache_split_path(cache_root, "boolq", "train")
    validation_path = dataset_cache_split_path(cache_root, "boolq", "validation")
    train_path.parent.mkdir(parents=True)
    train_path.write_text('{"answer": "yes", "context": "P cache", "question": "Q cache"}\n')
    validation_path.write_text(
        '{"answer": "yes", "context": "PV cache", "question": "QV cache"}\n'
    )
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        """
datasets:
  - id: boolq
    hf_args: [google/boolq]
    train_split: train
    eval_split: validation
"""
    )
    formats_path = tmp_path / "formats.yaml"
    formats_path.write_text(
        """
datasets:
  - dataset_id: boolq
    task_family: qa_reasoning
    train_split: train
    eval_split: validation
    field_map:
      question: question
      context: context
      answer: answer
"""
    )
    run_row = _run(tmp_path).to_dict()
    run_row["runner"] = {
        "entrypoint": "weighttraits.cli run-training-row",
        "run_list_path": str(tmp_path / "runs.jsonl"),
        "status": "planned",
        "options": {
            "registry_path": str(registry_path),
            "formats_path": str(formats_path),
            "data_cache_root": str(cache_root),
            "require_data_cache": True,
            "allow_missing_eval": True,
        },
    }
    run_list_path = tmp_path / "runs.jsonl"
    run_list_path.write_text(json.dumps(run_row) + "\n")

    args = build_parser().parse_args(
        [
            "audit-training-row-data",
            "--run-list",
            str(run_list_path),
            "--node-id",
            "n0",
            "--max-train-samples",
            "1",
            "--max-eval-samples",
            "1",
        ]
    )

    assert args.func(args) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["valid"]
    assert output["require_data_cache"]
    assert output["streaming"] is False
    assert output["data"]["n_train_records"] == 1
    assert output["data"]["n_eval_records"] == 1
    assert output["data"]["issues"] == []


def test_audit_training_row_data_parser_accepts_overrides():
    args = build_parser().parse_args(
        [
            "audit-training-row-data",
            "--run-list",
            "/tmp/runs.jsonl",
            "--index",
            "0",
            "--registry",
            "/tmp/task_data.yaml",
            "--formats",
            "/tmp/formats.yaml",
            "--max-train-samples",
            "2",
            "--max-eval-samples",
            "1",
            "--data-cache-root",
            "/tmp/cache",
            "--require-data-cache",
            "--allow-missing-eval",
            "--streaming",
            "--allow-issues",
        ]
    )

    assert args.run_list == Path("/tmp/runs.jsonl")
    assert args.registry == Path("/tmp/task_data.yaml")
    assert args.formats == Path("/tmp/formats.yaml")
    assert args.max_train_samples == 2
    assert args.max_eval_samples == 1
    assert args.data_cache_root == Path("/tmp/cache")
    assert args.require_data_cache
    assert args.allow_missing_eval
    assert args.streaming
    assert args.allow_issues


def test_trainer_tokenizer_kwargs_support_transformers_constructor_versions():
    class OldTrainer:
        def __init__(self, tokenizer=None):
            pass

    class NewTrainer:
        def __init__(self, processing_class=None):
            pass

    class MinimalTrainer:
        def __init__(self):
            pass

    tokenizer = object()

    assert _trainer_tokenizer_kwargs(OldTrainer, tokenizer) == {"tokenizer": tokenizer}
    assert _trainer_tokenizer_kwargs(NewTrainer, tokenizer) == {"processing_class": tokenizer}
    assert _trainer_tokenizer_kwargs(MinimalTrainer, tokenizer) == {}


def test_training_arguments_use_seq2seq_class_for_seq2seq_jobs(tmp_path):
    class BaseArgs:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Seq2SeqArgs(BaseArgs):
        pass

    deps = {
        "TrainingArguments": BaseArgs,
        "Seq2SeqTrainingArguments": Seq2SeqArgs,
    }

    seq2seq_args = _training_arguments(deps, _run(tmp_path), has_eval=True, model_task="seq2seq")
    causal_args = _training_arguments(deps, _run(tmp_path), has_eval=False, model_task="causal_lm")

    assert isinstance(seq2seq_args, Seq2SeqArgs)
    assert seq2seq_args.kwargs["eval_strategy"] == "epoch"
    assert isinstance(causal_args, BaseArgs)
    assert not isinstance(causal_args, Seq2SeqArgs)
    assert causal_args.kwargs["eval_strategy"] == "no"


def test_training_seed_is_applied_before_backend_construction():
    seen = []

    assert _set_training_seed({"set_seed": seen.append}, {"trainer": {"seed": 17}}) == 17
    assert seen == [17]
    assert _set_training_seed({"set_seed": seen.append}, {}) == 42
    assert seen == [17, 42]


class _FakeModuleGraph:
    def __init__(self, modules, parameters=()):
        self._modules = modules
        self._parameters = parameters

    def named_modules(self):
        return list(self._modules)

    def named_parameters(self):
        return list(self._parameters)


class _FakeLoraLayer:
    lora_A = object()
    lora_B = object()


class _FakeParameter:
    def __init__(self, n, *, requires_grad=True):
        self.n = n
        self.requires_grad = requires_grad

    def numel(self):
        return self.n


def test_requested_lora_targets_rejects_empty_and_duplicate_values():
    for raw in (None, [], ["q", ""], ["q", "q"]):
        try:
            _requested_lora_targets({"target_modules": raw})
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected invalid LoRA targets to fail: {raw}")


def test_lora_target_resolution_rejects_unmatched_gated_ffn_alias():
    model = _FakeModuleGraph(
        [
            ("", object()),
            ("encoder.block.0.SelfAttention.q", object()),
            ("encoder.block.0.DenseReluDense.wi_0", object()),
            ("encoder.block.0.DenseReluDense.wi_1", object()),
            ("encoder.block.0.DenseReluDense.wo", object()),
        ]
    )

    try:
        _resolve_lora_target_modules(model, ["q", "wi", "wo"])
    except ValueError as exc:
        message = str(exc)
        assert "['wi']" in message
        assert "wi_0" in message
        assert "wi_1" in message
    else:
        raise AssertionError("expected unmatched Flan-T5 wi alias to fail")


def test_lora_target_audit_records_resolved_modules_and_parameter_counts():
    model = _FakeModuleGraph(
        [
            ("", object()),
            ("base_model.model.encoder.block.0.SelfAttention.q", _FakeLoraLayer()),
            ("base_model.model.encoder.block.0.DenseReluDense.wi_0", _FakeLoraLayer()),
            ("base_model.model.encoder.block.0.DenseReluDense.wi_1", _FakeLoraLayer()),
            ("base_model.model.encoder.block.0.DenseReluDense.wo", _FakeLoraLayer()),
        ],
        [
            ("base_model.model.encoder.q.lora_A.default.weight", _FakeParameter(10)),
            ("base_model.model.encoder.q.lora_B.default.weight", _FakeParameter(12)),
            ("base_model.model.shared.weight", _FakeParameter(100, requires_grad=False)),
        ],
    )

    audit = _audit_wrapped_lora_model(model, ["q", "wi_0", "wi_1", "wo"])

    assert audit["n_resolved_modules"] == 4
    assert audit["n_trainable_lora_parameters"] == 22
    assert audit["n_trainable_model_parameters"] == 22
    assert audit["n_lora_parameter_tensors"] == 2
    assert audit["unmatched_target_modules"] == []
    assert audit["matched_module_names_by_target"]["wi_0"] == [
        "base_model.model.encoder.block.0.DenseReluDense.wi_0"
    ]


class _FakeCausalTokenizer:
    eos_token_id = 99
    padding_side = "right"

    def __call__(
        self,
        value,
        *,
        max_length=None,
        truncation=False,
        add_special_tokens=True,
    ):
        def encode(text):
            ids = ([1] if add_special_tokens else []) + [ord(char) % 50 + 2 for char in text]
            return ids[:max_length] if truncation and max_length is not None else ids

        if isinstance(value, list):
            return {"input_ids": [encode(text) for text in value]}
        return {"input_ids": encode(value)}

    def pad(self, features, *, padding, return_tensors):
        import torch

        assert padding is True
        assert return_tensors == "pt"
        width = max(len(feature["input_ids"]) for feature in features)
        return {
            "input_ids": torch.tensor(
                [feature["input_ids"] + [0] * (width - len(feature["input_ids"])) for feature in features]
            ),
            "attention_mask": torch.tensor(
                [feature["attention_mask"] + [0] * (width - len(feature["attention_mask"])) for feature in features]
            ),
        }


def test_causal_completion_loss_masks_prompt_and_preserves_completion():
    tokenizer = _FakeCausalTokenizer()

    encoded = _tokenize_causal_batch(
        tokenizer,
        ["long prompt", "p"],
        ["answer", "x"],
        max_length=10,
        loss_scope="completion",
    )

    assert encoded["labels"][0][-1] == tokenizer.eos_token_id
    prompt_length = encoded["labels"][0].count(-100)
    assert prompt_length == 3
    assert all(label != -100 for label in encoded["labels"][0][prompt_length:])
    assert len(encoded["input_ids"][0]) == 10

    batch = _CausalDataCollator(tokenizer)(
        [{key: rows[index] for key, rows in encoded.items()} for index in range(2)]
    )
    assert batch["labels"].shape == batch["input_ids"].shape
    assert batch["labels"][1, -1].item() == -100


def test_causal_all_token_loss_keeps_prompt_labels():
    encoded = _tokenize_causal_batch(
        _FakeCausalTokenizer(),
        ["prompt"],
        ["answer"],
        max_length=20,
        loss_scope="all_tokens",
    )

    assert -100 not in encoded["labels"][0]
    assert encoded["labels"][0] == encoded["input_ids"][0]
