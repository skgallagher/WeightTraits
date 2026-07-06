from pathlib import Path

from weighttraits.cli import build_parser
from weighttraits.training.data_formats import DatasetFormatSpec
from weighttraits.training.datasets import DatasetRegistryEntry
from weighttraits.training.executor import (
    BackendTrainResult,
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
    def __init__(self, events, status="completed"):
        self.events = events
        self.status = status
        self.seen_data = None

    def train(self, run, data, event_callback):
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
    assert backend.seen_data.train_records[1].target == "no"


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
            "--allow-missing-eval",
        ]
    )
    dry = build_parser().parse_args(
        ["run-training-row", "--run-list", "/tmp/runs.jsonl", "--node-id", "n0", "--dry-run"]
    )

    assert args.run_list == Path("/tmp/runs.jsonl")
    assert args.registry == Path("/tmp/task_data.yaml")
    assert args.formats == Path("/tmp/formats.yaml")
    assert args.max_train_samples == 8
    assert args.allow_missing_eval
    assert dry.node_id == "n0"
    assert dry.dry_run


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
