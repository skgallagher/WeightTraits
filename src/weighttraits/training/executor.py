"""Per-row training execution behind run-list contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from weighttraits.training.data_formats import DatasetFormatSpec
from weighttraits.training.datasets import DatasetLoader, DatasetRegistryEntry
from weighttraits.training.ledger import TrainingLedgerEvent, append_ledger_event
from weighttraits.training.monitor import (
    LossMonitor,
    LossMonitorConfig,
    MonitorDecision,
    TrainingEvent,
)
from weighttraits.training.prompts import render_prompt
from weighttraits.training.runlist import TrainingRunSpec


MonitorEventCallback = Callable[[TrainingEvent], MonitorDecision]


@dataclass(frozen=True)
class RenderedTrainingRecord:
    text: str
    target: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedTrainingData:
    train_records: tuple[RenderedTrainingRecord, ...]
    eval_records: tuple[RenderedTrainingRecord, ...] = ()
    train_split: str | None = None
    eval_split: str | None = None
    n_dropped_train: int = 0
    n_dropped_eval: int = 0
    issues: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return bool(self.train_records) and not self.issues

    def summary(self) -> dict[str, Any]:
        return {
            "n_train_records": len(self.train_records),
            "n_eval_records": len(self.eval_records),
            "train_split": self.train_split,
            "eval_split": self.eval_split,
            "n_dropped_train": self.n_dropped_train,
            "n_dropped_eval": self.n_dropped_eval,
            "issues": list(self.issues),
        }


@dataclass(frozen=True)
class BackendTrainResult:
    status: str
    step: int | None = None
    train_loss: float | None = None
    eval_loss: float | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingRunResult:
    node_id: str
    status: str
    data: dict[str, Any]
    artifacts: dict[str, str]
    step: int | None = None
    train_loss: float | None = None
    eval_loss: float | None = None
    warnings: tuple[str, ...] = ()
    stop_reasons: tuple[str, ...] = ()
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["warnings"] = list(self.warnings)
        out["stop_reasons"] = list(self.stop_reasons)
        return out


class TrainingBackend(Protocol):
    def train(
        self,
        run: TrainingRunSpec,
        data: PreparedTrainingData,
        event_callback: MonitorEventCallback,
    ) -> BackendTrainResult:
        """Train and save artifacts for one run row."""


def prepare_training_data(
    run: TrainingRunSpec,
    registry: Mapping[str, DatasetRegistryEntry],
    format_specs: Mapping[str, DatasetFormatSpec],
    *,
    loader: DatasetLoader | None = None,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    allow_missing_eval: bool = True,
) -> PreparedTrainingData:
    """Load, field-map, and render data for one planned training run."""
    if run.dataset_id is None:
        raise ValueError(f"run {run.node_id} has no dataset_id")
    entry = registry.get(run.dataset_id)
    if entry is None:
        raise ValueError(f"dataset id not found in registry: {run.dataset_id}")
    spec = format_specs.get(run.dataset_id)
    if spec is None:
        raise ValueError(f"dataset id not found in format specs: {run.dataset_id}")
    dataset_loader = loader or _default_hf_loader()
    dataset = dataset_loader(*entry.hf_args, **(entry.hf_kwargs or {}))
    train_split = spec.train_split or entry.train_split or "train"
    eval_split = spec.eval_split or entry.eval_split

    train_dataset = _required_split(dataset, train_split)
    eval_dataset = _optional_split(dataset, eval_split) if eval_split else None
    if eval_split and eval_dataset is None and not allow_missing_eval:
        raise ValueError(f"missing eval split for {run.dataset_id}: {eval_split}")

    train_records, n_dropped_train, train_issues = _render_dataset_split(
        train_dataset,
        run=run,
        spec=spec,
        max_samples=max_train_samples,
        split_name=train_split,
    )
    eval_records: tuple[RenderedTrainingRecord, ...] = ()
    n_dropped_eval = 0
    eval_issues: tuple[str, ...] = ()
    if eval_dataset is not None:
        eval_records, n_dropped_eval, eval_issues = _render_dataset_split(
            eval_dataset,
            run=run,
            spec=spec,
            max_samples=max_eval_samples,
            split_name=eval_split,
        )
    issues = tuple(train_issues + eval_issues)
    if not train_records:
        issues = issues + (f"{train_split}: no renderable training records",)
    return PreparedTrainingData(
        train_records=train_records,
        eval_records=eval_records,
        train_split=train_split,
        eval_split=eval_split,
        n_dropped_train=n_dropped_train,
        n_dropped_eval=n_dropped_eval,
        issues=issues,
    )


def run_training_run(
    run: TrainingRunSpec,
    registry: Mapping[str, DatasetRegistryEntry],
    format_specs: Mapping[str, DatasetFormatSpec],
    *,
    backend: TrainingBackend | None = None,
    loader: DatasetLoader | None = None,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    allow_missing_eval: bool = True,
    write_ledger: bool = True,
) -> TrainingRunResult:
    """Run one training row and write ledger events for lifecycle and monitor state."""
    if write_ledger:
        append_ledger_event(
            run.ledger_path,
            TrainingLedgerEvent(
                node_id=run.node_id,
                status="started",
                message="training row started",
                extra={"run_id": run.run_id, "array_index": run.array_index},
            ),
        )
    try:
        data = prepare_training_data(
            run,
            registry,
            format_specs,
            loader=loader,
            max_train_samples=max_train_samples,
            max_eval_samples=max_eval_samples,
            allow_missing_eval=allow_missing_eval,
        )
        if not data.valid:
            raise ValueError(f"training data is not valid for {run.node_id}: {data.issues}")
        if write_ledger:
            append_ledger_event(
                run.ledger_path,
                TrainingLedgerEvent(
                    node_id=run.node_id,
                    status="running",
                    message="training data prepared",
                    extra=data.summary(),
                ),
            )

        monitor = LossMonitor(_monitor_config(run.job.get("stopping", {})))
        decisions: list[MonitorDecision] = []

        def on_event(event: TrainingEvent) -> MonitorDecision:
            decision = monitor.update(event)
            decisions.append(decision)
            if write_ledger:
                append_ledger_event(
                    run.ledger_path,
                    TrainingLedgerEvent(
                        node_id=run.node_id,
                        status="running",
                        step=event.step,
                        train_loss=event.train_loss,
                        eval_loss=event.eval_loss,
                        warnings=tuple(decision.warnings),
                        stop_reasons=tuple(decision.reasons),
                        message="training monitor update",
                        extra=decision.state,
                    ),
                )
            return decision

        train_backend = backend or HfPeftTrainingBackend()
        backend_result = train_backend.train(run, data, on_event)
        final_decision = decisions[-1] if decisions else None
        status = backend_result.status
        if status == "completed" and final_decision is not None and final_decision.should_stop:
            status = "stopped_early"
        warnings = _unique_items(decision.warnings for decision in decisions)
        stop_reasons = _unique_items(decision.reasons for decision in decisions)
        artifacts = dict(run.expected_artifacts)
        artifacts.update(backend_result.artifacts)
        result = TrainingRunResult(
            node_id=run.node_id,
            status=status,
            data=data.summary(),
            artifacts=artifacts,
            step=backend_result.step,
            train_loss=backend_result.train_loss,
            eval_loss=backend_result.eval_loss,
            warnings=tuple(warnings),
            stop_reasons=tuple(stop_reasons),
            message=backend_result.message,
        )
        if write_ledger:
            append_ledger_event(
                run.ledger_path,
                TrainingLedgerEvent(
                    node_id=run.node_id,
                    status=status,
                    step=result.step,
                    train_loss=result.train_loss,
                    eval_loss=result.eval_loss,
                    warnings=result.warnings,
                    stop_reasons=result.stop_reasons,
                    message=result.message or "training row finished",
                    extra={"artifacts": artifacts, "data": data.summary()},
                ),
            )
        return result
    except Exception as exc:
        if write_ledger:
            append_ledger_event(
                run.ledger_path,
                TrainingLedgerEvent(
                    node_id=run.node_id,
                    status="failed",
                    message=str(exc),
                ),
            )
        raise


def dry_run_training_row(run: TrainingRunSpec) -> TrainingRunResult:
    """Return the selected row without loading datasets or models."""
    return TrainingRunResult(
        node_id=run.node_id,
        status="dry_run",
        data={},
        artifacts=dict(run.expected_artifacts),
        message="selected training row without loading datasets or models",
    )


class HfPeftTrainingBackend:
    """Hugging Face / PEFT backend imported lazily for real training environments."""

    def train(
        self,
        run: TrainingRunSpec,
        data: PreparedTrainingData,
        event_callback: MonitorEventCallback,
    ) -> BackendTrainResult:
        deps = _load_hf_deps()
        model_task = _model_task(run.job)
        tokenizer = deps["AutoTokenizer"].from_pretrained(run.init_from)
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None):
            tokenizer.pad_token = tokenizer.eos_token
        model_cls = (
            deps["AutoModelForSeq2SeqLM"]
            if model_task == "seq2seq"
            else deps["AutoModelForCausalLM"]
        )
        model = model_cls.from_pretrained(run.init_from)
        if run.method == "lora":
            model = self._wrap_lora(model, run, deps, model_task)

        train_dataset = _hf_dataset_from_records(
            deps,
            tokenizer,
            model_task,
            data.train_records,
            run.job,
        )
        eval_dataset = (
            _hf_dataset_from_records(deps, tokenizer, model_task, data.eval_records, run.job)
            if data.eval_records
            else None
        )
        args = _training_arguments(deps, run, has_eval=eval_dataset is not None)
        callback = _make_loss_monitor_callback(deps["TrainerCallback"], event_callback)
        trainer_cls = deps["Seq2SeqTrainer"] if model_task == "seq2seq" else deps["Trainer"]
        collator = _data_collator(deps, tokenizer, model, model_task)
        trainer = trainer_cls(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=collator,
            callbacks=[callback],
        )
        output = trainer.train()
        artifacts = self._save_artifacts(run, trainer, model, tokenizer)
        train_loss = _metric_value(getattr(output, "metrics", {}), "train_loss")
        eval_loss = _metric_value(getattr(trainer.state, "log_history", []), "eval_loss")
        status = "stopped_early" if callback.stopped else "completed"
        return BackendTrainResult(
            status=status,
            step=int(getattr(trainer.state, "global_step", 0)),
            train_loss=train_loss,
            eval_loss=eval_loss,
            artifacts=artifacts,
        )

    def _wrap_lora(
        self,
        model: Any,
        run: TrainingRunSpec,
        deps: dict[str, Any],
        model_task: str,
    ) -> Any:
        lora = dict(run.job.get("lora") or {})
        task_type = (
            deps["TaskType"].SEQ_2_SEQ_LM
            if model_task == "seq2seq"
            else deps["TaskType"].CAUSAL_LM
        )
        config = deps["LoraConfig"](
            r=int(lora.get("r", 8)),
            lora_alpha=int(lora.get("lora_alpha", 16)),
            lora_dropout=float(lora.get("lora_dropout", 0.05)),
            target_modules=list(lora.get("target_modules", [])),
            task_type=task_type,
        )
        return deps["get_peft_model"](model, config)

    def _save_artifacts(
        self,
        run: TrainingRunSpec,
        trainer: Any,
        model: Any,
        tokenizer: Any,
    ) -> dict[str, str]:
        artifacts: dict[str, str] = {}
        if run.method == "lora":
            adapter = run.expected_artifacts["adapter"]
            merged = run.expected_artifacts["merged"]
            Path(adapter).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(adapter)
            tokenizer.save_pretrained(adapter)
            artifacts["adapter"] = adapter
            lora = dict(run.job.get("lora") or {})
            if lora.get("merge_after_train", True):
                Path(merged).mkdir(parents=True, exist_ok=True)
                merged_model = model.merge_and_unload()
                merged_model.save_pretrained(merged)
                tokenizer.save_pretrained(merged)
                artifacts["merged"] = merged
            return artifacts
        model_path = run.expected_artifacts["model"]
        Path(model_path).mkdir(parents=True, exist_ok=True)
        trainer.save_model(model_path)
        tokenizer.save_pretrained(model_path)
        artifacts["model"] = model_path
        return artifacts


def _make_loss_monitor_callback(callback_base: type, event_callback: MonitorEventCallback) -> Any:
    class LossMonitorCallback(callback_base):
        def __init__(self) -> None:
            self.stopped = False

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: dict[str, Any] | None = None,
            **_: Any,
        ) -> Any:
            logs = logs or {}
            event = TrainingEvent(
                step=int(getattr(state, "global_step", 0)),
                train_loss=_maybe_float(logs.get("loss") or logs.get("train_loss")),
                eval_loss=_maybe_float(logs.get("eval_loss")),
                learning_rate=_maybe_float(logs.get("learning_rate")),
            )
            decision = event_callback(event)
            if decision.should_stop:
                self.stopped = True
                control.should_training_stop = True
            return control

    return LossMonitorCallback()


def _render_dataset_split(
    dataset_split: Any,
    *,
    run: TrainingRunSpec,
    spec: DatasetFormatSpec,
    max_samples: int | None,
    split_name: str | None,
) -> tuple[tuple[RenderedTrainingRecord, ...], int, tuple[str, ...]]:
    records: list[RenderedTrainingRecord] = []
    issues: list[str] = []
    dropped = 0
    for index, row in enumerate(dataset_split):
        if max_samples is not None and index >= max_samples:
            break
        if not isinstance(row, Mapping) and not hasattr(row, "items"):
            dropped += 1
            issues.append(f"{split_name}[{index}]: row is not a mapping")
            continue
        try:
            canonical = _canonical_example(row, spec)
            text = render_prompt(str(run.job["prompt_template"]), canonical)
            target = _render_target(run.job, canonical, text)
        except Exception as exc:
            dropped += 1
            issues.append(f"{split_name}[{index}]: {exc}")
            continue
        if not text.strip() or not target.strip():
            dropped += 1
            issues.append(f"{split_name}[{index}]: empty rendered text or target")
            continue
        records.append(RenderedTrainingRecord(text=text, target=target))
    return tuple(records), dropped, tuple(issues)


def _render_target(job: Mapping[str, Any], example: dict[str, Any], rendered_prompt: str) -> str:
    trainer = dict(job.get("trainer", {}))
    if trainer.get("target_template"):
        return render_prompt(str(trainer["target_template"]), example)
    target_field = trainer.get("target_field") or trainer.get("label_field")
    if target_field:
        if target_field not in example:
            raise KeyError(f"target field missing: {target_field}")
        return str(example[target_field])
    return rendered_prompt


def _canonical_example(row: Any, spec: DatasetFormatSpec) -> dict[str, Any]:
    row_map = {str(key): value for key, value in row.items()}
    canonical = dict(row_map)
    for prompt_field, raw_field in (spec.field_map or {}).items():
        if raw_field in row_map:
            canonical[prompt_field] = row_map[raw_field]
    return canonical


def _required_split(dataset: Any, split_name: str) -> Any:
    if isinstance(dataset, Mapping) or hasattr(dataset, "keys"):
        if split_name not in dataset:
            raise ValueError(f"missing required train split: {split_name}")
        return dataset[split_name]
    return dataset


def _optional_split(dataset: Any, split_name: str | None) -> Any | None:
    if split_name is None:
        return None
    if isinstance(dataset, Mapping) or hasattr(dataset, "keys"):
        return dataset[split_name] if split_name in dataset else None
    return dataset


def _monitor_config(stopping: Mapping[str, Any]) -> LossMonitorConfig:
    return LossMonitorConfig(
        metric=str(stopping.get("metric", "eval_loss")),
        mode=str(stopping.get("mode", "min")),
        min_delta=float(stopping.get("min_delta", 0.0)),
        patience=stopping.get("patience"),
        plateau_window=stopping.get("plateau_window"),
        plateau_min_delta=float(stopping.get("plateau_min_delta", 0.0)),
        loss_increase_relative=stopping.get("loss_increase_relative"),
        loss_increase_patience=int(stopping.get("loss_increase_patience", 1)),
    )


def _load_hf_deps() -> dict[str, Any]:
    try:
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )
    except ImportError as exc:  # pragma: no cover - exercised only without optional deps.
        raise RuntimeError(
            "install WeightTraits with the training extra to run HF/PEFT training"
        ) from exc
    return {
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "TaskType": TaskType,
        "get_peft_model": get_peft_model,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoModelForSeq2SeqLM": AutoModelForSeq2SeqLM,
        "AutoTokenizer": AutoTokenizer,
        "DataCollatorForLanguageModeling": DataCollatorForLanguageModeling,
        "DataCollatorForSeq2Seq": DataCollatorForSeq2Seq,
        "Seq2SeqTrainer": Seq2SeqTrainer,
        "Trainer": Trainer,
        "TrainerCallback": TrainerCallback,
        "TrainingArguments": TrainingArguments,
    }


def _hf_dataset_from_records(
    deps: dict[str, Any],
    tokenizer: Any,
    model_task: str,
    records: Sequence[RenderedTrainingRecord],
    job: Mapping[str, Any],
) -> Any:
    dataset = deps["Dataset"].from_list([record.to_dict() for record in records])
    trainer = dict(job.get("trainer", {}))
    source_len = int(trainer.get("max_source_length", trainer.get("max_length", 512)))
    target_len = int(trainer.get("max_target_length", 128))

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        if model_task == "seq2seq":
            model_inputs = tokenizer(
                batch["text"],
                max_length=source_len,
                truncation=True,
            )
            labels = tokenizer(
                text_target=batch["target"],
                max_length=target_len,
                truncation=True,
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        full_text = [
            text if text == target else f"{text}\n{target}"
            for text, target in zip(batch["text"], batch["target"])
        ]
        model_inputs = tokenizer(full_text, max_length=source_len, truncation=True)
        model_inputs["labels"] = [list(ids) for ids in model_inputs["input_ids"]]
        return model_inputs

    return dataset.map(tokenize, batched=True, remove_columns=["text", "target"])


def _training_arguments(deps: dict[str, Any], run: TrainingRunSpec, *, has_eval: bool) -> Any:
    trainer = dict(run.job.get("trainer", {}))
    ignored = {
        "target_field",
        "label_field",
        "target_template",
        "max_source_length",
        "max_target_length",
        "max_length",
        "model_task",
    }
    kwargs = {key: value for key, value in trainer.items() if key not in ignored}
    kwargs["output_dir"] = run.output_dir
    kwargs.setdefault("report_to", [])
    kwargs.setdefault("save_strategy", "steps" if kwargs.get("save_steps") else "no")
    if has_eval:
        kwargs.setdefault("eval_strategy", "steps" if kwargs.get("eval_steps") else "epoch")
    else:
        kwargs.setdefault("eval_strategy", "no")
    try:
        return deps["TrainingArguments"](**kwargs)
    except TypeError:
        if "eval_strategy" in kwargs:
            kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
        return deps["TrainingArguments"](**kwargs)


def _data_collator(deps: dict[str, Any], tokenizer: Any, model: Any, model_task: str) -> Any:
    if model_task == "seq2seq":
        return deps["DataCollatorForSeq2Seq"](tokenizer=tokenizer, model=model)
    return deps["DataCollatorForLanguageModeling"](tokenizer=tokenizer, mlm=False)


def _model_task(job: Mapping[str, Any]) -> str:
    trainer = dict(job.get("trainer", {}))
    explicit = trainer.get("model_task") or trainer.get("task_type")
    if explicit in {"seq2seq", "causal_lm"}:
        return str(explicit)
    model_name = " ".join(
        str(job.get(key, "")).lower() for key in ("base_model", "model_family", "init_from")
    )
    if any(name in model_name for name in ("t5", "flan", "bart", "pegasus")):
        return "seq2seq"
    return "causal_lm"


def _metric_value(source: Any, key: str) -> float | None:
    if isinstance(source, Mapping):
        return _maybe_float(source.get(key))
    if isinstance(source, list):
        for row in reversed(source):
            if isinstance(row, Mapping) and key in row:
                return _maybe_float(row[key])
    return None


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _unique_items(items: Sequence[Sequence[str]]) -> list[str]:
    out: list[str] = []
    for group in items:
        for item in group:
            if item not in out:
                out.append(item)
    return out


def _default_hf_loader() -> DatasetLoader:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only without optional dependency.
        raise RuntimeError(
            "datasets is not installed; install WeightTraits with the training extra"
        ) from exc
    return load_dataset
