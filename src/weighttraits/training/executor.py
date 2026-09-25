"""Per-row training execution behind run-list contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import shutil
from typing import Any, Callable, Mapping, Protocol, Sequence

from weighttraits.training.data_formats import DatasetFormatSpec
from weighttraits.training.datasets import (
    DatasetCacheRecipe,
    DatasetLoader,
    DatasetRegistryEntry,
    canonical_dataset_example,
    dataset_cache_split_path,
    dataset_example_passes_filter,
    dataset_format_spec_fingerprint,
    dataset_registry_entry_fingerprint,
    load_cached_dataset_splits,
)
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
    metadata: dict[str, Any] = field(default_factory=dict)
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingRunResult:
    node_id: str
    status: str
    data: dict[str, Any]
    artifacts: dict[str, str]
    execution_overrides: dict[str, Any] = field(default_factory=dict)
    backend_metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
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
    data_cache_root: str | Path | None = None,
    require_data_cache: bool = False,
    expected_cache_recipe: DatasetCacheRecipe | None = None,
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
    train_split = spec.train_split or entry.train_split or "train"
    eval_split = spec.eval_split or entry.eval_split
    dataset = None
    if data_cache_root is not None:
        dataset = load_cached_dataset_splits(
            data_cache_root,
            dataset_id=run.dataset_id,
            train_split=train_split,
            eval_split=eval_split,
            require=require_data_cache,
            registry_entry=entry,
            format_spec=spec,
            expected_recipe=expected_cache_recipe,
        )
    if dataset is None:
        if require_data_cache:
            raise FileNotFoundError(f"missing cached dataset for {run.dataset_id}")
        dataset_loader = loader or _default_hf_loader()
        dataset = dataset_loader(*entry.hf_args, **(entry.hf_kwargs or {}))

    train_dataset = _required_split(dataset, train_split)
    eval_dataset = _optional_split(dataset, eval_split) if eval_split else None
    if eval_split and eval_dataset is None and not allow_missing_eval:
        raise ValueError(f"missing eval split for {run.dataset_id}: {eval_split}")

    train_records, n_dropped_train, train_issues = _render_dataset_split(
        train_dataset,
        run=run,
        spec=spec,
        filter_spec=entry.filter,
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
            filter_spec=entry.filter,
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
    data_cache_root: str | Path | None = None,
    require_data_cache: bool = False,
    expected_cache_recipe: DatasetCacheRecipe | None = None,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    allow_missing_eval: bool = True,
    execution_overrides: Mapping[str, Any] | None = None,
    write_ledger: bool = True,
) -> TrainingRunResult:
    """Run one training row and write ledger events for lifecycle and monitor state."""
    overrides = _execution_overrides(execution_overrides)
    provenance = _training_run_provenance(
        run,
        registry,
        format_specs,
        data_cache_root=data_cache_root,
        expected_cache_recipe=expected_cache_recipe,
        execution_overrides=overrides,
    )
    provenance_ref: dict[str, Any] = {}
    if write_ledger:
        extra = {
            "run_id": run.run_id,
            "array_index": run.array_index,
            "provenance": provenance,
        }
        if overrides:
            extra["execution_overrides"] = overrides
        started_event = TrainingLedgerEvent(
            node_id=run.node_id,
            status="started",
            message="training row started",
            extra=extra,
        )
        append_ledger_event(run.ledger_path, started_event)
        _append_training_log(run, started_event)
    try:
        _required_max_steps(run)
        data = prepare_training_data(
            run,
            registry,
            format_specs,
            loader=loader,
            data_cache_root=data_cache_root,
            require_data_cache=require_data_cache,
            expected_cache_recipe=expected_cache_recipe,
            max_train_samples=max_train_samples,
            max_eval_samples=max_eval_samples,
            allow_missing_eval=allow_missing_eval,
        )
        if not data.valid:
            raise ValueError(f"training data is not valid for {run.node_id}: {data.issues}")
        provenance["cache_splits"] = _cache_split_provenance(
            run,
            registry,
            format_specs,
            data_cache_root=data_cache_root,
        )
        provenance_ref = _write_training_provenance(run, provenance)
        if write_ledger:
            prepared_event = TrainingLedgerEvent(
                node_id=run.node_id,
                status="running",
                message="training data prepared",
                extra={
                    "data": data.summary(),
                    "provenance": provenance_ref,
                },
            )
            append_ledger_event(run.ledger_path, prepared_event)
            _append_training_log(run, prepared_event)

        monitor = LossMonitor(_monitor_config(run.job.get("stopping", {})))
        decisions: list[MonitorDecision] = []

        def on_event(event: TrainingEvent) -> MonitorDecision:
            decision = monitor.update(event)
            decisions.append(decision)
            if write_ledger:
                monitor_event = TrainingLedgerEvent(
                    node_id=run.node_id,
                    status="running",
                    step=event.step,
                    train_loss=event.train_loss,
                    eval_loss=event.eval_loss,
                    warnings=tuple(decision.warnings),
                    stop_reasons=tuple(decision.reasons),
                    message="training monitor update",
                    extra={
                        **decision.state,
                        "provenance_sha256": provenance_ref.get("sha256"),
                    },
                )
                append_ledger_event(run.ledger_path, monitor_event)
                _append_training_log(run, monitor_event)
            return decision

        train_backend = backend or HfPeftTrainingBackend()
        backend_result = train_backend.train(run, data, on_event)
        _enforce_required_max_steps(run, backend_result)
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
            execution_overrides=overrides,
            backend_metadata=dict(backend_result.metadata),
            provenance=provenance_ref,
            step=backend_result.step,
            train_loss=backend_result.train_loss,
            eval_loss=backend_result.eval_loss,
            warnings=tuple(warnings),
            stop_reasons=tuple(stop_reasons),
            message=backend_result.message,
        )
        if write_ledger:
            finished_event = TrainingLedgerEvent(
                node_id=run.node_id,
                status=status,
                step=result.step,
                train_loss=result.train_loss,
                eval_loss=result.eval_loss,
                warnings=result.warnings,
                stop_reasons=result.stop_reasons,
                message=result.message or "training row finished",
                extra={
                    "artifacts": artifacts,
                    "data": data.summary(),
                    "execution_overrides": overrides,
                    "backend_metadata": backend_result.metadata,
                    "provenance": provenance_ref,
                },
            )
            append_ledger_event(run.ledger_path, finished_event)
            _append_training_log(run, finished_event)
        return result
    except Exception as exc:
        if write_ledger:
            failed_event = TrainingLedgerEvent(
                node_id=run.node_id,
                status="failed",
                message=str(exc),
                extra={
                    "provenance": provenance_ref or provenance,
                },
            )
            append_ledger_event(run.ledger_path, failed_event)
            _append_training_log(run, failed_event)
        raise


def dry_run_training_row(
    run: TrainingRunSpec,
    *,
    execution_overrides: Mapping[str, Any] | None = None,
) -> TrainingRunResult:
    """Return the selected row without loading datasets or models."""
    return TrainingRunResult(
        node_id=run.node_id,
        status="dry_run",
        data={},
        artifacts=dict(run.expected_artifacts),
        execution_overrides=_execution_overrides(execution_overrides),
        message="selected training row without loading datasets or models",
    )


def _append_training_log(run: TrainingRunSpec, event: TrainingLedgerEvent) -> None:
    path = run.expected_artifacts.get("training_log")
    if not path:
        return
    log = Path(path)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")


class HfPeftTrainingBackend:
    """Hugging Face / PEFT backend imported lazily for real training environments."""

    def train(
        self,
        run: TrainingRunSpec,
        data: PreparedTrainingData,
        event_callback: MonitorEventCallback,
    ) -> BackendTrainResult:
        deps = _load_hf_deps()
        _set_training_seed(deps, run.job)
        model_task = _model_task(run.job)
        pretrained_kwargs = _pretrained_kwargs(run)
        tokenizer = deps["AutoTokenizer"].from_pretrained(run.init_from, **pretrained_kwargs)
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None):
            tokenizer.pad_token = tokenizer.eos_token
        _apply_tokenizer_padding_side(tokenizer, run.job)
        model_cls = (
            deps["AutoModelForSeq2SeqLM"]
            if model_task == "seq2seq"
            else deps["AutoModelForCausalLM"]
        )
        model = model_cls.from_pretrained(
            run.init_from,
            **_model_pretrained_kwargs(run, deps["torch"]),
        )
        backend_metadata: dict[str, Any] = {
            "base_model": str(run.job.get("base_model", "")),
            "base_model_revision": run.job.get("base_model_revision"),
            "init_from": run.init_from,
            "runtime_versions": _training_runtime_versions(),
        }
        preflight_artifacts: dict[str, str] = {}
        if run.method == "lora":
            model, lora_audit = self._wrap_lora(model, run, deps, model_task)
            audit_path = _write_lora_target_audit(run, lora_audit)
            backend_metadata["lora_target_audit"] = lora_audit
            preflight_artifacts["lora_target_audit"] = str(audit_path)

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
        args = _training_arguments(
            deps,
            run,
            has_eval=eval_dataset is not None,
            model_task=model_task,
        )
        callback = _make_loss_monitor_callback(deps["TrainerCallback"], event_callback)
        trainer_cls = deps["Seq2SeqTrainer"] if model_task == "seq2seq" else deps["Trainer"]
        collator = _data_collator(deps, tokenizer, model, model_task, run.job)
        trainer = trainer_cls(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            callbacks=[callback],
            **_trainer_tokenizer_kwargs(trainer_cls, tokenizer),
        )
        output = trainer.train()
        step = int(getattr(trainer.state, "global_step", 0))
        status = "stopped_early" if callback.stopped else "completed"
        _enforce_actual_training_step(run, step=step, status=status)
        artifacts = self._save_artifacts(run, trainer, model, tokenizer)
        artifacts.update(preflight_artifacts)
        removed_checkpoints = _cleanup_trainer_checkpoints(run)
        if removed_checkpoints:
            backend_metadata["removed_trainer_checkpoints"] = removed_checkpoints
        train_loss = _metric_value(getattr(output, "metrics", {}), "train_loss")
        eval_loss = _metric_value(getattr(trainer.state, "log_history", []), "eval_loss")
        return BackendTrainResult(
            status=status,
            step=step,
            train_loss=train_loss,
            eval_loss=eval_loss,
            artifacts=artifacts,
            metadata=backend_metadata,
        )

    def _wrap_lora(
        self,
        model: Any,
        run: TrainingRunSpec,
        deps: dict[str, Any],
        model_task: str,
    ) -> tuple[Any, dict[str, Any]]:
        lora = dict(run.job.get("lora") or {})
        requested_targets = _requested_lora_targets(lora)
        _resolve_lora_target_modules(model, requested_targets)
        task_type = (
            deps["TaskType"].SEQ_2_SEQ_LM if model_task == "seq2seq" else deps["TaskType"].CAUSAL_LM
        )
        config = deps["LoraConfig"](
            r=int(lora.get("r", 8)),
            lora_alpha=int(lora.get("lora_alpha", 16)),
            lora_dropout=float(lora.get("lora_dropout", 0.05)),
            target_modules=requested_targets,
            task_type=task_type,
        )
        wrapped = deps["get_peft_model"](model, config)
        return wrapped, _audit_wrapped_lora_model(wrapped, requested_targets)

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


def _requested_lora_targets(lora: Mapping[str, Any]) -> list[str]:
    raw_targets = lora.get("target_modules")
    if not isinstance(raw_targets, Sequence) or isinstance(raw_targets, (str, bytes)):
        raise ValueError("LoRA target_modules must be a non-empty sequence of module names")
    targets = [str(target).strip() for target in raw_targets]
    if not targets or any(not target for target in targets):
        raise ValueError("LoRA target_modules must be a non-empty sequence of module names")
    if len(targets) != len(set(targets)):
        raise ValueError(f"LoRA target_modules contains duplicates: {targets}")
    return targets


def _resolve_lora_target_modules(
    model: Any,
    requested_targets: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    """Resolve PEFT list-style target suffixes and reject every unmatched request."""
    module_names = [str(name) for name, _ in model.named_modules() if name]
    matches = {
        target: tuple(
            sorted(name for name in module_names if _module_name_matches_target(name, target))
        )
        for target in requested_targets
    }
    missing = [target for target, names in matches.items() if not names]
    if missing:
        suggestions = {
            target: sorted(
                name for name in module_names if name.rsplit(".", 1)[-1].startswith(f"{target}_")
            )[:8]
            for target in missing
        }
        suggestion_text = "; ".join(
            f"{target} -> {names}" for target, names in suggestions.items() if names
        )
        detail = f" Close model-name matches: {suggestion_text}." if suggestion_text else ""
        raise ValueError(
            "LoRA target modules matched no model modules: "
            f"{missing}. Requested targets: {list(requested_targets)}.{detail}"
        )
    return matches


def _audit_wrapped_lora_model(
    model: Any,
    requested_targets: Sequence[str],
) -> dict[str, Any]:
    adapter_module_names = sorted(
        str(name)
        for name, module in model.named_modules()
        if name and _module_has_lora_factors(module)
    )
    matched_by_target = {
        target: [name for name in adapter_module_names if _module_name_matches_target(name, target)]
        for target in requested_targets
    }
    missing = [target for target, names in matched_by_target.items() if not names]
    if missing:
        raise ValueError(
            "PEFT created no adapter layers for requested LoRA targets: "
            f"{missing}. Resolved adapter modules: {adapter_module_names}"
        )

    lora_parameter_names: list[str] = []
    n_trainable_lora_parameters = 0
    n_trainable_model_parameters = 0
    for name, parameter in model.named_parameters():
        if not getattr(parameter, "requires_grad", False):
            continue
        n_parameters = int(parameter.numel())
        n_trainable_model_parameters += n_parameters
        if ".lora_" in str(name):
            lora_parameter_names.append(str(name))
            n_trainable_lora_parameters += n_parameters

    return {
        "schema_version": 1,
        "requested_target_modules": list(requested_targets),
        "matched_module_names_by_target": matched_by_target,
        "resolved_module_names": adapter_module_names,
        "n_resolved_modules": len(adapter_module_names),
        "n_trainable_lora_parameters": n_trainable_lora_parameters,
        "n_trainable_model_parameters": n_trainable_model_parameters,
        "n_lora_parameter_tensors": len(lora_parameter_names),
        "lora_parameter_names": sorted(lora_parameter_names),
        "unmatched_target_modules": [],
    }


def _module_name_matches_target(module_name: str, target: str) -> bool:
    return module_name == target or module_name.endswith(f".{target}")


def _module_has_lora_factors(module: Any) -> bool:
    return hasattr(module, "lora_A") and hasattr(module, "lora_B")


def _write_lora_target_audit(run: TrainingRunSpec, audit: Mapping[str, Any]) -> Path:
    path = Path(
        run.expected_artifacts.get(
            "lora_target_audit",
            str(Path(run.output_dir) / "lora_target_audit.json"),
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(audit), indent=2, sort_keys=True) + "\n")
    return path


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
    filter_spec: Mapping[str, Any] | None,
    max_samples: int | None,
    split_name: str | None,
) -> tuple[tuple[RenderedTrainingRecord, ...], int, tuple[str, ...]]:
    records: list[RenderedTrainingRecord] = []
    issues: list[str] = []
    dropped = 0
    for index, row in enumerate(dataset_split):
        if max_samples is not None and len(records) >= max_samples:
            break
        if not isinstance(row, Mapping) and not hasattr(row, "items"):
            dropped += 1
            issues.append(f"{split_name}[{index}]: row is not a mapping")
            continue
        try:
            canonical = _canonical_example(row, spec)
            keep, _ = dataset_example_passes_filter(canonical, filter_spec)
            if not keep:
                dropped += 1
                continue
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
    return canonical_dataset_example(row, spec)


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


def _enforce_required_max_steps(
    run: TrainingRunSpec,
    backend_result: BackendTrainResult,
) -> None:
    _enforce_actual_training_step(
        run,
        step=backend_result.step,
        status=backend_result.status,
    )


def _required_max_steps(run: TrainingRunSpec) -> int | None:
    """Validate and return an exact-step contract before expensive training starts."""

    trainer = dict(run.job.get("trainer", {}))
    required = trainer.get("require_max_steps", False)
    if not isinstance(required, bool):
        raise ValueError("trainer.require_max_steps must be a boolean")
    if not required:
        return None

    expected = trainer.get("max_steps")
    if isinstance(expected, bool) or not isinstance(expected, int) or expected < 1:
        raise ValueError(
            "trainer.max_steps must be a positive integer when require_max_steps is true"
        )
    return expected


def _enforce_actual_training_step(
    run: TrainingRunSpec,
    *,
    step: int | None,
    status: str,
) -> None:
    """Reject a short run before final model artifacts are published."""

    expected = _required_max_steps(run)
    if expected is None:
        return
    if step != expected:
        raise RuntimeError(
            f"run {run.node_id} requires exactly {expected} training steps, but backend "
            f"reported {step!r} with status {status!r}"
        )


def _load_hf_deps() -> dict[str, Any]:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            Trainer,
            TrainerCallback,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:  # pragma: no cover - exercised only without optional deps.
        raise RuntimeError(
            "install WeightTraits with the training extra to run HF/PEFT training"
        ) from exc
    return {
        "Dataset": Dataset,
        "torch": torch,
        "LoraConfig": LoraConfig,
        "TaskType": TaskType,
        "get_peft_model": get_peft_model,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoModelForSeq2SeqLM": AutoModelForSeq2SeqLM,
        "AutoTokenizer": AutoTokenizer,
        "DataCollatorForSeq2Seq": DataCollatorForSeq2Seq,
        "Seq2SeqTrainer": Seq2SeqTrainer,
        "Seq2SeqTrainingArguments": Seq2SeqTrainingArguments,
        "Trainer": Trainer,
        "TrainerCallback": TrainerCallback,
        "TrainingArguments": TrainingArguments,
        "set_seed": set_seed,
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
    causal_len = int(trainer.get("max_seq_length", trainer.get("max_length", source_len)))
    causal_loss_scope = _causal_loss_scope(trainer)
    min_prompt_tokens = trainer.get("min_prompt_tokens", 0)

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
        return _tokenize_causal_batch(
            tokenizer,
            batch["text"],
            batch["target"],
            max_length=causal_len,
            loss_scope=causal_loss_scope,
            min_prompt_tokens=min_prompt_tokens,
        )

    return dataset.map(tokenize, batched=True, remove_columns=["text", "target"])


def _training_arguments(
    deps: dict[str, Any],
    run: TrainingRunSpec,
    *,
    has_eval: bool,
    model_task: str,
) -> Any:
    trainer = dict(run.job.get("trainer", {}))
    ignored = {
        "target_field",
        "label_field",
        "target_template",
        "max_source_length",
        "max_target_length",
        "max_length",
        "max_seq_length",
        "model_task",
        "causal_loss_scope",
        "min_prompt_tokens",
        "require_max_steps",
        "padding_side",
        "model_dtype",
        "pad_to_multiple_of",
        "cleanup_checkpoints_on_success",
    }
    kwargs = {key: value for key, value in trainer.items() if key not in ignored}
    kwargs["output_dir"] = run.output_dir
    kwargs.setdefault("report_to", [])
    kwargs.setdefault("save_strategy", "steps" if kwargs.get("save_steps") else "no")
    if has_eval:
        kwargs.setdefault("eval_strategy", "steps" if kwargs.get("eval_steps") else "epoch")
    else:
        kwargs.setdefault("eval_strategy", "no")
    args_cls = (
        deps.get("Seq2SeqTrainingArguments", deps["TrainingArguments"])
        if model_task == "seq2seq"
        else deps["TrainingArguments"]
    )
    try:
        return args_cls(**kwargs)
    except TypeError:
        if "eval_strategy" in kwargs:
            kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
        return args_cls(**kwargs)


def _data_collator(
    deps: dict[str, Any],
    tokenizer: Any,
    model: Any,
    model_task: str,
    job: Mapping[str, Any] | None = None,
) -> Any:
    trainer = dict((job or {}).get("trainer", {}))
    if model_task == "seq2seq":
        pad_to_multiple_of = trainer.get("pad_to_multiple_of")
        if pad_to_multiple_of is not None and (
            isinstance(pad_to_multiple_of, bool)
            or not isinstance(pad_to_multiple_of, int)
            or pad_to_multiple_of < 1
        ):
            raise ValueError("trainer.pad_to_multiple_of must be a positive integer or null")
        kwargs: dict[str, Any] = {
            "tokenizer": tokenizer,
            "model": model,
            "label_pad_token_id": -100,
        }
        if pad_to_multiple_of is not None:
            kwargs["pad_to_multiple_of"] = pad_to_multiple_of
        return deps["DataCollatorForSeq2Seq"](**kwargs)
    return _CausalDataCollator(
        tokenizer,
        pad_to_multiple_of=trainer.get("pad_to_multiple_of"),
    )


def _cleanup_trainer_checkpoints(run: TrainingRunSpec) -> list[str]:
    """Remove resumable Trainer checkpoints only after final artifacts were saved."""

    trainer = dict(run.job.get("trainer", {}))
    if not bool(trainer.get("cleanup_checkpoints_on_success", False)):
        return []
    removed: list[str] = []
    for checkpoint in sorted(Path(run.output_dir).glob("checkpoint-*")):
        if not checkpoint.is_dir():
            continue
        shutil.rmtree(checkpoint)
        removed.append(str(checkpoint))
    return removed


class _CausalDataCollator:
    """Pad causal examples without replacing prompt-masked labels."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        pad_to_multiple_of: int | None = None,
    ) -> None:
        if pad_to_multiple_of is not None and (
            isinstance(pad_to_multiple_of, bool)
            or not isinstance(pad_to_multiple_of, int)
            or pad_to_multiple_of < 1
        ):
            raise ValueError("trainer.pad_to_multiple_of must be a positive integer or null")
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        labels = [list(feature["labels"]) for feature in features]
        inputs = [
            {key: value for key, value in feature.items() if key != "labels"}
            for feature in features
        ]
        pad_kwargs: dict[str, Any] = {"padding": True, "return_tensors": "pt"}
        if self.pad_to_multiple_of is not None:
            pad_kwargs["pad_to_multiple_of"] = self.pad_to_multiple_of
        batch = self.tokenizer.pad(inputs, **pad_kwargs)
        sequence_length = int(batch["input_ids"].shape[1])
        padding_side = getattr(self.tokenizer, "padding_side", "right")
        padded_labels = []
        for row in labels:
            padding = [-100] * (sequence_length - len(row))
            padded_labels.append(padding + row if padding_side == "left" else row + padding)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def _set_training_seed(deps: Mapping[str, Any], job: Mapping[str, Any]) -> int:
    """Seed RNGs before model and PEFT adapter construction."""
    trainer = dict(job.get("trainer", {}))
    seed = int(trainer.get("seed", 42))
    deps["set_seed"](seed)
    return seed


def _pretrained_kwargs(run: TrainingRunSpec) -> dict[str, str]:
    """Pin remote root initialization while leaving local lineage paths untouched."""

    base_model = str(run.job.get("base_model", ""))
    revision = run.job.get("base_model_revision")
    if run.init_from != base_model or revision is None:
        return {}
    revision_text = str(revision).strip()
    return {"revision": revision_text} if revision_text else {}


def _training_runtime_versions() -> dict[str, str]:
    """Capture the numerical software environment in every terminal ledger row."""

    versions = {"python": platform.python_version()}
    for distribution in ("torch", "transformers", "datasets", "accelerate", "tokenizers"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def _training_run_provenance(
    run: TrainingRunSpec,
    registry: Mapping[str, DatasetRegistryEntry],
    format_specs: Mapping[str, DatasetFormatSpec],
    *,
    data_cache_root: str | Path | None,
    expected_cache_recipe: DatasetCacheRecipe | None,
    execution_overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a compact, deterministic receipt for the effective training row."""

    entry = registry.get(run.dataset_id or "")
    spec = format_specs.get(run.dataset_id or "")
    run_list_path = run.runner.get("run_list_path")
    run_list_file = Path(str(run_list_path)) if run_list_path else None
    return {
        "schema_version": 1,
        "protocol_id": run.job.get("protocol_id"),
        "planned_training_config_sha256": run.job.get("training_config_sha256"),
        "effective_job_sha256": _canonical_json_sha256(run.job),
        "run_spec_sha256": _canonical_json_sha256(run.to_dict()),
        "run_list_path": None if run_list_file is None else str(run_list_file),
        "run_list_sha256": (
            _file_sha256(run_list_file)
            if run_list_file is not None and run_list_file.is_file()
            else None
        ),
        "dataset_id": run.dataset_id,
        "dataset_registry_entry_sha256": (
            dataset_registry_entry_fingerprint(entry) if entry is not None else None
        ),
        "dataset_format_spec_sha256": (
            dataset_format_spec_fingerprint(spec) if spec is not None else None
        ),
        "data_cache_root": None if data_cache_root is None else str(data_cache_root),
        "expected_cache_recipe": (
            None if expected_cache_recipe is None else expected_cache_recipe.to_dict()
        ),
        "execution_overrides": dict(execution_overrides),
        "runtime_versions": _training_runtime_versions(),
    }


def _cache_split_provenance(
    run: TrainingRunSpec,
    registry: Mapping[str, DatasetRegistryEntry],
    format_specs: Mapping[str, DatasetFormatSpec],
    *,
    data_cache_root: str | Path | None,
) -> dict[str, dict[str, Any]]:
    """Read the small, already-validated cache metadata files into the receipt."""

    if data_cache_root is None or run.dataset_id is None:
        return {}
    entry = registry.get(run.dataset_id)
    spec = format_specs.get(run.dataset_id)
    if entry is None or spec is None:
        return {}
    train_split = spec.train_split or entry.train_split or "train"
    eval_split = spec.eval_split or entry.eval_split
    split_names = [train_split]
    if eval_split:
        split_names.append(eval_split)
    receipt: dict[str, dict[str, Any]] = {}
    recorded_fields = (
        "cache_metadata_version",
        "dataset_id",
        "split",
        "sample_strategy",
        "sample_seed",
        "shuffle_buffer_size",
        "cache_limit",
        "cache_n_filtered",
        "cache_n_dropped",
        "source_split_fingerprint",
        "source_split_num_rows",
        "dataset_registry_entry_sha256",
        "dataset_format_spec_sha256",
        "cache_jsonl_sha256",
        "cache_row_count",
    )
    for split_name in split_names:
        cache_path = dataset_cache_split_path(data_cache_root, run.dataset_id, split_name)
        metadata_path = cache_path.with_suffix(".metadata.json")
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text())
        receipt[split_name] = {
            "cache_path": str(cache_path),
            "metadata_path": str(metadata_path),
            "metadata_sha256": _file_sha256(metadata_path),
            **{field: metadata.get(field) for field in recorded_fields},
        }
    return receipt


def _write_training_provenance(
    run: TrainingRunSpec,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically publish one immutable provenance receipt before model training."""

    path = Path(run.output_dir) / "provenance.json"
    payload = json.dumps(dict(provenance), indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != payload:
            raise RuntimeError(
                f"refusing to mix a different training protocol in existing output {path}"
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(payload)
        temporary.replace(path)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_pretrained_kwargs(
    run: TrainingRunSpec,
    torch_module: Any,
) -> dict[str, Any]:
    """Build model-only loading kwargs, including the requested parameter dtype."""

    kwargs: dict[str, Any] = dict(_pretrained_kwargs(run))
    trainer = dict(run.job.get("trainer", {}))
    if "model_dtype" not in trainer:
        return kwargs
    model_dtype = trainer["model_dtype"]
    allowed = {"auto", "bfloat16", "float16", "float32"}
    if not isinstance(model_dtype, str) or model_dtype not in allowed:
        raise ValueError("trainer.model_dtype must be one of: auto, bfloat16, float16, float32")
    kwargs["torch_dtype"] = "auto" if model_dtype == "auto" else getattr(torch_module, model_dtype)
    return kwargs


def _apply_tokenizer_padding_side(
    tokenizer: Any,
    job: Mapping[str, Any],
) -> None:
    """Apply an explicitly requested tokenizer padding direction."""

    trainer = dict(job.get("trainer", {}))
    if "padding_side" not in trainer:
        return
    padding_side = trainer["padding_side"]
    if not isinstance(padding_side, str) or padding_side not in {"left", "right"}:
        raise ValueError("trainer.padding_side must be 'left' or 'right'")
    tokenizer.padding_side = padding_side


def _causal_loss_scope(trainer: Mapping[str, Any]) -> str:
    scope = str(trainer.get("causal_loss_scope", "completion"))
    if scope not in {"completion", "all_tokens"}:
        raise ValueError("trainer.causal_loss_scope must be 'completion' or 'all_tokens'")
    return scope


def _tokenize_causal_batch(
    tokenizer: Any,
    texts: Sequence[str],
    targets: Sequence[str],
    *,
    max_length: int,
    loss_scope: str,
    min_prompt_tokens: int = 0,
) -> dict[str, list[list[int]]]:
    if max_length < 1:
        raise ValueError("causal max sequence length must be positive")
    if isinstance(min_prompt_tokens, bool) or not isinstance(min_prompt_tokens, int):
        raise ValueError("trainer.min_prompt_tokens must be an integer")
    if min_prompt_tokens < 0 or min_prompt_tokens >= max_length:
        raise ValueError(
            "trainer.min_prompt_tokens must satisfy "
            f"0 <= min_prompt_tokens < max_length ({max_length})"
        )
    if loss_scope == "all_tokens":
        full_text = [
            text if text == target else f"{text}\n{target}" for text, target in zip(texts, targets)
        ]
        model_inputs = tokenizer(full_text, max_length=max_length, truncation=True)
        model_inputs["labels"] = [list(ids) for ids in model_inputs["input_ids"]]
        return model_inputs

    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    labels: list[list[int]] = []
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    for text, target in zip(texts, targets):
        if text == target:
            encoded = tokenizer(text, max_length=max_length, truncation=True)
            ids = _single_input_ids(encoded)
            row_labels = list(ids)
        else:
            prompt = tokenizer(f"{text}\n", add_special_tokens=True, truncation=False)
            completion = tokenizer(target, add_special_tokens=False, truncation=False)
            prompt_ids = _single_input_ids(prompt)
            completion_ids = _single_input_ids(completion)
            if eos_token_id is not None and (
                not completion_ids or completion_ids[-1] != eos_token_id
            ):
                completion_ids.append(int(eos_token_id))
            completion_ids = completion_ids[: max_length - min_prompt_tokens]
            prompt_budget = max_length - len(completion_ids)
            prompt_ids = prompt_ids[:prompt_budget]
            ids = prompt_ids + completion_ids
            row_labels = [-100] * len(prompt_ids) + completion_ids
        input_ids.append(ids)
        attention_mask.append([1] * len(ids))
        labels.append(row_labels)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _single_input_ids(encoded: Mapping[str, Any]) -> list[int]:
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("expected a single tokenized example")
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def _trainer_tokenizer_kwargs(trainer_cls: Any, tokenizer: Any) -> dict[str, Any]:
    parameters = inspect.signature(trainer_cls.__init__).parameters
    if "processing_class" in parameters:
        return {"processing_class": tokenizer}
    if "tokenizer" in parameters:
        return {"tokenizer": tokenizer}
    return {}


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


def _execution_overrides(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    if not overrides:
        return {}
    return {str(key): value for key, value in overrides.items()}


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
