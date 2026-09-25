"""Training job planning from enriched lineage manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from weighttraits.taskdata.assignment import load_manifest_rows
from weighttraits.training.prompts import template_fields


@dataclass(frozen=True)
class PromptResolution:
    template: str
    source: str


@dataclass(frozen=True)
class TrainingJob:
    node_id: str
    parent_id: str
    depth: int
    path: tuple[str, ...]
    base_model: str
    base_model_revision: str | None
    model_family: str
    method: str
    task_family: str | None
    dataset_id: str | None
    prompt_template: str
    prompt_source: str
    prompt_fields: tuple[str, ...]
    output_dir: str
    init_from: str
    expected_artifacts: dict[str, str]
    trainer: dict[str, Any]
    stopping: dict[str, Any]
    lora: dict[str, Any] | None
    protocol_id: str | None = None
    training_config_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["path"] = list(self.path)
        return out


def load_training_config(path: str | Path) -> dict[str, Any]:
    """Load one effective training config, resolving an optional relative parent.

    A config may declare ``extends: path/to/base.yaml`` at the document root.
    Nested mappings are merged recursively while scalars and lists in the child
    replace the parent.  This keeps shared paper protocols in one immutable file
    while allowing cohort-specific output roots and identifiers.
    """

    return _load_training_config(Path(path), stack=())


def _load_training_config(path: Path, *, stack: tuple[Path, ...]) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in stack:
        cycle = " -> ".join(str(item) for item in (*stack, resolved))
        raise ValueError(f"training config extends cycle: {cycle}")
    config = yaml.safe_load(resolved.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"training config must be a mapping: {path}")
    current = config.get("training", config)
    if not isinstance(current, dict):
        raise ValueError(f"training config body must be a mapping: {path}")
    parent_path = config.get("extends")
    if parent_path is None:
        return dict(current)
    if not isinstance(parent_path, str) or not parent_path.strip():
        raise ValueError(f"training config extends must be a non-empty path: {path}")
    parent = Path(parent_path)
    if not parent.is_absolute():
        parent = resolved.parent / parent
    base = _load_training_config(parent, stack=(*stack, resolved))
    return _deep_merge_mappings(base, current)


def _deep_merge_mappings(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge_mappings(existing, value)
        else:
            merged[key] = value
    return merged


def build_training_jobs(
    manifest_rows: list[dict[str, Any]],
    training_config: dict[str, Any],
) -> list[TrainingJob]:
    base_model = _required_str(training_config, "base_model")
    base_model_revision = _optional_str(training_config.get("base_model_revision"))
    model_family = str(training_config.get("model_family", base_model))
    method = str(training_config.get("method", training_config.get("adapter", "full"))).lower()
    if method in {"none", "full_finetune"}:
        method = "full"
    if method not in {"full", "lora"}:
        raise ValueError(f"unsupported training method: {method}")

    output_root = Path(_required_str(training_config, "output_root"))
    protocol_id = _optional_str(training_config.get("protocol_id"))
    training_config_sha256 = _training_config_fingerprint(training_config)
    trainer = dict(training_config.get("trainer", {}))
    if "max_steps" in training_config and "max_steps" not in trainer:
        trainer["max_steps"] = training_config["max_steps"]
    stopping = _stopping_config(training_config)
    lora = _lora_config(training_config, method)

    rows_by_id = {
        str(row["node_id"]): row for row in manifest_rows if row.get("grow", "train") == "train"
    }
    jobs: list[TrainingJob] = []
    for row in manifest_rows:
        if row.get("grow", "train") != "train":
            continue
        node_id = str(row["node_id"])
        parent_id = str(row.get("parent_id", "root"))
        if parent_id != "root" and parent_id not in rows_by_id:
            raise ValueError(f"node {node_id} references missing trained parent {parent_id}")
        prompt = resolve_prompt_template(row, training_config)
        output_dir = output_root / node_id
        jobs.append(
            TrainingJob(
                node_id=node_id,
                parent_id=parent_id,
                depth=int(row.get("depth", 0)),
                path=tuple(str(item) for item in row.get("path", ["root", node_id])),
                base_model=base_model,
                base_model_revision=base_model_revision,
                model_family=model_family,
                method=method,
                protocol_id=protocol_id,
                training_config_sha256=training_config_sha256,
                task_family=_optional_str(row.get("task_family")),
                dataset_id=_optional_str(row.get("dataset_id")),
                prompt_template=prompt.template,
                prompt_source=prompt.source,
                prompt_fields=tuple(template_fields(prompt.template)),
                output_dir=str(output_dir),
                init_from=_init_source(parent_id, output_root, method, base_model),
                expected_artifacts=_expected_artifacts(output_dir, method),
                trainer=trainer,
                stopping=stopping,
                lora=lora,
            )
        )
    return jobs


def build_training_jobs_from_files(
    manifest_path: str | Path,
    config_path: str | Path,
) -> list[TrainingJob]:
    return build_training_jobs(load_manifest_rows(manifest_path), load_training_config(config_path))


def write_training_plan(jobs: list[TrainingJob], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for job in jobs:
            handle.write(json.dumps(job.to_dict(), sort_keys=True) + "\n")


def resolve_prompt_template(
    manifest_row: dict[str, Any],
    training_config: dict[str, Any],
) -> PromptResolution:
    prompt_config = dict(training_config.get("prompt", {}))
    model_keys = [
        str(training_config.get("base_model", "")),
        str(training_config.get("model_family", "")),
    ]
    task_family = _optional_str(manifest_row.get("task_family"))
    dataset_id = _optional_str(manifest_row.get("dataset_id"))

    if manifest_row.get("prompt_template"):
        return PromptResolution(str(manifest_row["prompt_template"]), "manifest.prompt_template")

    model_override_items = _model_override_items(prompt_config, model_keys)
    search = []
    for model_key, model_overrides in model_override_items:
        search.extend(
            [
                (
                    f"model[{model_key}].dataset",
                    _nested_template(model_overrides, "dataset_templates", dataset_id),
                ),
                (
                    f"model[{model_key}].task",
                    _nested_template(model_overrides, "task_templates", task_family),
                ),
            ]
        )
    search.extend(
        [
            ("dataset", _nested_template(prompt_config, "dataset_templates", dataset_id)),
            ("task", _nested_template(prompt_config, "task_templates", task_family)),
        ]
    )
    for model_key, model_overrides in model_override_items:
        search.append((f"model[{model_key}].default", model_overrides.get("default_template")))
    search.append(
        ("default", prompt_config.get("default_template")),
    )
    for source, template in search:
        if template:
            return PromptResolution(str(template), f"prompt.{source}")
    raise ValueError(
        "no prompt template found; configure prompt.default_template or a task/dataset/model override"
    )


def _stopping_config(training_config: dict[str, Any]) -> dict[str, Any]:
    stopping = dict(training_config.get("stopping", {}))
    if stopping.get("enabled") is False:
        return {
            "enabled": False,
            "metric": str(stopping.get("metric", "eval_loss")),
            "mode": str(stopping.get("mode", "min")),
            "min_delta": 0.0,
            "patience": None,
            "plateau_window": None,
            "plateau_min_delta": 0.0,
            "loss_increase_relative": None,
            "loss_increase_patience": 1,
        }
    early = dict(stopping.get("early_stopping", {}))
    plateau = dict(stopping.get("plateau", {}))
    warnings = dict(stopping.get("warnings", {}))
    return {
        "enabled": True,
        "metric": str(stopping.get("metric", early.get("metric", "eval_loss"))),
        "mode": str(stopping.get("mode", early.get("mode", "min"))),
        "min_delta": float(early.get("min_delta", stopping.get("min_delta", 0.0))),
        "patience": early.get("patience"),
        "plateau_window": plateau.get("window"),
        "plateau_min_delta": float(plateau.get("min_delta", 0.0)),
        "loss_increase_relative": warnings.get("loss_increase_relative"),
        "loss_increase_patience": int(warnings.get("loss_increase_patience", 1)),
    }


def _training_config_fingerprint(training_config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        training_config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _lora_config(training_config: dict[str, Any], method: str) -> dict[str, Any] | None:
    if method != "lora":
        return None
    lora = dict(training_config.get("lora", {}))
    lora.setdefault("r", 8)
    lora.setdefault("lora_alpha", 16)
    lora.setdefault("lora_dropout", 0.05)
    lora.setdefault("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
    lora.setdefault("merge_after_train", True)
    return lora


def _init_source(parent_id: str, output_root: Path, method: str, base_model: str) -> str:
    if parent_id == "root":
        return base_model
    if method == "lora":
        return str(output_root / parent_id / "merged")
    return str(output_root / parent_id / "model")


def _expected_artifacts(output_dir: Path, method: str) -> dict[str, str]:
    if method == "lora":
        return {
            "adapter": str(output_dir / "adapter"),
            "merged": str(output_dir / "merged"),
            "lora_target_audit": str(output_dir / "lora_target_audit.json"),
            "training_log": str(output_dir / "training_log.jsonl"),
        }
    return {
        "model": str(output_dir / "model"),
        "training_log": str(output_dir / "training_log.jsonl"),
    }


def _nested_template(config: dict[str, Any], group: str, key: str | None) -> Any:
    if key is None:
        return None
    values = config.get(group, {})
    return values.get(key) if isinstance(values, dict) else None


def _model_override_items(
    prompt_config: dict[str, Any],
    model_keys: list[str],
) -> list[tuple[str, dict[str, Any]]]:
    overrides = prompt_config.get("model_overrides", {})
    if not isinstance(overrides, dict):
        return []
    out = []
    seen = set()
    for key in model_keys:
        if not key or key in seen:
            continue
        value = overrides.get(key)
        if isinstance(value, dict):
            out.append((key, value))
            seen.add(key)
    return out


def _required_str(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not value:
        raise ValueError(f"training config requires {key}")
    return str(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)
