"""Fail-closed checkpoint resolution and native held-out probe inference.

No architecture is inferred from adapters, checkpoint contents, or model configuration.  Callers
must name ``model_task`` explicitly, and every training row must independently attest the same
base-model pin and task before a leaf checkpoint can be loaded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Protocol, Sequence

from weighttraits.behavior.contracts import (
    BehaviorProtocolRegistry,
    MODEL_TASKS,
    RenderedBehaviorPrompt,
)
from weighttraits.behavior.responses import (
    BehaviorResponse,
    ResponseProvenance,
    audit_response_grid,
    expected_response_coordinates,
    load_behavior_responses,
    load_rendered_behavior_prompts,
    response_from_text,
)
from weighttraits.manifests.reference import leaf_ids, load_manifest
from weighttraits.training.ledger import latest_status_by_node, load_ledger_events
from weighttraits.training.runlist import TrainingRunSpec, load_training_run_specs


INFERENCE_SCHEMA_VERSION = 2
_PINNED_REVISION = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_WEIGHT_FILE_NAMES = {
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
}


@dataclass(frozen=True)
class ResolvedLeafCheckpoint:
    cohort_id: str
    tree_id: str
    leaf_id: str
    leaf_ordinal: int
    method: str
    artifact_name: str
    checkpoint: Path
    base_model_id: str
    base_model_revision: str
    model_task: str
    training_summary: Path
    training_summary_sha256: str
    run_list: Path
    run_list_sha256: str
    ledger: Path
    ledger_sha256: str
    truth_manifest: Path
    truth_manifest_sha256: str

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        for key in ("checkpoint", "training_summary", "run_list", "ledger", "truth_manifest"):
            row[key] = str(row[key])
        return row


@dataclass(frozen=True)
class ResolvedCheckpointTree:
    cohort_id: str
    tree_id: str
    method: str
    artifact_name: str
    base_model_id: str
    base_model_revision: str
    model_task: str
    leaves: tuple[ResolvedLeafCheckpoint, ...]

    def leaf(self, leaf_id: str) -> ResolvedLeafCheckpoint:
        matches = [leaf for leaf in self.leaves if leaf.leaf_id == leaf_id]
        if not matches:
            raise ValueError(f"leaf {leaf_id!r} is not present in tree {self.tree_id!r}")
        if len(matches) != 1:
            raise ValueError(f"tree {self.tree_id!r} contains duplicate leaf {leaf_id!r}")
        return matches[0]


class LoadedProbeBackend(Protocol):
    """One already-loaded checkpoint used across every requested probe."""

    def generate_batch(
        self,
        prompts: Sequence[str],
        *,
        model_task: str,
        generation_options: Mapping[str, Any],
    ) -> Sequence[str]:
        """Generate one ordered probe/draw group after seeding exactly once."""

    def provenance(self) -> Mapping[str, Any]:
        """Return deterministic loader/generation settings for request provenance."""


BackendFactory = Callable[[ResolvedLeafCheckpoint], LoadedProbeBackend]


def validate_inference_receipt_schema(
    payload: Mapping[str, Any],
    *,
    context: str,
) -> None:
    """Require the one receipt schema emitted by this inference implementation."""

    version = payload.get("schema_version")
    if isinstance(version, bool) or version != INFERENCE_SCHEMA_VERSION:
        raise ValueError(
            f"{context} schema_version must be exactly "
            f"{INFERENCE_SCHEMA_VERSION}, got {version!r}"
        )


def resolve_leaf_checkpoints(
    training_summary: str | Path,
    *,
    cohort_id: str,
    tree_id: str,
    model_task: str,
    base_model_id: str,
    base_model_revision: str,
    path_base: str | Path = ".",
) -> ResolvedCheckpointTree:
    """Resolve an exact ordered leaf set from summary, run list, ledger, and truth manifest."""

    task = _explicit_model_task(model_task)
    cohort = _nonempty(cohort_id, "cohort_id")
    expected_model = _nonempty(base_model_id, "base_model_id")
    expected_revision = _pinned_revision(base_model_revision, "base_model_revision")
    base = _absolute(path_base)
    summary_path = _declared_path(training_summary, base)
    summary_sha = sha256_file(summary_path)
    summary = _json_object(summary_path)
    _validate_summary(summary, path=summary_path, base=base)
    tree_rows = summary.get("trees")
    assert isinstance(tree_rows, list)
    matches = [row for row in tree_rows if isinstance(row, dict) and row.get("tree_id") == tree_id]
    if not matches:
        raise ValueError(f"training summary has no tree {tree_id!r}")
    if len(matches) != 1:
        raise ValueError(f"training summary contains duplicate tree {tree_id!r}")
    tree_row = matches[0]
    _validate_tree_summary_row(tree_row, tree_id=tree_id)

    run_list = _declared_path(_required_string(tree_row, "run_list"), base)
    run_list_sha = sha256_file(run_list)
    _require_declared_sha(tree_row, "run_list_sha256", run_list_sha, context=tree_id)
    ledger = _declared_path(_required_string(tree_row, "ledger"), base)
    ledger_sha = sha256_file(ledger)
    truth_manifest = _declared_path(_required_string(tree_row, "manifest"), base)
    truth_sha = sha256_file(truth_manifest)
    _require_declared_sha(tree_row, "manifest_sha256", truth_sha, context=tree_id)
    _verify_optional_hashed_path(tree_row, "report", base=base, context=tree_id)

    runs = load_training_run_specs(run_list)
    if not runs:
        raise ValueError(f"tree {tree_id!r} run list is empty")
    run_by_node = _unique_runs_by_node(runs, tree_id=tree_id)
    records = load_manifest(truth_manifest)
    manifest_nodes = [record.node_id for record in records]
    duplicates = _duplicates(manifest_nodes)
    if duplicates:
        raise ValueError(f"tree {tree_id!r} truth manifest has duplicate nodes: {duplicates}")
    trained_nodes = {record.node_id for record in records if record.is_trained}
    if set(run_by_node) != trained_nodes:
        missing = sorted(trained_nodes - set(run_by_node))
        extra = sorted(set(run_by_node) - trained_nodes)
        raise ValueError(
            f"tree {tree_id!r} run-list/truth nodes differ: missing={missing}, extra={extra}"
        )
    leaf_set = set(leaf_ids(records))
    ordered_leaf_ids = [record.node_id for record in records if record.node_id in leaf_set]
    if len(ordered_leaf_ids) != len(leaf_set):
        raise ValueError(f"tree {tree_id!r} leaf order is not one-to-one")
    if not ordered_leaf_ids:
        raise ValueError(f"tree {tree_id!r} has no trained leaves")

    events = load_ledger_events(ledger)
    latest = latest_status_by_node(events)
    if set(latest) != set(run_by_node):
        missing = sorted(set(run_by_node) - set(latest))
        extra = sorted(set(latest) - set(run_by_node))
        raise ValueError(
            f"tree {tree_id!r} ledger/run-list nodes differ: missing={missing}, extra={extra}"
        )

    method_values: set[str] = set()
    for run in runs:
        _validate_run_contract(
            run,
            run_list=run_list,
            ledger=ledger,
            base=base,
            model_task=task,
            base_model_id=expected_model,
            base_model_revision=expected_revision,
        )
        method_values.add(run.method)
        event = latest[run.node_id]
        if event.status != "completed":
            raise ValueError(
                f"tree {tree_id!r} node {run.node_id!r} must be terminal COMPLETED, "
                f"got {event.status!r}"
            )
    if len(method_values) != 1:
        raise ValueError(f"tree {tree_id!r} mixes training methods: {sorted(method_values)}")
    method = next(iter(method_values))
    artifact_name = _artifact_for_method(method)

    checkpoints_by_node: dict[str, Path] = {}
    for run in runs:
        checkpoint_path = _resolve_exact_artifact(
            run,
            latest[run.node_id].extra,
            artifact_name=artifact_name,
            base=base,
        )
        _validate_model_checkpoint(
            checkpoint_path,
            context=f"{tree_id}/{run.node_id}",
        )
        checkpoints_by_node[run.node_id] = checkpoint_path

    leaves: list[ResolvedLeafCheckpoint] = []
    for ordinal, leaf_id in enumerate(ordered_leaf_ids):
        checkpoint = checkpoints_by_node[leaf_id]
        leaves.append(
            ResolvedLeafCheckpoint(
                cohort_id=cohort,
                tree_id=tree_id,
                leaf_id=leaf_id,
                leaf_ordinal=ordinal,
                method=method,
                artifact_name=artifact_name,
                checkpoint=checkpoint,
                base_model_id=expected_model,
                base_model_revision=expected_revision,
                model_task=task,
                training_summary=summary_path,
                training_summary_sha256=summary_sha,
                run_list=run_list,
                run_list_sha256=run_list_sha,
                ledger=ledger,
                ledger_sha256=ledger_sha,
                truth_manifest=truth_manifest,
                truth_manifest_sha256=truth_sha,
            )
        )
    if [leaf.leaf_id for leaf in leaves] != ordered_leaf_ids:
        raise AssertionError("internal leaf-order corruption")
    return ResolvedCheckpointTree(
        cohort_id=cohort,
        tree_id=tree_id,
        method=method,
        artifact_name=artifact_name,
        base_model_id=expected_model,
        base_model_revision=expected_revision,
        model_task=task,
        leaves=tuple(leaves),
    )


def run_leaf_probe_inference(
    checkpoint: ResolvedLeafCheckpoint,
    *,
    registry: BehaviorProtocolRegistry,
    prompts_by_protocol: Mapping[str, Sequence[RenderedBehaviorPrompt]],
    prompt_artifacts: Mapping[str, Mapping[str, str]],
    output_path: str | Path,
    receipt_path: str | Path,
    backend: LoadedProbeBackend | None = None,
    backend_factory: BackendFactory | None = None,
    backend_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate or resume one leaf's complete requested grid with one checkpoint load."""

    if backend is not None and backend_factory is not None:
        raise ValueError("pass either a loaded backend or a backend_factory, not both")
    registry.validate_model_binding(
        model_task=checkpoint.model_task,
        model_id=checkpoint.base_model_id,
        model_revision=checkpoint.base_model_revision,
    )
    coordinates = expected_response_coordinates(registry, prompts_by_protocol)
    _validate_prompt_tasks(
        prompts_by_protocol,
        model_task=checkpoint.model_task,
    )
    prompt_artifact_pins = _normalize_prompt_artifacts(
        prompt_artifacts,
        prompts_by_protocol=prompts_by_protocol,
    )
    output = _absolute(output_path)
    receipt = _absolute(receipt_path)
    if output == receipt:
        raise ValueError("response JSONL and receipt must be different paths")
    checkpoint_sha = sha256_path(checkpoint.checkpoint)
    settings = _backend_settings(backend, backend_settings)
    request_payload = _request_payload(
        checkpoint,
        registry=registry,
        prompts_by_protocol=prompts_by_protocol,
        coordinates=coordinates,
        checkpoint_sha256=checkpoint_sha,
        backend_settings=settings,
        prompt_artifacts=prompt_artifact_pins,
    )
    request_sha = canonical_json_sha256(request_payload)
    provenance = ResponseProvenance(
        request_sha256=request_sha,
        training_summary=str(checkpoint.training_summary),
        training_summary_sha256=checkpoint.training_summary_sha256,
        run_list=str(checkpoint.run_list),
        run_list_sha256=checkpoint.run_list_sha256,
        ledger=str(checkpoint.ledger),
        ledger_sha256=checkpoint.ledger_sha256,
        truth_manifest=str(checkpoint.truth_manifest),
        truth_manifest_sha256=checkpoint.truth_manifest_sha256,
        checkpoint=str(checkpoint.checkpoint),
        checkpoint_artifact=checkpoint.artifact_name,
        checkpoint_sha256=checkpoint_sha,
        registry=str(registry.path),
        registry_sha256=registry.sha256,
        prompt_artifacts=prompt_artifact_pins,
    )

    if receipt.exists():
        return _validate_completed_cache(
            output,
            receipt,
            registry=registry,
            prompts_by_protocol=prompts_by_protocol,
            checkpoint=checkpoint,
            request_sha256=request_sha,
            provenance=provenance,
        )

    existing = load_behavior_responses(output)
    _validate_resume_prefix(
        existing,
        coordinates=coordinates,
        registry=registry,
        checkpoint=checkpoint,
        provenance=provenance,
    )
    generated_by_index: dict[int, str] = {}
    if len(existing) < len(coordinates):
        loaded = backend
        if loaded is None:
            if backend_factory is None:
                raise ValueError("inference requires a loaded backend or backend_factory")
            loaded = backend_factory(checkpoint)
        if dict(loaded.provenance()) != settings:
            raise ValueError("loaded backend provenance does not match the frozen backend settings")
        groups: dict[
            tuple[str, str, int], list[tuple[int, RenderedBehaviorPrompt]]
        ] = {}
        for coordinate_index, (protocol_id, prompt, sample_id) in enumerate(coordinates):
            groups.setdefault(
                (protocol_id, prompt.probe_id, sample_id), []
            ).append((coordinate_index, prompt))
        for (protocol_id, probe_id, sample_id), group in groups.items():
            if all(index < len(existing) for index, _prompt in group):
                continue
            options = registry.protocol(protocol_id).generation_options(
                model_task=checkpoint.model_task,
                sample_id=sample_id,
            )
            texts = list(
                loaded.generate_batch(
                    [prompt.prompt for _index, prompt in group],
                    model_task=checkpoint.model_task,
                    generation_options=options,
                )
            )
            if len(texts) != len(group):
                raise ValueError(
                    f"backend returned {len(texts)} responses for "
                    f"{protocol_id}/{probe_id}/draw{sample_id}, expected {len(group)}"
                )
            for (coordinate_index, _prompt), text in zip(group, texts, strict=True):
                if not isinstance(text, str):
                    raise TypeError(
                        f"backend response {protocol_id}/{probe_id}/draw{sample_id} "
                        f"at index {coordinate_index} is not text"
                    )
                if coordinate_index < len(existing):
                    if existing[coordinate_index].text != text:
                        raise ValueError(
                            "partial response resume does not replay exactly at "
                            f"row {coordinate_index}: {existing[coordinate_index].key}"
                        )
                else:
                    generated_by_index[coordinate_index] = text
        output.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if output.exists() else "w"
        with output.open(mode, encoding="utf-8", buffering=1) as handle:
            for coordinate_index, (protocol_id, prompt, sample_id) in enumerate(
                coordinates[len(existing) :], start=len(existing)
            ):
                options = registry.protocol(protocol_id).generation_options(
                    model_task=checkpoint.model_task,
                    sample_id=sample_id,
                )
                try:
                    text = generated_by_index[coordinate_index]
                except KeyError as exc:
                    raise AssertionError(
                        f"missing generated response for coordinate {coordinate_index}"
                    ) from exc
                response = response_from_text(
                    cohort_id=checkpoint.cohort_id,
                    tree_id=checkpoint.tree_id,
                    model_id=checkpoint.leaf_id,
                    base_model_id=checkpoint.base_model_id,
                    base_model_revision=checkpoint.base_model_revision,
                    model_task=checkpoint.model_task,
                    protocol_id=protocol_id,
                    prompt=prompt,
                    sample_id=sample_id,
                    generation_options=options,
                    text=text,
                    provenance=provenance,
                )
                handle.write(json.dumps(response.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
            os.fsync(handle.fileno())

    responses = load_behavior_responses(output)
    audit = audit_response_grid(
        responses,
        registry=registry,
        prompts_by_protocol=prompts_by_protocol,
        cohort_id=checkpoint.cohort_id,
        tree_id=checkpoint.tree_id,
        model_id=checkpoint.leaf_id,
        base_model_id=checkpoint.base_model_id,
        base_model_revision=checkpoint.base_model_revision,
        model_task=checkpoint.model_task,
        request_sha256=request_sha,
        expected_provenance=provenance,
    )
    if not audit.valid:
        raise ValueError(f"completed response grid is invalid: {list(audit.errors)}")
    response_sha = sha256_file(output)
    receipt_payload = {
        "schema_version": INFERENCE_SCHEMA_VERSION,
        "valid": True,
        "status": "completed",
        "request_sha256": request_sha,
        "request": request_payload,
        "provenance": provenance.to_dict(),
        "prompt_artifacts": prompt_artifact_pins,
        "responses": str(output),
        "responses_sha256": response_sha,
        "audit": audit.to_dict(),
    }
    _write_json_atomic(receipt, receipt_payload)
    return receipt_payload


class HuggingFaceProbeBackend:
    """A single local Hugging Face checkpoint plus its independently pinned base tokenizer."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        torch_module: Any,
        transformers_module: Any,
        model_task: str,
        device: str,
        torch_dtype: str,
        base_model_id: str,
        base_model_revision: str,
        checkpoint: Path,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._torch = torch_module
        self._transformers = transformers_module
        self._model_task = _explicit_model_task(model_task)
        self._device = device
        self._torch_dtype = torch_dtype
        self._base_model_id = base_model_id
        self._base_model_revision = base_model_revision
        self._checkpoint = checkpoint

    @classmethod
    def planned_provenance(
        cls,
        checkpoint: ResolvedLeafCheckpoint,
        *,
        device: str,
        torch_dtype: str,
    ) -> dict[str, Any]:
        """Resolve deterministic runtime settings without loading checkpoint weights."""

        try:
            import torch
            import transformers
        except ImportError as exc:  # pragma: no cover - exercised only in production environments
            raise RuntimeError("probe inference requires torch and transformers") from exc
        _torch_dtype(torch, torch_dtype)
        return {
            "backend": "huggingface",
            "device": device,
            "torch_dtype": torch_dtype,
            "local_files_only": True,
            "tokenizer_model_id": checkpoint.base_model_id,
            "tokenizer_revision": checkpoint.base_model_revision,
            "checkpoint": str(checkpoint.checkpoint),
            "transformers_version": getattr(transformers, "__version__", None),
            "torch_version": getattr(torch, "__version__", None),
            "decode_skip_special_tokens": True,
            "decode_clean_up_tokenization_spaces": False,
            "generation_api": "ordered_probe_draw_batches",
            "generation_seed_api": "transformers.set_seed_once_per_probe_draw",
            "tokenizer_padding_side": (
                "left" if checkpoint.model_task == "causal_lm" else "right"
            ),
            "tokenizer_truncation": True,
            "tokenizer_max_length": 512,
            "generation_pad_token_id": "tokenizer.pad_token_id",
        }

    @classmethod
    def load(
        cls,
        checkpoint: ResolvedLeafCheckpoint,
        *,
        device: str,
        torch_dtype: str,
    ) -> "HuggingFaceProbeBackend":
        """Load locally only; tokenizer identity always comes from the pinned base model."""

        try:
            import torch
            import transformers
        except ImportError as exc:  # pragma: no cover - exercised only in production environments
            raise RuntimeError("probe inference requires torch and transformers") from exc
        task = _explicit_model_task(checkpoint.model_task)
        dtype = _torch_dtype(torch, torch_dtype)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            checkpoint.base_model_id,
            revision=checkpoint.base_model_revision,
            local_files_only=True,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left" if task == "causal_lm" else "right"
        model_class = (
            transformers.AutoModelForSeq2SeqLM
            if task == "seq2seq"
            else transformers.AutoModelForCausalLM
        )
        model = model_class.from_pretrained(
            str(checkpoint.checkpoint),
            local_files_only=True,
            torch_dtype=dtype,
        )
        model.to(device)
        model.eval()
        return cls(
            model=model,
            tokenizer=tokenizer,
            torch_module=torch,
            transformers_module=transformers,
            model_task=task,
            device=device,
            torch_dtype=torch_dtype,
            base_model_id=checkpoint.base_model_id,
            base_model_revision=checkpoint.base_model_revision,
            checkpoint=checkpoint.checkpoint,
        )

    def provenance(self) -> Mapping[str, Any]:
        return self.planned_provenance_from_modules(
            torch_module=self._torch,
            transformers_module=self._transformers,
            model_task=self._model_task,
            device=self._device,
            torch_dtype=self._torch_dtype,
            base_model_id=self._base_model_id,
            base_model_revision=self._base_model_revision,
            checkpoint=self._checkpoint,
        )

    @staticmethod
    def planned_provenance_from_modules(
        *,
        torch_module: Any,
        transformers_module: Any,
        model_task: str,
        device: str,
        torch_dtype: str,
        base_model_id: str,
        base_model_revision: str,
        checkpoint: Path,
    ) -> dict[str, Any]:
        return {
            "backend": "huggingface",
            "device": device,
            "torch_dtype": torch_dtype,
            "local_files_only": True,
            "tokenizer_model_id": base_model_id,
            "tokenizer_revision": base_model_revision,
            "checkpoint": str(checkpoint),
            "transformers_version": getattr(transformers_module, "__version__", None),
            "torch_version": getattr(torch_module, "__version__", None),
            "decode_skip_special_tokens": True,
            "decode_clean_up_tokenization_spaces": False,
            "generation_api": "ordered_probe_draw_batches",
            "generation_seed_api": "transformers.set_seed_once_per_probe_draw",
            "tokenizer_padding_side": "left" if model_task == "causal_lm" else "right",
            "tokenizer_truncation": True,
            "tokenizer_max_length": 512,
            "generation_pad_token_id": "tokenizer.pad_token_id",
        }

    def generate_batch(
        self,
        prompts: Sequence[str],
        *,
        model_task: str,
        generation_options: Mapping[str, Any],
    ) -> Sequence[str]:
        task = _explicit_model_task(model_task)
        if task != self._model_task:
            raise ValueError(
                f"loaded backend task is {self._model_task!r}, requested {task!r}"
            )
        options = _exact_generation_options(generation_options)
        if options["seed_scope"] != "probe_draw":
            raise ValueError("behavior generation seed_scope must be probe_draw")
        prompt_texts = list(prompts)
        if not prompt_texts or not all(isinstance(prompt, str) for prompt in prompt_texts):
            raise ValueError("behavior generation requires a nonempty ordered text panel")
        self._transformers.set_seed(options["seed"])
        responses: list[str] = []
        batch_size = options["generation_batch_size"]
        for start in range(0, len(prompt_texts), batch_size):
            batch = prompt_texts[start : start + batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(self._device) if hasattr(value, "to") else value
                for key, value in encoded.items()
            }
            input_length = _input_length(encoded.get("input_ids"))
            generate_kwargs = {
                "do_sample": options["do_sample"],
                "min_new_tokens": options["min_new_tokens"],
                "max_new_tokens": options["max_new_tokens"],
                "pad_token_id": self._tokenizer.pad_token_id,
            }
            if options["do_sample"]:
                generate_kwargs["temperature"] = options["temperature"]
                generate_kwargs["top_p"] = options["top_p"]
            with self._torch.inference_mode():
                generated = self._model.generate(**encoded, **generate_kwargs)
            if task == "causal_lm":
                generated = generated[:, input_length:]
            decoded = self._tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if len(decoded) != len(batch):
                raise ValueError(
                    "behavior generation output count differs from its ordered input batch"
                )
            responses.extend(str(text) for text in decoded)
        if len(responses) != len(prompt_texts):
            raise AssertionError("behavior generation lost ordered responses")
        return responses


def causal_continuation_token_ids(generated_ids: Any, *, input_length: int) -> Any:
    """Return causal continuation IDs only, never the prompt IDs."""

    if isinstance(input_length, bool) or not isinstance(input_length, int) or input_length < 0:
        raise ValueError("input_length must be a non-negative integer")
    try:
        return generated_ids[input_length:]
    except (IndexError, TypeError) as exc:
        raise ValueError("generated_ids does not support causal continuation slicing") from exc


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"required file is missing: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: str | Path) -> str:
    """Hash one file or a directory's complete, sorted relative-path/file-hash manifest."""

    source = Path(path)
    if source.is_file():
        return sha256_file(source)
    if not source.is_dir():
        raise FileNotFoundError(f"required checkpoint artifact is missing: {source}")
    root = source.resolve(strict=True)
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"checkpoint directory is empty: {source}")
    rows: list[dict[str, Any]] = []
    for item in files:
        if item.is_symlink():
            raise ValueError(f"checkpoint contains a symlink: {item}")
        rows.append(
            {
                "path": item.relative_to(root).as_posix(),
                "size": item.stat().st_size,
                "sha256": sha256_file(item),
            }
        )
    return canonical_json_sha256(rows)


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_summary(summary: Mapping[str, Any], *, path: Path, base: Path) -> None:
    trees = summary.get("trees")
    if not isinstance(trees, list) or not trees:
        raise ValueError(f"training summary must contain a non-empty trees list: {path}")
    if summary.get("n_errors") != 0 or summary.get("n_warnings") != 0:
        raise ValueError(f"training summary is not clean: {path}")
    if summary.get("n_trees") != len(trees):
        raise ValueError(f"training summary n_trees does not match tree rows: {path}")
    ids = [row.get("tree_id") for row in trees if isinstance(row, dict)]
    if len(ids) != len(trees) or len(ids) != len(set(ids)):
        raise ValueError(f"training summary tree IDs are missing or duplicated: {path}")
    for path_key in ("assignment_summary", "config", "formats", "registry"):
        _verify_optional_hashed_path(summary, path_key, base=base, context=str(path))


def _validate_tree_summary_row(row: Mapping[str, Any], *, tree_id: str) -> None:
    if row.get("valid") is not True:
        raise ValueError(f"training summary row {tree_id!r} is not valid")
    if row.get("n_errors") != 0 or row.get("n_warnings") != 0:
        raise ValueError(f"training summary row {tree_id!r} is not clean")


def _validate_run_contract(
    run: TrainingRunSpec,
    *,
    run_list: Path,
    ledger: Path,
    base: Path,
    model_task: str,
    base_model_id: str,
    base_model_revision: str,
) -> None:
    declared_ledger = _declared_path(run.ledger_path, base)
    if declared_ledger != ledger:
        raise ValueError(
            f"run {run.node_id!r} ledger path changed: expected {ledger}, got {declared_ledger}"
        )
    runner_run_list = run.runner.get("run_list_path")
    if not isinstance(runner_run_list, str) or not runner_run_list:
        raise ValueError(f"run {run.node_id!r} lacks runner.run_list_path")
    declared_run_list = _declared_path(runner_run_list, base)
    if declared_run_list != run_list:
        raise ValueError(
            f"run {run.node_id!r} run-list path changed: expected {run_list}, "
            f"got {declared_run_list}"
        )
    job = run.job
    if job.get("method") != run.method:
        raise ValueError(
            f"run {run.node_id!r} method declarations differ: "
            f"row={run.method!r}, job={job.get('method')!r}"
        )
    job_artifacts = job.get("expected_artifacts")
    if not isinstance(job_artifacts, dict) or job_artifacts != run.expected_artifacts:
        raise ValueError(
            f"run {run.node_id!r} expected_artifacts declarations differ between row and job"
        )
    if job.get("base_model") != base_model_id:
        raise ValueError(
            f"run {run.node_id!r} base_model mismatch: expected {base_model_id!r}, "
            f"got {job.get('base_model')!r}"
        )
    if job.get("base_model_revision") != base_model_revision:
        raise ValueError(
            f"run {run.node_id!r} base_model_revision mismatch: expected "
            f"{base_model_revision!r}, got {job.get('base_model_revision')!r}"
        )
    trainer = job.get("trainer")
    if not isinstance(trainer, dict):
        raise ValueError(f"run {run.node_id!r} lacks job.trainer")
    if trainer.get("model_task") != model_task:
        raise ValueError(
            f"run {run.node_id!r} model_task mismatch: expected {model_task!r}, "
            f"got {trainer.get('model_task')!r}"
        )
    _artifact_for_method(run.method)


def _resolve_exact_artifact(
    run: TrainingRunSpec,
    event_extra: Mapping[str, Any],
    *,
    artifact_name: str,
    base: Path,
) -> Path:
    if artifact_name not in run.expected_artifacts:
        raise ValueError(
            f"run {run.node_id!r} lacks expected {artifact_name!r} artifact"
        )
    expected = _declared_path(run.expected_artifacts[artifact_name], base)
    raw_artifacts = event_extra.get("artifacts")
    if not isinstance(raw_artifacts, dict):
        raise ValueError(f"completed ledger node {run.node_id!r} lacks artifacts")
    raw_ledger_artifact = raw_artifacts.get(artifact_name)
    if not isinstance(raw_ledger_artifact, str) or not raw_ledger_artifact:
        raise ValueError(
            f"completed ledger node {run.node_id!r} lacks {artifact_name!r} artifact"
        )
    observed = _declared_path(raw_ledger_artifact, base)
    if observed != expected:
        raise ValueError(
            f"run {run.node_id!r} {artifact_name} artifact path changed: "
            f"expected {expected}, got {observed}"
        )
    return expected


def _validate_model_checkpoint(path: Path, *, context: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{context} checkpoint directory is missing: {path}")
    config = path / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"{context} checkpoint lacks config.json: {path}")
    weight_names = {item.name for item in path.iterdir() if item.is_file()}
    if not (weight_names & _WEIGHT_FILE_NAMES):
        raise FileNotFoundError(f"{context} checkpoint lacks saved model weights: {path}")


def _artifact_for_method(method: str) -> str:
    if method == "full":
        return "model"
    if method == "lora":
        return "merged"
    raise ValueError(f"behavior inference does not support training method {method!r}")


def _unique_runs_by_node(
    runs: Sequence[TrainingRunSpec], *, tree_id: str
) -> dict[str, TrainingRunSpec]:
    by_node: dict[str, TrainingRunSpec] = {}
    for run in runs:
        if run.node_id in by_node:
            raise ValueError(f"tree {tree_id!r} run list duplicates node {run.node_id!r}")
        by_node[run.node_id] = run
    return by_node


def _validate_prompt_tasks(
    prompts_by_protocol: Mapping[str, Sequence[RenderedBehaviorPrompt]], *, model_task: str
) -> None:
    for protocol_id, prompts in prompts_by_protocol.items():
        for prompt in prompts:
            if prompt.metadata.get("model_task") != model_task:
                raise ValueError(
                    f"protocol {protocol_id!r} prompt {prompt.prompt_id!r} was rendered for "
                    f"{prompt.metadata.get('model_task')!r}, not explicit task {model_task!r}"
                )


def _request_payload(
    checkpoint: ResolvedLeafCheckpoint,
    *,
    registry: BehaviorProtocolRegistry,
    prompts_by_protocol: Mapping[str, Sequence[RenderedBehaviorPrompt]],
    coordinates: Sequence[tuple[str, RenderedBehaviorPrompt, int]],
    checkpoint_sha256: str,
    backend_settings: Mapping[str, Any],
    prompt_artifacts: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    protocol_rows = []
    for protocol_id in prompts_by_protocol:
        protocol = registry.protocol(protocol_id)
        prompt_rows = []
        for prompt in prompts_by_protocol[protocol_id]:
            prompt_rows.append(
                {
                    "probe_id": prompt.probe_id,
                    "prompt_id": prompt.prompt_id,
                    "source_index": prompt.source_index,
                    "source_row_sha256": prompt.source_row_sha256,
                    "prompt_sha256": prompt.prompt_sha256,
                    "reference_sha256": None
                    if prompt.reference is None
                    else hashlib.sha256(prompt.reference.encode("utf-8")).hexdigest(),
                    "fixture_id": prompt.metadata.get("fixture_id"),
                    "fixture_sha256": prompt.metadata.get("fixture_sha256"),
                    "source_indices_sha256": prompt.metadata.get("source_indices_sha256"),
                    "dataset_revision": prompt.metadata.get("dataset_revision"),
                }
            )
        protocol_rows.append(
            {
                "protocol_id": protocol_id,
                "probe_ids": list(protocol.probe_ids),
                "prompt_counts": dict(protocol.prompt_counts),
                "samples_per_prompt": protocol.samples_per_prompt,
                "draws": [
                    protocol.generation_options(
                        model_task=checkpoint.model_task, sample_id=sample_id
                    )
                    for sample_id in range(protocol.samples_per_prompt)
                ],
                "prompts": prompt_rows,
            }
        )
    return {
        "schema_version": INFERENCE_SCHEMA_VERSION,
        "cohort_id": checkpoint.cohort_id,
        "tree_id": checkpoint.tree_id,
        "model_id": checkpoint.leaf_id,
        "leaf_ordinal": checkpoint.leaf_ordinal,
        "base_model_id": checkpoint.base_model_id,
        "base_model_revision": checkpoint.base_model_revision,
        "model_task": checkpoint.model_task,
        "checkpoint": str(checkpoint.checkpoint),
        "checkpoint_artifact": checkpoint.artifact_name,
        "checkpoint_sha256": checkpoint_sha256,
        "training_summary": str(checkpoint.training_summary),
        "training_summary_sha256": checkpoint.training_summary_sha256,
        "run_list": str(checkpoint.run_list),
        "run_list_sha256": checkpoint.run_list_sha256,
        "ledger": str(checkpoint.ledger),
        "ledger_sha256": checkpoint.ledger_sha256,
        "truth_manifest": str(checkpoint.truth_manifest),
        "truth_manifest_sha256": checkpoint.truth_manifest_sha256,
        "registry": str(registry.path),
        "registry_sha256": registry.sha256,
        "backend_settings": dict(backend_settings),
        "prompt_artifacts": {
            protocol_id: dict(prompt_artifacts[protocol_id])
            for protocol_id in prompts_by_protocol
        },
        "n_expected_responses": len(coordinates),
        "protocols": protocol_rows,
    }


def _backend_settings(
    backend: LoadedProbeBackend | None,
    supplied: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if supplied is not None:
        settings = dict(supplied)
        if backend is not None and dict(backend.provenance()) != settings:
            raise ValueError("loaded backend provenance does not match backend_settings")
        return settings
    if backend is None:
        raise ValueError("backend_settings are required when using a lazy backend_factory")
    return dict(backend.provenance())


def _normalize_prompt_artifacts(
    raw: Mapping[str, Mapping[str, str]],
    *,
    prompts_by_protocol: Mapping[str, Sequence[RenderedBehaviorPrompt]],
) -> dict[str, dict[str, str]]:
    if set(raw) != set(prompts_by_protocol):
        raise ValueError(
            "prompt_artifacts must declare exactly the requested protocols: "
            f"expected {list(prompts_by_protocol)}, got {list(raw)}"
        )
    normalized: dict[str, dict[str, str]] = {}
    for protocol_id in prompts_by_protocol:
        spec = raw[protocol_id]
        if not isinstance(spec, Mapping) or set(spec) != {"path", "sha256"}:
            raise ValueError(
                f"prompt_artifacts.{protocol_id} must contain exactly path and sha256"
            )
        path = _absolute(_required_string(spec, "path"))
        declared_sha = _required_string(spec, "sha256")
        if not _SHA256.fullmatch(declared_sha):
            raise ValueError(
                f"prompt_artifacts.{protocol_id}.sha256 must be a lowercase SHA256"
            )
        actual_sha = sha256_file(path)
        if actual_sha != declared_sha:
            raise ValueError(
                f"prompt_artifacts.{protocol_id} SHA256 mismatch: "
                f"expected {declared_sha}, got {actual_sha}"
            )
        loaded = load_rendered_behavior_prompts(path)
        supplied = list(prompts_by_protocol[protocol_id])
        if [prompt.to_dict() for prompt in loaded] != [prompt.to_dict() for prompt in supplied]:
            raise ValueError(
                f"prompt_artifacts.{protocol_id} content does not match supplied prompts"
            )
        normalized[protocol_id] = {"path": str(path), "sha256": actual_sha}
    return normalized


def _validate_resume_prefix(
    existing: Sequence[BehaviorResponse],
    *,
    coordinates: Sequence[tuple[str, RenderedBehaviorPrompt, int]],
    registry: BehaviorProtocolRegistry,
    checkpoint: ResolvedLeafCheckpoint,
    provenance: ResponseProvenance,
) -> None:
    if len(existing) > len(coordinates):
        raise ValueError("cached response artifact has more rows than the requested grid")
    for index, row in enumerate(existing):
        protocol_id, prompt, sample_id = coordinates[index]
        options = registry.protocol(protocol_id).generation_options(
            model_task=checkpoint.model_task, sample_id=sample_id
        )
        expected = response_from_text(
            cohort_id=checkpoint.cohort_id,
            tree_id=checkpoint.tree_id,
            model_id=checkpoint.leaf_id,
            base_model_id=checkpoint.base_model_id,
            base_model_revision=checkpoint.base_model_revision,
            model_task=checkpoint.model_task,
            protocol_id=protocol_id,
            prompt=prompt,
            sample_id=sample_id,
            generation_options=options,
            text=row.text,
            provenance=provenance,
        )
        if row.to_dict() != expected.to_dict():
            raise ValueError(
                f"cached response provenance/content drift at row {index}, coordinate {row.key}"
            )


def _validate_completed_cache(
    output: Path,
    receipt: Path,
    *,
    registry: BehaviorProtocolRegistry,
    prompts_by_protocol: Mapping[str, Sequence[RenderedBehaviorPrompt]],
    checkpoint: ResolvedLeafCheckpoint,
    request_sha256: str,
    provenance: ResponseProvenance,
) -> dict[str, Any]:
    payload = _json_object(receipt)
    validate_inference_receipt_schema(payload, context=f"cached receipt {receipt}")
    if payload.get("valid") is not True or payload.get("status") != "completed":
        raise ValueError(f"cached receipt is not terminal valid: {receipt}")
    if payload.get("request_sha256") != request_sha256:
        raise ValueError(f"cached receipt request provenance drift: {receipt}")
    if payload.get("provenance") != provenance.to_dict():
        raise ValueError(f"cached receipt detailed provenance drift: {receipt}")
    if payload.get("prompt_artifacts") != provenance.prompt_artifacts:
        raise ValueError(f"cached receipt prompt-artifact provenance drift: {receipt}")
    if _absolute(payload.get("responses", "")) != output:
        raise ValueError(f"cached receipt response path drift: {receipt}")
    response_sha = sha256_file(output)
    if payload.get("responses_sha256") != response_sha:
        raise ValueError(f"cached response artifact hash drift: {output}")
    responses = load_behavior_responses(output)
    audit = audit_response_grid(
        responses,
        registry=registry,
        prompts_by_protocol=prompts_by_protocol,
        cohort_id=checkpoint.cohort_id,
        tree_id=checkpoint.tree_id,
        model_id=checkpoint.leaf_id,
        base_model_id=checkpoint.base_model_id,
        base_model_revision=checkpoint.base_model_revision,
        model_task=checkpoint.model_task,
        request_sha256=request_sha256,
        expected_provenance=provenance,
    )
    if not audit.valid or payload.get("audit") != audit.to_dict():
        raise ValueError(f"cached response grid audit drift: {output}")
    return dict(payload)


def _exact_generation_options(options: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "sample_id",
        "seed",
        "generation_batch_size",
        "seed_scope",
        "do_sample",
        "temperature",
        "top_p",
        "min_new_tokens",
        "max_new_tokens",
        "empty_policy",
    }
    if set(options) != required:
        raise ValueError(
            f"generation options must contain exactly {sorted(required)}, got {sorted(options)}"
        )
    if not isinstance(options["seed"], int) or isinstance(options["seed"], bool):
        raise ValueError("generation seed must be an integer")
    batch_size = options["generation_batch_size"]
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("generation_batch_size must be a positive integer")
    if options["seed_scope"] != "probe_draw":
        raise ValueError("seed_scope must be probe_draw")
    if not isinstance(options["do_sample"], bool):
        raise ValueError("do_sample must be explicit boolean")
    for name in ("min_new_tokens", "max_new_tokens"):
        value = options[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    for name in ("temperature", "top_p"):
        value = options[name]
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError(f"{name} must be a number or null")
    return dict(options)


def _input_length(input_ids: Any) -> int:
    if input_ids is None:
        raise ValueError("tokenizer output lacks input_ids")
    shape = getattr(input_ids, "shape", None)
    if shape is not None and len(shape) == 2:
        return int(shape[1])
    try:
        if len(input_ids) != 1:
            raise ValueError("probe inference requires a tokenizer batch size of one")
        return len(input_ids[0])
    except (TypeError, IndexError) as exc:
        raise ValueError("could not determine tokenizer input length") from exc


def _torch_dtype(torch: Any, name: str) -> Any:
    choices = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return choices[name]
    except KeyError as exc:
        raise ValueError(f"unsupported torch dtype {name!r}; choose {sorted(choices)}") from exc


def _json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required JSON file is missing: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return raw


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite stale receipt temporary: {temporary}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _verify_optional_hashed_path(
    row: Mapping[str, Any], path_key: str, *, base: Path, context: str
) -> None:
    hash_key = f"{path_key}_sha256"
    if path_key not in row and hash_key not in row:
        return
    raw_path = _required_string(row, path_key)
    expected = _required_string(row, hash_key)
    actual = sha256_file(_declared_path(raw_path, base))
    if actual != expected:
        raise ValueError(
            f"{context} {path_key} SHA256 mismatch: expected {expected}, got {actual}"
        )


def _require_declared_sha(
    row: Mapping[str, Any], key: str, actual: str, *, context: str
) -> None:
    expected = _required_string(row, key)
    if not _SHA256.fullmatch(expected):
        raise ValueError(f"{context} {key} is not a lowercase SHA256")
    if expected != actual:
        raise ValueError(f"{context} {key} mismatch: expected {expected}, got {actual}")


def _required_string(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"required field {key!r} must be a non-empty string")
    return value


def _declared_path(raw: str | Path, base: Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = base / path
    return Path(os.path.abspath(path))


def _absolute(raw: str | Path) -> Path:
    return Path(os.path.abspath(Path(raw)))


def _duplicates(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def _explicit_model_task(value: Any) -> str:
    if not isinstance(value, str) or value not in MODEL_TASKS:
        raise ValueError(f"model_task must be explicit and one of {list(MODEL_TASKS)}")
    return value


def _nonempty(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _pinned_revision(value: Any, context: str) -> str:
    result = _nonempty(value, context)
    if not _PINNED_REVISION.fullmatch(result):
        raise ValueError(f"{context} must be a pinned lowercase commit or SHA256")
    return result
