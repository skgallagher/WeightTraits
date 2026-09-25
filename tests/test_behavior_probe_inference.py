from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from weighttraits.behavior.contracts import (
    AnalysisContract,
    ArchitectureContract,
    BehaviorProtocol,
    BehaviorProtocolRegistry,
    ModelPin,
    RenderedBehaviorPrompt,
)
from weighttraits.behavior.probe_inference import (
    HuggingFaceProbeBackend,
    INFERENCE_SCHEMA_VERSION,
    ResolvedLeafCheckpoint,
    causal_continuation_token_ids,
    resolve_leaf_checkpoints,
    run_leaf_probe_inference,
    sha256_file,
    validate_inference_receipt_schema,
)
from weighttraits.behavior.responses import (
    ResponseProvenance,
    audit_response_grid,
    expected_response_coordinates,
    load_behavior_responses,
    response_from_text,
)
from weighttraits.training.ledger import TrainingLedgerEvent, append_ledger_event


REVISION = "a" * 40
SEQ2SEQ_REVISION = "b" * 40
EMBEDDING_REVISION = "c" * 40
HASH = "d" * 64


def test_inference_receipt_schema_validator_is_exact() -> None:
    validate_inference_receipt_schema(
        {"schema_version": INFERENCE_SCHEMA_VERSION},
        context="test receipt",
    )
    for version in (1, INFERENCE_SCHEMA_VERSION + 1, True, None):
        with pytest.raises(
            ValueError,
            match=rf"schema_version must be exactly {INFERENCE_SCHEMA_VERSION}",
        ):
            validate_inference_receipt_schema(
                {"schema_version": version},
                context="test receipt",
            )


def test_resolver_preserves_truth_leaf_order_and_requires_exact_full_artifacts(
    tmp_path: Path,
) -> None:
    fixture = _write_training_tree(tmp_path, method="full")

    resolved = resolve_leaf_checkpoints(
        fixture["summary"],
        cohort_id="ordinary-full",
        tree_id="tree-001",
        model_task="causal_lm",
        base_model_id="base/causal",
        base_model_revision=REVISION,
        path_base=tmp_path,
    )

    assert resolved.artifact_name == "model"
    assert [leaf.leaf_id for leaf in resolved.leaves] == ["n4", "n2", "n5", "n3"]
    assert [leaf.leaf_ordinal for leaf in resolved.leaves] == [0, 1, 2, 3]
    assert all(leaf.checkpoint.name == "model" for leaf in resolved.leaves)


def test_resolver_uses_merged_full_weight_checkpoint_for_lora(tmp_path: Path) -> None:
    fixture = _write_training_tree(tmp_path, method="lora")

    resolved = resolve_leaf_checkpoints(
        fixture["summary"],
        cohort_id="ordinary-r8",
        tree_id="tree-001",
        model_task="causal_lm",
        base_model_id="base/causal",
        base_model_revision=REVISION,
        path_base=tmp_path,
    )

    assert resolved.artifact_name == "merged"
    assert all(leaf.checkpoint.name == "merged" for leaf in resolved.leaves)
    assert all("adapter" not in str(leaf.checkpoint) for leaf in resolved.leaves)


def test_resolver_requires_explicit_matching_task(tmp_path: Path) -> None:
    fixture = _write_training_tree(tmp_path, method="full")
    with pytest.raises(ValueError, match="model_task must be explicit"):
        resolve_leaf_checkpoints(
            fixture["summary"],
            cohort_id="ordinary-full",
            tree_id="tree-001",
            model_task=None,  # type: ignore[arg-type]
            base_model_id="base/causal",
            base_model_revision=REVISION,
            path_base=tmp_path,
        )
    with pytest.raises(ValueError, match="model_task mismatch"):
        resolve_leaf_checkpoints(
            fixture["summary"],
            cohort_id="ordinary-full",
            tree_id="tree-001",
            model_task="seq2seq",
            base_model_id="base/causal",
            base_model_revision=REVISION,
            path_base=tmp_path,
        )


def test_resolver_rejects_missing_and_duplicate_leaf_contracts(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing"
    missing = _write_training_tree(missing_root, method="full")
    run_rows = _read_jsonl(missing["run_list"])
    _write_jsonl(missing["run_list"], [row for row in run_rows if row["node_id"] != "n4"])
    _update_summary_hash(missing["summary"], "run_list_sha256", missing["run_list"])
    with pytest.raises(ValueError, match="run-list/truth nodes differ"):
        _resolve_default(missing["summary"], missing_root)

    duplicate_root = tmp_path / "duplicate"
    duplicate = _write_training_tree(duplicate_root, method="full")
    manifest_rows = _read_jsonl(duplicate["manifest"])
    _write_jsonl(duplicate["manifest"], [*manifest_rows, manifest_rows[-1]])
    _update_summary_hash(duplicate["summary"], "manifest_sha256", duplicate["manifest"])
    with pytest.raises(ValueError, match="truth manifest has duplicate nodes"):
        _resolve_default(duplicate["summary"], duplicate_root)


def test_resolver_rejects_changed_runlist_ledger_and_artifact_paths(tmp_path: Path) -> None:
    runlist_root = tmp_path / "runlist"
    runlist = _write_training_tree(runlist_root, method="full")
    rows = _read_jsonl(runlist["run_list"])
    rows[0]["runner"]["run_list_path"] = str(runlist_root / "other.runs.jsonl")
    _write_jsonl(runlist["run_list"], rows)
    _update_summary_hash(runlist["summary"], "run_list_sha256", runlist["run_list"])
    with pytest.raises(ValueError, match="run-list path changed"):
        _resolve_default(runlist["summary"], runlist_root)

    ledger_root = tmp_path / "ledger"
    ledger = _write_training_tree(ledger_root, method="full")
    rows = _read_jsonl(ledger["run_list"])
    rows[0]["ledger_path"] = str(ledger_root / "other.ledger.jsonl")
    _write_jsonl(ledger["run_list"], rows)
    _update_summary_hash(ledger["summary"], "run_list_sha256", ledger["run_list"])
    with pytest.raises(ValueError, match="ledger path changed"):
        _resolve_default(ledger["summary"], ledger_root)

    artifact_root = tmp_path / "artifact"
    artifact = _write_training_tree(artifact_root, method="full")
    append_ledger_event(
        artifact["ledger"],
        TrainingLedgerEvent(
            node_id="n4",
            status="completed",
            extra={"artifacts": {"model": str(artifact_root / "changed/model")}},
        ),
    )
    with pytest.raises(ValueError, match="artifact path changed"):
        _resolve_default(artifact["summary"], artifact_root)


def test_resolver_rejects_non_completed_or_missing_checkpoint_content(tmp_path: Path) -> None:
    status_root = tmp_path / "status"
    status = _write_training_tree(status_root, method="full")
    append_ledger_event(
        status["ledger"],
        TrainingLedgerEvent(
            node_id="n2",
            status="stopped_early",
            extra={"artifacts": {"model": str(status["checkpoints"]["n2"])}},
        ),
    )
    with pytest.raises(ValueError, match="must be terminal COMPLETED"):
        _resolve_default(status["summary"], status_root)

    content_root = tmp_path / "content"
    content = _write_training_tree(content_root, method="full")
    (content["checkpoints"]["n3"] / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="lacks saved model weights"):
        _resolve_default(content["summary"], content_root)


def test_causal_decode_slices_prompt_tokens_only() -> None:
    assert causal_continuation_token_ids([10, 11, 12, 20, 21], input_length=3) == [20, 21]
    assert causal_continuation_token_ids([10, 11], input_length=2) == []


def test_huggingface_loader_uses_pinned_base_tokenizer_and_local_checkpoint_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _standalone_checkpoint(tmp_path)
    tokenizer_calls: list[tuple[str, dict[str, Any]]] = []
    model_calls: list[tuple[str, dict[str, Any]]] = []

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> FakeTokenizer:
            tokenizer_calls.append((model_id, kwargs))
            return FakeTokenizer()

    class FakeModel:
        def __init__(self) -> None:
            self.device = None
            self.is_eval = False

        def to(self, device: str) -> None:
            self.device = device

        def eval(self) -> None:
            self.is_eval = True

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> FakeModel:
            model_calls.append((model_id, kwargs))
            return FakeModel()

    fake_torch = SimpleNamespace(
        __version__="test-torch",
        float32="float32",
        float16="float16",
        bfloat16="bfloat16",
    )
    fake_transformers = SimpleNamespace(
        __version__="test-transformers",
        AutoTokenizer=FakeAutoTokenizer,
        AutoModelForCausalLM=FakeAutoModel,
        AutoModelForSeq2SeqLM=FakeAutoModel,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    backend = HuggingFaceProbeBackend.load(
        checkpoint,
        device="cpu",
        torch_dtype="float32",
    )

    assert tokenizer_calls == [
        (
            "base/causal",
            {"revision": REVISION, "local_files_only": True},
        )
    ]
    assert model_calls == [
        (
            str(checkpoint.checkpoint),
            {"local_files_only": True, "torch_dtype": "float32"},
        )
    ]
    assert backend.provenance()["tokenizer_revision"] == REVISION


def test_exact_registry_grids_are_200x1_100x3_and_30x3(tmp_path: Path) -> None:
    protocols = {
        "translation": _protocol(
            "translation",
            probes=("wmt14_fr_en",),
            prompt_counts={"wmt14_fr_en": 200},
            draw_seeds=(42,),
            do_sample=False,
            max_new_tokens=64,
        ),
        "mc": _protocol(
            "mc",
            probes=("hellaswag",),
            prompt_counts={"hellaswag": 100},
            draw_seeds=(42, 43, 44),
            do_sample=True,
            max_new_tokens=64,
        ),
        "dolly": _protocol(
            "dolly",
            probes=("dolly",),
            prompt_counts={"dolly": 30},
            draw_seeds=(42, 43, 44),
            do_sample=True,
            max_new_tokens=96,
        ),
    }
    registry = _registry(tmp_path, protocols)
    prompts = {
        "translation": _prompts("wmt14_fr_en", 200),
        "mc": _prompts("hellaswag", 100),
        "dolly": _prompts("dolly", 30),
    }
    coordinates = expected_response_coordinates(registry, prompts)
    assert len(coordinates) == 200 + (100 * 3) + (30 * 3)
    provenance = _response_provenance()
    responses = []
    for protocol_id, prompt, sample_id in coordinates:
        options = registry.protocol(protocol_id).generation_options(
            model_task="causal_lm", sample_id=sample_id
        )
        responses.append(
            response_from_text(
                cohort_id="cohort",
                tree_id="tree-001",
                model_id="n4",
                base_model_id="base/causal",
                base_model_revision=REVISION,
                model_task="causal_lm",
                protocol_id=protocol_id,
                prompt=prompt,
                sample_id=sample_id,
                generation_options=options,
                text="",
                provenance=provenance,
            )
        )
    audit = audit_response_grid(
        responses,
        registry=registry,
        prompts_by_protocol=prompts,
        cohort_id="cohort",
        tree_id="tree-001",
        model_id="n4",
        base_model_id="base/causal",
        base_model_revision=REVISION,
        model_task="causal_lm",
        request_sha256=HASH,
    )
    assert audit.valid
    assert audit.protocol_counts == {"dolly": 90, "mc": 300, "translation": 200}
    assert all(row.empty and row.text == "" and row.completed for row in responses)


def test_inference_preserves_empty_response_loads_once_and_resumes_strictly(
    tmp_path: Path,
) -> None:
    protocol = _protocol(
        "tiny",
        probes=("probe",),
        prompt_counts={"probe": 1},
        draw_seeds=(42, 43),
        do_sample=True,
        max_new_tokens=64,
    )
    registry = _registry(tmp_path, {"tiny": protocol})
    prompts = {"tiny": _prompts("probe", 1)}
    prompt_artifacts = _write_prompt_artifacts(tmp_path, prompts)
    checkpoint = _standalone_checkpoint(tmp_path)
    settings = {"backend": "fake", "version": 1}
    backend = _FakeBackend(["", "  exact text  "], settings)
    factory_calls: list[str] = []

    def factory(selected: ResolvedLeafCheckpoint) -> _FakeBackend:
        factory_calls.append(selected.leaf_id)
        return backend

    output = tmp_path / "responses.jsonl"
    receipt = tmp_path / "responses.receipt.json"
    first = run_leaf_probe_inference(
        checkpoint,
        registry=registry,
        prompts_by_protocol=prompts,
        prompt_artifacts=prompt_artifacts,
        output_path=output,
        receipt_path=receipt,
        backend_factory=factory,
        backend_settings=settings,
    )
    assert first["valid"] is True
    assert factory_calls == ["n4"]
    rows = load_behavior_responses(output)
    assert rows[0].text == ""
    assert rows[0].empty is True
    assert rows[1].text == "  exact text  "

    poison = _FakeBackend([], settings)
    cached = run_leaf_probe_inference(
        checkpoint,
        registry=registry,
        prompts_by_protocol=prompts,
        prompt_artifacts=prompt_artifacts,
        output_path=output,
        receipt_path=receipt,
        backend=poison,
    )
    assert cached == first
    assert poison.calls == []

    receipt.unlink()
    output.write_text(json.dumps(rows[0].to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    resume_backend = _FakeBackend(["  exact text  "], settings)
    run_leaf_probe_inference(
        checkpoint,
        registry=registry,
        prompts_by_protocol=prompts,
        prompt_artifacts=prompt_artifacts,
        output_path=output,
        receipt_path=receipt,
        backend=resume_backend,
    )
    assert len(resume_backend.calls) == 1
    assert len(load_behavior_responses(output)) == 2


def test_cache_refuses_prompt_backend_and_response_provenance_drift(tmp_path: Path) -> None:
    protocol = _protocol(
        "tiny",
        probes=("probe",),
        prompt_counts={"probe": 1},
        draw_seeds=(42,),
        do_sample=False,
        max_new_tokens=64,
    )
    registry = _registry(tmp_path, {"tiny": protocol})
    prompts = {"tiny": _prompts("probe", 1)}
    prompt_artifacts = _write_prompt_artifacts(tmp_path, prompts)
    checkpoint = _standalone_checkpoint(tmp_path)
    settings = {"backend": "fake", "version": 1}
    output = tmp_path / "responses.jsonl"
    receipt = tmp_path / "responses.receipt.json"
    run_leaf_probe_inference(
        checkpoint,
        registry=registry,
        prompts_by_protocol=prompts,
        prompt_artifacts=prompt_artifacts,
        output_path=output,
        receipt_path=receipt,
        backend=_FakeBackend(["answer"], settings),
    )

    alternate_prompt_path = tmp_path / "tiny.prompts.alternate.jsonl"
    original_prompt_path = Path(prompt_artifacts["tiny"]["path"])
    alternate_prompt_path.write_text(original_prompt_path.read_text() + "\n")
    alternate_prompt_artifacts = {
        "tiny": {
            "path": str(alternate_prompt_path),
            "sha256": sha256_file(alternate_prompt_path),
        }
    }
    with pytest.raises(ValueError, match="request provenance drift"):
        run_leaf_probe_inference(
            checkpoint,
            registry=registry,
            prompts_by_protocol=prompts,
            prompt_artifacts=alternate_prompt_artifacts,
            output_path=output,
            receipt_path=receipt,
            backend=_FakeBackend([], settings),
        )

    with pytest.raises(ValueError, match="request provenance drift"):
        run_leaf_probe_inference(
            checkpoint,
            registry=registry,
            prompts_by_protocol=prompts,
            prompt_artifacts=prompt_artifacts,
            output_path=output,
            receipt_path=receipt,
            backend_factory=lambda _: _FakeBackend([], {"backend": "fake", "version": 2}),
            backend_settings={"backend": "fake", "version": 2},
        )

    receipt_payload = json.loads(receipt.read_text())
    output.write_text(output.read_text().replace("answer", "tampered", 1))
    with pytest.raises(ValueError, match="artifact hash drift"):
        run_leaf_probe_inference(
            checkpoint,
            registry=registry,
            prompts_by_protocol=prompts,
            prompt_artifacts=prompt_artifacts,
            output_path=output,
            receipt_path=receipt,
            backend=_FakeBackend([], settings),
        )
    assert json.loads(receipt.read_text()) == receipt_payload


class _FakeBackend:
    def __init__(self, outputs: list[str], settings: Mapping[str, Any]) -> None:
        self.outputs = list(outputs)
        self.settings = dict(settings)
        self.calls: list[dict[str, Any]] = []

    def provenance(self) -> Mapping[str, Any]:
        return self.settings

    def generate_batch(
        self,
        prompts: list[str],
        *,
        model_task: str,
        generation_options: Mapping[str, Any],
    ) -> list[str]:
        assert model_task == "causal_lm"
        assert set(generation_options) == {
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
        self.calls.append({"prompts": list(prompts), "options": dict(generation_options)})
        if len(self.outputs) < len(prompts):
            raise AssertionError("fake backend was loaded/generated unexpectedly")
        outputs = self.outputs[: len(prompts)]
        del self.outputs[: len(prompts)]
        return outputs


def _resolve_default(summary: Path, root: Path):
    return resolve_leaf_checkpoints(
        summary,
        cohort_id="ordinary-full",
        tree_id="tree-001",
        model_task="causal_lm",
        base_model_id="base/causal",
        base_model_revision=REVISION,
        path_base=root,
    )


def _write_training_tree(root: Path, *, method: str) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    run_list = root / "tree.runs.jsonl"
    ledger = root / "tree.ledger.jsonl"
    manifest = root / "tree.manifest.jsonl"
    summary = root / "summary.json"
    manifest_rows = [
        {"node_id": "n0", "path": ["root", "n0"], "grow": "train"},
        {"node_id": "n1", "path": ["root", "n1"], "grow": "train"},
        {"node_id": "n4", "path": ["root", "n1", "n4"], "grow": "train"},
        {"node_id": "n2", "path": ["root", "n0", "n2"], "grow": "train"},
        {"node_id": "n5", "path": ["root", "n1", "n5"], "grow": "train"},
        {"node_id": "n3", "path": ["root", "n0", "n3"], "grow": "train"},
    ]
    _write_jsonl(manifest, manifest_rows)
    parents = {"n0": "root", "n1": "root", "n2": "n0", "n3": "n0", "n4": "n1", "n5": "n1"}
    checkpoints: dict[str, Path] = {}
    run_rows = []
    artifact_name = "model" if method == "full" else "merged"
    for index, (node_id, parent_id) in enumerate(parents.items()):
        checkpoint = root / "outputs" / node_id / artifact_name
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.json").write_text("{}\n")
        (checkpoint / "model.safetensors").write_bytes(f"weights-{node_id}".encode())
        checkpoints[node_id] = checkpoint
        expected = {artifact_name: str(checkpoint)}
        if method == "lora":
            expected["adapter"] = str(root / "outputs" / node_id / "adapter")
        run_rows.append(
            {
                "array_index": index,
                "run_id": node_id,
                "node_id": node_id,
                "parent_id": parent_id,
                "depth": 1 if parent_id == "root" else 2,
                "method": method,
                "dataset_id": "dataset",
                "task_family": "classification",
                "init_from": "base/causal",
                "output_dir": str(checkpoint.parent),
                "expected_artifacts": expected,
                "ledger_path": str(ledger),
                "runner": {"run_list_path": str(run_list)},
                "job": {
                    "base_model": "base/causal",
                    "base_model_revision": REVISION,
                    "method": method,
                    "expected_artifacts": expected,
                    "trainer": {"model_task": "causal_lm"},
                },
            }
        )
    _write_jsonl(run_list, run_rows)
    for node_id in parents:
        append_ledger_event(
            ledger,
            TrainingLedgerEvent(
                node_id=node_id,
                status="completed",
                extra={"artifacts": {artifact_name: str(checkpoints[node_id])}},
            ),
        )
    summary.write_text(
        json.dumps(
            {
                "n_errors": 0,
                "n_warnings": 0,
                "n_trees": 1,
                "trees": [
                    {
                        "tree_id": "tree-001",
                        "valid": True,
                        "n_errors": 0,
                        "n_warnings": 0,
                        "run_list": str(run_list),
                        "run_list_sha256": sha256_file(run_list),
                        "ledger": str(ledger),
                        "manifest": str(manifest),
                        "manifest_sha256": sha256_file(manifest),
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return {
        "summary": summary,
        "run_list": run_list,
        "ledger": ledger,
        "manifest": manifest,
        "checkpoints": checkpoints,
    }


def _update_summary_hash(summary_path: Path, key: str, source: Path) -> None:
    summary = json.loads(summary_path.read_text())
    summary["trees"][0][key] = sha256_file(source)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _protocol(
    protocol_id: str,
    *,
    probes: tuple[str, ...],
    prompt_counts: Mapping[str, int],
    draw_seeds: tuple[int, ...],
    do_sample: bool,
    max_new_tokens: int,
) -> BehaviorProtocol:
    architecture = ArchitectureContract(
        max_new_tokens=max_new_tokens,
        prompt_templates={probe: "{text}" for probe in probes},
    )
    return BehaviorProtocol(
        protocol_id=protocol_id,
        fixture_id=f"fixture-{probes[0]}",
        probe_ids=probes,
        prompt_counts=dict(prompt_counts),
        samples_per_prompt=len(draw_seeds),
        base_seed=42,
        draw_seeds=draw_seeds,
        generation_batch_size=32 if do_sample else 16,
        seed_scope="probe_draw",
        do_sample=do_sample,
        temperature=1.0 if do_sample else None,
        top_p=1.0 if do_sample else None,
        min_new_tokens=0,
        empty_policy="preserve_exact",
        analysis=AnalysisContract(
            primary_empty_policy="preserve_exact",
            sensitivity_empty_policy="remove_exact_empty",
            natural_language_only=False,
        ),
        architectures={"causal_lm": architecture, "seq2seq": architecture},
    )


def _registry(
    tmp_path: Path, protocols: Mapping[str, BehaviorProtocol]
) -> BehaviorProtocolRegistry:
    path = tmp_path / f"registry-{len(list(tmp_path.glob('registry-*')))}.json"
    path.write_text("{}\n")
    return BehaviorProtocolRegistry(
        model_pins={
            "causal_lm": ModelPin("base/causal", REVISION),
            "seq2seq": ModelPin("base/seq2seq", SEQ2SEQ_REVISION),
        },
        embedding_pin=ModelPin("embedding/model", EMBEDDING_REVISION),
        fixtures={},
        protocols=dict(protocols),
        path=path,
        sha256=sha256_file(path),
    )


def _prompts(probe_id: str, count: int) -> list[RenderedBehaviorPrompt]:
    prompts = []
    source_indices_hash = hashlib.sha256(f"indices-{probe_id}".encode()).hexdigest()
    fixture_hash = hashlib.sha256(f"fixture-{probe_id}".encode()).hexdigest()
    for index in range(count):
        prompt_text = f"{probe_id} prompt {index}"
        prompt_hash = hashlib.sha256(prompt_text.encode()).hexdigest()
        source_hash = hashlib.sha256(f"source-{probe_id}-{index}".encode()).hexdigest()
        prompts.append(
            RenderedBehaviorPrompt(
                probe_id=probe_id,
                prompt_id=f"{probe_id}-{index:04d}",
                prompt=prompt_text,
                reference=None,
                source_index=index,
                source_row_sha256=source_hash,
                prompt_sha256=prompt_hash,
                metadata={
                    "fixture_id": f"fixture-{probe_id}",
                    "dataset": f"dataset-{probe_id}",
                    "dataset_config": None,
                    "split": "test",
                    "dataset_revision": REVISION,
                    "source_index": index,
                    "source_row_sha256": source_hash,
                    "prompt_sha256": prompt_hash,
                    "source_indices_sha256": source_indices_hash,
                    "fixture_sha256": fixture_hash,
                    "model_task": "causal_lm",
                    "selection_seed": 42,
                },
            )
        )
    return prompts


def _response_provenance() -> ResponseProvenance:
    return ResponseProvenance(
        request_sha256=HASH,
        training_summary="summary.json",
        training_summary_sha256=HASH,
        run_list="runs.jsonl",
        run_list_sha256=HASH,
        ledger="ledger.jsonl",
        ledger_sha256=HASH,
        truth_manifest="truth.jsonl",
        truth_manifest_sha256=HASH,
        checkpoint="model",
        checkpoint_artifact="model",
        checkpoint_sha256=HASH,
        registry="registry.json",
        registry_sha256=HASH,
        prompt_artifacts={"protocol": {"path": "prompts.jsonl", "sha256": HASH}},
    )


def _standalone_checkpoint(tmp_path: Path) -> ResolvedLeafCheckpoint:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(exist_ok=True)
    (checkpoint / "config.json").write_text("{}\n")
    (checkpoint / "model.safetensors").write_text("weights\n")
    provenance_files = {}
    for name in ("summary.json", "runs.jsonl", "ledger.jsonl", "truth.jsonl"):
        path = tmp_path / name
        path.write_text(f"{name}\n")
        provenance_files[name] = path
    return ResolvedLeafCheckpoint(
        cohort_id="cohort",
        tree_id="tree-001",
        leaf_id="n4",
        leaf_ordinal=0,
        method="full",
        artifact_name="model",
        checkpoint=checkpoint,
        base_model_id="base/causal",
        base_model_revision=REVISION,
        model_task="causal_lm",
        training_summary=provenance_files["summary.json"],
        training_summary_sha256=sha256_file(provenance_files["summary.json"]),
        run_list=provenance_files["runs.jsonl"],
        run_list_sha256=sha256_file(provenance_files["runs.jsonl"]),
        ledger=provenance_files["ledger.jsonl"],
        ledger_sha256=sha256_file(provenance_files["ledger.jsonl"]),
        truth_manifest=provenance_files["truth.jsonl"],
        truth_manifest_sha256=sha256_file(provenance_files["truth.jsonl"]),
    )


def _write_prompt_artifacts(
    tmp_path: Path,
    prompts_by_protocol: Mapping[str, list[RenderedBehaviorPrompt]],
) -> dict[str, dict[str, str]]:
    artifacts = {}
    for protocol_id, prompts in prompts_by_protocol.items():
        path = tmp_path / f"{protocol_id}.prompts.jsonl"
        path.write_text(
            "".join(
                json.dumps(prompt.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
                for prompt in prompts
            )
        )
        artifacts[protocol_id] = {"path": str(path), "sha256": sha256_file(path)}
    return artifacts
