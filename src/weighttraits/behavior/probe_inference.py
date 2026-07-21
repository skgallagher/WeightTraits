"""Generic held-out behavior collection and sentence-embedding export."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from weighttraits.behavior.probes import BehaviorPrompt
from weighttraits.behavior.responses import BehaviorResponse, audit_behavior_responses
from weighttraits.distances.manifest import DistanceInputSpec


DEFAULT_SENTENCE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SENTENCE_EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def behavior_collection_provenance(
    spec: DistanceInputSpec,
    *,
    run_id: str,
    prompts: Sequence[BehaviorPrompt],
    model_task: str,
    base_model: str | None,
    base_revision: str | None,
    samples_per_prompt: int,
    min_new_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    batch_size: int,
    seed: int,
    empty_policy: str,
) -> dict[str, Any]:
    """Return a stable fingerprint for one model's response artifact."""

    prompt_json = json.dumps(
        [prompt.to_dict() for prompt in prompts],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "model_id": spec.model_id,
        "checkpoint": None if spec.checkpoint is None else str(spec.checkpoint.resolve()),
        "adapter_chain": [str(path.resolve()) for path in spec.adapter_chain],
        "model_task": model_task,
        "base_model": base_model,
        "base_revision": base_revision,
        "prompt_sha256": hashlib.sha256(prompt_json).hexdigest(),
        "n_prompts": len(prompts),
        "samples_per_prompt": samples_per_prompt,
        "min_new_tokens": min_new_tokens,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "top_p": top_p if do_sample else None,
        "batch_size": batch_size,
        "seed": seed,
        "empty_policy": empty_policy,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**payload, "sha256": hashlib.sha256(canonical).hexdigest()}


def audit_cached_behavior_responses(
    records: Sequence[BehaviorResponse],
    *,
    provenance: dict[str, Any],
    prompts: Sequence[BehaviorPrompt],
    samples_per_prompt: int,
    allow_empty_completed: bool,
) -> dict[str, Any]:
    """Fail closed unless a cached response file exactly matches the requested run."""

    expected_keys = {
        (
            str(provenance["run_id"]),
            str(provenance["model_id"]),
            prompt.probe_id,
            prompt.prompt_id,
            sample_id,
        )
        for prompt in prompts
        for sample_id in range(samples_per_prompt)
    }
    observed_keys = {record.key for record in records}
    observed_hashes = {
        str(record.metadata.get("collection_provenance", {}).get("sha256"))
        for record in records
    }
    issues = []
    if observed_keys != expected_keys:
        issues.append(
            {
                "issue": "response_grid_mismatch",
                "missing_keys": [list(key) for key in sorted(expected_keys - observed_keys)],
                "unexpected_keys": [list(key) for key in sorted(observed_keys - expected_keys)],
            }
        )
    if observed_hashes != {str(provenance["sha256"])}:
        issues.append(
            {
                "issue": "collection_provenance_mismatch",
                "expected_sha256": provenance["sha256"],
                "observed_sha256": sorted(observed_hashes),
            }
        )
    response_audit = audit_behavior_responses(
        records,
        expected_samples_per_prompt=samples_per_prompt,
        allow_empty_completed=allow_empty_completed,
    )
    if not response_audit["valid"]:
        issues.append({"issue": "cached_response_audit_failed", "details": response_audit["issues"]})
    return {
        "valid": not issues,
        "issues": issues,
        "expected_provenance": provenance,
        "response_audit": response_audit,
    }


def load_behavior_leaf_model(
    spec: DistanceInputSpec,
    *,
    model_task: str,
    base_model: str | None,
    base_revision: str | None,
    local_files_only: bool = True,
) -> tuple[Any, Any, dict[str, Any]]:
    if model_task not in {"seq2seq", "causal_lm"}:
        raise ValueError("behavior model_task must be seq2seq or causal_lm")
    try:
        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("behavior collection requires transformers") from exc
    model_class = AutoModelForCausalLM if model_task == "causal_lm" else AutoModelForSeq2SeqLM
    common = {"torch_dtype": "auto", "device_map": "auto", "local_files_only": local_files_only}
    if spec.checkpoint is not None:
        model = model_class.from_pretrained(str(spec.checkpoint), **common)
        tokenizer = AutoTokenizer.from_pretrained(
            str(spec.checkpoint), local_files_only=local_files_only
        )
        return model, tokenizer, {
            "representation": "full_checkpoint",
            "checkpoint": str(spec.checkpoint),
        }
    if not spec.adapter_chain or not base_model:
        raise ValueError("adapter-chain behavior collection requires --base-model")
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError("adapter-chain behavior collection requires peft") from exc
    model = model_class.from_pretrained(base_model, revision=base_revision, **common)
    for adapter in spec.adapter_chain:
        model = PeftModel.from_pretrained(model, str(adapter), local_files_only=local_files_only)
        model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, revision=base_revision, local_files_only=local_files_only
    )
    return model, tokenizer, {
        "representation": "cumulative_lora_rematerialized",
        "base_model": base_model,
        "base_revision": base_revision,
        "adapter_chain": [str(path) for path in spec.adapter_chain],
    }


def extract_behavior_pipeline_outputs(
    prompts: Sequence[str], outputs: Sequence[Any], *, model_task: str
) -> list[str]:
    if len(prompts) != len(outputs):
        raise ValueError("behavior generation output count does not match prompt count")
    responses: list[str] = []
    for prompt, output in zip(prompts, outputs):
        record = output[0] if isinstance(output, list) else output
        if not isinstance(record, dict) or "generated_text" not in record:
            raise ValueError("unexpected behavior generation pipeline output")
        text = str(record["generated_text"])
        if model_task == "causal_lm":
            if not text.startswith(prompt):
                raise ValueError("causal generation output does not preserve its prompt")
            text = text[len(prompt) :]
        responses.append(text.strip())
    return responses


def generate_seq2seq_responses(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    samples_per_batch: int,
    min_new_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> list[str]:
    """Generate seq2seq text without relying on version-specific pipeline task names."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("behavior collection requires torch") from exc
    responses: list[str] = []
    for start in range(0, len(prompts), samples_per_batch):
        batch = list(prompts[start : start + samples_per_batch])
        encoded = tokenizer(batch, padding=True, return_tensors="pt")
        device = getattr(model, "device", None)
        if device is not None:
            encoded = {key: value.to(device) for key, value in encoded.items()}
        generation_options = {
            "do_sample": do_sample,
            "min_new_tokens": min_new_tokens,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if do_sample:
            generation_options["temperature"] = temperature
            generation_options["top_p"] = top_p
        with torch.inference_mode():
            generated = model.generate(**encoded, **generation_options)
        responses.extend(
            text.strip()
            for text in tokenizer.batch_decode(generated, skip_special_tokens=True)
        )
    if len(responses) != len(prompts):
        raise ValueError("seq2seq generation output count does not match prompt count")
    return responses


def generate_causal_responses(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    samples_per_batch: int,
    min_new_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> list[str]:
    """Generate and decode only new causal tokens using the stable model API."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("behavior collection requires torch") from exc
    responses: list[str] = []
    for start in range(0, len(prompts), samples_per_batch):
        batch = list(prompts[start : start + samples_per_batch])
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        device = getattr(model, "device", None)
        if device is not None:
            encoded = {key: value.to(device) for key, value in encoded.items()}
        generation_options = {
            "do_sample": do_sample,
            "min_new_tokens": min_new_tokens,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if do_sample:
            generation_options["temperature"] = temperature
            generation_options["top_p"] = top_p
        with torch.inference_mode():
            generated = model.generate(**encoded, **generation_options)
        prompt_tokens = encoded["input_ids"].shape[1]
        responses.extend(
            text.strip()
            for text in tokenizer.batch_decode(
                generated[:, prompt_tokens:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )
    if len(responses) != len(prompts):
        raise ValueError("causal generation output count does not match prompt count")
    return responses


def collect_behavior_model_responses(
    spec: DistanceInputSpec,
    *,
    run_id: str,
    prompts: Sequence[BehaviorPrompt],
    out_path: str | Path,
    model_task: str,
    base_model: str | None,
    base_revision: str | None,
    samples_per_prompt: int,
    min_new_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    batch_size: int,
    seed: int,
    empty_policy: str = "drop",
    local_files_only: bool = True,
    collection_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if samples_per_prompt <= 0 or max_new_tokens <= 0 or batch_size <= 0:
        raise ValueError("samples_per_prompt, max_new_tokens, and batch_size must be positive")
    if min_new_tokens < 0 or min_new_tokens > max_new_tokens:
        raise ValueError("min_new_tokens must be between zero and max_new_tokens")
    if not prompts:
        raise ValueError("behavior collection requires at least one prompt")
    if do_sample and temperature <= 0:
        raise ValueError("behavior sampling temperature must be positive")
    if do_sample and not 0 < top_p <= 1:
        raise ValueError("behavior sampling top_p must be in (0, 1]")
    if empty_policy not in {"drop", "preserve"}:
        raise ValueError("empty_policy must be drop or preserve")
    provenance = collection_provenance or behavior_collection_provenance(
        spec,
        run_id=run_id,
        prompts=prompts,
        model_task=model_task,
        base_model=base_model,
        base_revision=base_revision,
        samples_per_prompt=samples_per_prompt,
        min_new_tokens=min_new_tokens,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        batch_size=batch_size,
        seed=seed,
        empty_policy=empty_policy,
    )
    model, tokenizer, model_metadata = load_behavior_leaf_model(
        spec,
        model_task=model_task,
        base_model=base_model,
        base_revision=base_revision,
        local_files_only=local_files_only,
    )
    try:
        from transformers import set_seed

        model.eval()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if model_task == "causal_lm":
            tokenizer.padding_side = "left"
        prompt_texts = [prompt.prompt for prompt in prompts]
        generated_by_sample: list[list[str]] = []
        for sample_id in range(samples_per_prompt):
            # A separate, recorded seed makes each Monte Carlo draw reproducible and
            # aligns draw IDs across every model in the same behavioral run.
            set_seed(seed + sample_id)
            generator = (
                generate_causal_responses
                if model_task == "causal_lm"
                else generate_seq2seq_responses
            )
            generated_by_sample.append(
                generator(
                    model,
                    tokenizer,
                    prompt_texts,
                    samples_per_batch=batch_size,
                    min_new_tokens=min_new_tokens,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                )
            )
    finally:
        del model
        del tokenizer
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    records: list[BehaviorResponse] = []
    for prompt_index, prompt in enumerate(prompts):
        for sample_id in range(samples_per_prompt):
            response = generated_by_sample[sample_id][prompt_index]
            sample_seed = seed + sample_id
            records.append(
                BehaviorResponse(
                    run_id=run_id,
                    model_id=spec.model_id,
                    probe_id=prompt.probe_id,
                    prompt_id=prompt.prompt_id,
                    sample_id=sample_id,
                    prompt=prompt.prompt,
                    response=response,
                    status="completed" if response or empty_policy == "preserve" else "dropped",
                    reference=prompt.reference,
                    seed=sample_seed,
                    reason=None if response else "empty_generation",
                    metadata={
                        **model_metadata,
                        "model_task": model_task,
                        "do_sample": do_sample,
                        "temperature": temperature if do_sample else None,
                        "top_p": top_p if do_sample else None,
                        "base_seed": seed,
                        "batch_size": batch_size,
                        "min_new_tokens": min_new_tokens,
                        "max_new_tokens": max_new_tokens,
                        "empty_policy": empty_policy,
                        "collection_provenance": provenance,
                    },
                )
            )
    from weighttraits.behavior.responses import write_behavior_responses

    write_behavior_responses(records, out_path)
    return audit_behavior_responses(
        records,
        expected_samples_per_prompt=samples_per_prompt,
        allow_empty_completed=empty_policy == "preserve",
    )


def build_embedding_tensor(
    records: Sequence[BehaviorResponse],
    *,
    encode: Callable[[list[str]], np.ndarray],
    natural_language_only: bool = False,
) -> tuple[np.ndarray, list[str], list[str], np.ndarray, dict[str, Any]]:
    if not records:
        raise ValueError("at least one behavior response is required")
    run_ids = {record.run_id for record in records}
    probe_ids = {record.probe_id for record in records}
    if len(run_ids) != 1 or len(probe_ids) != 1:
        raise ValueError("embedding export requires exactly one run and one probe")
    model_ids = sorted({record.model_id for record in records})
    observation_ids = sorted({f"{record.prompt_id}:{record.sample_id}" for record in records})
    model_index = {model_id: index for index, model_id in enumerate(model_ids)}
    observation_index = {value: index for index, value in enumerate(observation_ids)}
    usable = [
        record
        for record in records
        if record.status == "completed" and (record.response or "").strip()
    ]
    if natural_language_only:
        from weighttraits.behavior.surface import is_natural_language_response

        usable = [record for record in usable if is_natural_language_response(record.response)]
    if not usable:
        raise ValueError("behavior embedding export found no completed non-empty responses")
    encoded = np.asarray(encode([str(record.response) for record in usable]), dtype=np.float64)
    if encoded.ndim != 2 or encoded.shape[0] != len(usable):
        raise ValueError("sentence encoder returned an unexpected embedding shape")
    embeddings = np.zeros((len(model_ids), len(observation_ids), encoded.shape[1]), dtype=np.float32)
    valid = np.zeros((len(model_ids), len(observation_ids)), dtype=bool)
    for record, vector in zip(usable, encoded):
        left = model_index[record.model_id]
        right = observation_index[f"{record.prompt_id}:{record.sample_id}"]
        if valid[left, right]:
            raise ValueError(f"duplicate completed behavior observation: {record.key}")
        embeddings[left, right] = vector
        valid[left, right] = True
    return embeddings, model_ids, observation_ids, valid, {
        "schema_version": 1,
        "run_id": next(iter(run_ids)),
        "probe_id": next(iter(probe_ids)),
        "n_models": len(model_ids),
        "n_observations": len(observation_ids),
        "n_valid_embeddings": int(np.sum(valid)),
        "natural_language_only": natural_language_only,
        "semantic_coverage_by_model": {
            model_id: int(np.sum(valid[index])) / len(observation_ids)
            for index, model_id in enumerate(model_ids)
        },
        "embedding_dimension": int(encoded.shape[1]),
    }


def sentence_transformer_embedding_tensor(
    records: Sequence[BehaviorResponse],
    *,
    model_name: str,
    model_revision: str,
    batch_size: int = 64,
    local_files_only: bool = True,
    natural_language_only: bool = False,
):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("behavior embedding export requires sentence-transformers") from exc
    model = SentenceTransformer(
        model_name,
        revision=model_revision,
        local_files_only=local_files_only,
    )
    return build_embedding_tensor(
        records,
        encode=lambda texts: model.encode(
            texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False
        ),
        natural_language_only=natural_language_only,
    )
