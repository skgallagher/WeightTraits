"""GPU population collection for the locked corrected PhyloLM protocol."""

from __future__ import annotations

import gc
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from weighttraits.behavior.phylolm import (
    LOCKED_PHYLOLM_PROTOCOL,
    PHYLOLM_GENERATION_SCHEMA,
    build_population_receipt,
    compute_population,
    load_genome_receipt,
    write_population_receipt,
)
from weighttraits.behavior.probe_inference import (
    ResolvedLeafCheckpoint,
    sha256_path,
)
from weighttraits.paper.analysis_contracts import (
    canonical_tree_id,
    sha256,
    validate_strict_training_completion,
)


PipelineFactory = Callable[..., Any]
SeedSetter = Callable[[int], None]


def extract_raw_continuation_alleles(
    prompts: Sequence[str], outputs: Sequence[Any]
) -> list[str]:
    """Extract exactly the first four characters after each raw prompt."""

    if len(prompts) != len(outputs):
        raise ValueError("PhyloLM output count differs from prompt count")
    alleles: list[str] = []
    for index, (prompt, output) in enumerate(zip(prompts, outputs, strict=True)):
        record = output[0] if isinstance(output, list) else output
        if not isinstance(record, Mapping) or not isinstance(
            record.get("generated_text"), str
        ):
            raise ValueError(f"PhyloLM output {index} has no generated_text")
        generated = record["generated_text"]
        if not generated.startswith(prompt):
            raise ValueError(f"PhyloLM output {index} does not preserve the raw prompt")
        alleles.append(
            generated[len(prompt) :][: LOCKED_PHYLOLM_PROTOCOL.allele_characters]
        )
    return alleles


def generate_phylolm_alleles(
    model: Any,
    tokenizer: Any,
    genes: Sequence[str],
    *,
    batch_size: int = 64,
    pipeline_factory: PipelineFactory | None = None,
    set_seed: SeedSetter | None = None,
) -> np.ndarray:
    """Generate the exact G=128 by N=32 raw-continuation allele grid."""

    protocol = LOCKED_PHYLOLM_PROTOCOL
    if (
        len(genes) != protocol.n_genes
        or not all(isinstance(gene, str) and gene for gene in genes)
    ):
        raise ValueError("PhyloLM inference requires exactly 128 nonempty genes")
    if batch_size != protocol.batch_size or isinstance(batch_size, bool):
        raise ValueError(
            f"corrected PhyloLM requires locked batch_size={protocol.batch_size}"
        )
    if pipeline_factory is None or set_seed is None:
        try:
            from transformers import pipeline as transformers_pipeline
            from transformers import set_seed as transformers_set_seed
        except ImportError as exc:  # pragma: no cover - production dependency
            raise RuntimeError("PhyloLM inference requires transformers") from exc
        pipeline_factory = pipeline_factory or transformers_pipeline
        set_seed = set_seed or transformers_set_seed
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("PhyloLM tokenizer has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
        # Some tokenizer implementations expose ``pad_token_id`` as a
        # derived property, while lightweight/test implementations do not.
        # Pin the actual generation argument explicitly in either case.
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    generator = pipeline_factory("text-generation", model=model, tokenizer=tokenizer)
    set_seed(protocol.sampling_seed)
    prompts = [gene for gene in genes for _ in range(protocol.samples_per_gene)]
    outputs = generator(
        prompts,
        do_sample=True,
        min_new_tokens=protocol.new_tokens,
        max_new_tokens=protocol.new_tokens,
        temperature=protocol.temperature,
        return_full_text=True,
        pad_token_id=tokenizer.pad_token_id,
        batch_size=batch_size,
    )
    alleles = extract_raw_continuation_alleles(prompts, outputs)
    return np.asarray(alleles, dtype=object).reshape(
        protocol.n_genes, protocol.samples_per_gene
    )


def collect_phylolm_leaf_population(
    checkpoint: ResolvedLeafCheckpoint,
    *,
    genome_receipt_path: str | Path,
    completion_receipt_path: str | Path,
    stage_manifest_path: str | Path,
    output_path: str | Path,
    batch_size: int = 64,
    torch_dtype: str = "auto",
    model_loader: Callable[[ResolvedLeafCheckpoint, str], tuple[Any, Any, Mapping[str, Any]]]
    | None = None,
) -> dict[str, Any]:
    """Load one merged/model checkpoint once and atomically publish its population."""

    if checkpoint.model_task != "causal_lm":
        raise ValueError("corrected PhyloLM supports causal_lm checkpoints only")
    if checkpoint.artifact_name not in {"model", "merged"}:
        raise ValueError("corrected PhyloLM requires model or merged full weights")
    protocol = LOCKED_PHYLOLM_PROTOCOL
    if batch_size != protocol.batch_size or isinstance(batch_size, bool):
        raise ValueError(
            f"corrected PhyloLM requires locked batch_size={protocol.batch_size}"
        )
    if torch_dtype != protocol.torch_dtype:
        raise ValueError(
            f"corrected PhyloLM requires locked torch_dtype={protocol.torch_dtype!r}"
        )
    canonical_tree_id(checkpoint.tree_id)
    genome_path = Path(genome_receipt_path).resolve()
    genome = load_genome_receipt(genome_path)
    genes = genome.get("genes")
    if not isinstance(genes, list):
        raise ValueError("PhyloLM genome receipt lacks genes")
    completion = Path(completion_receipt_path).resolve()
    stage_manifest = Path(stage_manifest_path).resolve()
    if not completion.is_file():
        raise FileNotFoundError(f"PhyloLM completion receipt is missing: {completion}")
    if not stage_manifest.is_file():
        raise FileNotFoundError(f"PhyloLM stage manifest is missing: {stage_manifest}")
    validate_strict_training_completion(
        json.loads(completion.read_text()), cohort_id=checkpoint.cohort_id
    )
    source_pins = _validated_checkpoint_source_pins(checkpoint)
    completion_digest = sha256(completion)
    stage_digest = sha256(stage_manifest)
    checkpoint_digest_before = sha256_path(checkpoint.checkpoint)
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite PhyloLM population: {output}")
    started = time.monotonic()
    loader = model_loader or _load_huggingface_checkpoint
    model, tokenizer, loader_receipt = loader(checkpoint, torch_dtype)
    try:
        alleles = generate_phylolm_alleles(
            model,
            tokenizer,
            genes,
            batch_size=batch_size,
        )
        population = compute_population(alleles.tolist())
    finally:
        del model
        del tokenizer
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover - production dependency
            pass
    checkpoint_digest = sha256_path(checkpoint.checkpoint)
    if checkpoint_digest != checkpoint_digest_before:
        raise ValueError("PhyloLM checkpoint changed during model loading/generation")
    for field, pin in source_pins.items():
        if sha256(pin[0]) != pin[1]:
            raise ValueError(f"PhyloLM {field} changed during generation")
    if sha256(completion) != completion_digest or sha256(stage_manifest) != stage_digest:
        raise ValueError("PhyloLM completion/stage provenance changed during generation")
    generation_receipt = {
        "schema": PHYLOLM_GENERATION_SCHEMA,
        "valid": True,
        "suite_id": checkpoint.cohort_id,
        "tree_id": canonical_tree_id(checkpoint.tree_id),
        "model_id": checkpoint.leaf_id,
        "protocol": LOCKED_PHYLOLM_PROTOCOL.to_dict(),
        "checkpoint": str(checkpoint.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_digest,
        "training_summary": str(checkpoint.training_summary.resolve()),
        "training_summary_sha256": checkpoint.training_summary_sha256,
        "run_list": str(checkpoint.run_list.resolve()),
        "run_list_sha256": checkpoint.run_list_sha256,
        "ledger": str(checkpoint.ledger.resolve()),
        "ledger_sha256": checkpoint.ledger_sha256,
        "truth_manifest": str(checkpoint.truth_manifest.resolve()),
        "truth_manifest_sha256": checkpoint.truth_manifest_sha256,
        "completion_receipt": str(completion),
        "completion_receipt_sha256": completion_digest,
        "stage_manifest": str(stage_manifest),
        "stage_manifest_sha256": stage_digest,
        "loader": dict(loader_receipt),
        "batch_size": batch_size,
        "torch_dtype": torch_dtype,
        "elapsed_seconds": time.monotonic() - started,
    }
    receipt = build_population_receipt(
        suite_id=checkpoint.cohort_id,
        tree_id=checkpoint.tree_id,
        model_id=checkpoint.leaf_id,
        population=population,
        genome_receipt_path=genome_path,
        checkpoint_path=checkpoint.checkpoint,
        checkpoint_sha256=checkpoint_digest,
        generation_receipt=generation_receipt,
    )
    write_population_receipt(receipt, output)
    return receipt


def _load_huggingface_checkpoint(
    checkpoint: ResolvedLeafCheckpoint, torch_dtype: str
) -> tuple[Any, Any, Mapping[str, Any]]:
    try:
        import torch
        import transformers
    except ImportError as exc:  # pragma: no cover - production dependency
        raise RuntimeError("PhyloLM inference requires torch and transformers") from exc
    if torch_dtype != LOCKED_PHYLOLM_PROTOCOL.torch_dtype:
        raise ValueError("corrected PhyloLM requires torch_dtype='auto'")
    dtype: Any = torch_dtype
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(checkpoint.checkpoint),
        local_files_only=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        checkpoint.base_model_id,
        revision=checkpoint.base_model_revision,
        local_files_only=True,
    )
    return model, tokenizer, {
        "backend": "huggingface_text_generation_pipeline",
        "checkpoint": str(checkpoint.checkpoint),
        "checkpoint_artifact": checkpoint.artifact_name,
        "tokenizer_model_id": checkpoint.base_model_id,
        "tokenizer_revision": checkpoint.base_model_revision,
        "local_files_only": True,
        "torch_dtype": torch_dtype,
        "torch_version": getattr(torch, "__version__", None),
        "transformers_version": getattr(transformers, "__version__", None),
    }


def _validated_checkpoint_source_pins(
    checkpoint: ResolvedLeafCheckpoint,
) -> dict[str, tuple[Path, str]]:
    pins = {
        "training_summary": (
            checkpoint.training_summary.resolve(),
            checkpoint.training_summary_sha256,
        ),
        "run_list": (checkpoint.run_list.resolve(), checkpoint.run_list_sha256),
        "ledger": (checkpoint.ledger.resolve(), checkpoint.ledger_sha256),
        "truth_manifest": (
            checkpoint.truth_manifest.resolve(),
            checkpoint.truth_manifest_sha256,
        ),
    }
    for field, (path, expected) in pins.items():
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"PhyloLM checkpoint {field} pin does not verify")
    return pins
