"""GPU collection for PhyloLM, including cumulative LoRA leaf reconstruction."""

from __future__ import annotations

import gc
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from weighttraits.behavior.phylolm import (
    DEFAULT_ALLELE_CHARACTERS,
    DEFAULT_NEW_TOKENS,
    DEFAULT_SAMPLES_PER_GENE,
    compute_population,
    save_population,
)
from weighttraits.distances.manifest import DistanceInputSpec


def load_sampled_genome(path: str | Path) -> tuple[list[str], dict[str, Any]]:
    row = json.loads(Path(path).read_text())
    genes = row.get("genes")
    if not isinstance(genes, list) or not genes or not all(isinstance(gene, str) for gene in genes):
        raise ValueError("sampled PhyloLM genome must contain a non-empty genes list")
    return genes, row


def extract_pipeline_alleles(
    prompts: Sequence[str], outputs: Sequence[Any], *, allele_characters: int
) -> list[str]:
    if allele_characters <= 0:
        raise ValueError("allele_characters must be positive")
    if len(prompts) != len(outputs):
        raise ValueError("PhyloLM pipeline output count does not match prompt count")
    alleles: list[str] = []
    for prompt, output in zip(prompts, outputs):
        record = output[0] if isinstance(output, list) else output
        if not isinstance(record, dict) or "generated_text" not in record:
            raise ValueError("unexpected text-generation pipeline output")
        generated = str(record["generated_text"])
        if not generated.startswith(prompt):
            raise ValueError("text-generation output does not preserve the raw PhyloLM prompt")
        alleles.append(generated[len(prompt) :][:allele_characters])
    return alleles


def generate_phylolm_alleles(
    model: Any,
    tokenizer: Any,
    genes: Sequence[str],
    *,
    samples_per_gene: int = DEFAULT_SAMPLES_PER_GENE,
    new_tokens: int = DEFAULT_NEW_TOKENS,
    allele_characters: int = DEFAULT_ALLELE_CHARACTERS,
    temperature: float = 1.0,
    batch_size: int = 64,
    seed: int = 0,
) -> np.ndarray:
    """Run PhyloLM's raw-prompt fixed-token sampling contract."""

    if samples_per_gene <= 0 or new_tokens <= 0 or batch_size <= 0:
        raise ValueError("samples_per_gene, new_tokens, and batch_size must be positive")
    if temperature <= 0:
        raise ValueError("PhyloLM sampling temperature must be positive")
    try:
        from transformers import pipeline, set_seed
    except ImportError as exc:
        raise ImportError("PhyloLM collection requires transformers") from exc

    set_seed(seed)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    prompts = [gene for gene in genes for _ in range(samples_per_gene)]
    outputs = generator(
        prompts,
        do_sample=True,
        min_new_tokens=new_tokens,
        max_new_tokens=new_tokens,
        temperature=temperature,
        return_full_text=True,
        pad_token_id=tokenizer.pad_token_id,
        batch_size=batch_size,
    )
    alleles = extract_pipeline_alleles(
        prompts, outputs, allele_characters=allele_characters
    )
    return np.asarray(alleles, dtype=object).reshape(len(genes), samples_per_gene)


def load_causal_leaf_model(
    spec: DistanceInputSpec,
    *,
    base_model: str | None,
    base_revision: str | None,
    local_files_only: bool = True,
    torch_dtype: str = "auto",
) -> tuple[Any, Any, dict[str, Any]]:
    """Load a full checkpoint or rematerialize a cumulative LoRA leaf from base + chain."""

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("PhyloLM collection requires transformers") from exc

    common = {
        "torch_dtype": torch_dtype,
        "device_map": "auto",
        "local_files_only": local_files_only,
    }
    if spec.checkpoint is not None:
        model = AutoModelForCausalLM.from_pretrained(str(spec.checkpoint), **common)
        tokenizer = AutoTokenizer.from_pretrained(
            str(spec.checkpoint), local_files_only=local_files_only
        )
        return model, tokenizer, {
            "representation": "full_checkpoint",
            "checkpoint": str(spec.checkpoint),
            "adapter_chain": [],
        }

    if not spec.adapter_chain:
        raise ValueError(f"model {spec.model_id} has neither checkpoint nor adapter chain")
    if not base_model:
        raise ValueError("cumulative LoRA PhyloLM collection requires --base-model")
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError("cumulative LoRA PhyloLM collection requires peft") from exc

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        revision=base_revision,
        **common,
    )
    for adapter_path in spec.adapter_chain:
        model = PeftModel.from_pretrained(model, str(adapter_path), local_files_only=local_files_only)
        model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        revision=base_revision,
        local_files_only=local_files_only,
    )
    return model, tokenizer, {
        "representation": "cumulative_lora_rematerialized",
        "base_model": base_model,
        "base_revision": base_revision,
        "adapter_chain": [str(path) for path in spec.adapter_chain],
    }


def collect_phylolm_population(
    spec: DistanceInputSpec,
    *,
    genome_path: str | Path,
    out_path: str | Path,
    base_model: str | None = None,
    base_revision: str | None = None,
    samples_per_gene: int = DEFAULT_SAMPLES_PER_GENE,
    new_tokens: int = DEFAULT_NEW_TOKENS,
    allele_characters: int = DEFAULT_ALLELE_CHARACTERS,
    temperature: float = 1.0,
    batch_size: int = 64,
    seed: int = 0,
    local_files_only: bool = True,
) -> dict[str, Any]:
    genes, genome_metadata = load_sampled_genome(genome_path)
    started = time.monotonic()
    model, tokenizer, model_metadata = load_causal_leaf_model(
        spec,
        base_model=base_model,
        base_revision=base_revision,
        local_files_only=local_files_only,
    )
    try:
        alleles = generate_phylolm_alleles(
            model,
            tokenizer,
            genes,
            samples_per_gene=samples_per_gene,
            new_tokens=new_tokens,
            allele_characters=allele_characters,
            temperature=temperature,
            batch_size=batch_size,
            seed=seed,
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
        except ImportError:
            pass

    mean_distinct = float(np.mean([len(gene) for gene in population]))
    mean_empty = float(np.mean([gene.get("", 0.0) for gene in population]))
    metadata = {
        **model_metadata,
        "genome_path": str(genome_path),
        "genome_source_sha256": genome_metadata.get("source_sha256"),
        "n_genes": len(genes),
        "samples_per_gene": samples_per_gene,
        "new_tokens": new_tokens,
        "allele_characters": allele_characters,
        "temperature": temperature,
        "seed": seed,
        "mean_distinct_alleles_per_gene": mean_distinct,
        "mean_empty_allele_frequency": mean_empty,
        "elapsed_seconds": time.monotonic() - started,
    }
    save_population(out_path, model_id=spec.model_id, population=population, metadata=metadata)
    return metadata
