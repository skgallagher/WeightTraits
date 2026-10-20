#!/usr/bin/env python3
"""PhyloLM core for the WeightTraits PhyloLM scripts (method: Yax, Oudeyer et al., "PhyloLM:
Inferring the Phylogeny of Large Language Models and Predicting their Performances in
Benchmarks", arXiv:2404.04671; https://github.com/Nicolas-Yax/PhyloLM at commit
8c70edf062a0adce2a3e6c8c79cd23a645fd0905).

This is our code: gene sampling, batched allele generation from local checkpoints, and
population (de)serialization. Allele frequencies and Nei similarity come from WeightTraits'
own implementation (weighttraits.behavior.phylolm); no upstream source is included (the
upstream repository is GPL-3.0). The upstream gene pool is not redistributed either: obtain the
``math`` genome (``data/kl_pop/math.zip`` at the pinned commit; 4,096 genes) and place it at
``scripts/phylolm/genes_math.json`` or point ``PHYLOLM_GENE_POOL`` at it. Its sha256 is checked.

PhyloLM contract kept here: prompt = raw gene text (no chat template), do_sample=True,
min_new_tokens = max_new_tokens = T, allele = the first ``allele_chars`` characters of the
newly generated text. Paper defaults: G=128 genes, N=32 probes per gene, T=4, four-character
alleles, temperature 1.0. Distance for trees: -log(max(similarity, 1e-3)), then neighbor joining.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from weighttraits.behavior import phylolm as _wt  # noqa: E402

PHYLOLM_COMMIT = "8c70edf062a0adce2a3e6c8c79cd23a645fd0905"

# PhyloLM paper defaults (the `math` dendrogram setting, Figs 3-4).
DEFAULT_G = 128          # number of genes sampled from the gene pool
DEFAULT_N = 32           # number of probes (completions) per gene
DEFAULT_T = 4            # new tokens generated per probe (min==max)
DEFAULT_ALLELE_CHARS = 4  # allele = first 4 chars of the completion

_GENES_MATH = Path(__file__).with_name("genes_math.json")

# The released ``math`` gene pool (not redistributed; see the module docstring).
GENE_POOL_SHA256 = "98d553eb4042cb084a7a17ce3a932ee147629698315e8cc2910dde6b05f84445"


def resolve_gene_pool(configured: Path | str | None = None, repo: Path | None = None) -> Path:
    """Return the gene pool: the configured path if it exists, else ``PHYLOLM_GENE_POOL``, else
    ``scripts/phylolm/genes_math.json``. A file named genes_math.json must match the released hash."""

    candidates = []
    if configured:
        configured = Path(configured)
        candidates.append(configured if configured.is_absolute() or repo is None else repo / configured)
    if os.environ.get("PHYLOLM_GENE_POOL"):
        candidates.append(Path(os.environ["PHYLOLM_GENE_POOL"]))
    candidates.append(_GENES_MATH)
    for path in candidates:
        if path.is_file():
            if path.name == "genes_math.json":
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != GENE_POOL_SHA256:
                    raise ValueError(f"{path} is not the released math gene pool ({digest})")
            return path
    raise FileNotFoundError("PhyloLM gene pool not found; see scripts/phylolm/README.md")




# ---------------------------------------------------------------------------
# Gene bank (the released `math` genome, extracted from
# data/kl_pop/math.zip in the pinned repo -> 4096 open-web-math snippets,
# each a 20-99 char prefix of a document, exactly as PhyloLM's gene_maker built
# them). The paper samples G of these uniformly at random.
# ---------------------------------------------------------------------------

def load_gene_pool(path: Path | str = _GENES_MATH) -> list[str]:
    return json.loads(Path(path).read_text())


def sample_genes(G: int = DEFAULT_G, seed: int = 0,
                 path: Path | str = _GENES_MATH) -> list[str]:
    """Deterministically sample G genes from the released `math` pool.

    Uses np.random.choice(replace=False) with a fixed seed so every model in the
    zoo is probed on the IDENTICAL gene set (required: the Nei similarity is only
    meaningful across models that saw the same genes)."""
    pool = load_gene_pool(path)
    if G > len(pool):
        raise ValueError(f"G={G} > pool size {len(pool)}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=G, replace=False)
    return [pool[int(i)] for i in idx]


# ---------------------------------------------------------------------------
# Allele generation (local-checkpoint, batched port of PhyloLM.generate_alleles)
# ---------------------------------------------------------------------------

def generate_alleles(model_path: str,
                     genome: list[str],
                     N: int = DEFAULT_N,
                     T: int = DEFAULT_T,
                     allele_chars: int = DEFAULT_ALLELE_CHARS,
                     temperature: float = 1.0,
                     batch_size: int = 64,
                     local_files_only: bool = True,
                     torch_dtype: str = "auto",
                     adapter_path: str | None = None,
                     adapter_paths: list[str] | None = None,
                     base_model: str | None = None,
                     model_revision: str | None = None,
                     sampling_seed: int | None = None):
    """Generate N alleles for each gene in `genome` from the model at `model_path`.

    Returns an (G, N) numpy array of string alleles. Mirrors PhyloLM's
    generate_alleles: prompt = raw gene text (no chat template), do_sample=True,
    min_new_tokens=max_new_tokens=T, allele = generated_text[len(gene):][:allele_chars].

    LoRA support (for the training-tree groups whose leaf checkpoints are PEFT
    adapters, not full models): pass `adapter_path` for one adapter or
    `adapter_paths` for an ordered root-to-leaf chain. Each adapter in a chain is
    applied to the current cumulative model and merged before the next adapter is
    loaded. This reconstructs the actual cumulative leaf state when intermediate
    merged directories were cleaned after training. `sampling_seed`, when set,
    is reset immediately before generation so repeated runs are reproducible and
    every compared model uses the same random stream.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed

    if adapter_path is not None and adapter_paths is not None:
        raise ValueError("pass adapter_path or adapter_paths, not both")
    chain = list(adapter_paths or ([] if adapter_path is None else [adapter_path]))

    # Resolve where model weights vs. tokenizer come from (full-FT: same path;
    # LoRA: base for weights, adapter dir usually carries the tokenizer too).
    if chain:
        from peft import PeftConfig, PeftModel
        pcfg = PeftConfig.from_pretrained(chain[0])
        base = base_model or pcfg.base_model_name_or_path
        tok_src = chain[-1]
        model = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=torch_dtype, device_map="auto",
            local_files_only=local_files_only, revision=model_revision,
        )
        for apath in chain:
            model = PeftModel.from_pretrained(model, apath)
            model = model.merge_and_unload()
    else:
        tok_src = model_path
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map="auto",
            local_files_only=local_files_only, revision=model_revision,
        )

    tok = AutoTokenizer.from_pretrained(tok_src, local_files_only=local_files_only)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # Left padding so the completion of each prompt is contiguous at the end.
    tok.padding_side = "left"
    gen = pipeline("text-generation", model=model, tokenizer=tok)

    if sampling_seed is not None:
        set_seed(sampling_seed)

    G = len(genome)
    # Duplicate each gene N times (PhyloLM: np.repeat(genome[:,None], N).flatten()).
    prompts = [g for g in genome for _ in range(N)]
    alleles_flat: list[str] = []
    gen_kwargs = dict(
        do_sample=True,
        min_new_tokens=T,
        max_new_tokens=T,
        temperature=temperature if temperature and temperature > 0 else None,
        return_full_text=True,   # need len(gene) offset like PhyloLM
        pad_token_id=tok.pad_token_id,
        batch_size=batch_size,
    )
    outputs = gen(prompts, **gen_kwargs)
    for prompt, out in zip(prompts, outputs):
        # pipeline returns a list[dict] per prompt when given a list of prompts
        text = out[0]["generated_text"] if isinstance(out, list) else out["generated_text"]
        allele = text[len(prompt):][:allele_chars]
        alleles_flat.append(allele)
    return np.reshape(np.array(alleles_flat, dtype=object), (G, N))


# ---------------------------------------------------------------------------
# Populations and Nei similarity (WeightTraits' implementation)
# ---------------------------------------------------------------------------

def compute_P(alleles):
    """Allele frequencies per gene, P(allele | gene), from a (G, N) array of alleles."""
    return _wt.compute_population(np.asarray(alleles).tolist())


def match(population_gene, allele):
    """Frequency of ``allele`` in one gene's population (0 if absent)."""
    return population_gene.get(allele, 0)


def nei_similarity(P1, P2):
    return _wt.nei_similarity(P1, P2)


def compute_sim_matrix(model_ids, all_P):
    """Nei similarity between every pair of ``model_ids`` (diagonal included)."""
    n = len(model_ids)
    sim = np.zeros((n, n))
    for i, m1 in enumerate(model_ids):
        for j, m2 in enumerate(model_ids):
            sim[i][j] = nei_similarity(all_P[m1], all_P[m2])
    return sim


# ---------------------------------------------------------------------------
# Population (de)serialization: P is a list[dict[str->float]] per model.
# ---------------------------------------------------------------------------

def save_population(path: Path | str, model_id: str, P, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "model_id": model_id,
        "phylolm_commit": PHYLOLM_COMMIT,
        "meta": meta or {},
        "P": P,
    }, ensure_ascii=False))


def load_population(path: Path | str) -> dict:
    return json.loads(Path(path).read_text())
