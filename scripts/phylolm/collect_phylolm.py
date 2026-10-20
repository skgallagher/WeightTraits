#!/usr/bin/env python3
"""Collect PhyloLM allele populations for one (or all) Mistral-zoo checkpoint(s).

Faithful PhyloLM baseline (see phylolm_core.py for the vendored algorithm and the
pinned upstream commit). For each model this samples the SAME G genes from the
released `math` pool, generates N probes of T tokens per gene as RAW completions
(no chat template -- PhyloLM's design), extracts the 4-char allele, computes the
per-gene allele population P, and writes it to

    blackbox/data/<name>/populations/<model_id>.json

Designed for a Slurm job array (one model per task via --model-index), mirroring
codex/submit_mistral_blackbox_tasks.sh. The registry (model paths / ids) is
reused unchanged from blackbox/configs/models_mistral_zoo.yaml.

Usage
-----
    python codex/phylolm/collect_phylolm.py \
        --config blackbox/configs/mistral_zoo_phylolm.yaml \
        --registry blackbox/configs/models_mistral_zoo.yaml \
        --model-index $SLURM_ARRAY_TASK_ID
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
import phylolm_core as pl

REPO = Path(os.environ.get("PHYLOLM_ROOT") or Path(__file__).resolve().parents[2])


def load_yaml(p: Path) -> dict:
    return yaml.safe_load(Path(p).read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--registry", required=True, type=Path)
    ap.add_argument("--model-index", type=int, default=None,
                    help="0-based index into config model_ids (for Slurm arrays). Default: all.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--no-resume", action="store_true",
                    help="Re-generate even if a population file already exists.")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    reg = {m["id"]: m for m in load_yaml(args.registry)["models"]}

    name = cfg["name"]
    G = int(cfg.get("G", pl.DEFAULT_G))
    N = int(cfg.get("N", pl.DEFAULT_N))
    T = int(cfg.get("T", pl.DEFAULT_T))
    allele_chars = int(cfg.get("allele_chars", pl.DEFAULT_ALLELE_CHARS))
    temperature = float(cfg.get("temperature", 1.0))
    seed = int(cfg.get("gene_seed", 0))
    gene_pool = pl.resolve_gene_pool(cfg.get("gene_pool"), REPO)

    data_dir = REPO / "blackbox" / "data" / name
    pop_dir = data_dir / "populations"
    pop_dir.mkdir(parents=True, exist_ok=True)

    # Sample the shared genome once and persist it (all models must use the same genes).
    genome = pl.sample_genes(G=G, seed=seed, path=gene_pool)
    genome_path = data_dir / "genome.json"
    if not genome_path.exists():
        genome_path.write_text(json.dumps(
            {"G": G, "seed": seed, "gene_pool": str(gene_pool),
             "phylolm_commit": pl.PHYLOLM_COMMIT, "genes": genome},
            ensure_ascii=False))

    model_ids = cfg["model_ids"]
    if args.model_index is not None:
        if not (0 <= args.model_index < len(model_ids)):
            sys.exit(f"--model-index {args.model_index} out of range 0..{len(model_ids)-1}")
        model_ids = [model_ids[args.model_index]]

    for mid in model_ids:
        if mid not in reg:
            sys.exit(f"model '{mid}' not in registry {args.registry}")
        out_path = pop_dir / f"{mid}.json"
        if out_path.exists() and not args.no_resume:
            print(f"[skip] {mid}: population exists ({out_path})", flush=True)
            continue
        model_path = reg[mid].get("model_path") or reg[mid]["hf_model_id"]
        print(f"[gen] {mid}  G={G} N={N} T={T} temp={temperature}  path={model_path}", flush=True)
        t0 = time.monotonic()
        alleles = pl.generate_alleles(
            model_path=model_path, genome=genome, N=N, T=T,
            allele_chars=allele_chars, temperature=temperature,
            batch_size=args.batch_size,
        )
        P = pl.compute_P(alleles)
        # Quick health signal: mean #distinct alleles per gene, empty-allele rate.
        n_distinct = sum(len(d) for d in P) / len(P)
        empty_rate = sum(pl.match(d, "") for d in P) / len(P)
        meta = {
            "G": G, "N": N, "T": T, "allele_chars": allele_chars,
            "temperature": temperature, "gene_seed": seed,
            "model_path": model_path,
            "mean_distinct_alleles_per_gene": round(n_distinct, 3),
            "mean_empty_allele_freq": round(float(empty_rate), 4),
            "seconds": round(time.monotonic() - t0, 1),
        }
        pl.save_population(out_path, mid, P, meta=meta)
        print(f"[done] {mid}: distinct/gene={n_distinct:.2f} empty_freq={empty_rate:.3f} "
              f"({meta['seconds']}s) -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
