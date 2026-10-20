#!/usr/bin/env python3
"""Run faithful PhyloLM on one fixed-2,000-step WeightTraits Llama-3.2-1B training tree.

This bridge is deliberately separate from the historical controlled-tree runner.
It consumes the exact fresh production run lists used by the paper audit:

* LoRA r8/r64: reconstruct each cumulative leaf from the pinned base revision and
  every root-to-leaf adapter, in order. Never treat the leaf adapter as a delta
  from the original base and never fall back to a historical CheckpointStore.
* Full fine-tuning: load the leaf's recorded ``model`` artifact directly.

The output uses the existing ELLMTrees scoring layout under uniquely named
``*_fresh_phylolm`` groups, and every run includes hashes and resolved artifact
paths in ``run_provenance.json``.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("PHYLOLM_ROOT") or HERE.parents[1])
sys.path.insert(0, str(HERE))
import phylolm_core as pl
from phylolm_distance import nj_newick

SUITES = {
    "r8": {
        "family": "llama32_1b_lora_qkv_r8_legacy_causal_2000",
        "method": "lora_chain",
        "group": "runs_weighttraits_llama32_1b_lora_qkv_r8_fixed2000_phylolm",
    },
    "r64": {
        "family": "llama32_1b_lora_qkv_r64_legacy_causal_2000",
        "method": "lora_chain",
        "group": "runs_weighttraits_llama32_1b_lora_qkv_r64_fixed2000_phylolm",
    },
    "full_ft": {
        "family": "llama32_1b_full_finetune_legacy_causal_2000",
        "method": "full_model",
        "group": "runs_weighttraits_llama32_1b_full_finetune_fixed2000_phylolm",
    },
}
METRICS = ("cosine", "l1", "l2", "correlation", "threshold")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty run list: {path}")
    ids = [r["node_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate node_id in {path}")
    return rows


def runlist_path(root: Path, suite: str, tree_index: int) -> Path:
    family = SUITES[suite]["family"]
    return (root / "examples/training/confirm_paper_numbers" /
            f"{family}_training_runlists/run_lists" /
            f"confirm_paper_tree_{tree_index:03d}.runs.jsonl")


def topology(rows: list[dict]) -> tuple[list[str], str]:
    by_id = {r["node_id"]: r for r in rows}
    children: dict[str, list[str]] = {"root": []}
    for r in rows:
        nid, parent = r["node_id"], r.get("parent_id") or "root"
        if parent != "root" and parent not in by_id:
            raise ValueError(f"node {nid} has missing parent {parent}")
        children.setdefault(parent, []).append(nid)
        children.setdefault(nid, [])

    leaves = [r["node_id"] for r in rows if not children[r["node_id"]]]

    def emit(nid: str) -> str:
        kids = children[nid]
        return f"({','.join(emit(k) for k in kids)}){nid}" if kids else nid

    truth = "(" + ",".join(emit(k) for k in children["root"]) + ")root;"
    return leaves, truth


def validate_tree(root: Path, suite: str, tree_index: int) -> dict:
    rp = runlist_path(root, suite, tree_index)
    if not rp.exists():
        raise FileNotFoundError(rp)
    rows = load_rows(rp)
    leaves, truth = topology(rows)
    if len(leaves) < 4:
        raise ValueError(f"tree {tree_index:03d} has only {len(leaves)} leaves")
    by_id = {r["node_id"]: r for r in rows}
    base_models = {r["job"]["base_model"] for r in rows}
    revisions = {r["job"].get("base_model_revision") for r in rows}
    if len(base_models) != 1 or len(revisions) != 1 or None in revisions:
        raise ValueError(f"non-unique base model/revision in {rp}")

    resolved: dict[str, list[Path]] = {}
    if SUITES[suite]["method"] == "lora_chain":
        for leaf in leaves:
            path = by_id[leaf]["job"].get("path") or by_id[leaf].get("path")
            if not path or path[0] != "root" or path[-1] != leaf:
                raise ValueError(f"bad root-to-leaf path for {leaf} in {rp}")
            chain = []
            for depth, nid in enumerate(path[1:]):
                row = by_id[nid]
                expected_init = (next(iter(base_models)) if depth == 0 else
                                 by_id[path[depth]]["expected_artifacts"]["merged"])
                if row.get("init_from") != expected_init:
                    raise ValueError(
                        f"chain discontinuity for {nid}: init_from={row.get('init_from')!r}, "
                        f"expected {expected_init!r}")
                rel = by_id[nid]["expected_artifacts"]["adapter"]
                artifact = root / rel
                for required in ("adapter_config.json", "adapter_model.safetensors"):
                    if not (artifact / required).is_file():
                        raise FileNotFoundError(artifact / required)
                chain.append(artifact)
            resolved[leaf] = chain
    else:
        for leaf in leaves:
            artifact = root / by_id[leaf]["expected_artifacts"]["model"]
            for required in ("config.json", "model.safetensors"):
                if not (artifact / required).is_file():
                    raise FileNotFoundError(artifact / required)
            resolved[leaf] = [artifact]

    return {
        "runlist": rp, "runlist_sha256": sha256(rp), "rows": rows,
        "leaves": leaves, "truth": truth, "resolved": resolved,
        "base_model": next(iter(base_models)), "base_revision": next(iter(revisions)),
    }


def preflight(root: Path, suite: str, out: Path) -> None:
    trees = []
    unique_artifacts: set[str] = set()
    for idx in range(1, 51):
        rec = validate_tree(root, suite, idx)
        for paths in rec["resolved"].values():
            unique_artifacts.update(str(p) for p in paths)
        trees.append({
            "tree_id": f"confirm_paper_tree_{idx:03d}",
            "runlist": str(rec["runlist"]),
            "runlist_sha256": rec["runlist_sha256"],
            "n_nodes": len(rec["rows"]), "n_leaves": len(rec["leaves"]),
            "truth_sha256": hashlib.sha256(rec["truth"].encode()).hexdigest(),
        })
    payload = {
        "status": "pass", "suite": suite, "family": SUITES[suite]["family"],
        "method": SUITES[suite]["method"], "n_trees": len(trees),
        "n_leaves": sum(t["n_leaves"] for t in trees),
        "n_unique_required_artifacts": len(unique_artifacts), "trees": trees,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "trees"}, indent=2))
    print(f"wrote {out}")


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", choices=sorted(SUITES), required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tree-index", type=int)
    mode.add_argument("--preflight-all", action="store_true")
    ap.add_argument("--weighttraits-root", type=Path, required=True)
    ap.add_argument("--config", type=Path,
                    default=REPO / "blackbox/configs/weighttraits_fresh_phylolm.yaml")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    wt_root = args.weighttraits_root.resolve()
    suite_meta = SUITES[args.suite]
    group = suite_meta["group"]
    if args.preflight_all:
        preflight(wt_root, args.suite,
                  REPO / "results/aggregate/weighttraits_fixed2000_phylolm" /
                  f"input_audit_{args.suite}.json")
        return
    if not 1 <= args.tree_index <= 50:
        ap.error("--tree-index must be in 1..50")

    cfg = load_yaml(args.config)
    G, N, T = (int(cfg.get(k, getattr(pl, f"DEFAULT_{k}"))) for k in ("G", "N", "T"))
    allele_chars = int(cfg.get("allele_chars", pl.DEFAULT_ALLELE_CHARS))
    temperature = float(cfg.get("temperature", 1.0))
    gene_seed = int(cfg.get("gene_seed", 0))
    sampling_seed = int(cfg.get("sampling_seed", 0))
    gene_pool = pl.resolve_gene_pool(cfg.get("gene_pool"), REPO)

    src = validate_tree(wt_root, args.suite, args.tree_index)
    run = f"run_{args.tree_index:03d}"
    tree_id = f"confirm_paper_tree_{args.tree_index:03d}"
    genome = pl.sample_genes(G=G, seed=gene_seed, path=gene_pool)
    provenance_fingerprint = hashlib.sha256(json.dumps({
        "suite": args.suite, "runlist_sha256": src["runlist_sha256"],
        "config_sha256": sha256(args.config), "gene_pool_sha256": sha256(gene_pool),
        "runner_sha256": sha256(Path(__file__)),
        "core_sha256": sha256(HERE / "phylolm_core.py"),
        "phylolm_commit": pl.PHYLOLM_COMMIT, "G": G, "N": N, "T": T,
        "allele_chars": allele_chars, "temperature": temperature,
        "gene_seed": gene_seed, "sampling_seed": sampling_seed,
        "batch_size": args.batch_size,
    }, sort_keys=True).encode()).hexdigest()

    data_dir = REPO / "blackbox/data" / group
    genome_path = data_dir / "genome.json"
    genome_path.parent.mkdir(parents=True, exist_ok=True)
    genome_record = {
        "G": G, "gene_seed": gene_seed, "gene_pool": str(gene_pool),
        "gene_pool_sha256": sha256(gene_pool), "phylolm_commit": pl.PHYLOLM_COMMIT,
        "genes": genome,
    }
    if genome_path.exists() and json.loads(genome_path.read_text()) != genome_record:
        raise RuntimeError(f"existing genome provenance differs: {genome_path}")
    if not genome_path.exists():
        tmp_genome = genome_path.with_name(f"{genome_path.name}.{os.getpid()}.tmp")
        tmp_genome.write_text(json.dumps(genome_record, ensure_ascii=False) + "\n")
        os.replace(tmp_genome, genome_path)

    pop_dir = data_dir / "populations" / run
    pop_dir.mkdir(parents=True, exist_ok=True)
    all_P, health = {}, {}
    for leaf in src["leaves"]:
        pf = pop_dir / f"{leaf}.json"
        if pf.exists() and not args.no_resume:
            rec = pl.load_population(pf)
            if rec.get("meta", {}).get("provenance_fingerprint") != provenance_fingerprint:
                raise RuntimeError(f"refusing stale/incompatible population: {pf}")
            all_P[leaf], health[leaf] = rec["P"], rec.get("meta", {})
            print(f"[skip] {tree_id}/{leaf}: matching population exists", flush=True)
            continue

        paths = src["resolved"][leaf]
        t0 = time.monotonic()
        if suite_meta["method"] == "lora_chain":
            print(f"[gen] {tree_id}/{leaf}: cumulative chain of {len(paths)} adapters", flush=True)
            alleles = pl.generate_alleles(
                model_path=src["base_model"], base_model=src["base_model"],
                model_revision=src["base_revision"], adapter_paths=[str(p) for p in paths],
                genome=genome, N=N, T=T, allele_chars=allele_chars,
                temperature=temperature, batch_size=args.batch_size,
                sampling_seed=sampling_seed,
            )
        else:
            print(f"[gen] {tree_id}/{leaf}: full model {paths[0]}", flush=True)
            alleles = pl.generate_alleles(
                model_path=str(paths[0]), genome=genome, N=N, T=T,
                allele_chars=allele_chars, temperature=temperature,
                batch_size=args.batch_size, sampling_seed=sampling_seed,
            )
        P = pl.compute_P(alleles)
        empty = sum(pl.match(d, "") for d in P) / len(P)
        distinct = sum(len(d) for d in P) / len(P)
        meta = {
            "tree_id": tree_id, "node": leaf, "method": suite_meta["method"],
            "resolved_artifacts": [str(p) for p in paths],
            "base_model": src["base_model"], "base_revision": src["base_revision"],
            "G": G, "N": N, "T": T, "temperature": temperature,
            "sampling_seed": sampling_seed, "batch_size": args.batch_size,
            "provenance_fingerprint": provenance_fingerprint,
            "mean_distinct_alleles_per_gene": round(distinct, 3),
            "mean_empty_allele_freq": round(float(empty), 4),
            "seconds": round(time.monotonic() - t0, 1),
        }
        pl.save_population(pf, leaf, P, meta=meta)
        all_P[leaf], health[leaf] = P, meta
        print(f"[done] {tree_id}/{leaf}: distinct/gene={distinct:.2f} empty={empty:.3f}", flush=True)
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    names = sorted(all_P)
    sim = pl.compute_sim_matrix(names, all_P)
    dist = -np.log(np.maximum(sim, 1e-3))
    np.fill_diagonal(dist, 0.0)
    newick = nj_newick(names, dist)

    wdir = REPO / "results" / group / run / "weight"
    wdir.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        (wdir / f"avg_reconstructed_{metric}.newick").write_text(newick + "\n")
        np.save(wdir / f"avg_distance_{metric}.npy", dist)
        np.save(wdir / f"full_distance_{metric}.npy", dist[None, :, :])
    np.save(wdir / "phylolm_sim.npy", sim)
    np.save(wdir / "phylolm_dist.npy", dist)
    (wdir / "distance_metadata.json").write_text(json.dumps({
        "layer_names": ["phylolm_nei_similarity"], "model_names": names,
        "nodes_mode": "leaves", "granularity": "phylolm_blackbox",
        "source": "codex/phylolm/run_weighttraits_fresh_tree.py",
        "suite": args.suite, "group": group, "run": run, "tree_id": tree_id,
        "phylolm_commit": pl.PHYLOLM_COMMIT, "provenance_fingerprint": provenance_fingerprint,
        "per_model_meta": health,
        "min_nei_offdiag": float(sim[~np.eye(len(names), dtype=bool)].min()),
    }, indent=2) + "\n")
    (wdir / "run_provenance.json").write_text(json.dumps({
        "suite": args.suite, "family": suite_meta["family"], "method": suite_meta["method"],
        "tree_id": tree_id, "weighttraits_root": str(wt_root),
        "runlist": str(src["runlist"]), "runlist_sha256": src["runlist_sha256"],
        "config": str(args.config), "config_sha256": sha256(args.config),
        "gene_pool": str(gene_pool), "gene_pool_sha256": sha256(gene_pool),
        "runner_sha256": sha256(Path(__file__)),
        "core_sha256": sha256(HERE / "phylolm_core.py"),
        "batch_size": args.batch_size, "sampling_seed": sampling_seed,
        "base_model": src["base_model"], "base_revision": src["base_revision"],
        "leaves": src["leaves"], "resolved_artifacts": {
            k: [str(p) for p in v] for k, v in src["resolved"].items()},
        "provenance_fingerprint": provenance_fingerprint,
    }, indent=2) + "\n")

    truth_dst = REPO / "outputs" / group / run / "tree.newick"
    truth_dst.parent.mkdir(parents=True, exist_ok=True)
    truth_dst.write_text(src["truth"] + "\n")
    off = sim[~np.eye(len(names), dtype=bool)]
    print(f"[tree] {tree_id}: {len(names)} leaves -> {wdir/'avg_reconstructed_cosine.newick'}")
    print(f"       Nei off-diag min/med/max={off.min():.4f}/{np.median(off):.4f}/{off.max():.4f}")


if __name__ == "__main__":
    main()
