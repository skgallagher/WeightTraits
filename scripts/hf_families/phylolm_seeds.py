#!/usr/bin/env python3
"""How much App. D's PhyloLM tree-likeness A varies with the sampling seed.

The original App. D PhyloLM run set no sampling seed. The Wright redo used a single seed
(20260803), and it moved Mistral's A from 0.68 to 0.81. This script repeats PhyloLM on one family
for several seeds and scores each seed against the documented tree. It reads checkpoints already on
local disk, as left by ``download_families.py`` and ``convert_bin_to_safetensors.py``.

For each seed it:

1. writes the family config with the seed appended to ``name`` and ``sampling_seed`` set;
2. runs ``scripts/phylolm/collect_phylolm.py`` over every model (existing populations are kept);
3. writes distances and the NJ tree with ``scripts/phylolm/phylolm_distance.py``;
4. scores them with ``weighttraits.paper.hf_families.score_family``.

Populations land under ``$PHYLOLM_ROOT/blackbox/data/<name>_seed<seed>/populations``. Distances
go to ``--out/seed_<seed>``. ``--out/seeds.json`` gives each seed's clades and A, with summary
statistics.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PHYLOLM = HERE.parent / "phylolm"
sys.path.insert(0, str(REPO / "src"))
from weighttraits.paper.hf_families import MethodInputs, score_family  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=["mistral_7b", "llama2_7b"])
    parser.add_argument("--config", type=Path, required=True, help="*_zoo_phylolm_wright.yaml")
    parser.add_argument("--registry", type=Path, required=True, help="models_*_zoo.yaml")
    parser.add_argument("--models-dir", type=Path, required=True, help="local <org>__<repo> dirs")
    parser.add_argument("--seeds", required=True, help="comma-separated sampling seeds")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    registry = yaml.safe_load(args.registry.read_text())
    for model in registry["models"]:
        model["model_path"] = str(args.models_dir / model["id"])
        if not Path(model["model_path"]).is_dir():
            raise FileNotFoundError(model["model_path"])
    (args.out / "configs").mkdir(parents=True, exist_ok=True)
    local_registry = args.out / "registry.yaml"
    local_registry.write_text(yaml.safe_dump(registry, sort_keys=False))
    root = Path(os.environ.get("PHYLOLM_ROOT") or REPO)
    truth = REPO / "examples/hf_families" / args.family / "tree.newick"

    results = []
    for seed in (int(s) for s in args.seeds.split(",")):
        seed_cfg = {**cfg, "name": f"{cfg['name']}_seed{seed}", "sampling_seed": seed}
        cfg_path = args.out / "configs" / f"{seed_cfg['name']}.yaml"
        cfg_path.write_text(yaml.safe_dump(seed_cfg, sort_keys=False))
        subprocess.run([sys.executable, str(PHYLOLM / "collect_phylolm.py"), "--config",
                        str(cfg_path), "--registry", str(local_registry)], check=True)
        dist = args.out / f"seed_{seed}"
        subprocess.run([sys.executable, str(PHYLOLM / "phylolm_distance.py"), "--populations",
                        str(root / "blackbox/data" / seed_cfg["name"] / "populations"),
                        "--out", str(dist)], check=True)
        scored = score_family(truth, {"phylolm": MethodInputs(
            dist / "phylolm_nj.newick", dist / "phylolm_dist.npy", dist / "distance_metadata.json"
        )})["methods"]["phylolm"]
        results.append({"seed": seed, **{k: scored[k] for k in (
            "clades_estimated", "n_documented_clades", "tree_likeness_all_quartets")}})
        print(f"seed {seed}: {scored['clades_estimated']}/{scored['n_documented_clades']} clades, "
              f"A={scored['tree_likeness_all_quartets']:.3f}", flush=True)

    a = np.array([r["tree_likeness_all_quartets"] for r in results])
    summary = {
        "family": args.family,
        "config": str(args.config),
        "seeds": results,
        "A_mean": float(a.mean()),
        "A_sd": float(a.std(ddof=1)) if len(a) > 1 else None,
        "A_min": float(a.min()),
        "A_max": float(a.max()),
        "n_seeds_all_clades": sum(r["clades_estimated"] == r["n_documented_clades"]
                                  for r in results),
    }
    (args.out / "seeds.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"A over {len(a)} seeds: mean {a.mean():.3f}, range {a.min():.3f}-{a.max():.3f}")


if __name__ == "__main__":
    main()
