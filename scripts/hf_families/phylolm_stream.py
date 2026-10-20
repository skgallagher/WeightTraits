#!/usr/bin/env python3
"""App. D PhyloLM on one GPU, streaming each family model through temporary storage.

For each model in the config's ``model_ids``, the script:

1. downloads the pinned revision (``download_families.FAMILIES``) into a temporary directory
   under ``--scratch`` and verifies the files;
2. converts sharded ``.bin`` weights to safetensors (transformers 5.8 will not ``torch.load``
   them under torch 2.5.1);
3. runs ``scripts/phylolm/collect_phylolm.py`` on that one model, with the registry's
   ``torch_dtype``;
4. deletes the temporary copy.

It then writes distances and the NJ tree with ``scripts/phylolm/phylolm_distance.py``.
Populations land under ``$PHYLOLM_ROOT/blackbox/data/<config name>/populations``, and existing
populations are kept (resume).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PHYLOLM = HERE.parent / "phylolm"
sys.path.insert(0, str(HERE))
from convert_bin_to_safetensors import convert_dir  # noqa: E402
from download_families import FAMILIES, download_repo  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=sorted(FAMILIES))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="PhyloLM distance/tree output dir")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    cfg = yaml.safe_load(args.config.read_text())
    registry = {m["id"]: m for m in yaml.safe_load(args.registry.read_text())["models"]}
    root = Path(os.environ.get("PHYLOLM_ROOT") or HERE.parents[1])
    pop_dir = root / "blackbox" / "data" / cfg["name"] / "populations"
    api = HfApi()
    for index, model_id in enumerate(cfg["model_ids"]):
        if (pop_dir / f"{model_id}.json").exists():
            print(f"[skip] {model_id}: population exists", flush=True)
            continue
        repo = registry[model_id]["hf_model_id"]
        tmp = Path(tempfile.mkdtemp(prefix="phylolm_model.", dir=args.scratch))
        try:
            record = download_repo(api, repo, FAMILIES[args.family][repo], tmp)
            if record["weight_format"] == "bin":
                convert_dir(tmp)
            one = tmp / "registry.yaml"
            one.write_text(yaml.safe_dump({"models": [{**registry[model_id], "model_path": str(tmp)}]}))
            subprocess.run(
                [sys.executable, str(PHYLOLM / "collect_phylolm.py"), "--config", str(args.config),
                 "--registry", str(one), "--model-index", str(index)],
                check=True,
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    subprocess.run(
        [sys.executable, str(PHYLOLM / "phylolm_distance.py"), "--populations", str(pop_dir),
         "--out", str(args.out)],
        check=True,
    )


if __name__ == "__main__":
    main()
