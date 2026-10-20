#!/usr/bin/env python3
"""Per-tensor weight-distance cube for one documented Hugging Face family (App. D).

The models of a family were released independently, so they need not hold exactly the same
tensors. The cube uses every tensor name the models share, with equal shapes, and skips tensors
with more than ``--max-elements`` entries (50,000,000, as in the paper). That skips the token
embedding and the output head of the 7B models.

Distances come from ``weighttraits.analysis.direct``. The paper's cube stored each per-tensor
matrix as float32 before averaging, so the default rounds them the same way;
``--float64`` keeps full precision. Layers are ordered by the first model's shard, then by name,
as in the paper.

Outputs, under ``--out``:

* ``full_distance_<metric>.npy``, shape (layers, n, n);
* ``avg_distance_<metric>.npy``, the mean over layers;
* ``distance_metadata.json``.

Models are ordered by name (``org__repo``).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from weighttraits.analysis.direct import _direct_distance_layers  # noqa: E402
from weighttraits.distances.readers import reader_from_path  # noqa: E402

METRICS = ("cosine", "l2", "correlation")


def _shard_of(reader, key: str) -> str:
    """The shard holding ``key`` (empty for single-file checkpoints)."""
    index = getattr(reader, "_index", None)
    return index().get(key, "") if callable(index) else ""


def build_cube(paths: dict[str, Path], out: Path, *, max_elements: int = 50_000_000,
               float32: bool = True, chunk_size: int = 4_000_000) -> dict:
    names = sorted(paths)
    readers = [reader_from_path(paths[name], model_id=name) for name in names]
    common = set.intersection(*(set(reader.keys()) for reader in readers))
    keys, skipped_shape, skipped_size = [], 0, 0
    for key in sorted(common, key=lambda k: (_shard_of(readers[0], k), k)):
        infos = [reader.tensor_info(key) for reader in readers]
        if len({info.shape for info in infos}) != 1:
            skipped_shape += 1
        elif max_elements and infos[0].numel > max_elements:
            skipped_size += 1
        else:
            keys.append(key)
    distances, layers, models, audit = _direct_distance_layers(
        readers, metrics=list(METRICS), chunk_size=chunk_size, eps=1e-3, artifact="model", keys=keys
    )
    assert models == names and layers == keys
    out.mkdir(parents=True, exist_ok=True)
    for metric, cube in distances.items():
        cube = cube.astype(np.float32).astype(np.float64) if float32 else cube
        np.save(out / f"full_distance_{metric}.npy", cube)
        np.save(out / f"avg_distance_{metric}.npy", cube.mean(axis=0))
    meta = {
        "layer_names": layers,
        "model_names": names,
        "nodes_mode": "leaves",
        "granularity": "weight_common",
        "source": "scripts/hf_families/build_family_cube.py",
        "checkpoint_paths": [str(paths[name]) for name in names],
        "n_key_intersection": len(common),
        "n_skipped_shape_mismatch": skipped_shape,
        "n_skipped_size": skipped_size,
        "max_elements": max_elements,
        "per_layer_storage": "float32" if float32 else "float64",
    }
    (out / "distance_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="download_manifest.json")
    parser.add_argument("--family", required=True, choices=["mistral_7b", "llama2_7b"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-elements", type=int, default=50_000_000)
    parser.add_argument("--float64", action="store_true", help="keep per-layer matrices in float64")
    args = parser.parse_args()
    family = json.loads(args.manifest.read_text())["families"][args.family]
    paths = {repo.replace("/", "__"): Path(record["path"]) for repo, record in family.items()}
    meta = build_cube(paths, args.out, max_elements=args.max_elements, float32=not args.float64)
    print(f"{args.family}: {len(meta['layer_names'])} layers of {meta['n_key_intersection']} shared "
          f"tensors ({meta['n_skipped_size']} over the size limit, "
          f"{meta['n_skipped_shape_mismatch']} with differing shapes) -> {args.out}")


if __name__ == "__main__":
    main()
