#!/usr/bin/env python3
"""Convert sharded PyTorch ``.bin`` checkpoints to safetensors, tensor for tensor, and verify.

The WeightTraits readers stream safetensors one tensor at a time; sharded ``pytorch_model*.bin``
files would have to be unpickled whole. Each shard is loaded with ``weights_only=True``, written
as ``model-*.safetensors`` with a matching ``model.safetensors.index.json``, then re-read and
compared with ``torch.equal`` (same dtype, shape, and bits). The ``.bin`` files are left alone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def convert_dir(ckpt_dir: Path, overwrite: bool = False) -> int:
    index_path = ckpt_dir / "pytorch_model.bin.index.json"
    if index_path.exists():
        weight_map = dict(json.loads(index_path.read_text())["weight_map"])
        index = json.loads(index_path.read_text())
    elif (ckpt_dir / "pytorch_model.bin").exists():
        index = {"metadata": {}, "weight_map": {}}
        weight_map = None
    else:
        raise FileNotFoundError(f"no pytorch_model.bin[.index.json] in {ckpt_dir}")
    out_index = ckpt_dir / "model.safetensors.index.json"
    if out_index.exists() and not overwrite:
        print(f"skip existing {out_index}")
        return 0
    shards = sorted(set(weight_map.values())) if weight_map else ["pytorch_model.bin"]
    new_map, n_tensors = {}, 0
    for shard in shards:
        dst_name = shard.replace("pytorch_model", "model").removesuffix(".bin") + ".safetensors"
        state = torch.load(str(ckpt_dir / shard), map_location="cpu", weights_only=True)
        state = {k: v.contiguous() for k, v in state.items()}
        save_file(state, str(ckpt_dir / dst_name), metadata={"format": "pt"})
        back = load_file(str(ckpt_dir / dst_name))
        assert back.keys() == state.keys(), f"{shard}: tensor names changed"
        for key, value in state.items():
            assert back[key].dtype == value.dtype and torch.equal(back[key], value), f"{shard}:{key}"
            new_map[key] = dst_name
        n_tensors += len(state)
        print(f"{shard} -> {dst_name}: {len(state)} tensors identical", flush=True)
        del state, back
    index["weight_map"] = new_map
    out_index.write_text(json.dumps(index, indent=2) + "\n")
    return n_tensors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dirs", nargs="+", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for ckpt_dir in args.checkpoint_dirs:
        print(f"{ckpt_dir}: {convert_dir(ckpt_dir, args.overwrite)} tensors converted and verified")


if __name__ == "__main__":
    main()
