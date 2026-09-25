#!/usr/bin/env python3
"""Independent recomputation of a sealed distance cube — run ON WRIGHT, read-only.

usage: wright_recompute_distances.py <analysis_dir> <stage_root> <cohort_output_root> <assigned_manifest>
                                     <mode: model|merged|adapter_chain> <base_snapshot_dir|-> [n_keys|all] [--bf16]

Loads the SAME leaf checkpoints the pipeline used, with only `safetensors` + numpy
(no WeightTraits code), recomputes per-tensor cosine distance 1 - a.b/(|a||b|) between
every pair of leaves, and compares with the sealed cube direct_distance_layers.npz.
  model         : read <node>/model/model.safetensors directly (full fine-tuning)
  merged        : re-materialise W = W_base + sum_{edges on root->node path} (alpha/r) * B @ A
                  from the retained adapters (the merged copies were pruned on 2026-09-15);
                  --bf16 rounds to bfloat16 after every merge step, as the bf16 Llama merge did
  adapter_chain : compare cumulative displacements Delta = sum (alpha/r) * B @ A (no base)
Prints per-key max |recomputed - sealed| and an overall verdict.
"""
import json, sys, os
import numpy as np
import torch
from pathlib import Path
from safetensors import safe_open

def get(f, key, dtype=np.float64):
    """Read a tensor through torch (numpy cannot decode bfloat16) and return it as a numpy array."""
    t = f.get_tensor(key)
    return t.to(torch.float64).numpy().astype(dtype) if dtype == np.float64 else t.to(torch.float32).numpy()

analysis_dir, stage, out_root, manifest, mode, base_dir = sys.argv[1:7]
n_keys = sys.argv[7] if len(sys.argv) > 7 and not sys.argv[7].startswith("--") else "all"
bf16 = "--bf16" in sys.argv
analysis_dir = Path(analysis_dir); stage = Path(stage)
models = json.load(open(analysis_dir / "models.json")); layers = json.load(open(analysis_dir / "layers.json"))
cube = np.load(analysis_dir / "direct_distance_layers.npz")["cosine"]
assert cube.shape == (len(layers), len(models), len(models)), cube.shape

rows = [json.loads(l) for l in open(stage / manifest) if l.strip()]
path_of = {r["node_id"]: [p for p in r["path"] if p != "root"] for r in rows}
tree = Path(manifest).name.split(".")[0]
node_dir = lambda n: stage / out_root / tree / n

def bf16_round(x):
    """Round a float32 array to bfloat16 precision exactly as torch does."""
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(torch.bfloat16).to(torch.float32).numpy()

adapters = {}
def adapter(n):
    if n not in adapters:
        d = node_dir(n) / "adapter"; cfg = json.load(open(d / "adapter_config.json"))
        adapters[n] = (safe_open(str(d / "adapter_model.safetensors"), "pt"), cfg["lora_alpha"] / cfg["r"])
    return adapters[n]

def lora_delta(n, key, dtype):
    f, s = adapter(n); stem = key[:-len(".weight")]
    ka, kb = f"base_model.model.{stem}.lora_A.weight", f"base_model.model.{stem}.lora_B.weight"
    if ka not in f.keys(): return None
    A = get(f, ka, dtype); B = get(f, kb, dtype)
    return s * (B @ A)

base = safe_open(str(Path(base_dir) / "model.safetensors"), "pt") if base_dir != "-" else None
files = {}
def model_file(n):
    if n not in files: files[n] = safe_open(str(node_dir(n) / "model" / "model.safetensors"), "pt")
    return files[n]

def tensor(n, key):
    if mode == "model":
        return get(model_file(n), key)
    if mode == "merged":
        w = get(base, key, np.float32)
        for e in path_of[n]:
            d = lora_delta(e, key, np.float32)
            if d is None: break            # not a LoRA target: identical to base for every model
            w = w + d
            if bf16: w = bf16_round(w)
        return w.astype(np.float64)
    if mode == "adapter_chain":
        acc = None
        for e in path_of[n]:
            d = lora_delta(e, key, np.float64)
            acc = d if acc is None else acc + d
        return acc

def cosine_matrix(vs):
    V = np.vstack([v.reshape(1, -1) for v in vs]); G = V @ V.T
    norms = np.sqrt(np.maximum(np.diag(G), 0.0)); den = np.outer(norms, norms)
    with np.errstate(divide="ignore", invalid="ignore"):
        C = 1.0 - np.where(den > 0, G / np.where(den > 0, den, 1.0), 0.0)
    C = np.nan_to_num(C); C = (C + C.T) / 2; np.fill_diagonal(C, 0.0); return C

idx = list(range(len(layers))) if n_keys == "all" else [int(round(i)) for i in np.linspace(0, len(layers) - 1, int(n_keys))]
print(f"tree={tree} mode={mode} bf16={bf16} leaves={models} keys_checked={len(idx)}/{len(layers)}")
worst = 0.0; worst_key = None; n_nonzero = 0; per = []
for i in idx:
    key = layers[i]
    vs = [tensor(m, key) for m in models]
    if any(v is None for v in vs):
        print(f"  {key}: not a LoRA target in adapter_chain mode (skipped)"); continue
    C = cosine_matrix(vs); diff = np.abs(C - cube[i]).max()
    if cube[i].max() > 1e-12: n_nonzero += 1
    per.append((diff, key, cube[i].max()))
    if diff > worst: worst, worst_key = diff, key
per.sort(reverse=True)
print(f"keys with nonzero sealed distance: {n_nonzero}; largest recompute-vs-sealed differences:")
for diff, key, mx in per[:5]:
    print(f"  {diff:.3e}  (sealed max {mx:.3e})  {key}")
print(f"OVERALL max |recomputed - sealed| = {worst:.3e} at {worst_key}")
