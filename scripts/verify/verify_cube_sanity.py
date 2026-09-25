#!/usr/bin/env python
"""Sanity + active-tensor + fingerprint pass over every sealed distance cube in the local mirror.

usage: verify_cube_sanity.py <local_mirror_direct_dir> <assigned_manifest_dir> <fingerprints_out.tsv>
For each of the 9 receipted cohorts x 50 trees: files present; matrix symmetric, zero-diagonal,
finite, non-negative; n == number of trained leaves in the truth manifest == models.json; the
per-tensor cube has shape (n_layers, n, n); the saved matrix equals the mean over all layers;
the set of ACTIVE tensors (max distance > 1e-12) is exactly the LoRA scope (or all tensors for
full FT).  Writes one fingerprint row per cohort/tree and checks they are all distinct.
"""
import csv, hashlib, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

D, MAN, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
SCOPE = {  # cohort: (n_layers, expected active count, regex every active tensor must match)
  "flan_full": (282, 282, r".*"), "flan_k": (282, 36, r".*Attention\.k\.weight$"), "flan_qkv": (282, 108, r".*Attention\.[qkv]\.weight$"),
  "flan_qkvo": (282, 144, r".*Attention\.[qkvo]\.weight$"), "flan_correctedall": (216, 216, r".*(Attention\.[qkvo]|DenseReluDense\.(wi_0|wi_1|wo))\.weight$"),
  "flan_legacyall": (282, 168, r".*(Attention\.[qkvo]|DenseReluDense\.wo)\.weight$"),
  "llama_r8": (146, 48, r".*self_attn\.[qkv]_proj\.weight$"), "llama_r64": (146, 48, r".*self_attn\.[qkv]_proj\.weight$"),
  # Llama full FT ran in pure bf16: 24 RMSNorm weights (13 input_layernorm, 10 post_attention_layernorm, model.norm) never
  # move because their lr*grad updates fall below bf16 resolution near 1.0 -> exactly zero distance in every tree.
  # (9 other norm weights did move.)  For this cohort the rule is: every INACTIVE tensor must be a norm weight.
  "llama_full": (146, 122, "INACTIVE_MUST_MATCH:" + r".*(input_layernorm|post_attention_layernorm|^model\.norm)\.weight$"),
}
def leaves_of(tree):
    rows = [json.loads(l) for l in open(MAN / f"{tree}.manifest.jsonl") if l.strip()]
    kids = defaultdict(list)
    for r in rows: kids[r["parent_id"]].append(r["node_id"])
    return sorted(r["node_id"] for r in rows if not kids[r["node_id"]])
fails = []; fp = []; per_cohort = {}
for cohort, (n_layers, n_active, pat) in SCOPE.items():
    trees = sorted(p for p in (D / cohort / "analysis").iterdir() if p.is_dir())
    bad = Counter(); shas = []
    for t in trees:
        a = next(t.glob("*_leaf_analysis"))
        try:
            m = np.load(a / "distance_matrix_cosine.npy"); z = np.load(a / "direct_distance_layers.npz")["cosine"]
            models = json.load(open(a / "models.json")); layers = json.load(open(a / "layers.json"))
        except Exception as e:
            bad["missing files"] += 1; continue
        lv = leaves_of(t.name); n = len(lv)
        if models != lv: bad["models.json != truth leaves"] += 1
        if m.shape != (n, n): bad["matrix shape"] += 1
        if not (np.isfinite(m).all() and np.allclose(m, m.T) and np.allclose(np.diag(m), 0) and m.min() > -1e-12): bad["matrix not a distance matrix"] += 1
        if z.shape != (n_layers, n, n) or len(layers) != n_layers: bad[f"cube shape (got {z.shape}, {len(layers)} names)"] += 1
        if not np.allclose(m, z.mean(0)): bad["matrix != mean over layers"] += 1
        active = [layers[i] for i in range(z.shape[0]) if np.abs(z[i]).max() > 1e-12]
        inactive = [x for x in layers if x not in active]
        if pat.startswith("INACTIVE_MUST_MATCH:"):
            scope_ok = all(re.match(pat.split(":", 1)[1], x) for x in inactive)
        else:
            scope_ok = all(re.match(pat, x) for x in active)
        if len(active) != n_active or not scope_ok: bad[f"active tensors {len(active)} (expected {n_active}) or names outside scope"] += 1
        sha = hashlib.sha256(m.tobytes()).hexdigest(); shas.append(sha)
        off = m[~np.eye(n, dtype=bool)]
        fp.append((cohort, t.name, n, len(active), f"{off.mean():.6f}", sha))
    dup = sum(c - 1 for c in Counter(shas).values() if c > 1)
    per_cohort[cohort] = (len(trees), dict(bad), dup)
    if len(trees) != 50 or bad or dup: fails.append(cohort)
    print(f"{cohort:18s} trees={len(trees):2d}  problems={dict(bad) if bad else 'none'}  duplicate matrices within cohort={dup}")
with open(OUT, "w") as fh:
    w = csv.writer(fh, delimiter="\t"); w.writerow(["cohort", "tree", "n_leaves", "n_active_tensors", "mean_offdiag_cosine", "matrix_sha256"]); w.writerows(fp)
all_sha = Counter(s for *_, s in fp); cross = sum(c - 1 for c in all_sha.values() if c > 1)
print(f"\n{len(fp)} cube fingerprints written to {OUT}; identical matrices across ALL cohorts/trees: {cross}")
if cross: fails.append("cross-cohort identical matrices")
print("PASS verify_cube_sanity" if not fails else f"FAIL verify_cube_sanity: {fails}"); sys.exit(1 if fails else 0)
