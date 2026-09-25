#!/usr/bin/env python
"""Known-answer test of the production distance-cube builder (`wt build-distance-cube`).

Three tiny synthetic "checkpoints" whose per-tensor cosine distances are known in closed
form: tensor w: A.B=0 -> distance 1; A.C = B.C = cos 45 deg -> 1-1/sqrt(2); tensor m: C is a
scalar multiple of A and B -> distance 0.   usage: verify_cube_known_answer.py <wt_root> <scratch>
"""
import json, math, os, subprocess, sys
import numpy as np, torch
from safetensors.torch import save_file
WT, K = sys.argv[1], sys.argv[2]; os.makedirs(K, exist_ok=True)
a = torch.tensor([1., 0., 0., 0.]); b = torch.tensor([0., 1., 0., 0.]); c = torch.tensor([1., 1., 0., 0.]) / math.sqrt(2); m = torch.ones(2, 3)
save_file({"w": a, "m": m}, f"{K}/A.safetensors"); save_file({"w": b, "m": m}, f"{K}/B.safetensors"); save_file({"w": c, "m": 2 * m}, f"{K}/C.safetensors")
r = subprocess.run([sys.executable, "-m", "weighttraits.cli", "build-distance-cube", "--checkpoint", f"{K}/A.safetensors", "--checkpoint", f"{K}/B.safetensors",
                    "--checkpoint", f"{K}/C.safetensors", "--metric", "cosine", "--out", f"{K}/out"], cwd=WT, env={**os.environ, "PYTHONPATH": "src"}, capture_output=True, text=True)
if r.returncode: print(r.stderr[-1500:]); sys.exit("FAIL verify_cube_known_answer: builder exited nonzero")
z = np.load(f"{K}/out/distance_cube.npz")["cosine"]; layers = json.load(open(f"{K}/out/layers.json")); models = json.load(open(f"{K}/out/models.json"))
d = 1 - 1 / math.sqrt(2)
expected = {"w": np.array([[0, 1, d], [1, 0, d], [d, d, 0]]), "m": np.zeros((3, 3))}
worst = max(np.abs(z[i] - expected[name]).max() for i, name in enumerate(layers))
print("models:", models, "layers:", layers); print("cube:\n", np.round(z, 6)); print(f"max |cube - closed form| = {worst:.2e}")
print("PASS verify_cube_known_answer" if worst < 1e-12 else "FAIL verify_cube_known_answer"); sys.exit(0 if worst < 1e-12 else 1)
