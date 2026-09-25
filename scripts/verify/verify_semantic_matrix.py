#!/usr/bin/env python
"""Recompute one tree's semantic distance matrix from the RAW generated text.

usage: verify_semantic_matrix.py <responses_dir(with <node>/responses.jsonl)> <semantic_matrix.json> <probe> <minilm_snapshot>
Embeds every response with the pinned all-MiniLM-L6-v2 snapshot (normalised), takes the cosine
distance between the two nodes' responses to the same (prompt_id, sample_id), and averages over
identities — exactly the paper's endpoint — then compares with the sealed semantic_matrix.json.
Also verifies each response's stored sha256 against the text, and prints three nodes' answers
to the same prompt so a human can see different models answering differently.
"""
import glob, hashlib, json, sys
import numpy as np
from sentence_transformers import SentenceTransformer
R, SM, PROBE, SNAP = sys.argv[1:5]
sm = json.load(open(SM)); ids = sm["matrix_receipt"]["ordered_node_ids"]; ref = np.array(sm["matrix"])
texts = {}; sha_ok = 0; sha_n = 0
for node in ids:
    for line in open(f"{R}/{node}/responses.jsonl"):
        r = json.loads(line); rr = r["response_row"]
        if rr["probe_id"] != PROBE: continue
        texts[(node, rr["prompt_id"], rr["sample_id"])] = r["text"]; sha_n += 1
        sha_ok += hashlib.sha256(r["text"].encode("utf-8")).hexdigest() == rr["response_sha256"]
identities = sorted({(p, s) for (_, p, s) in texts})
model = SentenceTransformer(SNAP, local_files_only=True)
E = {}
for node in ids:
    e = model.encode([texts[(node, p, s)] for p, s in identities], batch_size=128, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True).astype(np.float64)
    E[node] = e / np.linalg.norm(e, axis=1, keepdims=True)
n = len(ids); M = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        M[i, j] = M[j, i] = float(np.mean(1.0 - np.sum(E[ids[i]] * E[ids[j]], axis=1)))
diff = np.abs(M - ref).max(); empty = sum(1 for t in texts.values() if t.strip() == "")
print(f"probe={PROBE} nodes={n} identities={len(identities)} responses={sha_n} (stored sha256 matches text: {sha_ok}/{sha_n}; empty responses: {empty})")
print(f"max |recomputed - sealed semantic matrix| = {diff:.3e}")
p0 = identities[3]
print(f"\nprompt {p0[0]} sample {p0[1]} — three models' answers:")
for node in ids[:3]: print(f"  {node:4s}: {texts[(node,)+p0][:110]!r}")
ok = diff < 1e-5 and sha_ok == sha_n
print("PASS verify_semantic_matrix" if ok else "FAIL verify_semantic_matrix"); sys.exit(0 if ok else 1)
