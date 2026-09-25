#!/usr/bin/env python
"""Summarise the per-node training audit TSVs produced on Wright by wright_audit_training.py.

usage: summarize_training_audit.py <dir_with_audit_*.tsv> [expected_labels...]
For every cohort: 641 nodes / 50 trees, all completed at step 2000 with no stop reason,
child init_from == its manifest parent, pinned base revision, the final (last-attempt) run has
exactly 9 eval points (every 250 steps) and is clean, <=10,000/1,000 rows, finite losses that
vary, weight file present.  Weight fingerprints must be distinct EXCEPT between nodes with the
identical training recipe (same dataset chain from the base), which deterministic training may
legitimately make bit-identical.  Across cohorts, a fingerprint may only be shared between a
cohort and its matched no-translation twin, and only for nodes whose recipe is unchanged.
"""
from __future__ import annotations
import csv, math, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

D = Path(sys.argv[1]); want = sys.argv[2:]
REV = {"flan": "7bcac572ce56", "llama": "4e20de362430"}
fails = []
by_key = defaultdict(dict)   # (tree,node) -> label -> sha
print(f"{'cohort':18s} {'nodes':>5s} {'trees':>5s} {'done@2000':>9s} {'noStop':>6s} {'parentOK':>8s} {'rev':>4s} {'eval9':>5s} {'clean':>5s} {'restarted':>9s} {'rows<=cap':>9s} {'distinct':>8s} {'dup groups (all same recipe?)':>29s} {'train_loss med [min,max]':>26s} {'weight GB':>9s}")
files = sorted(f for f in D.glob("audit_*.tsv") if not f.name.startswith("audit_behavior"))
seen = []
for f in files:
    rows = list(csv.DictReader(open(f), delimiter="\t")); label = rows[0]["label"]; seen.append(label)
    n = len(rows); trees = len({r["tree"] for r in rows})
    done = sum(1 for r in rows if r["status"] == "completed" and r["step"] == "2000")
    nostop = sum(1 for r in rows if r["stop_reasons"] == "-")
    pok = sum(1 for r in rows if r["init_from_ok"] in ("ok", "root"))
    rev = sum(1 for r in rows if r["base_revision"] == REV["flan" if label.startswith("flan") else "llama"])
    ev9 = sum(1 for r in rows if r["n_eval"] == "9")
    clean = sum(1 for r in rows if r["last_run_clean"] == "yes"); restarted = sum(1 for r in rows if r["n_starts"] not in ("1", "?"))
    groups = defaultdict(list)
    for r in rows: groups[r["slice_sha256"]].append(r)
    dup_groups = [g for g in groups.values() if len(g) > 1]
    dup_same_recipe = sum(1 for g in dup_groups if len({r["chain"] for r in g}) == 1)
    cap = sum(1 for r in rows if r["n_train"].isdigit() and int(r["n_train"]) <= 10000 and int(r["n_eval_rows"]) <= 1000)
    shas = [r["slice_sha256"] for r in rows]; distinct = len(set(shas)); nofile = sum(1 for s in shas if s == "NO_WEIGHT_FILE")
    losses = [float(r["train_loss"]) for r in rows if r["train_loss"] not in ("", "None", "?")]
    finite = all(math.isfinite(x) for x in losses)
    gb = sum(int(r["size_bytes"]) for r in rows if r["size_bytes"].lstrip("-").isdigit() and int(r["size_bytes"]) > 0) / 1e9
    for r in rows: by_key[(r["tree"], r["node"])][label] = (r["slice_sha256"], r["chain"])
    ok = (n == 641 and trees == 50 and done == 641 and nostop == 641 and pok == 641 and rev == 641 and ev9 == 641 and clean == 641
          and cap == 641 and dup_same_recipe == len(dup_groups) and nofile == 0 and finite and len(losses) == 641 and statistics.pstdev(losses) > 0)
    if not ok: fails.append(label)
    print(f"{label:18s} {n:>5d} {trees:>5d} {done:>9d} {nostop:>6d} {pok:>8d} {rev:>4d} {ev9:>5d} {clean:>5d} {restarted:>9d} {cap:>9d} {distinct:>8d} "
          f"{dup_same_recipe:>3d}/{len(dup_groups):<3d} {'yes' if dup_same_recipe == len(dup_groups) else 'NO ':>21s} "
          f"{statistics.median(losses):>8.3f} [{min(losses):.3f},{max(losses):.3f}] {gb:>8.1f} {'' if ok else '  <-- PROBLEM'}")
missing = [w for w in want if w not in seen]
if missing: fails.append("missing audits: " + ",".join(missing))
# cross-cohort: a fingerprint may be shared only between a cohort and its matched no-translation twin
# (same base model, same recipe for nodes whose whole chain avoids translation); anything else is a copy.
twins = {("flan_full", "flan_notrans"), ("llama_full", "llama_notrans")}
allowed = 0; forbidden = []
for k, m in by_key.items():
    inv = defaultdict(list)
    for lab, (sha, chain) in m.items(): inv[sha].append(lab)
    for sha, labs in inv.items():
        if len(labs) > 1:
            if tuple(sorted(labs)) in twins: allowed += 1
            else: forbidden.append((k, labs))
print(f"\n(tree,node) keys audited across cohorts: {len(by_key)}; fingerprints shared between a cohort and its matched"
      f" no-translation twin (unchanged recipe, expected): {allowed}; shared between any OTHER cohort pair: {len(forbidden)}")
if forbidden: fails.append(f"cross-cohort duplicate fingerprints outside matched twins: {forbidden[:3]}")
# a few example fingerprints so the difference is visible to a human
ex = [k for k in sorted(by_key) if len(by_key[k]) >= 3][:3]
for k in ex:
    print("  ", k, {lab: v[0][:10] for lab, v in sorted(by_key[k].items())})
print("\nPASS summarize_training_audit" if not fails else f"\nFAIL summarize_training_audit: {fails}"); sys.exit(1 if fails else 0)
