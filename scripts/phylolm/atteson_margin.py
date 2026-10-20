#!/usr/bin/env python3
"""Atteson-margin: a governing variable for topology recovery (2026-06-18).

Coherence (lambda_1/n) is a one-spike statistic and saturates inside the
recoverable regime (the star counterexample, notes/what_governs_recovery.md).
This computes the classical NJ-robustness quantity instead, in distance space:

  For each run, fit additive edge lengths L (NNLS) to the layer-averaged
  pairwise weight-distance matrix D GIVEN the known truth topology, then

      margin_s = L_s / (2 * ||D - D_fit||_inf)      [Atteson 1999]

  NJ provably recovers the tree when margin_s > 1 for the SHORTEST edge. Atteson's
  bound is over ALL edges (internal AND pendant), so bottleneck = min over all
  fitted edges. (Restricting the min to internal edges is optimistic: a pendant
  edge is the shortest in ~96% of our groups, so an internal-only bottleneck can
  sit above 1 while the true Atteson bottleneck is below it -- see 2026-07-07.)
  The ratio is scale-free, so no within-run z-scoring.
    - bottleneck   = min over ALL edges          (Atteson-faithful certificate)
    - mean_margin  = mean over INTERNAL edges    (topology-edge diagnostic; the
                     mean over all edges is dominated by a few huge pendant edges,
                     e.g. full-FT groups, and is uninformative)

Result (current data, post-2026-06-25 cumulative rescore + 2026-07-07 balanced-split
dedup fix; >=6-leaf runs, 24 groups): bottleneck (all-edge) Spearman with clade
recovery +0.46; mean_margin (internal) ~+0.38. The dedup fix raised every group's
bottleneck ~10-25% (removed phantom zero-length duplicates of balanced splits) and
moved rho +0.50 -> +0.46. NB the older "+0.92" quoted here predated the 2026-06-25
cumulative fix that rewrote clade_recovery for the Llama LoRA groups -- don't trust it.

Usage:  python scripts/atteson_margin.py            # table + correlations
        FIG=1 python scripts/atteson_margin.py       # also write CSV + PNG
Env:    ELLM_ROOT (repo root, default: parent of scripts/),  MIN_LEAVES (default 6)
"""
import json, glob, os, re
import numpy as np
from scipy.optimize import nnls

ROOT = os.environ.get("ELLM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MIN_LEAVES = int(os.environ.get("MIN_LEAVES", "6"))
COH_CSV = os.path.join(ROOT, "results/aggregate/recovery_rescore/shared_direction_coherence.csv")


def parse_newick(s):
    s = s.strip().rstrip(";"); pos = 0
    def clade():
        nonlocal pos
        ch = []
        if s[pos] == "(":
            pos += 1
            while True:
                ch.append(clade())
                if s[pos] == ",": pos += 1; continue
                if s[pos] == ")": pos += 1; break
        m = re.match(r"[^,()\s:]*", s[pos:]); label = m.group(0); pos += len(label)
        if pos < len(s) and s[pos] == ":":
            pos += 1; pos += len(re.match(r"[^,()\s]*", s[pos:]).group(0))
        return {"label": label, "children": ch}
    return clade()


def leaves_under(node):
    if not node["children"]:
        return [node["label"]]
    out = []
    for c in node["children"]:
        out += leaves_under(c)
    return out


def split_key(cl, full):
    """Orientation-invariant key for an unrooted bipartition cl | (full-cl).
    Canonicalize to the smaller side; on a BALANCED split (|cl| == |full-cl|)
    both sides tie, so break the tie by the reference (min-labeled) leaf -- else
    cl and its complement get counted as two distinct edges, NNLS splits the one
    real edge length across the duplicate columns, and the phantom copy reads as a
    spurious zero-length edge (see 2026-07-07: this zeroed the balanced-tree
    bottleneck, e.g. the organic Llama chat|vicuna central split)."""
    comp = full - cl
    if len(cl) != len(comp):
        return cl if len(cl) < len(comp) else comp
    return cl if min(full) in cl else comp


def all_splits(tree, leaves):
    full = frozenset(leaves); out = []
    def walk(n, root):
        cl = frozenset(leaves_under(n))
        if not root and 0 < len(cl) < len(full):
            out.append(cl)
        for c in n["children"]:
            walk(c, False)
    walk(tree, True)
    uniq, seen = [], set()
    for c in out:
        k = split_key(c, full)
        if k not in seen:
            seen.add(k); uniq.append(c)
    return uniq


def run_margin(D, leaves, tree):
    full = frozenset(leaves)
    splits, seen = [], set()
    for s in (x & full for x in all_splits(tree, leaves)):
        if not (0 < len(s) < len(full)):
            continue
        k = split_key(s, full)
        if k not in seen:
            seen.add(k); splits.append(s)
    if not splits:
        return None
    m = len(leaves)
    pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
    A = np.zeros((len(pairs), len(splits)))
    for p, (i, j) in enumerate(pairs):
        for k, cl in enumerate(splits):
            if (leaves[i] in cl) ^ (leaves[j] in cl):
                A[p, k] = 1.0
    d = np.array([D[i, j] for i, j in pairs])
    L, _ = nnls(A, d)
    noise = float(np.max(np.abs(A @ L - d)))
    internal = [k for k, cl in enumerate(splits) if 2 <= len(cl) <= m - 2]
    if not internal:
        return None
    denom = 2 * noise + 1e-12
    mar_all = L / denom                                   # every fitted edge (pendant + internal)
    mar_int = np.array([L[k] / denom for k in internal])  # topology-defining edges only
    return dict(bottleneck=float(mar_all.min()),          # Atteson: min over ALL edges
                mean_margin=float(mar_int.mean()),        # diagnostic: mean over internal edges
                frac_resolvable=float((mar_int > 1).mean()))


def collect():
    groups = {}
    with open(COH_CSV) as f:
        hdr = f.readline().strip().split(",")
        for line in f:
            r = dict(zip(hdr, line.strip().split(",")))
            groups[r["group"]] = dict(coherence=float(r["coherence"]),
                                      clade=float(r["clade_recovery"]),
                                      sig=float(r["pct_sig_vs_label"]))
    rows = []
    for g in sorted(groups):
        b, mm, fr = [], [], []
        for wd in sorted(glob.glob(os.path.join(ROOT, "results", g, "run_*/weight"))):
            try:
                D = np.abs(np.load(os.path.join(wd, "avg_distance_cosine.npy")))
                meta = json.load(open(os.path.join(wd, "distance_metadata.json")))
                names = [n.split("/")[-1] for n in meta["model_names"]]
                tnk = os.path.join(ROOT, "outputs", g, os.path.basename(os.path.dirname(wd)), "tree.newick")
                if not os.path.exists(tnk) or D.shape[0] != len(names) or D.shape[0] < MIN_LEAVES:
                    continue
                tree = parse_newick(open(tnk).read())
                if not set(names).issubset(set(leaves_under(tree))):
                    continue
                res = run_margin(D, names, tree)
                if res is None:
                    continue
                b.append(res["bottleneck"]); mm.append(res["mean_margin"]); fr.append(res["frac_resolvable"])
            except Exception:
                continue
        if len(b) < 3:
            continue
        rows.append(dict(group=g, n=len(b), coherence=groups[g]["coherence"],
                         clade=groups[g]["clade"], sig=groups[g]["sig"],
                         bottleneck=float(np.mean(b)), mean_margin=float(np.mean(mm)),
                         frac_resolvable=float(np.mean(fr))))
    return rows


def spear(a, b):
    a = np.argsort(np.argsort(a)); b = np.argsort(np.argsort(b))
    return float(np.corrcoef(a, b)[0, 1])


def main():
    rows = collect()
    cols = ["group", "n", "coherence", "clade", "sig", "bottleneck", "mean_margin", "frac_resolvable"]
    print(",".join(cols))
    for r in sorted(rows, key=lambda x: x["clade"]):
        print(",".join(str(round(r[c], 4)) if isinstance(r[c], float) else str(r[c]) for c in cols))
    A = {c: np.array([r[c] for r in rows]) for c in ["coherence", "clade", "sig", "bottleneck", "mean_margin", "frac_resolvable"]}
    print(f"\n=== Spearman vs clade_recovery (>= {MIN_LEAVES} leaves, {len(rows)} groups) ===")
    for nm in ["coherence", "bottleneck", "mean_margin", "frac_resolvable"]:
        print(f"  {nm:14s}: {spear(A[nm], A['clade']):+.3f}")
    print("=== Spearman vs pct_sig_vs_label ===")
    for nm in ["coherence", "bottleneck", "mean_margin", "frac_resolvable"]:
        print(f"  {nm:14s}: {spear(A[nm], A['sig']):+.3f}")
    hi = A["coherence"] > 0.9
    if hi.sum() >= 4:
        print(f"\n=== WITHIN saturated-coherence cluster (n={int(hi.sum())}) ===")
        print(f"  clade {A['clade'][hi].min():.0f}-{A['clade'][hi].max():.0f}%, "
              f"coherence {A['coherence'][hi].min():.4f}-{A['coherence'][hi].max():.4f} (flat)")
        for nm in ["coherence", "bottleneck", "mean_margin", "frac_resolvable"]:
            print(f"  {nm:14s} vs clade: {spear(A[nm][hi], A['clade'][hi]):+.3f}")

    if os.environ.get("FIG"):
        outdir = os.path.join(ROOT, "results/aggregate/recovery_rescore")
        import csv
        with open(os.path.join(outdir, "atteson_margin_by_group.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
            for r in sorted(rows, key=lambda x: x["clade"]):
                w.writerow(r)
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        coh, clade, mm = A["coherence"], A["clade"], A["mean_margin"]
        recov = coh > 0.9
        fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
        ax[0].scatter(mm, clade, c=np.where(recov, "#2a7fb8", "#c0392b"), s=70, edgecolor="k", linewidth=.5, zorder=3)
        ax[0].axvline(1.0, color="gray", ls="--", lw=1.2)
        ax[0].set_xscale("log"); ax[0].grid(alpha=.25)
        ax[0].set_xlabel("mean Atteson margin = edge length / 2·non-additivity error")
        ax[0].set_ylabel("clade recovery (%)")
        ax[0].set_title("A. Margin orders recovery across ALL groups")
        ax[0].legend(handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor='#2a7fb8',markeredgecolor='k',label='coherence > 0.9'),
                              Line2D([0],[0],marker='o',color='w',markerfacecolor='#c0392b',markeredgecolor='k',label='dead (~0.2)')], fontsize=8, loc="lower right")
        b = ax[1]
        b.scatter(coh[recov], clade[recov], c="#999", s=70, marker="s", edgecolor="k", linewidth=.5, label=f"coherence ({spear(coh[recov],clade[recov]):+.2f})")
        b2 = b.twiny()
        b2.scatter(mm[recov], clade[recov], c="#2a7fb8", s=70, edgecolor="k", linewidth=.5, label=f"margin ({spear(mm[recov],clade[recov]):+.2f})")
        b.set_xlabel("coherence (saturated)", color="#555"); b2.set_xlabel("mean Atteson margin", color="#2a7fb8")
        b.set_ylabel("clade recovery (%)"); b.grid(alpha=.25)
        b.set_title("B. Where coherence saturates, margin still ranks")
        h1,l1=b.get_legend_handles_labels(); h2,l2=b2.get_legend_handles_labels()
        b.legend(h1+h2, l1+l2, fontsize=8, loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "fig_atteson_margin_vs_recovery.png"), dpi=150, bbox_inches="tight")
        print("\nwrote results/aggregate/recovery_rescore/atteson_margin_by_group.csv + fig_atteson_margin_vs_recovery.png")


if __name__ == "__main__":
    main()
