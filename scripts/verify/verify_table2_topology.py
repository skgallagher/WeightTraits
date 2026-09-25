#!/usr/bin/env python
"""Compare Table 2's topology columns (clade, PAER, RF, FN with SEs) in the tex against the BHV
summary CSV (scale_mode=raw, seed 0), at the paper's display rounding.  Also recomputes the
caption's exact binomial intervals and the prose ranges, and explains the identical cells.
usage: verify_table2_topology.py <paper.tex> <controlled_bhv_summary.csv> <reconstruction_rescore.tsv>
"""
import csv, re, sys
from math import comb
TEX, CSV, RESC = sys.argv[1:4]
tex = open(TEX).read()
ROWS = {  # tex row label fragment -> BHV suite id
  r"LoRA: key only": "flan_key", r"LoRA: q/k/v  ": "flan_qkv", r"LoRA: q/k/v/o (all attention)": "flan_qkvo",
  r"LoRA: q/k/v/o + FFN projections": "flan_all_projections", r"Full fine-tuning                  &": "flan_full",
  r"LoRA qkv ($r{=}8$)": "llama_r8", r"LoRA qkv ($r{=}64$)": "llama_r64", r"Full fine-tuning                   &": "llama_full"}
bhv = {r["suite_id"]: r for r in csv.DictReader(open(CSV)) if r["scale_mode"] == "raw" and r["seed"] == "0"}
def cell(s):  # "\textbf{92\%} (3\%)" or "\textbf{100\%}$^\star$" -> (value, se or None)
    nums = re.findall(r"(\d+(?:\.\d+)?)", s.replace("\\%", ""))
    return (float(nums[0]), float(nums[1]) if len(nums) > 1 else None)
fails = 0
print(f"{'row':34s} {'clade':>12s} {'PAER':>12s} {'RF (SE)':>14s} {'FN (SE)':>12s}  vs BHV raw seed0")
for frag, suite in ROWS.items():
    line = next((l for l in tex.splitlines() if frag in l and "46/26" in l), None)
    if line is None: print("ROW NOT FOUND:", frag); fails += 1; continue
    cells = [c.strip() for c in line.split("&")]
    clade, paer, rffn = cell(cells[4]), cell(cells[6]), cells[7]
    rf, fn = [cell(x) for x in rffn.replace("\\\\", "").split(";")]
    b = bhv[suite]
    exp = dict(clade=round(100 * float(b["clade_recovery_mean"])), clade_se=round(100 * float(b["clade_recovery_se"])),
               paer=round(100 * float(b["polytomy_aware_exact_recovery_rate"])), paer_se=round(100 * float(b["polytomy_aware_exact_recovery_rate_se"])),
               rf=round(float(b["rf_mean"]), 2), rf_se=round(float(b["rf_se"]), 2), fn=round(float(b["false_negative_mean"]), 2), fn_se=round(float(b["false_negative_se"]), 2))
    ok = (clade[0] == exp["clade"] and (clade[1] is None or clade[1] == exp["clade_se"]) and paer[0] == exp["paer"] and (paer[1] is None or paer[1] == exp["paer_se"])
          and rf[0] == exp["rf"] and rf[1] == exp["rf_se"] and fn[0] == exp["fn"] and fn[1] == exp["fn_se"])
    fails += 0 if ok else 1
    print(f"{suite:34s} {clade[0]:>5.0f} ({'-' if clade[1] is None else int(clade[1])})   {paer[0]:>5.0f} ({'-' if paer[1] is None else int(paer[1])})   {rf[0]:.2f} ({rf[1]:.2f})   {fn[0]:.2f} ({fn[1]:.2f})   {'ok' if ok else 'MISMATCH: ' + str(exp)}")
# caption: 117/117 recovered splits, 46/46 PAER, exact two-sided 95% binomial intervals
def clopper_pearson_lower(k, n, alpha=0.05):
    # exact lower bound for k == n: alpha^(1/n)... use the general Beta quantile via bisection on the binomial tail
    lo, hi = 0.0, 1.0
    for _ in range(200):
        p = (lo + hi) / 2
        tail = sum(comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1))
        if tail < alpha / 2: lo = p
        else: hi = p
    return lo
ci117 = clopper_pearson_lower(117, 117); ci46 = clopper_pearson_lower(46, 46)
ok = ("117/117" in tex and "46/46 PAER" in tex and "96.9--100" in tex and "92.3--100" in tex and round(100 * ci117, 1) == 96.9 and round(100 * ci46, 1) == 92.3)
fails += 0 if ok else 1
print(f"\ncaption: 117/117 splits -> exact 95% CI lower {100*ci117:.1f}% (paper 96.9); 46/46 PAER -> {100*ci46:.1f}% (paper 92.3)  {'ok' if ok else 'MISMATCH'}")
# prose ranges
clades = [round(100 * float(b["clade_recovery_mean"])) for b in bhv.values()]; paers = [round(100 * float(b["polytomy_aware_exact_recovery_rate"])) for b in bhv.values()]
flan_lora = [round(100 * float(bhv[s]["clade_recovery_mean"])) for s in ("flan_key", "flan_qkv", "flan_qkvo", "flan_all_projections")]
ok2 = ("clade recovery ranges from 92--100\\%" in tex and min(clades) == 92 and max(clades) == 100 and "PAER from 85--100\\%" in tex and min(paers) == 85 and max(paers) == 100
       and "92--99\\% of clades for Flan" in tex and min(flan_lora) == 92 and max(flan_lora) == 99)
fails += 0 if ok2 else 1
print(f"prose: clade range {min(clades)}-{max(clades)} (paper 92-100), PAER {min(paers)}-{max(paers)} (paper 85-100), Flan LoRA clades {min(flan_lora)}-{max(flan_lora)} (paper 92-99)  {'ok' if ok2 else 'MISMATCH'}")
# why four rows share 100% / 2.04 / 0.00: with FN = 0 and a binary NJ tree, RF = (n_leaves - 3) - n_truth_splits, a property of the truth set
rows = list(csv.DictReader(open(RESC), delimiter="\t")); ok3 = True; expl = {}
for c in ("flan_full", "llama_r8", "llama_r64"):
    rs = [r for r in rows if r["cohort"] == c and int(r["truth_splits"]) > 0]
    formula = all(int(r["FN"]) == 0 and int(r["RF"]) == (int(r["n_leaves"]) - 3) - int(r["truth_splits"]) for r in rs)
    expl[c] = (formula, round(sum(int(r["RF"]) for r in rs) / len(rs), 2)); ok3 &= formula and expl[c][1] == 2.04
print(f"identical cells: FN=0 and RF=(leaves-3)-truth_splits on every eligible tree, mean-NJ RF over 46 trees: {expl}  {'ok' if ok3 else 'MISMATCH'}")
fails += 0 if ok3 else 1
print("PASS verify_table2_topology" if fails == 0 else f"FAIL verify_table2_topology ({fails})"); sys.exit(1 if fails else 0)
