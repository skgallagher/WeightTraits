#!/usr/bin/env python
"""Compare every number in the behavior tables of the paper with the sealed artifacts, at display rounding.

usage: verify_behavior_paper.py <paper.tex> <behavior_trees_dir> <table8_dir>
  tab:behavior_holdout (12 cells: r_DL + CI), tab:behavior_het (12 rows: r_DL, CI, I^2, k),
  tab:eos-sensitivity (5 probes: EOS-included / EOS-excluded r_DL + CI, empty %),
  tab:eos-depth (5 probes x 4 depths), and the prose numbers (38.64%, -0.17 / -0.27, |r_DL| 0.17-0.23).
"""
import json, re, sys
TEX, B, T8 = sys.argv[1:4]
tex = open(TEX).read()
def dl(cohort, probe, stratum="all_trained_node_pairs"):
    rec = json.load(open(f"{B}/semantic/{cohort}/{probe}/runset_receipt.json"))
    return {s["stratum"]: s for s in rec["analysis"]["strata"]}[stratum]["dl"]
CELLS = {("Llama-3.2-1B", "trained"): ("llama_full", "translation"), ("Llama-3.2-1B", "held-out"): ("llama_notrans", "translation"),
         ("Llama-3.2-1B", "hellaswag"): ("llama_full", "hellaswag"), ("Llama-3.2-1B", "arc"): ("llama_full", "arc_challenge"),
         ("Llama-3.2-1B", "mmlu"): ("llama_full", "mmlu"), ("Llama-3.2-1B", "tqa"): ("llama_full", "truthfulqa"),
         ("Flan-T5-base", "trained"): ("flan_full", "translation"), ("Flan-T5-base", "held-out"): ("flan_notrans", "translation"),
         ("Flan-T5-base", "hellaswag"): ("flan_full", "hellaswag"), ("Flan-T5-base", "arc"): ("flan_full", "arc_challenge"),
         ("Flan-T5-base", "mmlu"): ("flan_full", "mmlu"), ("Flan-T5-base", "tqa"): ("flan_full", "truthfulqa")}
fails = 0
def num(s): return float(s.replace("$", "").replace("\\", "").replace("{", "").replace("}", "").replace("+", "").replace("−", "-").strip())
def r2(x): return round(x + 0.0, 2)
# --- Table 4 -------------------------------------------------------------------------------------
print("tab:behavior_holdout")
for model in ("Llama-3.2-1B", "Flan-T5-base"):
    line = next(l for l in tex.splitlines() if l.startswith(model + " &"))
    cells = re.findall(r"\\shortstack\{\$([+-]?[0-9.]+)(?:\^\{\\ddagger\})?\$\\\\\{\\scriptsize\$\[([+-]?\.[0-9]+),([+-]?\.[0-9]+)\]\$\}\}", line)
    assert len(cells) == 6, (model, len(cells), line[:200])
    for (val, lo, hi), key in zip(cells, ("trained", "held-out", "hellaswag", "arc", "mmlu", "tqa")):
        d = dl(*CELLS[(model, key)]); ok = (num(val), num(lo), num(hi)) == (r2(d["correlation"]), r2(d["ci_low_r"]), r2(d["ci_high_r"]))
        fails += 0 if ok else 1
        print(f"  {model:13s} {key:9s} paper {val} [{lo},{hi}]   artifact {d['correlation']:+.4f} [{d['ci_low_r']:+.3f},{d['ci_high_r']:+.3f}]  {'ok' if ok else 'MISMATCH'}")
# --- heterogeneity table -------------------------------------------------------------------------
print("tab:behavior_het")
HET = [("Translation (trained)", "trained"), ("Translation (held-out)", "held-out"), ("Translation (withheld)", "held-out"), ("HellaSwag", "hellaswag"), ("ARC-C", "arc"), ("MMLU", "mmlu"), ("TruthfulQA", "tqa")]
for line in tex.splitlines():
    m = re.match(r"(Llama-3\.2-1B|Flan-T5-base) & ([A-Za-z\- ()]+?)\$?\^?\\?d?d?a?g?g?e?r?\$?\s*& \$([+-]?[0-9.]+)\$ \[\$([+-]?[0-9.]+),([+-]?[0-9.]+)\$\] & ([0-9]+)\\% & ([0-9]+) \\\\", line.strip())
    if not m: continue
    model, probe, val, lo, hi, i2, k = m.groups(); key = next(key for lab, key in HET if probe.strip().startswith(lab))
    d = dl(*CELLS[(model, key)]); ok = (num(val), num(lo), num(hi), int(i2), int(k)) == (r2(d["correlation"]), r2(d["ci_low_r"]), r2(d["ci_high_r"]), round(100 * d["i2"]), d["n_trees"])
    fails += 0 if ok else 1
    print(f"  {model:13s} {probe.strip():26s} paper {val} [{lo},{hi}] I2={i2}% k={k}   artifact {d['correlation']:+.3f} I2={100*d['i2']:.1f}% k={d['n_trees']}  {'ok' if ok else 'MISMATCH'}")
# --- EOS sensitivity table -----------------------------------------------------------------------
print("tab:eos-sensitivity")
NAMES = {"ARC-Challenge": "arc_challenge", "Dolly open-ended": "dolly_open_ended", "HellaSwag": "hellaswag", "MMLU": "mmlu", "TruthfulQA": "truthfulqa"}
for line in tex.splitlines():
    m = re.match(r"(ARC-Challenge|Dolly open-ended|HellaSwag|MMLU|TruthfulQA) & \$([+-]?[0-9.]+)\$ \[\$([+-]?[0-9.]+),([+-]?[0-9.]+)\$\] & \$([+-]?[0-9.]+)\$ \[\$([+-]?[0-9.]+),([+-]?[0-9.]+)\$\] & \$([0-9.]+)\\%\$", line.strip())
    if not m: continue
    name, inc, ilo, ihi, exc, elo, ehi, pct = m.groups(); t8 = json.load(open(f"{T8}/{NAMES[name]}.json"))
    inc_d = dl("llama_full", NAMES[name]); exc_d = t8["eos_excluded_all_node_dl"]; p = t8["counts"]["pooled_leaf_empty_percent"]
    r3 = lambda x: round(x, 3)
    ok = ((num(inc), num(ilo), num(ihi)) == (r3(inc_d["correlation"]), r3(inc_d["ci_low_r"]), r3(inc_d["ci_high_r"])) and
          (num(exc), num(elo), num(ehi)) == (r3(exc_d["correlation"]), r3(exc_d["ci_low_r"]), r3(exc_d["ci_high_r"])) and num(pct) == round(p, 2))
    fails += 0 if ok else 1
    print(f"  {name:17s} paper incl {inc} [{ilo},{ihi}] excl {exc} [{elo},{ehi}] empty {pct}%   artifact incl {inc_d['correlation']:+.4f} excl {exc_d['correlation']:+.4f} empty {p:.4f}%  {'ok' if ok else 'MISMATCH'}")
# --- depth table ---------------------------------------------------------------------------------
print("tab:eos-depth")
dep = json.load(open(f"{T8}/depth_summary.json"))["depth"]
for line in tex.splitlines():
    m = re.match(r"(ARC-Challenge|Dolly open-ended|HellaSwag|MMLU|TruthfulQA) & ([0-9.]+)\\% & ([0-9.]+)\\% & ([0-9.]+)\\% & \$?<?([0-9.]+)\\%\$? \\\\", line.strip())
    if not m: continue
    name = m.group(1); vals = [num(x) for x in m.groups()[1:]]; art = [dep[NAMES[name]][str(k)]["percent"] for k in (1, 2, 3, 4)]
    ok = all(v == round(a, 2) or (v == 0.01 and a < 0.01) for v, a in zip(vals, art)); fails += 0 if ok else 1
    print(f"  {name:17s} paper {vals}   artifact {[round(a, 3) for a in art]}  {'ok' if ok else 'MISMATCH'}")
# --- prose ---------------------------------------------------------------------------------------
hs = json.load(open(f"{T8}/hellaswag.json")); inc = dl("llama_full", "hellaswag"); exc = hs["eos_excluded_all_node_dl"]
p1 = "38.64\\% of HellaSwag leaf outputs were empty" in tex and round(hs["counts"]["pooled_leaf_empty_percent"], 2) == 38.64
p2 = "$r_{\\mathrm{DL}}=-0.17$ [$-0.24,-0.10$]" in tex and (r2(inc["correlation"]), r2(inc["ci_low_r"]), r2(inc["ci_high_r"])) == (-0.17, -0.24, -0.10)
p3 = "$-0.27$ [$-0.34,-0.19$]" in tex and (r2(exc["correlation"]), r2(exc["ci_low_r"]), r2(exc["ci_high_r"])) == (-0.27, -0.34, -0.19)
llama = [abs(dl(*CELLS[("Llama-3.2-1B", k)])["correlation"]) for k in ("held-out", "hellaswag", "arc", "mmlu", "tqa")]
p4 = "0.17$--$0.23" in tex and (round(min(llama), 2), round(max(llama), 2)) == (0.17, 0.23)
for lab, ok in (("prose: 38.64% HellaSwag leaf outputs empty", p1), ("prose: HellaSwag incl. EOS r_DL=-0.17 [-0.24,-0.10]", p2), ("prose: excluding EOS -0.27 [-0.34,-0.19]", p3), ("prose: |r_DL| = 0.17-0.23 across held-out Llama probes", p4)):
    fails += 0 if ok else 1; print(f"  {lab:60s} {'ok' if ok else 'MISMATCH'}")
print("PASS verify_behavior_paper" if fails == 0 else f"FAIL verify_behavior_paper ({fails})"); sys.exit(1 if fails else 0)
