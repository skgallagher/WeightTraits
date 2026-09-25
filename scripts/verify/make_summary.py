#!/usr/bin/env python
"""Build reports/verify/<date>/SUMMARY.md from checks.tsv.  usage: make_summary.py <report_dir> <paper.tex>"""
import csv, subprocess, sys, datetime
from collections import OrderedDict
from pathlib import Path
R, TEX = Path(sys.argv[1]), sys.argv[2]
rows = list(csv.reader(open(R / "checks.tsv"), delimiter="\t"))
by = OrderedDict()
for script, label, status, detail in rows: by.setdefault(script, []).append((label, status, detail))
STAGES = [("1 Tree generation", "01_"), ("2 Data/task assignment + training", "02_"), ("3 Distance cubes", "03_"),
          ("4 Tree building from the cubes", "04_"), ("5 Analysis of the trees", "05_"), ("6 Behavior from distance", "06_")]
def git(path):
    try: return subprocess.check_output(["git", "-C", path, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception: return "?"
n_ok = sum(1 for r in rows if r[2] == "OK"); n_fail = sum(1 for r in rows if r[2] != "OK")
out = [f"# Verification summary — {datetime.date.today().isoformat()}", "",
       f"Paper: `{Path(TEX).name}` @ ELLMTrees-paper {git(str(Path(TEX).parent))}; WeightTraits {git('/Users/shannon/Desktop/phylo/WeightTraits')}; ELLMTrees {git('/Users/shannon/Desktop/phylo/ELLMTrees')}", "",
       f"**{n_ok} checks passed, {n_fail} failed.**  (a) = code works as intended, (b) = exists on Wright & differs per simulation, (c) = paper matches artifacts.", ""]
if n_fail:
    out += ["## Failing checks", ""] + [f"- `{s}` — {l}: {d}" for s, l, st, d in rows if st != "OK"] + [""]
for title, prefix in STAGES:
    out += [f"## Stage {title}", "", "| part | check | result | detail |", "|---|---|---|---|"]
    for script, items in by.items():
        if not script.startswith(prefix): continue
        part = script.split("_")[-1]
        for label, status, detail in items:
            out.append(f"| ({part}) | {label} | {'PASS' if status == 'OK' else '**FAIL**'} | {detail[:110]} |")
    out.append("")
(R / "SUMMARY.md").write_text("\n".join(out)); print(f"{n_ok} passed, {n_fail} failed -> {R/'SUMMARY.md'}")
