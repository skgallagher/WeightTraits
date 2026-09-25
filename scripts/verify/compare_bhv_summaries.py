#!/usr/bin/env python
"""Compare the paper's BHV summary (raw, seed 0, 300 epochs) with a re-run at new seeds / more epochs.
usage: compare_bhv_summaries.py <paper_summary.csv> <rerun_summary.csv>"""
import csv, sys
ref = {r["suite_id"]: r for r in csv.DictReader(open(sys.argv[1])) if r["scale_mode"] == "raw" and r["seed"] == "0"}
new = [r for r in csv.DictReader(open(sys.argv[2])) if r["scale_mode"] == "raw"]
cols = ["clade_recovery_mean", "polytomy_aware_exact_recovery_rate", "rf_mean", "false_negative_mean"]
worst = 0.0; print(f"{'suite':22s} seed epochs  d(clade)  d(PAER)   d(RF)    d(FN)")
for r in new:
    b = ref[r["suite_id"]]; ds = [abs(float(r[c]) - float(b[c])) for c in cols]; worst = max(worst, *ds)
    print(f"{r['suite_id']:22s} {r['seed']:>4s} {r['epochs']:>6s}  " + "  ".join(f"{d:7.4f}" for d in ds))
print(f"largest change in any reported mean across {len(new)} re-runs: {worst:.4f}")
print("PASS compare_bhv_summaries (paper values reproduced at new seeds; RF jitter <= 0.05)" if worst <= 0.05 else "FAIL compare_bhv_summaries"); sys.exit(0 if worst <= 0.05 else 1)
