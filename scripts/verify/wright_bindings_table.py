#!/usr/bin/env python3
"""Print one line per sealed training-completion binding on Wright (read-only)."""
import glob, json
R = "/home/export/sgallagh/WeightTraits-downstream-20260831-native"
files = sorted(glob.glob(R + "/runtime_receipts/*_COMPLETION_BINDING.json")) + \
        sorted(glob.glob(R + "/outputs/final_paper_20260908/completion/*.binding.json"))
for f in files:
    d = json.load(open(f)); rows = d["rows"]
    name = f.split("/")[-1]
    print(f"{name[:70]:70s} cohort={d['cohort_id'][:55]:55s} ok={d['n_ok_nodes']}/{d['n_total_runs']} trees={d['n_trees']} "
          f"valid={d['valid']} err={d['n_errors']} warn={d['n_warnings']} rows_valid={sum(1 for r in rows if r['valid'])}/{len(rows)}")
