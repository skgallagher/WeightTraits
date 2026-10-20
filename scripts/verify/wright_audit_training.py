#!/usr/bin/env python3
"""Per-node training audit, run ON WRIGHT against the sealed stage (read-only).

usage: wright_audit_training.py <stage_root> <cohort_output_root> <assigned_manifest_dir> <label>

For every <tree>/<node> under the cohort output root it reads the node's
training_log.jsonl completion record and the weight file, and prints one TSV row:

  label tree node parent_in_manifest status step n_eval stop_reasons init_from_ok
  base_revision n_train n_eval_rows train_loss eval_loss weight_file size_bytes slice_sha256
  n_starts last_run_clean chain

n_eval counts evaluation records AFTER the last "training row started" event (a Slurm
requeue appends a second attempt to the same log; no intermediate checkpoints are saved,
so the final model always comes from the last, complete attempt).  last_run_clean is
"yes" when those records are monotone 250..2000 and the log ends with status=completed.
chain is the dataset sequence along the root->node path: two nodes with the same chain
have the same training recipe and MAY be bit-identical (deterministic training).

slice_sha256 = sha256 over three 4 MiB slices (head / middle / tail) of the weight
file: cheap, and two files with different slice hashes are certainly different.
Nothing is written anywhere.
"""
import hashlib, json, os, sys
from pathlib import Path

stage, out_root, man_dir, label = sys.argv[1:5]
stage = Path(stage); out_root = stage / out_root; man_dir = stage / man_dir
SL = 4 * 1024 * 1024

def slice_sha(path):
    sz = os.path.getsize(path); h = hashlib.sha256()
    with open(path, "rb") as fh:
        for off in (0, sz // 2, max(0, sz - SL)):
            fh.seek(off); h.update(fh.read(SL))
    return sz, h.hexdigest()

print("\t".join("label tree node parent status step n_eval stop_reasons init_from_ok base_revision n_train n_eval_rows train_loss eval_loss weight_file size_bytes slice_sha256 n_starts last_run_clean chain".split()))
for tree_dir in sorted(p for p in out_root.iterdir() if p.is_dir()):
    tree = tree_dir.name
    parents = {}; mrows = {}
    for line in open(man_dir / f"{tree}.manifest.jsonl"):
        if line.strip():
            r = json.loads(line); parents[r["node_id"]] = r["parent_id"]; mrows[r["node_id"]] = r
    chain_of = lambda n: ">".join(mrows[x]["dataset_id"] for x in mrows[n]["path"] if x != "root") if n in mrows else "?"
    for node_dir in sorted(p for p in tree_dir.iterdir() if p.is_dir()):
        node = node_dir.name
        log = node_dir / "training_log.jsonl"
        rec = None; events = []
        if log.exists():
            events = [json.loads(line) for line in open(log) if line.strip()]
            for d in events:
                if d.get("message") == "training row finished":
                    rec = d
        parent = parents.get(node, "?")
        if rec is None:
            print("\t".join(map(str, [label, tree, node, parent, "NO_COMPLETION_RECORD"] + ["?"] * 15 + [chain_of(node)]))); continue
        ex = rec.get("extra", {}); bm = ex.get("backend_metadata", {}); data = ex.get("data", {})
        init_from = bm.get("init_from") or ""
        if parent == "root":
            init_ok = "root" if (init_from == "" or f"/{tree}/" not in init_from) else f"BAD:{init_from}"
        else:
            init_ok = "ok" if f"/{tree}/{parent}/" in init_from else f"BAD:{init_from}"
        starts = [i for i, d in enumerate(events) if d.get("message") == "training row started"]
        last = starts[-1] if starts else 0
        steps = [d["step"] for d in events[last:] if d.get("eval_loss") is not None]
        n_eval = len(steps)
        last_run_clean = "yes" if (steps and steps == sorted(steps) and steps[-1] == 2000 and events[-1].get("status") == "completed") else "NO"
        wf = None
        for cand in ("model/model.safetensors", "adapter/adapter_model.safetensors", "merged/model.safetensors"):
            if (node_dir / cand).exists():
                wf = node_dir / cand; break
        if wf is None:
            size, sha = -1, "NO_WEIGHT_FILE"
        else:
            size, sha = slice_sha(wf)
        print("\t".join(map(str, [label, tree, node, parent, rec.get("status"), rec.get("step"),
              n_eval, ";".join(rec.get("stop_reasons") or []) or "-", init_ok,
              bm.get("base_model_revision", "?")[:12], data.get("n_train_records", "?"), data.get("n_eval_records", "?"),
              rec.get("train_loss"), rec.get("eval_loss"),
              str(wf.relative_to(stage)) if wf else "-", size, sha, len(starts), last_run_clean, chain_of(node)])))
