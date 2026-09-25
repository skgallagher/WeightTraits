#!/usr/bin/env python3
"""Behavior inference audit — run ON WRIGHT, read-only.  usage: wright_audit_behavior.py <cohort_key>
Walks outputs/allnode_v1/inference/<cohort>/<tree>/<node>/responses.jsonl for all 50 trees: one TSV
row per node (rows, probes, sha256 of the file, empty outputs per probe) and a JSON summary with the
pooled LEAF empty-output percentage per probe and per depth (the paper's Table 8/9 quantities),
recounted directly from the raw text with text.strip() == "".
"""
import collections, glob, hashlib, json, sys
cohort = sys.argv[1]
root = f"/home/export/sgallagh/WeightTraits-downstream-20260831-native/outputs/allnode_v1/inference/{cohort}"
leaf_empty = collections.Counter(); leaf_out = collections.Counter(); depth_empty = collections.Counter(); depth_out = collections.Counter()
rows_out = []; n_nodes = 0; trees = set()
for f in sorted(glob.glob(f"{root}/*/*/responses.jsonl")):
    tree, node = f.split("/")[-3], f.split("/")[-2]; trees.add(tree); n_nodes += 1
    h = hashlib.sha256(open(f, "rb").read()).hexdigest()
    n = 0; probes = collections.Counter(); empties = collections.Counter(); role = depth = None
    for line in open(f):
        r = json.loads(line); rr = r["response_row"]; n += 1; probes[rr["probe_id"]] += 1
        role, depth = r["node_role"], r["depth"]
        if r["text"].strip() == "":
            empties[rr["probe_id"]] += 1
    if role == "leaf":
        for p, c in probes.items():
            leaf_out[p] += c; leaf_empty[p] += empties[p]; depth_out[(p, depth)] += c; depth_empty[(p, depth)] += empties[p]
    rows_out.append((cohort, tree, node, role, depth, n, ";".join(f"{p}={c}" for p, c in sorted(probes.items())), ";".join(f"{p}={c}" for p, c in sorted(empties.items())) or "-", h))
with open(f"audit_behavior_{cohort}.tsv", "w") as fh:
    fh.write("cohort\ttree\tnode\trole\tdepth\trows\tprobes\tempty_by_probe\tsha256\n")
    for r in rows_out: fh.write("\t".join(map(str, r)) + "\n")
summary = {"cohort": cohort, "n_trees": len(trees), "n_nodes": n_nodes, "n_distinct_sha256": len({r[-1] for r in rows_out}),
           "rows_per_node": sorted(collections.Counter(r[5] for r in rows_out).items()),
           "leaf_empty_percent": {p: 100 * leaf_empty[p] / leaf_out[p] for p in leaf_out},
           "leaf_empty_counts": {p: [leaf_empty[p], leaf_out[p]] for p in leaf_out},
           "depth_empty_percent": {f"{p}@{d}": 100 * depth_empty[(p, d)] / depth_out[(p, d)] for (p, d) in sorted(depth_out)}}
json.dump(summary, open(f"audit_behavior_{cohort}.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
