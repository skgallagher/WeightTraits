"""App. F: empty-response rates over trained leaves, pooled and by leaf depth.

Ported from the Wright-only ``compute_table8_empty_counts.py`` that produced the paper's
``tab:eos-depth`` and the empty-output column of ``tab:eos-sensitivity``. Every leaf's response
file is checked against its terminal receipt before counting. A response is empty when
``text.strip() == ""``. Per depth, ``percent`` pools all outputs. ``mean_tree_percent`` and its
t interval average the per-tree rates.

``python -m weighttraits.behavior.allnode.empty_rates [--root ROOT] [--cohort llama_full]``
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

DOWNSTREAM_ROOT = Path("/home/export/sgallagh/WeightTraits-downstream-20260831-native")
PROBES = ("arc_challenge", "dolly_open_ended", "hellaswag", "mmlu", "truthfulqa")
# Outputs per leaf: 300 for the multiple-choice probes and 90 for Dolly (prompts x draws).
OUTPUTS_PER_LEAF = {"dolly_open_ended": 90}
DEFAULT_OUTPUTS_PER_LEAF = 300


def leaf_empty_rates(root: Path = DOWNSTREAM_ROOT, cohort: str = "llama_full",
                     probes=PROBES) -> dict[str, Any]:
    from scipy.stats import t as student_t

    root = Path(root)
    plan = json.loads((root / f"deployments/allnode_v1/runtime_inputs/{cohort}/inference_plan.json")
                      .read_text())
    totals = {probe: [0, 0] for probe in probes}
    depths = {probe: collections.defaultdict(lambda: [0, 0]) for probe in probes}
    tree_depths = {
        probe: collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
        for probe in probes
    }
    files = 0
    for tree_manifest in plan["checkpoint_manifests"]:
        tree_id = tree_manifest["tree_id"]
        for binding in tree_manifest["bindings"]:
            if binding["node_role"] != "leaf":
                continue
            depth = binding["depth"]
            directory = root / "outputs/allnode_v1/inference" / cohort / tree_id / binding["node_id"]
            receipt = json.loads((directory / "terminal_receipt.json").read_text())
            if receipt["valid"] is not True or receipt["status"] != "completed":
                raise ValueError(f"invalid leaf receipt: {directory}")
            raw = (directory / "responses.jsonl").read_bytes()
            if hashlib.sha256(raw).hexdigest() != receipt["responses"]["sha256"]:
                raise ValueError(f"response hash drift: {directory}")
            files += 1
            for line in raw.splitlines():
                row = json.loads(line)
                probe = row["response_row"]["probe_id"]
                if probe not in totals:
                    continue
                empty = int(not row["text"].strip())
                for counts in (totals[probe], depths[probe][depth],
                               tree_depths[probe][tree_id][depth]):
                    counts[0] += empty
                    counts[1] += 1

    depth_summary: dict[str, dict[str, Any]] = {}
    for probe, by_depth in depths.items():
        depth_summary[probe] = {}
        for depth, values in sorted(by_depth.items()):
            tree_percentages = np.asarray(
                [100 * counts[depth][0] / counts[depth][1]
                 for counts in tree_depths[probe].values() if depth in counts],
                dtype=float,
            )
            mean = float(np.mean(tree_percentages))
            sem = float(np.std(tree_percentages, ddof=1) / math.sqrt(len(tree_percentages)))
            margin = float(student_t.ppf(0.975, df=len(tree_percentages) - 1) * sem)
            depth_summary[probe][str(depth)] = {
                "trees": int(len(tree_percentages)),
                "leaves": values[1] // OUTPUTS_PER_LEAF.get(probe, DEFAULT_OUTPUTS_PER_LEAF),
                "empty": values[0],
                "outputs": values[1],
                "percent": 100 * values[0] / values[1],
                "mean_tree_percent": mean,
                "ci_low": mean - margin,
                "ci_high": mean + margin,
            }
    return {
        "validated_leaf_receipts": files,
        "totals": {
            probe: {"empty": v[0], "outputs": v[1], "percent": 100 * v[0] / v[1]}
            for probe, v in totals.items()
        },
        "depth": depth_summary,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DOWNSTREAM_ROOT)
    parser.add_argument("--cohort", default="llama_full")
    args = parser.parse_args(argv)
    print(json.dumps(leaf_empty_rates(args.root, args.cohort), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
