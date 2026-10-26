from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from weighttraits.behavior.allnode.checkpoints import ArtifactPin
from weighttraits.behavior.allnode.direct_compatibility import (
    BASELINE_SCHEMA,
    LeafCompatibilityAPI,
    build_leaf_compatibility_spec,
    execute_leaf_compatibility,
)
from weighttraits.behavior.allnode.direct_execution import DIRECT_METRICS
from weighttraits.behavior.allnode.manifest import canonical_sha256
from weighttraits.behavior.allnode.production import PRODUCTION_TREE_IDS


def file_pin(path):
    return ArtifactPin(str(path), hashlib.sha256(path.read_bytes()).hexdigest())


RECOVERY = {
    "n_truth_leaves": 2,
    "n_estimate_leaves": 2,
    "n_truth_splits": 0,
    "n_estimate_splits": 0,
    "true_positive": 0,
    "false_negative": 0,
    "false_positive": 0,
    "rf": 0,
    "normalized_rf": 0.0,
    "clade_recovery": 1.0,
    "split_precision": 1.0,
    "exact_tree_recovery": True,
    "polytomy_aware_exact_recovery": True,
}


class DirectCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.snapshot = self.root / "src" / "weighttraits" / "phylo"
        self.snapshot.mkdir(parents=True)
        self.engine = []
        for name in ("reconstruct.py", "recovery.py", "splits.py"):
            path = self.snapshot / name
            path.write_text(f"# pinned {name}\n", encoding="utf-8")
            self.engine.append(file_pin(path))
        self.rollup = self.root / "old_rollup.json"
        self.table = self.root / "old_table_source.json"
        self.matrix = self.root / "old.npy"
        self.models = self.root / "models.json"
        self.summary = self.root / "summary.json"
        self.truth = self.root / "truth.jsonl"
        self.rollup.write_text("{}", encoding="utf-8")
        self.table.write_text("{}", encoding="utf-8")
        self.matrix.write_bytes(b"fake-npy")
        self.models.write_text('["b","a"]', encoding="utf-8")
        self.summary.write_text("{}", encoding="utf-8")
        self.truth.write_text("{}\n", encoding="utf-8")
        eligible = list(PRODUCTION_TREE_IDS[:46])
        rows = [
            {
                "tree_id": tree_id,
                "metric": metric,
                "topology_eligible": tree_id in eligible,
                "legacy_matrix": file_pin(self.matrix).to_dict(),
                "legacy_model_ids": file_pin(self.models).to_dict(),
                "source_summary": file_pin(self.summary).to_dict(),
                "truth_manifest": file_pin(self.truth).to_dict(),
                "legacy_recovery": dict(RECOVERY),
            }
            for tree_id in PRODUCTION_TREE_IDS
            for metric in DIRECT_METRICS
        ]
        baseline_core = {
            "schema": BASELINE_SCHEMA,
            "valid": True,
            "comparison_available": True,
            "cohort_key": "llama_full",
            "population": "trained_leaves",
            "internal_nodes_as_tips": False,
            "source_direct_rollup": file_pin(self.rollup).to_dict(),
            "source_table_input": file_pin(self.table).to_dict(),
            "n_trees": 50,
            "n_metrics": 3,
            "n_rows": 150,
            "tree_ids": list(PRODUCTION_TREE_IDS),
            "metrics": list(DIRECT_METRICS),
            "matched_eligible_tree_ids": eligible,
            "rows": rows,
            "rows_sha256": canonical_sha256(rows),
        }
        baseline = {
            **baseline_core,
            "baseline_sha256": canonical_sha256(baseline_core),
        }
        self.baseline = self.root / "baseline.json"
        self.baseline.write_text(
            json.dumps(baseline, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        self.spec = build_leaf_compatibility_spec(
            comparison_available=True,
            baseline_manifest=file_pin(self.baseline),
            source_direct_rollup=file_pin(self.rollup),
            source_table_input=file_pin(self.table),
            reconstruct_source=self.engine[0],
            recovery_source=self.engine[1],
            splits_source=self.engine[2],
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_exact_leaf_reorder_nj_and_matched_aggregate(self):
        seen = []

        def neighbor(ids, matrix):
            seen.append((list(ids), matrix))
            return "(a,b);"

        api = LeafCompatibilityAPI(
            load_npy=lambda path: [[0.0, 1.0], [1.0, 0.0]],
            neighbor_joining_newick=neighbor,
            splits_from_manifest_path=lambda path: (set(), {"a", "b"}),
            splits_from_newick_text=lambda text: (set(), {"a", "b"}),
            score_split_recovery=lambda *args: {
                **RECOVERY,
                "n_common_leaves": 2,
                "n_missing_truth_leaves": 0,
                "n_extra_estimate_leaves": 0,
                "false_negative_rate": 0.0,
                "false_discovery_rate": 0.0,
                "exact_tree_recovery_numeric": 1.0,
                "polytomy_aware_exact_recovery_numeric": 1.0,
                "false_negative_splits": [],
                "false_positive_splits": [],
            },
        )

        def matrix_loader(tree_id, metric):
            # Primary order a,b is deliberately opposite the pinned legacy IDs.
            return {
                "schema": "weighttraits.all_node_distance_values.v1",
                "tree_id": tree_id,
                "metric": metric,
                "leaf_sensitivity_ids": ["a", "b"],
                "leaf_sensitivity_matrix": [[0.0, 1.0], [1.0, 0.0]],
            }

        receipt = execute_leaf_compatibility(
            self.spec,
            cohort_key="llama_full",
            direct_plan_sha256="a" * 64,
            matrix_loader=matrix_loader,
            snapshot_root=str(self.root / "src" / "weighttraits"),
            api=api,
        )
        self.assertEqual(receipt["status"], "completed")
        self.assertFalse(receipt["internal_nodes_as_tips"])
        self.assertEqual(receipt["n_rows"], 150)
        self.assertEqual(len(seen), 150)
        self.assertEqual(seen[0][0], ["b", "a"])
        self.assertEqual(receipt["rows"][0]["matrix_delta"]["max_abs"], 0.0)
        for aggregate in receipt["aggregate_matched_eligibility"]:
            self.assertEqual(aggregate["n_matched_eligible"], 46)
            self.assertEqual(aggregate["recovery"]["clade_recovery"]["delta"], 0.0)

    def test_unavailable_requires_and_binds_pinned_reason(self):
        reason = self.root / "reason.txt"
        reason.write_text("no defensible prior baseline\n", encoding="utf-8")
        spec = build_leaf_compatibility_spec(
            comparison_available=False, unavailable_reason=file_pin(reason)
        )
        receipt = execute_leaf_compatibility(
            spec,
            cohort_key="flan_notrans",
            direct_plan_sha256="b" * 64,
            matrix_loader=lambda *_: self.fail("matrix loader must not run"),
            snapshot_root=str(self.root),
        )
        self.assertEqual(receipt["status"], "comparison_unavailable")
        self.assertEqual(receipt["n_rows"], 0)
        with self.assertRaisesRegex(ValueError, "requires a pinned reason"):
            build_leaf_compatibility_spec(comparison_available=False)


if __name__ == "__main__":
    unittest.main()
