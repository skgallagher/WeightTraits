from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from weighttraits.behavior.allnode.compatibility_preflight import (
    materialize_leaf_compatibility_preflight,
    plan_available_leaf_compatibility,
    plan_unavailable_leaf_compatibility,
)
from weighttraits.behavior.allnode.direct_compatibility import BASELINE_SCHEMA
from weighttraits.behavior.allnode.direct_execution import DIRECT_METRICS
from weighttraits.behavior.allnode.manifest import canonical_sha256
from weighttraits.behavior.allnode.production import PRODUCTION_TREE_IDS


EXCLUDED = {
    "confirm_paper_tree_015",
    "confirm_paper_tree_020",
    "confirm_paper_tree_037",
    "confirm_paper_tree_047",
}


def recovery(n_truth_splits: int) -> dict[str, object]:
    return {
        "n_truth_leaves": 4,
        "n_estimate_leaves": 4,
        "n_truth_splits": n_truth_splits,
        "n_estimate_splits": n_truth_splits,
        "true_positive": n_truth_splits,
        "false_negative": 0,
        "false_positive": 0,
        "rf": 0,
        "normalized_rf": 0.0,
        "clade_recovery": 1.0,
        "split_precision": 1.0,
        "exact_tree_recovery": True,
        "polytomy_aware_exact_recovery": True,
    }


class CompatibilityPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.legacy = self.root / "legacy"
        self.analysis = self.legacy / "analysis"
        self.analysis.mkdir(parents=True)
        self.table = self.root / "iclr_draft_v4.tex"
        self.table.write_text("immutable old table source\n", encoding="utf-8")
        self.snapshot = self.root / "snapshot" / "src" / "weighttraits"
        engine = self.snapshot / "phylo"
        engine.mkdir(parents=True)
        for name in ("reconstruct.py", "recovery.py", "splits.py"):
            (engine / name).write_text(f"# frozen {name}\n", encoding="utf-8")
        self.runtime_inputs = self.root / "runtime_inputs" / "compatibility"
        self.runtime_inputs.mkdir(parents=True)
        self.destination = self.runtime_inputs / "flan_full"
        self._build_legacy()

    def tearDown(self):
        self.temporary.cleanup()

    def _build_legacy(self):
        rows = []
        model_ids = ["n1", "n2", "n3", "n4"]
        for tree_id in PRODUCTION_TREE_IDS:
            tree_root = self.analysis / tree_id / "model_leaf_analysis"
            tree_root.mkdir(parents=True)
            truth = self.root / "truth" / f"tree{tree_id}.jsonl"
            truth.parent.mkdir(exist_ok=True)
            truth.write_text('{"node_id":"root"}\n', encoding="utf-8")
            (tree_root / "models.json").write_text(
                json.dumps(model_ids) + "\n", encoding="utf-8"
            )
            score_paths = {}
            result_rows = []
            n_truth = 0 if tree_id in EXCLUDED else 1
            for metric in sorted(DIRECT_METRICS):
                score_path = tree_root / f"score_{metric}.json"
                score_path.write_text(
                    json.dumps(recovery(n_truth), sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                score_paths[metric] = score_path
                result_rows.append(
                    {
                        "metric": metric,
                        "score": str(score_path),
                        **{
                            key: value
                            for key, value in recovery(n_truth).items()
                            if key
                            in {
                                "rf",
                                "normalized_rf",
                                "clade_recovery",
                                "split_precision",
                                "exact_tree_recovery",
                                "polytomy_aware_exact_recovery",
                            }
                        },
                    }
                )
            summary = {
                "schema": "weighttraits.direct_analysis_summary.v2",
                "analysis_engine": "direct",
                "artifact": "model",
                "truth_manifest": str(truth),
                "n_models": 4,
                "model_ids": model_ids,
                "metrics": sorted(DIRECT_METRICS),
                "results": result_rows,
            }
            summary_path = tree_root / "summary.json"
            summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
            for metric in DIRECT_METRICS:
                matrix = np.array(
                    [
                        [0.0, 1.0, 2.0, 3.0],
                        [1.0, 0.0, 1.5, 2.5],
                        [2.0, 1.5, 0.0, 1.0],
                        [3.0, 2.5, 1.0, 0.0],
                    ],
                    dtype=np.float64,
                )
                np.save(tree_root / f"distance_matrix_{metric}.npy", matrix)
                rows.append(
                    {
                        "tree_id": tree_id,
                        "metric": metric,
                        "artifact": "model",
                        "n_models": 4,
                        "summary": str(summary_path),
                        "truth_manifest": str(truth),
                        **recovery(n_truth),
                    }
                )
        rollup = {
            "valid": True,
            "artifact": "model",
            "n_tree_summaries": 50,
            "n_rows": 150,
            "tree_ids": list(PRODUCTION_TREE_IDS),
            "metrics": sorted(DIRECT_METRICS),
            "rows": rows,
        }
        (self.legacy / "direct_run_set_summary.json").write_text(
            json.dumps(rollup) + "\n", encoding="utf-8"
        )

    def test_available_plan_is_no_write_and_binds_exact_50x3(self):
        plan = plan_available_leaf_compatibility(
            cohort_key="flan_full",
            legacy_output_root=self.legacy,
            old_table_source=self.table,
            destination_dir=self.destination,
            snapshot_root=self.snapshot,
        )
        self.assertFalse(self.destination.exists())
        self.assertTrue(plan.comparison_available)
        self.assertEqual(len(plan.planned_files), 2)
        self.assertTrue(plan.spec.comparison_available)
        baseline_bytes = next(
            row.content for row in plan.planned_files if row.pin.path.endswith("/baseline.json")
        )
        baseline = json.loads(baseline_bytes)
        self.assertEqual(baseline["schema"], BASELINE_SCHEMA)
        self.assertEqual(baseline["n_rows"], 150)
        self.assertEqual(
            baseline["matched_eligible_tree_ids"],
            [tree_id for tree_id in PRODUCTION_TREE_IDS if tree_id not in EXCLUDED],
        )
        self.assertEqual(
            baseline["rows_sha256"], canonical_sha256(baseline["rows"])
        )
        core = {key: value for key, value in baseline.items() if key != "baseline_sha256"}
        self.assertEqual(baseline["baseline_sha256"], canonical_sha256(core))
        self.assertEqual(
            plan.source_table_input.sha256,
            hashlib.sha256(self.table.read_bytes()).hexdigest(),
        )

    def test_exclusive_atomic_materialization(self):
        plan = plan_available_leaf_compatibility(
            cohort_key="flan_full",
            legacy_output_root=self.legacy,
            old_table_source=self.table,
            destination_dir=self.destination,
            snapshot_root=self.snapshot,
        )
        materialize_leaf_compatibility_preflight(plan)
        self.assertEqual(
            {path.name for path in self.destination.iterdir()},
            {"baseline.json", "old_table_source.tex"},
        )
        with self.assertRaises(FileExistsError):
            materialize_leaf_compatibility_preflight(plan)

    def test_unavailable_plan_is_explicit_pinned_and_no_write(self):
        destination = self.runtime_inputs / "llama_full"
        plan = plan_unavailable_leaf_compatibility(
            cohort_key="llama_full",
            reason="no exact matching legacy leaf-direct cohort exists",
            destination_dir=destination,
        )
        self.assertFalse(destination.exists())
        self.assertFalse(plan.spec.comparison_available)
        self.assertEqual(len(plan.planned_files), 1)
        self.assertIn(b"status=comparison_unavailable", plan.planned_files[0].content)
        materialize_leaf_compatibility_preflight(plan)
        self.assertTrue((destination / "comparison_unavailable.txt").is_file())

    def test_metric_specific_truth_eligibility_drift_fails_closed(self):
        score = (
            self.analysis
            / "confirm_paper_tree_001"
            / "model_leaf_analysis"
            / "score_l2.json"
        )
        score.write_text(json.dumps(recovery(0)) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "n_truth_splits"):
            plan_available_leaf_compatibility(
                cohort_key="flan_full",
                legacy_output_root=self.legacy,
                old_table_source=self.table,
                destination_dir=self.destination,
                snapshot_root=self.snapshot,
            )


if __name__ == "__main__":
    unittest.main()
