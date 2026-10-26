from __future__ import annotations

import unittest

from weighttraits.behavior.allnode.source_specs import (
    DOWNSTREAM_STAGE,
    FLAN_MODEL,
    FROZEN_STAGE,
    LLAMA_MODEL,
    production_source_specs,
)


class ProductionSourceSpecTests(unittest.TestCase):
    def test_exact_four_distinct_external_cohorts_and_pinned_roots(self) -> None:
        specs = production_source_specs()
        self.assertEqual(
            [spec.cohort_key for spec in specs],
            ["llama_full", "llama_notrans", "flan_full", "flan_notrans"],
        )
        self.assertEqual(len({spec.training_summary.path for spec in specs}), 4)
        self.assertEqual(len({spec.completion_receipt.path for spec in specs}), 4)
        for spec in specs:
            self.assertEqual(spec.stage_root, FROZEN_STAGE)
            self.assertEqual(spec.completion_receipt_root, DOWNSTREAM_STAGE)
            self.assertTrue(spec.top_config.path.startswith(FROZEN_STAGE + "/"))
            self.assertTrue(spec.training_summary.path.startswith(FROZEN_STAGE + "/"))
            self.assertTrue(spec.completion_receipt.path.startswith(DOWNSTREAM_STAGE + "/"))
            self.assertEqual(spec.expected_method, "full")
            self.assertEqual(spec.expected_artifact_name, "model")

    def test_conditions_share_architecture_identity_not_training_sources(self) -> None:
        by_key = {spec.cohort_key: spec for spec in production_source_specs()}
        self.assertEqual(
            by_key["llama_full"].training_cohort_id,
            by_key["llama_notrans"].training_cohort_id,
        )
        self.assertEqual(
            by_key["flan_full"].training_cohort_id,
            by_key["flan_notrans"].training_cohort_id,
        )
        self.assertEqual(by_key["llama_full"].base_model_id, LLAMA_MODEL)
        self.assertEqual(by_key["flan_full"].base_model_id, FLAN_MODEL)
        self.assertNotEqual(
            by_key["llama_full"].training_summary.sha256,
            by_key["llama_notrans"].training_summary.sha256,
        )
        self.assertNotEqual(
            by_key["flan_full"].training_summary.sha256,
            by_key["flan_notrans"].training_summary.sha256,
        )


if __name__ == "__main__":
    unittest.main()
