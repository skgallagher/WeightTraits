from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from test_allnode_inference import request
from weighttraits.behavior.allnode.runtime import execute_all_node_inference


def small_runtime_fixture(cohort_key: str = "flan_notrans"):
    """Use the frozen 200-row condition, avoiding the 1,490-row cohort in unit tests."""

    active = request(cohort_key, node_ordinal=0)
    payloads = {
        coordinate: {
            "rendered_prompt": coordinate.prompt_id,
            "protocol_id": coordinate.protocol_id,
            "coordinate": coordinate,
        }
        for coordinate in active.coordinates()
    }
    return active, payloads


class RuntimeExecutionTests(unittest.TestCase):
    def test_executes_internal_node_and_returns_terminal_atomic_write_plan(self) -> None:
        active, payloads = small_runtime_fixture()
        calls = []

        def generate_batch(prompt_batch, group, request_value):
            self.assertEqual(request_value.node_role, "internal")
            self.assertEqual(request_value.parent_id, "root")
            self.assertEqual(group.generation_options["seed_scope"], "probe_draw")
            self.assertEqual(group.generation_options["seed"], 42 + group.sample_id)
            calls.append(group)
            return [
                f"answer:{payload['rendered_prompt']}:{group.sample_id}"
                for payload in prompt_batch
            ]

        result = execute_all_node_inference(
            active,
            prompt_payloads=payloads,
            generate_batch=generate_batch,
            observed_backend_provenance=active.planned_backend_provenance,
            output_path="/planned/responses/tree001/a.jsonl",
            receipt_path="/planned/receipts/tree001/a.json",
        )

        self.assertFalse(result.cache_hit)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0].prompt_ids), 200)
        self.assertEqual(len(result.response_rows), 200)
        self.assertEqual(result.receipt["audit"]["n_expected"], 200)
        self.assertEqual(result.receipt["audit"]["n_missing"], 0)
        self.assertEqual(result.receipt["audit"]["n_extra"], 0)
        self.assertEqual([write.publish_order for write in result.writes], [0, 1])
        self.assertEqual([write.terminal for write in result.writes], [False, True])
        self.assertEqual(
            result.receipt["responses"]["sha256"], result.writes[0].sha256
        )

        first = json.loads(result.writes[0].payload.splitlines()[0])
        self.assertEqual(first["population"], "all_trained_nodes")
        self.assertEqual(first["internal_node_treatment"], "observed_ancestors")
        self.assertEqual(first["node_role"], "internal")
        self.assertEqual(first["parent_id"], "root")
        self.assertNotIn("leaf_ordinal", first)
        self.assertNotIn("leaf_ordinal", first["response_row"])
        terminal = json.loads(result.writes[1].payload)
        self.assertEqual(terminal, result.receipt)

    def test_exact_terminal_cache_is_reused_without_callback_or_writes(self) -> None:
        active, payloads = small_runtime_fixture()
        fresh = execute_all_node_inference(
            active,
            prompt_payloads=payloads,
            generate_batch=lambda batch, group, request_value: [
                payload["rendered_prompt"] for payload in batch
            ],
            observed_backend_provenance=active.planned_backend_provenance,
            output_path="/planned/responses/tree001/a.jsonl",
            receipt_path="/planned/receipts/tree001/a.json",
        )

        def forbidden(*args):
            raise AssertionError("cache validation must not invoke generation")

        cached = execute_all_node_inference(
            active,
            prompt_payloads={},
            generate_batch=forbidden,
            observed_backend_provenance=None,
            output_path="/planned/responses/tree001/a.jsonl",
            receipt_path="/planned/receipts/tree001/a.json",
            existing_output=fresh.writes[0].payload,
            existing_receipt=fresh.writes[1].payload,
        )
        self.assertTrue(cached.cache_hit)
        self.assertEqual(cached.writes, ())
        self.assertEqual(cached.receipt, fresh.receipt)
        self.assertEqual(cached.response_rows, fresh.response_rows)

    def test_refuses_partial_or_tampered_existing_cache(self) -> None:
        active, payloads = small_runtime_fixture()
        with self.assertRaisesRegex(FileExistsError, "exact terminal receipt/cache"):
            execute_all_node_inference(
                active,
                prompt_payloads=payloads,
                generate_batch=lambda batch, group, request_value: ["unused"] * len(batch),
                observed_backend_provenance=active.planned_backend_provenance,
                output_path="/planned/responses/tree001/a.jsonl",
                receipt_path="/planned/receipts/tree001/a.json",
                existing_output=b"occupied\n",
            )

        fresh = execute_all_node_inference(
            active,
            prompt_payloads=payloads,
            generate_batch=lambda batch, group, request_value: [
                payload["rendered_prompt"] for payload in batch
            ],
            observed_backend_provenance=active.planned_backend_provenance,
            output_path="/planned/responses/tree001/a.jsonl",
            receipt_path="/planned/receipts/tree001/a.json",
        )
        tampered = fresh.writes[0].payload.replace(b"translation-prompt-000", b"changed", 1)
        with self.assertRaisesRegex(ValueError, "artifact pin differs"):
            execute_all_node_inference(
                active,
                prompt_payloads={},
                generate_batch=lambda *args: ["unused"],
                observed_backend_provenance=None,
                output_path="/planned/responses/tree001/a.jsonl",
                receipt_path="/planned/receipts/tree001/a.json",
                existing_output=tampered,
                existing_receipt=fresh.writes[1].payload,
            )

    def test_callback_exception_yields_no_files_or_partial_terminal_plan(self) -> None:
        active, payloads = small_runtime_fixture("llama_full")
        calls = 0
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "responses.jsonl"
            receipt = Path(directory) / "receipt.json"

            def fail_after_two(prompt_batch, group, request_value):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise RuntimeError("backend failed")
                return ["temporary"] * len(prompt_batch)

            with self.assertRaisesRegex(RuntimeError, "backend failed"):
                execute_all_node_inference(
                    active,
                    prompt_payloads=payloads,
                    generate_batch=fail_after_two,
                    observed_backend_provenance=active.planned_backend_provenance,
                    output_path=str(output),
                    receipt_path=str(receipt),
                )
            self.assertEqual(calls, 3)
            self.assertFalse(output.exists())
            self.assertFalse(receipt.exists())

    def test_rejects_prompt_grid_and_callback_type_drift(self) -> None:
        active, payloads = small_runtime_fixture()
        missing = dict(payloads)
        missing.pop(next(iter(missing)))
        with self.assertRaisesRegex(ValueError, "prompt payload coordinates differ"):
            execute_all_node_inference(
                active,
                prompt_payloads=missing,
                generate_batch=lambda batch, group, request_value: ["unused"] * len(batch),
                observed_backend_provenance=active.planned_backend_provenance,
                output_path="/planned/responses/tree001/a.jsonl",
                receipt_path="/planned/receipts/tree001/a.json",
            )

        with self.assertRaisesRegex(TypeError, "generate_batch must return text"):
            execute_all_node_inference(
                active,
                prompt_payloads=payloads,
                generate_batch=lambda batch, group, request_value: [42] * len(batch),
                observed_backend_provenance=active.planned_backend_provenance,
                output_path="/planned/responses/tree001/a.jsonl",
                receipt_path="/planned/receipts/tree001/a.json",
            )


if __name__ == "__main__":
    unittest.main()
