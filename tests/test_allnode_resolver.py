from __future__ import annotations

from copy import deepcopy
import hashlib
import unittest

from weighttraits.behavior.allnode.checkpoints import ArtifactPin
from weighttraits.behavior.allnode.resolver import ResolverArtifactPins, resolve_all_node_tree


REVISION = "a" * 40
MODEL_ID = "pinned/base-model"
TREE_ID = "tree001"
STAGE_ROOT = "/frozen/stage"


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def pin(path: str, label: str) -> ArtifactPin:
    return ArtifactPin(path=path, sha256=digest(label))


def _run(
    index: int,
    node_id: str,
    parent_id: str,
    depth: int,
    path: list[str],
    dataset_id: str,
    task_family: str,
    checkpoint: ArtifactPin,
) -> dict:
    output_dir = f"outputs/{TREE_ID}/{node_id}"
    artifacts = {
        "model": f"{output_dir}/model",
        "training_log": f"{output_dir}/training_log.jsonl",
    }
    if checkpoint.path != f"{STAGE_ROOT}/{artifacts['model']}":
        raise AssertionError("fixture checkpoint pin/path drift")
    job = {
        "base_model": MODEL_ID,
        "base_model_revision": REVISION,
        "dataset_id": dataset_id,
        "depth": depth,
        "expected_artifacts": dict(artifacts),
        "init_from": MODEL_ID if parent_id == "root" else f"outputs/{TREE_ID}/{parent_id}/model",
        "lora": None,
        "method": "full",
        "model_family": "production-model-family",
        "node_id": node_id,
        "output_dir": output_dir,
        "parent_id": parent_id,
        "path": list(path),
        "prompt_fields": ["text"],
        "prompt_source": "prompt.dataset",
        "prompt_template": "Classify: {text}",
        "stopping": {"enabled": False},
        "task_family": task_family,
        "trainer": {
            "bf16": True,
            "learning_rate": 2e-5,
            "max_steps": 2000,
            "per_device_train_batch_size": 1,
            "seed": 20260713,
        },
        "training_config_sha256": digest("training-config"),
    }
    return {
        "array_index": index,
        "dataset_id": dataset_id,
        "depth": depth,
        "expected_artifacts": dict(artifacts),
        "init_from": job["init_from"],
        "job": job,
        "ledger_path": f"ledgers/{TREE_ID}.ledger.jsonl",
        "method": "full",
        "node_id": node_id,
        "output_dir": output_dir,
        "parent_id": parent_id,
        "run_id": node_id,
        "runner": {
            "entrypoint": "weighttraits.cli run-training-row",
            "options": {"require_data_cache": True},
            "run_list_path": f"run_lists/{TREE_ID}.runs.jsonl",
            "status": "planned",
        },
        "task_family": task_family,
    }


def _ledger_event(node_id: str, status: str, extra: dict) -> dict:
    return {
        "node_id": node_id,
        "status": status,
        "step": 2000 if status == "completed" else None,
        "train_loss": 0.1 if status == "completed" else None,
        "eval_loss": 0.2 if status == "completed" else None,
        "warnings": [],
        "stop_reasons": [],
        "message": "training row finished" if status == "completed" else "training row started",
        "timestamp": "2026-08-20T00:00:00+00:00",
        "extra": extra,
    }


def fixture() -> dict:
    run_pin = pin(f"{STAGE_ROOT}/run_lists/{TREE_ID}.runs.jsonl", "runs")
    ledger_pin = pin(f"{STAGE_ROOT}/ledgers/{TREE_ID}.ledger.jsonl", "ledger")
    truth_pin = pin(f"{STAGE_ROOT}/truth/{TREE_ID}.truth.jsonl", "truth")
    checkpoints = {
        node_id: pin(f"{STAGE_ROOT}/outputs/{TREE_ID}/{node_id}/model", f"checkpoint/{node_id}")
        for node_id in ("a", "b", "c", "d")
    }
    pins = ResolverArtifactPins(
        top_config=pin(f"{STAGE_ROOT}/config.json", "config"),
        training_summary=pin(f"{STAGE_ROOT}/summary.json", "summary"),
        completion_receipt=pin(f"{STAGE_ROOT}/completion.json", "completion"),
        run_list=run_pin,
        ledger=ledger_pin,
        truth_manifest=truth_pin,
        checkpoints=checkpoints,
    )
    shape = (
        ("a", "root", 1, ["root", "a"], "rte", "classification"),
        ("b", "a", 2, ["root", "a", "b"], "ag_news", "classification"),
        ("c", "a", 2, ["root", "a", "c"], "wmt14", "translation"),
        ("d", "c", 3, ["root", "a", "c", "d"], "xsum", "summarization"),
    )
    runs = [_run(index, *row, checkpoints[row[0]]) for index, row in enumerate(shape)]
    truth = [
        {
            "tree_id": TREE_ID,
            "node_id": node_id,
            "parent_id": parent_id,
            "depth": depth,
            "path": list(path),
            "dataset_id": dataset_id,
            "task_family": task_family,
            "grow": "train",
            "prompt_template": "unused production metadata is allowed",
        }
        for node_id, parent_id, depth, path, dataset_id, task_family in shape
    ]
    ledger = []
    for run in runs:
        node_id = run["node_id"]
        ledger.append(
            _ledger_event(
                node_id,
                "started",
                {
                    "run_id": node_id,
                    "array_index": run["array_index"],
                    "provenance": {"schema": "weighttraits.training_run_provenance.v1"},
                },
            )
        )
        ledger.append(
            _ledger_event(
                node_id,
                "completed",
                {
                    "artifacts": dict(run["expected_artifacts"]),
                    "data": {"valid": True},
                    "execution_overrides": {},
                    "backend_metadata": {"backend": "transformers"},
                    "provenance": {"path": f"{run['output_dir']}/training_provenance.json"},
                },
            )
        )
    summary = {
        "tree_id": TREE_ID,
        "run_list": f"run_lists/{TREE_ID}.runs.jsonl",
        "ledger": f"ledgers/{TREE_ID}.ledger.jsonl",
        "manifest": f"truth/{TREE_ID}.truth.jsonl",
        "output_root": "outputs",
        "valid": True,
        "ready_for_analysis": True,
        "n_runs": 4,
        "n_events": 8,
        "n_ledger_nodes": 4,
        "n_terminal_nodes": 4,
        "n_ok_nodes": 4,
        "n_failed_nodes": 0,
        "n_missing_nodes": 0,
        "n_expected_artifacts": 8,
        "n_existing_artifacts": 8,
        "n_errors": 0,
        "n_warnings": 0,
        "status_counts": {"completed": 4},
        "issue_counts": {},
        "source_extension": "allowed",
    }
    return {
        "cohort_key": "llama_full",
        "training_cohort_id": "shared-llama-r64-training",
        "path_base": STAGE_ROOT,
        "summary_row": summary,
        "run_list_rows": runs,
        "ledger_rows": ledger,
        "truth_rows": truth,
        "observed_pins": pins,
        "expected_method": "full",
        "expected_artifact_name": "model",
        "base_model_id": MODEL_ID,
        "base_model_revision": REVISION,
    }


class ResolverTests(unittest.TestCase):
    def test_resolves_production_rows_and_preserves_observed_ancestors(self) -> None:
        resolved = resolve_all_node_tree(**fixture())

        self.assertEqual(resolved.cohort_key, "llama_full")
        self.assertEqual(resolved.training_cohort_id, "shared-llama-r64-training")
        self.assertEqual(resolved.population, "all_trained_nodes")
        self.assertEqual(resolved.internal_node_treatment, "observed_ancestors")
        self.assertEqual(resolved.topology.ordered_node_ids, ("a", "b", "c", "d"))
        self.assertEqual(
            [(node.node_id, node.parent_id, node.depth, node.node_role) for node in resolved.topology.nodes],
            [
                ("a", "root", 1, "internal"),
                ("b", "a", 2, "leaf"),
                ("c", "a", 2, "internal"),
                ("d", "c", 3, "leaf"),
            ],
        )
        self.assertEqual(
            [row.node_id for row in resolved.checkpoint_manifest.bindings],
            ["a", "b", "c", "d"],
        )
        self.assertNotIn("root", resolved.topology.ordered_node_ids)

    def test_external_key_is_separate_from_shared_training_identity(self) -> None:
        first = resolve_all_node_tree(**fixture())
        second_args = fixture()
        second_args["cohort_key"] = "llama_notrans"
        second = resolve_all_node_tree(**second_args)

        self.assertEqual(first.training_cohort_id, second.training_cohort_id)
        self.assertEqual(
            first.checkpoint_manifest.checkpoint_set_sha256,
            second.checkpoint_manifest.checkpoint_set_sha256,
        )
        self.assertNotEqual(first.resolution_sha256, second.resolution_sha256)

    def test_rejects_reorder_duplicates_root_and_truth_topology_drift(self) -> None:
        args = fixture()
        args["run_list_rows"][0], args["run_list_rows"][1] = (
            args["run_list_rows"][1],
            args["run_list_rows"][0],
        )
        with self.assertRaisesRegex(ValueError, "array_index must equal exact input order"):
            resolve_all_node_tree(**args)

        args = fixture()
        args["run_list_rows"][1] = deepcopy(args["run_list_rows"][3])
        args["run_list_rows"][1]["array_index"] = 1
        with self.assertRaisesRegex(ValueError, "duplicate node IDs"):
            resolve_all_node_tree(**args)

        args = fixture()
        args["truth_rows"][0]["node_id"] = "root"
        args["truth_rows"][0]["path"] = ["root"]
        with self.assertRaisesRegex(ValueError, "run from root|root must not appear"):
            resolve_all_node_tree(**args)

        args = fixture()
        args["truth_rows"][2]["parent_id"] = "b"
        with self.assertRaisesRegex(ValueError, "truth path is inconsistent|parent_id drift"):
            resolve_all_node_tree(**args)

        args = fixture()
        args["run_list_rows"][3]["depth"] = 4
        args["run_list_rows"][3]["job"]["depth"] = 4
        args["truth_rows"][3]["depth"] = 4
        with self.assertRaisesRegex(ValueError, "path is inconsistent with parent/depth"):
            resolve_all_node_tree(**args)

    def test_latest_production_ledger_event_controls_completion(self) -> None:
        args = fixture()
        args["ledger_rows"].append(_ledger_event("c", "running", {"provenance": {}}))
        with self.assertRaisesRegex(ValueError, "latest ledger state.*not terminal completed"):
            resolve_all_node_tree(**args)

        args = fixture()
        for row in args["ledger_rows"]:
            if row["node_id"] == "d" and row["status"] == "completed":
                row["extra"]["artifacts"]["model"] = "outputs/wrong/model"
        with self.assertRaisesRegex(ValueError, "ledger checkpoint artifact drift"):
            resolve_all_node_tree(**args)

        args = fixture()
        for row in args["ledger_rows"]:
            if row["node_id"] == "d" and row["status"] == "completed":
                row["extra"]["execution_overrides"] = {"ROW_END": 1}
        with self.assertRaisesRegex(ValueError, "execution_overrides must be empty"):
            resolve_all_node_tree(**args)

        args = fixture()
        for row in args["ledger_rows"]:
            if row["node_id"] == "d" and row["status"] == "completed":
                del row["extra"]["execution_overrides"]
        with self.assertRaisesRegex(ValueError, "lacks execution_overrides"):
            resolve_all_node_tree(**args)

    def test_rejects_production_training_and_checkpoint_drift(self) -> None:
        mutations = (
            (
                lambda args: args["run_list_rows"][1]["job"]["trainer"].update(max_steps=1999),
                "max_steps=2000",
            ),
            (
                lambda args: (
                    args["run_list_rows"][1].update(method="lora"),
                    args["run_list_rows"][1]["job"].update(method="lora"),
                ),
                "method/artifact drift",
            ),
            (
                lambda args: args["run_list_rows"][1]["job"].update(base_model="other/model"),
                "base model ID drift",
            ),
            (
                lambda args: args["run_list_rows"][1]["job"].update(base_model_revision="b" * 40),
                "base model revision drift",
            ),
            (
                lambda args: (
                    args["run_list_rows"][1]["expected_artifacts"].update(model="wrong/model"),
                    args["run_list_rows"][1]["job"]["expected_artifacts"].update(model="wrong/model"),
                ),
                "checkpoint path drift",
            ),
        )
        for mutation, message in mutations:
            with self.subTest(message=message):
                args = fixture()
                mutation(args)
                with self.assertRaisesRegex(ValueError, message):
                    resolve_all_node_tree(**args)

        args = fixture()
        del args["run_list_rows"][0]["job"]["trainer"]
        with self.assertRaisesRegex(ValueError, "lacks required keys"):
            resolve_all_node_tree(**args)

        args = fixture()
        args["run_list_rows"][0]["execution_overrides"] = {}
        with self.assertRaisesRegex(ValueError, "must not contain execution_overrides"):
            resolve_all_node_tree(**args)

    def test_rejects_summary_truth_and_pin_drift(self) -> None:
        args = fixture()
        args["summary_row"]["valid"] = False
        with self.assertRaisesRegex(ValueError, "valid and ready_for_analysis"):
            resolve_all_node_tree(**args)

        args = fixture()
        args["summary_row"]["status_counts"] = {"completed": 3, "failed": 1}
        with self.assertRaisesRegex(ValueError, "completed-only"):
            resolve_all_node_tree(**args)

        args = fixture()
        args["summary_row"]["run_list"] = "other.jsonl"
        with self.assertRaisesRegex(ValueError, "run_list path differs"):
            resolve_all_node_tree(**args)

        args = fixture()
        del args["truth_rows"][1]["task_family"]
        with self.assertRaisesRegex(ValueError, "lacks required keys"):
            resolve_all_node_tree(**args)

        args = fixture()
        args["truth_rows"].append(deepcopy(args["truth_rows"][1]))
        with self.assertRaisesRegex(ValueError, "duplicate node IDs"):
            resolve_all_node_tree(**args)

        args = fixture()
        args["cohort_key"] = "unregistered"
        with self.assertRaisesRegex(ValueError, "unknown external cohort_key"):
            resolve_all_node_tree(**args)

    def test_stage_relative_resolution_is_bounded_and_absolute_compatible(self) -> None:
        args = fixture()
        args["summary_row"]["run_list"] = "../escape.runs.jsonl"
        with self.assertRaisesRegex(ValueError, "escapes.*path_base"):
            resolve_all_node_tree(**args)

        args = fixture()
        outside = "/frozen/other-stage/outputs/tree001/a/model"
        args["run_list_rows"][0]["expected_artifacts"]["model"] = outside
        args["run_list_rows"][0]["job"]["expected_artifacts"]["model"] = outside
        with self.assertRaisesRegex(ValueError, "not name an artifact under path_base"):
            resolve_all_node_tree(**args)

        args = fixture()
        args["path_base"] = "/frozen/wrong-stage"
        with self.assertRaisesRegex(ValueError, "not name an artifact under path_base"):
            resolve_all_node_tree(**args)

        args = fixture()
        pins = args["observed_pins"]
        args["summary_row"]["run_list"] = pins.run_list.path
        args["summary_row"]["ledger"] = pins.ledger.path
        args["summary_row"]["manifest"] = pins.truth_manifest.path
        for run in args["run_list_rows"]:
            for name, relative in tuple(run["expected_artifacts"].items()):
                absolute = f"{STAGE_ROOT}/{relative}"
                run["expected_artifacts"][name] = absolute
                run["job"]["expected_artifacts"][name] = absolute
        for event in args["ledger_rows"]:
            artifacts = event["extra"].get("artifacts")
            if artifacts:
                for name, relative in tuple(artifacts.items()):
                    artifacts[name] = f"{STAGE_ROOT}/{relative}"
        resolved = resolve_all_node_tree(**args)
        self.assertEqual(resolved.topology.ordered_node_ids, ("a", "b", "c", "d"))


if __name__ == "__main__":
    unittest.main()
