from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from weighttraits.behavior.allnode.checkpoints import ArtifactPin
from weighttraits.behavior.allnode.loader import (
    CohortSourceSpec,
    load_cohort_from_frozen_stage,
    sha256_file,
    sha256_path,
)
from weighttraits.behavior.allnode.symlink_inventory import (
    SourcePathAuthoritySpec,
    SourcePathMapping,
)


MODEL_ID = "pinned/base-model"
REVISION = "a" * 40
TREE_ID = "confirm_paper_tree_001"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _pin(path: Path) -> ArtifactPin:
    return ArtifactPin(path=str(path), sha256=sha256_file(path))


def build_stage(root: Path) -> tuple[CohortSourceSpec, dict[str, Path]]:
    config_path = root / "configs" / "llama.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("training:\n  method: full\n", encoding="utf-8")
    run_path = root / "runs" / f"{TREE_ID}.runs.jsonl"
    ledger_path = root / "ledgers" / f"{TREE_ID}.ledger.jsonl"
    truth_path = root / "truth" / f"{TREE_ID}.manifest.jsonl"
    summary_path = root / "summaries" / "llama.json"
    completion_path = root / "receipts" / "llama.json"

    shape = (
        ("a", "root", 1, ["root", "a"], "rte"),
        ("b", "a", 2, ["root", "a", "b"], "ag_news"),
        ("c", "a", 2, ["root", "a", "c"], "wmt14"),
        ("d", "c", 3, ["root", "a", "c", "d"], "xsum"),
    )
    runs: list[dict] = []
    ledgers: list[dict] = []
    truths: list[dict] = []
    for index, (node, parent, depth, node_path, dataset) in enumerate(shape):
        task_family = "classification" if node in {"a", "b"} else "generation"
        model_path = root / "outputs" / TREE_ID / node / "model"
        model_path.mkdir(parents=True)
        (model_path / "weights.bin").write_bytes(f"weights-{node}".encode())
        log_path = root / "outputs" / TREE_ID / node / "training_log.jsonl"
        log_path.write_text("{}\n", encoding="utf-8")
        artifacts = {
            "model": str(model_path.relative_to(root)),
            "training_log": str(log_path.relative_to(root)),
        }
        job = {
            "node_id": node,
            "parent_id": parent,
            "depth": depth,
            "path": node_path,
            "dataset_id": dataset,
            "task_family": task_family,
            "method": "full",
            "expected_artifacts": deepcopy(artifacts),
            "base_model": MODEL_ID,
            "base_model_revision": REVISION,
            "trainer": {"max_steps": 2000},
        }
        runs.append(
            {
                "array_index": index,
                "node_id": node,
                "parent_id": parent,
                "depth": depth,
                "dataset_id": dataset,
                "task_family": task_family,
                "method": "full",
                "expected_artifacts": deepcopy(artifacts),
                "job": job,
            }
        )
        ledgers.append(
            {
                "node_id": node,
                "status": "completed",
                "extra": {"artifacts": deepcopy(artifacts), "execution_overrides": {}},
            }
        )
        truths.append(
            {
                "tree_id": TREE_ID,
                "node_id": node,
                "parent_id": parent,
                "depth": depth,
                "path": node_path,
                "dataset_id": dataset,
                "task_family": task_family,
            }
        )
    _write_jsonl(run_path, runs)
    _write_jsonl(ledger_path, ledgers)
    _write_jsonl(truth_path, truths)
    summary_tree = {
        "tree_id": TREE_ID,
        "valid": True,
        "n_runs": 4,
        "n_errors": 0,
        "n_warnings": 0,
        "run_list": str(run_path.relative_to(root)),
        "run_list_sha256": sha256_file(run_path),
        "ledger": str(ledger_path.relative_to(root)),
        "manifest": str(truth_path.relative_to(root)),
        "manifest_sha256": sha256_file(truth_path),
    }
    _write_json(
        summary_path,
        {
            "valid": True,
            "config": str(config_path.relative_to(root)),
            "config_sha256": sha256_file(config_path),
            "n_trees": 1,
            "n_runs": 4,
            "n_errors": 0,
            "n_warnings": 0,
            "trees": [summary_tree],
        },
    )
    completion_row = {
        "tree_id": TREE_ID,
        "valid": True,
        "ready_for_analysis": True,
        "n_runs": 4,
        "n_events": 4,
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
        "run_list": str(run_path.relative_to(root)),
        "ledger": str(ledger_path.relative_to(root)),
        "manifest": str(truth_path.relative_to(root)),
    }
    _write_json(
        completion_path,
        {
            "schema": "weighttraits.training_completion_binding.v1",
            "cohort_id": "llama-production",
            "artifact_factor": 2,
            "valid": True,
            "path_base": str(root),
            "summary": str(summary_path.relative_to(root)),
            "n_trees": 1,
            "n_ready": 1,
            "n_failed": 0,
            "n_in_progress": 0,
            "n_not_started": 0,
            "n_total_runs": 4,
            "n_terminal_nodes": 4,
            "n_ok_nodes": 4,
            "n_failed_nodes": 0,
            "n_missing_nodes": 0,
            "n_errors": 0,
            "n_warnings": 0,
            "rows": [completion_row],
        },
    )
    spec = CohortSourceSpec(
        cohort_key="llama_full",
        training_cohort_id="llama-production",
        stage_root=str(root),
        top_config=_pin(config_path),
        training_summary=_pin(summary_path),
        completion_receipt=_pin(completion_path),
        expected_method="full",
        expected_artifact_name="model",
        base_model_id=MODEL_ID,
        base_model_revision=REVISION,
    )
    return spec, {
        "config": config_path,
        "run": run_path,
        "ledger": ledger_path,
        "truth": truth_path,
        "summary": summary_path,
        "completion": completion_path,
    }


def add_llama_output_symlink(
    root: Path,
    spec: CohortSourceSpec,
    *,
    external_root: Path,
) -> tuple[CohortSourceSpec, Path, Path]:
    """Convert the synthetic output tree to the frozen production shape."""

    link = root / "outputs" / TREE_ID
    target = external_root / "llama-production-output"
    target.parent.mkdir(parents=True, exist_ok=True)
    link.rename(target)
    link.symlink_to(target, target_is_directory=True)
    inventory = root / ".stage_provenance" / "SYMLINKS.txt"
    inventory.parent.mkdir(parents=True, exist_ok=True)
    inventory.write_text(
        f"{link.relative_to(root).as_posix()} -> {target}\n",
        encoding="utf-8",
    )
    manifest = root / ".stage_provenance" / "STAGE_SHA256SUMS"
    manifest.write_text(
        f"{sha256_file(inventory)}  .stage_provenance/SYMLINKS.txt\n",
        encoding="utf-8",
    )
    authority = SourcePathAuthoritySpec(
        stage_root=str(root),
        stage_manifest=_pin(manifest),
        symlink_inventory=_pin(inventory),
        mappings=(SourcePathMapping(str(link), str(target)),),
    )
    return (
        replace(
            spec,
            source_authority=authority,
            allowed_symlink_entries=(str(link),),
        ),
        link,
        target,
    )


class LoaderTests(unittest.TestCase):
    def test_loads_complete_read_only_source_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec, _ = build_stage(Path(directory))
            loaded = load_cohort_from_frozen_stage(spec)

        self.assertEqual(len(loaded.trees), 1)
        self.assertEqual(loaded.trees[0].topology.ordered_node_ids, ("a", "b", "c", "d"))
        self.assertEqual(loaded.receipt["n_nodes"], 4)
        self.assertEqual(loaded.receipt["n_internal"], 2)
        self.assertEqual(loaded.receipt["n_leaves"], 2)
        self.assertTrue(loaded.receipt["read_only"])
        self.assertEqual(loaded.receipt["scheduler_actions"], 0)
        self.assertEqual(loaded.receipt["declared_path_base"], str(Path(directory)))
        self.assertEqual(
            loaded.receipt["effective_path_base"],
            str(Path(directory).resolve()),
        )
        self.assertEqual(loaded.receipt["trees"][0]["n_verified_artifacts"], 8)

    def test_accepts_actual_flan_shape_with_absolute_completion_tree_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, paths = build_stage(root)
            completion = json.loads(paths["completion"].read_text())
            completion["path_base"] = "."
            completion["rows"][0]["run_list"] = str(paths["run"].resolve())
            completion["rows"][0]["ledger"] = str(paths["ledger"].resolve())
            # The real completion binding mixes absolute audited run/ledger
            # paths with the summary's relative paths and a relative manifest.
            _write_json(paths["completion"], completion)
            spec = replace(spec, completion_receipt=_pin(paths["completion"]))
            loaded = load_cohort_from_frozen_stage(spec)

        self.assertEqual(loaded.receipt["declared_path_base"], ".")
        self.assertEqual(loaded.receipt["effective_path_base"], str(root.resolve()))
        self.assertEqual(loaded.receipt["n_nodes"], 4)

    def test_binds_summary_config_path_and_hash_to_top_config(self) -> None:
        for field, replacement, message in (
            ("config", "configs/other.yaml", "config path differs"),
            ("config_sha256", "0" * 64, "config_sha256 differs"),
        ):
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    spec, paths = build_stage(root)
                    if field == "config":
                        other = root / replacement
                        other.write_text("training:\n  method: full\n", encoding="utf-8")
                    summary = json.loads(paths["summary"].read_text())
                    summary[field] = replacement
                    _write_json(paths["summary"], summary)
                    spec = replace(spec, training_summary=_pin(paths["summary"]))
                    with self.assertRaisesRegex(ValueError, message):
                        load_cohort_from_frozen_stage(spec)

    def test_binds_completion_schema_cohort_and_artifact_factor(self) -> None:
        cases = (
            ("schema", "weighttraits.training_completion_binding.v0", "schema differs"),
            ("cohort_id", "other-cohort", "cohort_id differs"),
            ("artifact_factor", 4, "artifact_factor differs"),
        )
        for field, replacement, message in cases:
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as directory:
                    spec, paths = build_stage(Path(directory))
                    completion = json.loads(paths["completion"].read_text())
                    completion[field] = replacement
                    _write_json(paths["completion"], completion)
                    spec = replace(spec, completion_receipt=_pin(paths["completion"]))
                    with self.assertRaisesRegex(ValueError, message):
                        load_cohort_from_frozen_stage(spec)

    def test_verifies_model_and_training_log_with_injected_digester(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec, _ = build_stage(Path(directory))
            calls: list[Path] = []

            def recording_digest(path: str | Path) -> str:
                source = Path(path)
                calls.append(source)
                return sha256_path(source)

            loaded = load_cohort_from_frozen_stage(spec, digest_path=recording_digest)

        models = [path for path in calls if path.name == "model"]
        logs = [path for path in calls if path.name == "training_log.jsonl"]
        self.assertEqual(len(models), 4)
        self.assertEqual(len(logs), 4)
        self.assertEqual(loaded.receipt["trees"][0]["n_verified_artifacts"], 8)
        self.assertRegex(loaded.receipt["trees"][0]["artifact_set_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(
            loaded.receipt["trees"][0]["training_log_set_sha256"],
            r"^[0-9a-f]{64}$",
        )

    def test_accepts_actual_llama_shape_only_through_pinned_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            root = base / "frozen"
            spec, _ = build_stage(root)
            spec, link, target = add_llama_output_symlink(
                root, spec, external_root=base / "external"
            )
            loaded = load_cohort_from_frozen_stage(spec)

        authority = loaded.receipt["source_path_authority"]
        self.assertEqual(
            authority["symlink_inventory"]["path"],
            str(root / ".stage_provenance/SYMLINKS.txt"),
        )
        self.assertEqual(len(authority["mappings"]), 1)
        self.assertEqual(
            loaded.receipt["allowed_symlink_entries"],
            [{"logical_prefix": str(link), "physical_root": str(target)}],
        )
        self.assertEqual(loaded.receipt["used_symlink_entries"], [str(link)])
        self.assertEqual(
            loaded.receipt["trees"][0]["symlink_entries_used"], [str(link)]
        )
        for binding in loaded.trees[0].checkpoint_manifest.bindings:
            self.assertTrue(binding.checkpoint.path.startswith(str(target) + "/"))
            self.assertFalse(binding.checkpoint.path.startswith(str(link) + "/"))

    def test_rejects_actual_llama_shape_when_inventory_entry_is_not_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            root = base / "frozen"
            spec, _ = build_stage(root)
            spec, _, _ = add_llama_output_symlink(
                root, spec, external_root=base / "external"
            )
            spec = replace(spec, allowed_symlink_entries=())
            with self.assertRaisesRegex(ValueError, "not allowed"):
                load_cohort_from_frozen_stage(spec)

    def test_rejects_live_llama_symlink_target_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            root = base / "frozen"
            spec, _ = build_stage(root)
            spec, link, target = add_llama_output_symlink(
                root, spec, external_root=base / "external"
            )
            other = base / "external" / "other-output"
            other.mkdir()
            link.unlink()
            link.symlink_to(other, target_is_directory=True)
            self.assertNotEqual(target, other)
            with self.assertRaisesRegex(ValueError, "target differs"):
                load_cohort_from_frozen_stage(spec)

    def test_physical_flan_paths_do_not_use_unrelated_pinned_links(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            root = base / "frozen"
            spec, _ = build_stage(root)
            target = base / "external" / "unused"
            target.mkdir(parents=True)
            link = root / "outputs" / "unused-llama-output"
            link.symlink_to(target, target_is_directory=True)
            inventory = root / ".stage_provenance" / "SYMLINKS.txt"
            inventory.parent.mkdir(parents=True)
            inventory.write_text(
                f"{link.relative_to(root).as_posix()} -> {target}\n",
                encoding="utf-8",
            )
            manifest = root / ".stage_provenance" / "STAGE_SHA256SUMS"
            manifest.write_text(
                f"{sha256_file(inventory)}  .stage_provenance/SYMLINKS.txt\n",
                encoding="utf-8",
            )
            spec = replace(
                spec,
                source_authority=SourcePathAuthoritySpec(
                    stage_root=str(root),
                    stage_manifest=_pin(manifest),
                    symlink_inventory=_pin(inventory),
                    mappings=(SourcePathMapping(str(link), str(target)),),
                ),
                allowed_symlink_entries=(),
            )
            loaded = load_cohort_from_frozen_stage(spec)

        self.assertEqual(loaded.receipt["allowed_symlink_entries"], [])
        self.assertEqual(loaded.receipt["used_symlink_entries"], [])
        self.assertEqual(len(loaded.receipt["source_path_authority"]["mappings"]), 1)

    def test_rejects_full_ledger_artifact_map_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, paths = build_stage(root)
            other = root / "outputs" / "other-training-log.jsonl"
            other.parent.mkdir(parents=True, exist_ok=True)
            other.write_text("{}\n", encoding="utf-8")
            ledger = [json.loads(line) for line in paths["ledger"].read_text().splitlines()]
            ledger[-1]["extra"]["artifacts"]["training_log"] = str(
                other.relative_to(root)
            )
            _write_jsonl(paths["ledger"], ledger)
            with self.assertRaisesRegex(ValueError, "full artifact map drift"):
                load_cohort_from_frozen_stage(spec)

    def test_rejects_model_identity_drift_after_run_list_is_rebound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec, paths = build_stage(Path(directory))
            runs = [json.loads(line) for line in paths["run"].read_text().splitlines()]
            runs[0]["job"]["base_model"] = "other/model"
            _write_jsonl(paths["run"], runs)
            summary = json.loads(paths["summary"].read_text())
            summary["trees"][0]["run_list_sha256"] = sha256_file(paths["run"])
            _write_json(paths["summary"], summary)
            spec = replace(spec, training_summary=_pin(paths["summary"]))
            with self.assertRaisesRegex(ValueError, "base model ID drift"):
                load_cohort_from_frozen_stage(spec)

    def test_rejects_declared_file_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec, paths = build_stage(Path(directory))
            paths["run"].write_text(paths["run"].read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from its declaration"):
                load_cohort_from_frozen_stage(spec)

    def test_rejects_checkpoint_hash_or_symlink_from_digest_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec, _ = build_stage(Path(directory))

            def invalid_digest(path: str | Path) -> str:
                if Path(path).is_dir():
                    return "BAD"
                return sha256_file(path)

            with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
                load_cohort_from_frozen_stage(spec, digest_path=invalid_digest)

    def test_rejects_completion_scope_or_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, paths = build_stage(root)
            completion = json.loads(paths["completion"].read_text())
            completion["n_failed"] = 1
            _write_json(paths["completion"], completion)
            spec = replace(spec, completion_receipt=_pin(paths["completion"]))
            with self.assertRaisesRegex(ValueError, "n_failed must be zero"):
                load_cohort_from_frozen_stage(spec)

    def test_allows_separately_pinned_downstream_completion_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            stage = base / "frozen"
            downstream = base / "downstream"
            spec, paths = build_stage(stage)
            downstream.mkdir()
            destination = downstream / "completion.json"
            destination.write_bytes(paths["completion"].read_bytes())
            paths["completion"].unlink()
            spec = replace(
                spec,
                completion_receipt=_pin(destination),
                completion_receipt_root=str(downstream),
            )
            loaded = load_cohort_from_frozen_stage(spec)

        self.assertEqual(
            loaded.receipt["completion_receipt_root"],
            str(downstream.resolve()),
        )
        self.assertEqual(loaded.receipt["n_nodes"], 4)

    def test_accepts_literal_dot_completion_path_base_as_pinned_stage_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            stage = base / "frozen"
            downstream = base / "downstream"
            spec, paths = build_stage(stage)
            downstream.mkdir()
            completion = json.loads(paths["completion"].read_text())
            completion["path_base"] = "."
            destination = downstream / "completion.json"
            _write_json(destination, completion)
            paths["completion"].unlink()
            spec = replace(
                spec,
                completion_receipt=_pin(destination),
                completion_receipt_root=str(downstream),
            )
            loaded = load_cohort_from_frozen_stage(spec)

        self.assertEqual(loaded.receipt["stage_root"], str(stage.resolve()))
        self.assertEqual(
            loaded.receipt["completion_receipt_root"],
            str(downstream.resolve()),
        )
        self.assertEqual(loaded.receipt["n_nodes"], 4)

    def test_rejects_every_other_relative_completion_path_base(self) -> None:
        for relative in ("frozen", "./", "../frozen", "receipts/.."):
            with self.subTest(path_base=relative):
                with tempfile.TemporaryDirectory() as directory:
                    spec, paths = build_stage(Path(directory))
                    completion = json.loads(paths["completion"].read_text())
                    completion["path_base"] = relative
                    _write_json(paths["completion"], completion)
                    spec = replace(spec, completion_receipt=_pin(paths["completion"]))
                    with self.assertRaisesRegex(
                        ValueError,
                        "relative path_base must be the literal",
                    ):
                        load_cohort_from_frozen_stage(spec)

    def test_rejects_absolute_completion_path_base_for_another_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, paths = build_stage(root / "frozen")
            completion = json.loads(paths["completion"].read_text())
            completion["path_base"] = str(root / "other-frozen")
            _write_json(paths["completion"], completion)
            spec = replace(spec, completion_receipt=_pin(paths["completion"]))
            with self.assertRaisesRegex(ValueError, "differs from frozen stage root"):
                load_cohort_from_frozen_stage(spec)


if __name__ == "__main__":
    unittest.main()
