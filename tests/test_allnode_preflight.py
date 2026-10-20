from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import hashlib
import tempfile
import unittest
from unittest.mock import patch

from weighttraits.behavior.allnode.audit import INFERENCE_PLAN_SCHEMA, RESPONSES_PER_NODE
from weighttraits.behavior.allnode.checkpoints import ArtifactPin
from weighttraits.behavior.allnode.checkpoint_identity import SCHEME
from weighttraits.behavior.allnode.gate import COHORTS
from weighttraits.behavior.allnode.gate import canonical_sha256
from weighttraits.behavior.allnode.preflight import (
    LazyCheckpointDigester,
    PlannedFile,
    _planned_file,
    _publish_exclusive,
    backend_template,
    code_manifest_payload,
    inference_plan_payload,
    matrix_plan_payload,
)
from weighttraits.behavior.allnode.symlink_inventory import (
    SourcePathAuthoritySpec,
    SourcePathMapping,
    load_source_path_authority,
)


SHA = "a" * 64


def _source_authority(root: Path, link: Path, target: Path):
    inventory_path = root / ".stage_provenance/SYMLINKS.txt"
    inventory_path.parent.mkdir(exist_ok=True)
    inventory_path.write_text(
        f"{link.relative_to(root).as_posix()} -> {target}\n",
        encoding="utf-8",
    )
    inventory_sha = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
    manifest_path = root / ".stage_provenance/STAGE_SHA256SUMS"
    manifest_path.write_text(
        f"{inventory_sha}  .stage_provenance/SYMLINKS.txt\n",
        encoding="utf-8",
    )
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return load_source_path_authority(
        SourcePathAuthoritySpec(
            stage_root=str(root),
            stage_manifest=ArtifactPin(str(manifest_path), manifest_sha),
            symlink_inventory=ArtifactPin(str(inventory_path), inventory_sha),
            mappings=(SourcePathMapping(str(link), str(target)),),
        )
    )


class ProductionPreflightTests(unittest.TestCase):
    def test_lazy_digester_full_hashes_files_and_samples_directories_once(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            source = root / "source.json"
            source.write_bytes(b"source")
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_bytes(b"{}")
            (checkpoint / "model.bin").write_bytes(b"model")
            unused_target = root.parent / f"{root.name}-unused-target"
            unused_target.mkdir()
            unused_link = root / "unused-link"
            unused_link.symlink_to(unused_target, target_is_directory=True)
            authority = _source_authority(root, unused_link, unused_target)
            digester = LazyCheckpointDigester(
                frozen_stage_root=str(root),
                stage_manifest_sha256=authority.stage_manifest_sha256,
                source_authority=authority,
            )
            self.assertEqual(
                digester(source), hashlib.sha256(b"source").hexdigest()
            )
            first = digester(checkpoint)
            second = digester(checkpoint)
            self.assertEqual(first, second)
            self.assertEqual(len(digester.identities), 1)
            identity = next(iter(digester.identities.values()))
            self.assertEqual(identity.identity_sha256, first)
            self.assertEqual(identity.to_dict()["scheme"], SCHEME)

    def test_lazy_digester_preserves_lexical_checkpoint_under_pinned_output_link(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw).resolve()
            root = base / "frozen"
            root.mkdir()
            target = base / "external/llama"
            checkpoint = target / "tree001/n3/model"
            checkpoint.mkdir(parents=True)
            (checkpoint / "model.bin").write_bytes(b"model")
            link = root / "outputs/llama"
            link.parent.mkdir()
            link.symlink_to(target, target_is_directory=True)
            authority = _source_authority(root, link, target)
            logical = link / "tree001/n3/model"
            digester = LazyCheckpointDigester(
                frozen_stage_root=str(root),
                stage_manifest_sha256=authority.stage_manifest_sha256,
                source_authority=authority,
            )
            digest = digester(checkpoint)

        self.assertEqual(set(digester.identities), {str(checkpoint)})
        identity = digester.identities[str(checkpoint)]
        self.assertEqual(identity.path, str(checkpoint))
        self.assertEqual(identity.logical_path, str(logical))
        self.assertEqual(identity.resolved_path, str(checkpoint))
        self.assertEqual(identity.identity_sha256, digest)

    def test_exact_inference_summary_counts_and_no_overrides(self) -> None:
        payload = inference_plan_payload()
        self.assertEqual(payload["schema"], INFERENCE_PLAN_SCHEMA)
        self.assertEqual(payload["cohort_ids"], list(COHORTS))
        self.assertEqual(payload["n_nodes_per_cohort"], 641)
        for cohort in COHORTS:
            row = payload["cohorts"][cohort]
            self.assertEqual(row["array"], {"start": 0, "end": 640, "max_concurrent": 4})
            self.assertEqual(
                row["expected_response_rows"], 641 * RESPONSES_PER_NODE[cohort]
            )
            self.assertEqual(row["execution_overrides"], {})

    def test_matrix_plan_binds_exact_topology_projection_and_order(self) -> None:
        trees = []
        projection = []
        for index in range(50):
            tree_id = f"confirm_paper_tree_{index + 1:03d}"
            n_nodes = 13 if index < 41 else 12
            n_internal = 6 if index < 26 else 5
            topology = SimpleNamespace(
                tree_id=tree_id,
                n_nodes=n_nodes,
                n_internal=n_internal,
                n_leaves=n_nodes - n_internal,
                topology_sha256=hashlib.sha256(f"t{index}".encode()).hexdigest(),
                nodes_sha256=hashlib.sha256(f"n{index}".encode()).hexdigest(),
                leaf_subset_sha256=hashlib.sha256(f"l{index}".encode()).hexdigest(),
                pair_index_sha256=hashlib.sha256(f"p{index}".encode()).hexdigest(),
                ordered_node_ids=tuple(f"node{j}" for j in range(n_nodes)),
            )
            trees.append(SimpleNamespace(topology=topology))
            projection.append(
                {
                    "tree_id": tree_id,
                    "topology_sha256": topology.topology_sha256,
                    "nodes_sha256": topology.nodes_sha256,
                    "leaf_subset_sha256": topology.leaf_subset_sha256,
                    "pair_index_sha256": topology.pair_index_sha256,
                }
            )
        cohorts = {cohort: trees for cohort in COHORTS}
        payload = matrix_plan_payload(
            cohorts,
            stage_root="/wright/stage",
            checkpoint_payload={"topology_scope_sha256": canonical_sha256(projection)},
        )
        self.assertEqual(payload["n_nodes"], 641)
        self.assertEqual(payload["n_internal"], 276)
        self.assertEqual(payload["n_leaves"], 365)
        self.assertEqual(payload["topology_scope_sha256"], canonical_sha256(projection))
        self.assertEqual(len(payload["topology_scope"]), 50)

    def test_backend_template_uses_runtime_module_versions_and_sentinel(self) -> None:
        modules = {
            "transformers": SimpleNamespace(__version__="5.8.0"),
            "torch": SimpleNamespace(__version__="2.5.1+cu121"),
        }
        binding = SimpleNamespace(base_model_id="model", base_model_revision="b" * 40)
        with patch("weighttraits.behavior.allnode.preflight.importlib.import_module", side_effect=modules.__getitem__):
            causal = backend_template(model_task="causal_lm", binding=binding)
        self.assertEqual(causal["torch_version"], "2.5.1+cu121")
        self.assertEqual(causal["transformers_version"], "5.8.0")
        self.assertEqual(causal["checkpoint"], "$CHECKPOINT_BINDING_PATH")
        self.assertEqual(causal["tokenizer_padding_side"], "left")

    def test_code_manifest_is_sorted_and_changes_with_source_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            for folder in ("weighttraits_allnode", "scripts/slurm", "tests"):
                (root / folder).mkdir(parents=True)
            (root / "weighttraits_allnode" / "b.py").write_text("b\n")
            (root / "weighttraits_allnode" / "a.py").write_text("a\n")
            (root / "scripts" / "slurm" / "job.sbatch").write_text("#!/bin/sh\n")
            (root / "tests" / "test_x.py").write_text("pass\n")
            first = code_manifest_payload(str(root))
            paths = [row["path"] for row in first["files"]]
            self.assertEqual(paths, sorted(paths))
            (root / "weighttraits_allnode" / "a.py").write_text("changed\n")
            second = code_manifest_payload(str(root))
            self.assertNotEqual(
                first["code_manifest_sha256"], second["code_manifest_sha256"]
            )

    def test_external_compatibility_file_and_exclusive_publish(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = str(Path(raw).resolve() / "planned.json")
            content = b"{}"
            pin = SimpleNamespace(
                pin=ArtifactPin(path, hashlib.sha256(content).hexdigest()),
                content=content,
            )
            planned = _planned_file(pin)
            self.assertEqual(planned, PlannedFile(path, content))
            _publish_exclusive(planned)
            self.assertEqual(Path(path).read_bytes(), content)
            with self.assertRaises(FileExistsError):
                _publish_exclusive(planned)


if __name__ == "__main__":
    unittest.main()
