from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from weighttraits.behavior.allnode.checkpoints import ArtifactPin
from weighttraits.behavior.allnode.provenance import (
    ALLNODE_CODE_MANIFEST_BASENAME,
    ALLNODE_CODE_MANIFEST_SCHEMA,
    DOWNSTREAM_CODE_MANIFEST_BASENAME,
    PRODUCTION_INTERPRETER,
    SOURCE_RESOLUTION_BASENAME,
    SOURCE_RESOLUTION_SCHEMA,
    attest_runtime_provenance,
    production_runtime,
    verify_allnode_code_manifest,
    verify_downstream_code_manifest,
    verify_source_resolution,
)
from weighttraits.behavior.allnode.symlink_inventory import (
    SourcePathAuthoritySpec,
    SourcePathMapping,
    load_source_path_authority,
)


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def canonical_sha(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.stage = Path(self.temp.name).resolve() / "stage"
        self.deployment = self.stage / "deployments" / "allnode_v1"
        (self.stage / "runtime_receipts").mkdir(parents=True)
        (self.deployment / "runtime_receipts").mkdir(parents=True)
        (self.deployment / "weighttraits_allnode").mkdir()
        (self.stage / "src").mkdir()

        self.frozen = Path(self.temp.name).resolve() / "frozen"
        self.frozen.mkdir()
        self.external = Path(self.temp.name).resolve() / "external/llama"
        self.external.mkdir(parents=True)
        self.source_link = self.frozen / "outputs/llama"
        self.source_link.parent.mkdir()
        self.source_link.symlink_to(self.external, target_is_directory=True)
        inventory = self.frozen / ".stage_provenance/SYMLINKS.txt"
        inventory.parent.mkdir()
        inventory.write_text(
            f"outputs/llama -> {self.external}\n",
            encoding="utf-8",
        )
        inventory_sha = file_sha(inventory)
        frozen_manifest = self.frozen / ".stage_provenance/STAGE_SHA256SUMS"
        frozen_manifest.write_text(
            f"{inventory_sha}  .stage_provenance/SYMLINKS.txt\n",
            encoding="utf-8",
        )
        self.authority_spec = SourcePathAuthoritySpec(
            stage_root=str(self.frozen),
            stage_manifest=ArtifactPin(str(frozen_manifest), file_sha(frozen_manifest)),
            symlink_inventory=ArtifactPin(str(inventory), inventory_sha),
            mappings=(
                SourcePathMapping(str(self.source_link), str(self.external)),
            ),
        )
        self.authority = load_source_path_authority(self.authority_spec)
        self.authority_patch = patch(
            "weighttraits.behavior.allnode.provenance.SOURCE_PATH_AUTHORITY",
            self.authority_spec,
        )
        self.authority_patch.start()
        self.addCleanup(self.authority_patch.stop)

        self.downstream_entry = self.stage / "src" / "base.py"
        self.downstream_entry.write_text("BASE = 1\n")
        downstream_text = f"{file_sha(self.downstream_entry)}  src/base.py\n"
        self.downstream_manifest = (
            self.stage / "runtime_receipts" / DOWNSTREAM_CODE_MANIFEST_BASENAME
        )
        self.downstream_manifest.write_text(downstream_text)
        self.downstream_sha = file_sha(self.downstream_manifest)

        self.allnode_entry = self.deployment / "weighttraits_allnode" / "runner.py"
        self.allnode_entry.write_text("VALUE = 1\n")
        rows = [{
            "path": "weighttraits_allnode/runner.py",
            "size_bytes": self.allnode_entry.stat().st_size,
            "sha256": file_sha(self.allnode_entry),
        }]
        core = {
            "schema": ALLNODE_CODE_MANIFEST_SCHEMA,
            "valid": True,
            "deployment_root": str(self.deployment),
            "n_files": 1,
            "files": rows,
            "files_sha256": canonical_sha(rows),
        }
        allnode_payload = {**core, "code_manifest_sha256": canonical_sha(core)}
        self.allnode_manifest = (
            self.deployment / "runtime_receipts" / ALLNODE_CODE_MANIFEST_BASENAME
        )
        self.allnode_manifest.write_bytes(canonical_bytes(allnode_payload))
        self.allnode_sha = file_sha(self.allnode_manifest)

        source_core = {
            "schema": SOURCE_RESOLUTION_SCHEMA,
            "valid": True,
            "read_only": True,
            "scheduler_actions": 0,
            "source_path_authority": self.authority.to_dict(),
            "loads": [{
                "cohort_key": "llama_full",
                "source_path_authority": self.authority.to_dict(),
            }],
            "production_resolution": {"valid": True},
        }
        source_payload = {
            **source_core,
            "source_scope_sha256": canonical_sha(source_core),
        }
        self.source = self.deployment / "runtime_receipts" / SOURCE_RESOLUTION_BASENAME
        self.source.write_bytes(canonical_bytes(source_payload))
        self.source_sha = file_sha(self.source)
        self.source_scope = source_payload["source_scope_sha256"]

    def tearDown(self):
        self.temp.cleanup()

    def test_replays_both_manifests_and_source_receipt(self):
        with patch(
            "weighttraits.behavior.allnode.provenance.DOWNSTREAM_CODE_MANIFEST_SHA256",
            self.downstream_sha,
        ):
            self.assertEqual(
                len(verify_downstream_code_manifest(
                    stage_root=self.stage,
                    manifest_path=self.downstream_manifest,
                    manifest_sha256=self.downstream_sha,
                )),
                1,
            )

    def test_wrong_process_interpreter_fails_closed(self):
        with patch(
            "weighttraits.behavior.allnode.provenance.sys.executable", "/wrong/python"
        ), self.assertRaisesRegex(ValueError, "production interpreter drift"):
            production_runtime()
            self.assertEqual(
                len(verify_allnode_code_manifest(
                    deployment_root=self.deployment,
                    manifest_path=self.allnode_manifest,
                    manifest_sha256=self.allnode_sha,
                )),
                1,
            )
            self.assertEqual(
                verify_source_resolution(
                    deployment_root=self.deployment,
                    receipt_path=self.source,
                    receipt_sha256=self.source_sha,
                    expected_scope_sha256=self.source_scope,
                )["source_scope_sha256"],
                self.source_scope,
            )

    def test_entry_or_scope_drift_fails_closed(self):
        self.allnode_entry.write_text("VALUE = 2\n")
        with self.assertRaisesRegex(ValueError, "entry drift"):
            verify_allnode_code_manifest(
                deployment_root=self.deployment,
                manifest_path=self.allnode_manifest,
                manifest_sha256=self.allnode_sha,
            )
        with self.assertRaisesRegex(ValueError, "scope differs"):
            verify_source_resolution(
                deployment_root=self.deployment,
                receipt_path=self.source,
                receipt_sha256=self.source_sha,
                expected_scope_sha256="0" * 64,
            )

    def test_live_source_link_drift_fails_closed(self):
        other = Path(self.temp.name).resolve() / "external/other"
        other.mkdir()
        self.source_link.unlink()
        self.source_link.symlink_to(other, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "target differs"):
            verify_source_resolution(
                deployment_root=self.deployment,
                receipt_path=self.source,
                receipt_sha256=self.source_sha,
                expected_scope_sha256=self.source_scope,
            )

    def test_combined_attestation_binds_version_and_all_three_receipts(self):
        with patch(
            "weighttraits.behavior.allnode.provenance.DOWNSTREAM_CODE_MANIFEST_SHA256",
            self.downstream_sha,
        ), patch(
                "weighttraits.behavior.allnode.provenance.production_runtime",
                return_value={
                    "interpreter": PRODUCTION_INTERPRETER,
                    "python_version": "3.11.9",
                    "python_implementation": "CPython",
                },
            ):
            receipt = attest_runtime_provenance(
                phase="before-runner",
                stage_root=self.stage,
                deployment_root=self.deployment,
                source_resolution_path=self.source,
                source_resolution_sha256=self.source_sha,
                source_scope_sha256=self.source_scope,
                allnode_code_manifest_path=self.allnode_manifest,
                allnode_code_manifest_sha256=self.allnode_sha,
                downstream_code_manifest_path=self.downstream_manifest,
                downstream_code_manifest_sha256=self.downstream_sha,
                expected_python_version="3.11.9",
            )
        self.assertTrue(receipt["valid"])
        self.assertEqual(receipt["phase"], "before-runner")
        self.assertEqual(receipt["n_allnode_code_entries"], 1)
        self.assertEqual(receipt["n_downstream_code_entries"], 1)


if __name__ == "__main__":
    unittest.main()
