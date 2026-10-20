from __future__ import annotations

import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from weighttraits.behavior.allnode.checkpoint_identity import (
    FULL_HASH_LIMIT_BYTES,
    SAMPLE_BYTES,
    build_checkpoint_identity,
    build_identity_index,
    read_identity_index,
    source_authority_from_identity,
    validate_checkpoint_identity,
)
from weighttraits.behavior.allnode.checkpoints import ArtifactPin
from weighttraits.behavior.allnode.symlink_inventory import (
    SourcePathAuthoritySpec,
    SourcePathMapping,
    load_source_path_authority,
)


SHA = "a" * 64


class CheckpointIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def checkpoint(self, name: str, *, large: bool = False) -> Path:
        path = self.root / name
        path.mkdir()
        (path / "config.json").write_text('{"x":1}', encoding="utf-8")
        if large:
            with (path / "model.safetensors").open("wb") as handle:
                handle.truncate(FULL_HASH_LIMIT_BYTES + SAMPLE_BYTES + 17)
        else:
            (path / "model.bin").write_bytes(b"model")
        return path

    def test_small_and_large_identity_round_trip_and_reattest(self) -> None:
        small = build_checkpoint_identity(
            self.checkpoint("small"),
            frozen_stage_root=self.root,
            stage_manifest_sha256=SHA,
        )
        large = build_checkpoint_identity(
            self.checkpoint("large", large=True),
            frozen_stage_root=self.root,
            stage_manifest_sha256=SHA,
        )
        self.assertEqual(
            [row["hash_kind"] for row in small.files],
            ["full_sha256", "full_sha256"],
        )
        sampled = [row for row in large.files if row["path"] == "model.safetensors"][0]
        self.assertEqual(sampled["hash_kind"], "fixed_window_sha256")
        self.assertEqual(len(sampled["sample_offsets"]), 5)
        self.assertEqual(validate_checkpoint_identity(small.to_dict(), reattest_filesystem=True), small)

    def test_sampled_drift_is_detected(self) -> None:
        path = self.checkpoint("large", large=True)
        identity = build_checkpoint_identity(path, frozen_stage_root=self.root, stage_manifest_sha256=SHA)
        with (path / "model.safetensors").open("r+b", buffering=0) as handle:
            handle.seek(0)
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError, "filesystem differs"):
            validate_checkpoint_identity(identity.to_dict(), reattest_filesystem=True)

    def test_symlink_and_escape_are_rejected(self) -> None:
        checkpoint = self.checkpoint("base")
        (checkpoint / "link").symlink_to(checkpoint / "config.json")
        with self.assertRaisesRegex(ValueError, "symlink"):
            build_checkpoint_identity(checkpoint, frozen_stage_root=self.root, stage_manifest_sha256=SHA)
        outside = Path(self.temp.name).parent.resolve() / "outside-checkpoint"
        outside.mkdir(exist_ok=True)
        self.addCleanup(lambda: outside.rmdir())
        with self.assertRaisesRegex(ValueError, "strictly below"):
            build_checkpoint_identity(outside, frozen_stage_root=self.root, stage_manifest_sha256=SHA)

    def test_exact_index_round_trip_and_tamper(self) -> None:
        one = build_checkpoint_identity(self.checkpoint("one"), frozen_stage_root=self.root, stage_manifest_sha256=SHA)
        two = build_checkpoint_identity(self.checkpoint("two"), frozen_stage_root=self.root, stage_manifest_sha256=SHA)
        payload = build_identity_index({"a": [one], "b": [two]}, cohort_order=("a", "b"))
        lookup = read_identity_index(payload)
        self.assertEqual(set(lookup), {one.path, two.path})
        tampered = json.loads(json.dumps(payload))
        tampered["cohorts"][0]["entries"][0]["total_bytes"] += 1
        with self.assertRaises(ValueError):
            read_identity_index(tampered)

    def test_external_checkpoint_binds_and_reattests_exact_inventory_entry(self) -> None:
        stage = self.root / "stage"
        stage.mkdir()
        target = self.root / "external-llama"
        checkpoint = target / "tree001/n3/model"
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.json").write_text("{}", encoding="utf-8")
        (checkpoint / "model.bin").write_bytes(b"model")
        link = stage / "outputs/llama"
        link.parent.mkdir()
        link.symlink_to(target, target_is_directory=True)
        inventory_path = stage / ".stage_provenance/SYMLINKS.txt"
        inventory_path.parent.mkdir()
        inventory_path.write_text(f"outputs/llama -> {target}\n", encoding="utf-8")
        inventory_sha = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
        manifest_path = stage / ".stage_provenance/STAGE_SHA256SUMS"
        manifest_path.write_text(
            f"{inventory_sha}  .stage_provenance/SYMLINKS.txt\n",
            encoding="utf-8",
        )
        manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        authority = load_source_path_authority(
            SourcePathAuthoritySpec(
                stage_root=str(stage),
                stage_manifest=ArtifactPin(str(manifest_path), manifest_sha),
                symlink_inventory=ArtifactPin(str(inventory_path), inventory_sha),
                mappings=(SourcePathMapping(str(link), str(target)),),
            )
        )
        logical = link / "tree001/n3/model"
        identity = build_checkpoint_identity(
            checkpoint,
            frozen_stage_root=stage,
            stage_manifest_sha256=manifest_sha,
            source_authority=authority,
            logical_path=logical,
        )
        self.assertEqual(identity.path, str(checkpoint))
        self.assertEqual(identity.logical_path, str(logical))
        self.assertEqual(identity.resolved_path, str(checkpoint))
        self.assertEqual(
            identity.source_authority["symlink_inventory"],
            {"path": str(inventory_path), "sha256": inventory_sha},
        )
        self.assertEqual(
            identity.source_mapping,
            {"logical_prefix": str(link), "physical_root": str(target)},
        )
        self.assertEqual(source_authority_from_identity(identity), authority)
        self.assertEqual(
            validate_checkpoint_identity(
                identity.to_dict(),
                reattest_filesystem=True,
                source_authority=authority,
            ),
            identity,
        )
        tampered = json.loads(json.dumps(identity.to_dict()))
        tampered["source_authority"]["mappings"][0]["physical_root"] = str(
            self.root / "unauthorized-root"
        )
        with self.assertRaisesRegex(ValueError, "digest drift"):
            validate_checkpoint_identity(tampered)

        other = self.root / "other-external"
        other.mkdir()
        link.unlink()
        link.symlink_to(other, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "target differs|authority"):
            validate_checkpoint_identity(identity.to_dict(), reattest_filesystem=True)

    def test_external_checkpoint_requires_exact_authority_and_rejects_nested_link(self) -> None:
        stage = self.root / "stage"
        stage.mkdir()
        target = self.root / "external"
        checkpoint = target / "tree/model"
        checkpoint.mkdir(parents=True)
        (checkpoint / "model.bin").write_bytes(b"model")
        logical_root = stage / "outputs/llama"
        logical_root.parent.mkdir()
        logical_root.symlink_to(target, target_is_directory=True)
        logical = logical_root / "tree/model"
        with self.assertRaisesRegex(ValueError, "authority"):
            build_checkpoint_identity(
                checkpoint,
                frozen_stage_root=stage,
                stage_manifest_sha256=SHA,
                logical_path=logical,
            )

        inventory = stage / ".stage_provenance/SYMLINKS.txt"
        inventory.parent.mkdir()
        inventory.write_text(f"outputs/llama -> {target}\n", encoding="utf-8")
        inventory_sha = hashlib.sha256(inventory.read_bytes()).hexdigest()
        manifest = stage / ".stage_provenance/STAGE_SHA256SUMS"
        manifest.write_text(
            f"{inventory_sha}  .stage_provenance/SYMLINKS.txt\n", encoding="utf-8"
        )
        manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
        authority = load_source_path_authority(
            SourcePathAuthoritySpec(
                str(stage),
                ArtifactPin(str(manifest), manifest_sha),
                ArtifactPin(str(inventory), inventory_sha),
                (SourcePathMapping(str(logical_root), str(target)),),
            )
        )
        unauthorized = self.root / "unauthorized/tree/model"
        unauthorized.mkdir(parents=True)
        (unauthorized / "model.bin").write_bytes(b"model")
        with self.assertRaisesRegex(ValueError, "differs from authority resolution"):
            build_checkpoint_identity(
                unauthorized,
                frozen_stage_root=stage,
                stage_manifest_sha256=manifest_sha,
                source_authority=authority,
                logical_path=logical,
            )
        nested_target = self.root / "nested-external"
        nested_target.mkdir()
        (target / "nested").symlink_to(nested_target, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink|canonical|resolution"):
            build_checkpoint_identity(
                nested_target,
                frozen_stage_root=stage,
                stage_manifest_sha256=manifest_sha,
                source_authority=authority,
                logical_path=logical_root / "nested",
            )


if __name__ == "__main__":
    unittest.main()
