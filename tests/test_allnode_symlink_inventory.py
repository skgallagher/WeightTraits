from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from weighttraits.behavior.allnode.checkpoints import ArtifactPin
from weighttraits.behavior.allnode.symlink_inventory import (
    SourcePathAuthoritySpec,
    SourcePathMapping,
    load_frozen_symlink_inventory,
    load_source_path_authority,
    source_path_authority_from_dict,
)


def _file_pin(path: Path) -> ArtifactPin:
    return ArtifactPin(str(path), hashlib.sha256(path.read_bytes()).hexdigest())


class FrozenSymlinkInventoryTests(unittest.TestCase):
    def fixture(self) -> tuple[Path, Path, Path, Path]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name).resolve()
        stage = base / "frozen"
        stage.mkdir()
        external = base / "external"
        full_target = external / "ordinary-llama"
        notrans_target = external / "matched-llama"
        full_target.mkdir(parents=True)
        notrans_target.mkdir()
        full_link = (
            stage
            / "outputs/confirm_paper_numbers/"
            "llama32_1b_full_finetune_legacy_causal_2000"
        )
        notrans_link = (
            stage
            / "outputs/translation_holdout_20260803/"
            "llama32_1b_full_finetune_legacy_causal_2000"
        )
        full_link.parent.mkdir(parents=True)
        notrans_link.parent.mkdir(parents=True)
        full_link.symlink_to(full_target, target_is_directory=True)
        notrans_link.symlink_to(notrans_target, target_is_directory=True)
        inventory = stage / ".stage_provenance/SYMLINKS.txt"
        inventory.parent.mkdir()
        # This is the production inventory's two-entry shape: sorted relative
        # output-root paths and exact absolute targets.
        inventory.write_text(
            f"{full_link.relative_to(stage).as_posix()} -> {full_target}\n"
            f"{notrans_link.relative_to(stage).as_posix()} -> {notrans_target}\n",
            encoding="utf-8",
        )
        return stage, inventory, full_link, notrans_link

    def test_loads_exact_two_llama_output_entries(self) -> None:
        stage, path, full, notrans = self.fixture()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        value = load_frozen_symlink_inventory(
            ArtifactPin(str(path), digest),
            stage_root=str(stage),
            allowed_entry_paths=(str(full), str(notrans)),
        )
        self.assertEqual(value.pin, ArtifactPin(str(path), digest))
        self.assertEqual([entry.path for entry in value.entries], [str(full), str(notrans)])
        self.assertEqual(value.to_dict()["n_entries"], 2)
        self.assertRegex(value.entries_sha256, r"^[0-9a-f]{64}$")

    def test_rejects_descendant_path_that_escapes_through_nested_link(self) -> None:
        stage, path, full, _ = self.fixture()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        value = load_frozen_symlink_inventory(
            ArtifactPin(str(path), digest),
            stage_root=str(stage),
            allowed_entry_paths=(str(full),),
        )
        nested_target = stage.parent / "other"
        nested_target.mkdir()
        (Path(value.entries[0].target) / "nested").symlink_to(
            nested_target, target_is_directory=True
        )
        with self.assertRaisesRegex(ValueError, "resolution differs"):
            value.resolve(str(full / "nested" / "checkpoint"))

    def test_rejects_tab_or_arrow_ambiguity_and_hash_drift(self) -> None:
        stage, path, full, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, "hash drift"):
            load_frozen_symlink_inventory(
                ArtifactPin(str(path), "0" * 64),
                stage_root=str(stage),
                allowed_entry_paths=(str(full),),
            )
        raw = path.read_text()
        path.write_text(raw.replace(" -> ", "\t -> ", 1))
        changed = hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "delimiter"):
            load_frozen_symlink_inventory(
                ArtifactPin(str(path), changed),
                stage_root=str(stage),
                allowed_entry_paths=(str(full),),
            )


class SourcePathAuthorityTests(FrozenSymlinkInventoryTests):
    def authority_fixture(
        self,
    ) -> tuple[SourcePathAuthoritySpec, Path, Path, Path, Path]:
        stage, inventory, full, notrans = self.fixture()
        manifest = stage / ".stage_provenance/STAGE_SHA256SUMS"
        manifest.write_text(
            f"{hashlib.sha256(inventory.read_bytes()).hexdigest()}  "
            ".stage_provenance/SYMLINKS.txt\n",
            encoding="utf-8",
        )
        full_target = Path(full.readlink())
        notrans_target = Path(notrans.readlink())
        spec = SourcePathAuthoritySpec(
            stage_root=str(stage),
            stage_manifest=_file_pin(manifest),
            symlink_inventory=_file_pin(inventory),
            mappings=(
                SourcePathMapping(str(full), str(full_target)),
                SourcePathMapping(str(notrans), str(notrans_target)),
            ),
        )
        return spec, stage, full, notrans, inventory

    def test_attests_exact_two_mapping_production_shape_and_inverts_paths(self) -> None:
        spec, _, full, _, _ = self.authority_fixture()
        checkpoint = Path(full.readlink()) / "confirm_paper_tree_001/n0/model"
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
        authority = load_source_path_authority(spec)
        logical = full / "confirm_paper_tree_001/n0/model"
        resolution = authority.resolve(str(logical))
        self.assertEqual(resolution.logical_path, str(logical))
        self.assertEqual(resolution.physical_path, str(checkpoint))
        self.assertEqual(authority.logical_for_physical(str(checkpoint)), str(logical))
        self.assertEqual(authority.authorize(str(logical), str(checkpoint)), spec.mappings[0])
        self.assertEqual(authority.spec(), spec)
        self.assertEqual(authority.reattest(), authority)
        self.assertEqual(authority.to_dict()["authority_sha256"], spec.authority_sha256)
        self.assertEqual(
            source_path_authority_from_dict(authority.to_dict()), authority
        )

        tampered = authority.to_dict()
        tampered["authority_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "differs from serialized"):
            source_path_authority_from_dict(tampered)

    def test_rejects_any_third_unlisted_stage_symlink(self) -> None:
        spec, stage, _, _, _ = self.authority_fixture()
        target = stage.parent / "external/third"
        target.mkdir()
        (stage / "third").symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink set differs"):
            load_source_path_authority(spec)

    def test_rejects_live_target_or_pinned_mapping_drift(self) -> None:
        spec, stage, full, _, _ = self.authority_fixture()
        other = stage.parent / "external/other"
        other.mkdir()
        full.unlink()
        full.symlink_to(other, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "target differs"):
            load_source_path_authority(spec)

    def test_rejects_nested_symlink_in_authorized_physical_path(self) -> None:
        spec, stage, full, _, _ = self.authority_fixture()
        authority = load_source_path_authority(spec)
        nested_target = stage.parent / "external/nested-target"
        nested_target.mkdir()
        nested = Path(full.readlink()) / "tree/n0/model"
        nested.parent.mkdir(parents=True)
        nested.symlink_to(nested_target, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "nested symlink"):
            authority.resolve(str(full / "tree/n0/model"))

    def test_rejects_manifest_that_does_not_bind_inventory(self) -> None:
        spec, stage, _, _, _ = self.authority_fixture()
        manifest = Path(spec.stage_manifest.path)
        manifest.write_text(f"{'0' * 64}  unrelated\n", encoding="utf-8")
        drifted = SourcePathAuthoritySpec(
            stage_root=spec.stage_root,
            stage_manifest=_file_pin(manifest),
            symlink_inventory=spec.symlink_inventory,
            mappings=spec.mappings,
        )
        with self.assertRaisesRegex(ValueError, "absent or duplicated"):
            load_source_path_authority(drifted)


if __name__ == "__main__":
    unittest.main()
