from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from weighttraits.behavior.allnode.checkpoints import ArtifactPin
from weighttraits.behavior.allnode.encoder_snapshot import (
    REFERENCE_IMPLEMENTATION,
    REFERENCE_STUDY_SOURCE_SHA256,
    SCHEME,
    encoder_snapshot_sha256,
    require_encoder_snapshot_sha256,
)
from weighttraits.behavior.allnode.semantic_runner import (
    ENCODER_MODE,
    ENCODER_RECEIPT_SCHEMA,
    MINILM_MODEL,
    MINILM_PACKAGE,
    MINILM_REVISION,
    MINILM_VERSION,
    validate_encoder_receipt,
)


EXPECTED_HF_FIXTURE_SHA256 = (
    "69d9c250c9c5012d725a5e6eddc8d8bd89f9444ffb9332360c9e687c5a50505a"
)


def frozen_weighttraits_reference(path: Path) -> str:
    """Independent transcription of the receipt-producing implementation."""

    root = path.resolve(strict=True)
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"encoder snapshot is empty: {root}")
    rows = []
    for item in files:
        resolved = item.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError(f"encoder snapshot entry is not a file: {item}")
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        rows.append(
            {
                "path": item.relative_to(root).as_posix(),
                "symlink_target": os.readlink(item) if item.is_symlink() else None,
                "size": resolved.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    payload = json.dumps(
        rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class EncoderSnapshotIdentityTests(unittest.TestCase):
    def _hf_snapshot(self, base: Path) -> tuple[Path, Path, Path]:
        cache = base / "cache"
        blobs = cache / "blobs"
        snapshot = cache / "snapshots" / "revision"
        (snapshot / "sub").mkdir(parents=True)
        blobs.mkdir(parents=True)
        config_blob = blobs / "config.json"
        model_blob = blobs / "model"
        config_blob.write_bytes(b'{"hidden_size":384}\n')
        model_blob.write_bytes(b"\x00\x01model-weights\xff")
        (snapshot / "modules.json").write_bytes(b"[]\n")
        os.symlink("../../blobs/config.json", snapshot / "config.json")
        os.symlink("../../../blobs/model", snapshot / "sub" / "model.safetensors")
        return snapshot, config_blob, model_blob

    def test_huggingface_symlink_snapshot_matches_frozen_algorithm_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            snapshot, _, _ = self._hf_snapshot(root)
            observed = encoder_snapshot_sha256(snapshot)
            self.assertEqual(observed, frozen_weighttraits_reference(snapshot))
            self.assertEqual(observed, EXPECTED_HF_FIXTURE_SHA256)

            # The frozen implementation resolves the snapshot root itself.
            alias = root / "snapshot-alias"
            os.symlink(snapshot, alias)
            self.assertEqual(encoder_snapshot_sha256(alias), observed)

        self.assertEqual(
            SCHEME, "weighttraits.behavior.study._encoder_snapshot_sha256.v1"
        )
        self.assertEqual(
            REFERENCE_IMPLEMENTATION,
            "weighttraits.behavior.study._encoder_snapshot_sha256",
        )
        self.assertEqual(len(REFERENCE_STUDY_SOURCE_SHA256), 64)

    def test_content_link_target_and_dangling_link_drift_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            snapshot, config_blob, _ = self._hf_snapshot(Path(raw).resolve())
            expected = encoder_snapshot_sha256(snapshot)
            self.assertEqual(
                require_encoder_snapshot_sha256(snapshot, expected), expected
            )

            config_blob.write_bytes(b'{"hidden_size":385}\n')
            with self.assertRaisesRegex(ValueError, "encoder snapshot hash drift"):
                require_encoder_snapshot_sha256(snapshot, expected)

            config_blob.write_bytes(b'{"hidden_size":384}\n')
            copied_blob = config_blob.with_name("config-copy.json")
            copied_blob.write_bytes(config_blob.read_bytes())
            config_link = snapshot / "config.json"
            config_link.unlink()
            os.symlink("../../blobs/config-copy.json", config_link)
            with self.assertRaisesRegex(ValueError, "encoder snapshot hash drift"):
                require_encoder_snapshot_sha256(snapshot, expected)

            copied_blob.unlink()
            with self.assertRaisesRegex(ValueError, "encoder snapshot hash drift"):
                require_encoder_snapshot_sha256(snapshot, expected)

    def test_empty_snapshot_and_malformed_expected_digest_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            empty = Path(raw).resolve() / "empty"
            empty.mkdir()
            with self.assertRaisesRegex(ValueError, "encoder snapshot is empty"):
                encoder_snapshot_sha256(empty)
            with self.assertRaisesRegex(ValueError, "lowercase hex"):
                require_encoder_snapshot_sha256(empty, "A" * 64)

    def test_semantic_runtime_validation_accepts_symlinks_and_rejects_blob_drift(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            snapshot, config_blob, _ = self._hf_snapshot(root)
            snapshot_pin = ArtifactPin(
                str(snapshot), encoder_snapshot_sha256(snapshot)
            )
            receipt_payload = {
                "schema": ENCODER_RECEIPT_SCHEMA,
                "valid": True,
                "mode": ENCODER_MODE,
                "model": MINILM_MODEL,
                "revision": MINILM_REVISION,
                "snapshot": snapshot_pin.to_dict(),
                "runtime": {
                    "package": MINILM_PACKAGE,
                    "version": MINILM_VERSION,
                },
                "input_config": {"path": "/pinned/minilm_encoder.json", "sha256": "0" * 64},
            }
            receipt = root / "MINILM_ENCODER.json"
            receipt.write_text(
                json.dumps(
                    receipt_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            receipt_pin = ArtifactPin(
                str(receipt), hashlib.sha256(receipt.read_bytes()).hexdigest()
            )
            plan = SimpleNamespace(
                encoder_receipt=receipt_pin,
                encoder_snapshot=snapshot_pin,
            )
            with patch(
                "weighttraits.behavior.allnode.semantic_runner.importlib.metadata.version",
                return_value=MINILM_VERSION,
            ):
                self.assertEqual(validate_encoder_receipt(plan), snapshot)

                config_blob.write_bytes(b'{"hidden_size":999}\n')
                with self.assertRaisesRegex(ValueError, "encoder snapshot hash drift"):
                    validate_encoder_receipt(plan)


if __name__ == "__main__":
    unittest.main()
