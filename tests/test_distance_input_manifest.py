import json
from pathlib import Path

import numpy as np
import pytest

from weighttraits.distances.manifest import (
    load_distance_input_manifest,
    readers_from_distance_manifest,
)
from weighttraits.distances.readers import CumulativeLoraReader, SafetensorsTensorReader


EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "distance_inputs"


def test_load_distance_input_manifest_resolves_relative_paths(tmp_path):
    manifest = tmp_path / "inputs.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"model_id": "n0", "checkpoint": "n0/model.safetensors"}),
                json.dumps({"node_id": "n1", "adapter_chain": ["e0", "e1"]}),
            ]
        )
        + "\n"
    )

    specs = load_distance_input_manifest(manifest)

    assert specs[0].model_id == "n0"
    assert specs[0].checkpoint == tmp_path / "n0/model.safetensors"
    assert specs[1].model_id == "n1"
    assert specs[1].adapter_chain == (tmp_path / "e0", tmp_path / "e1")


def test_distance_input_manifest_accepts_yaml_models_key(tmp_path):
    manifest = tmp_path / "inputs.yaml"
    manifest.write_text(
        """
models:
  - id: leaf_a
    path: checkpoints/a.safetensors
  - id: leaf_b
    adapter_chain: adapters/root_to_a,adapters/a_to_b
"""
    )

    specs = load_distance_input_manifest(manifest)

    assert [spec.model_id for spec in specs] == ["leaf_a", "leaf_b"]
    assert specs[0].checkpoint == tmp_path / "checkpoints/a.safetensors"
    assert specs[1].adapter_chain == (
        tmp_path / "adapters/root_to_a",
        tmp_path / "adapters/a_to_b",
    )


def test_readers_from_distance_manifest_builds_checkpoint_and_lora_readers(tmp_path):
    safetensors_np = pytest.importorskip("safetensors.numpy")
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "node0.safetensors"
    safetensors_np.save_file({"layer.weight": np.ones((2, 3), dtype=np.float32)}, str(checkpoint))

    edge0 = _write_adapter(tmp_path / "edge0", offset=0.0)
    edge1 = _write_adapter(tmp_path / "edge1", offset=1.0)
    manifest = tmp_path / "inputs.yaml"
    manifest.write_text(
        f"""
models:
  - model_id: node0
    checkpoint: {checkpoint.relative_to(tmp_path)}
  - model_id: node1
    adapter_chain:
      - {edge0.relative_to(tmp_path)}
      - {edge1.relative_to(tmp_path)}
"""
    )

    readers = readers_from_distance_manifest(manifest)

    assert len(readers) == 2
    assert isinstance(readers[0], SafetensorsTensorReader)
    assert isinstance(readers[1], CumulativeLoraReader)
    assert [reader.model_id for reader in readers] == ["node0", "node1"]


def test_distance_input_manifest_rejects_ambiguous_rows(tmp_path):
    manifest = tmp_path / "bad.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "model_id": "bad",
                    "checkpoint": "model.safetensors",
                    "adapter_chain": ["edge0"],
                }
            ]
        )
    )

    with pytest.raises(ValueError, match="both checkpoint and adapter_chain"):
        load_distance_input_manifest(manifest)


def test_tiny_lineage_output_examples_resolve_expected_paths():
    full = load_distance_input_manifest(EXAMPLE_DIR / "tiny_full_lineage_outputs.yaml")
    lora_merged = load_distance_input_manifest(
        EXAMPLE_DIR / "tiny_lora_merged_lineage_outputs.yaml"
    )
    lora_cumulative = load_distance_input_manifest(
        EXAMPLE_DIR / "tiny_lora_cumulative_lineage_outputs.yaml"
    )

    repo = EXAMPLE_DIR.parents[1]
    assert [spec.model_id for spec in full] == ["n0", "n1"]
    assert _normalized(full[0].checkpoint) == repo / "outputs/tiny_full_lineage_smoke/n0/model"
    assert _normalized(full[1].checkpoint) == repo / "outputs/tiny_full_lineage_smoke/n1/model"

    assert [spec.model_id for spec in lora_merged] == ["n0", "n1"]
    assert _normalized(lora_merged[0].checkpoint) == repo / "outputs/tiny_lora_lineage_smoke/n0/merged"
    assert _normalized(lora_merged[1].checkpoint) == repo / "outputs/tiny_lora_lineage_smoke/n1/merged"

    assert [spec.model_id for spec in lora_cumulative] == ["n0", "n1"]
    assert tuple(_normalized(path) for path in lora_cumulative[0].adapter_chain) == (
        repo / "outputs/tiny_lora_lineage_smoke/n0/adapter",
    )
    assert tuple(_normalized(path) for path in lora_cumulative[1].adapter_chain) == (
        repo / "outputs/tiny_lora_lineage_smoke/n0/adapter",
        repo / "outputs/tiny_lora_lineage_smoke/n1/adapter",
    )


def test_tiny_branching_leaf_output_examples_resolve_expected_paths():
    full = load_distance_input_manifest(EXAMPLE_DIR / "tiny_full_branching_leaf_outputs.yaml")
    lora_merged = load_distance_input_manifest(
        EXAMPLE_DIR / "tiny_lora_merged_branching_leaf_outputs.yaml"
    )
    lora_cumulative = load_distance_input_manifest(
        EXAMPLE_DIR / "tiny_lora_cumulative_branching_leaf_outputs.yaml"
    )

    repo = EXAMPLE_DIR.parents[1]
    assert [spec.model_id for spec in full] == ["n2", "n3", "n4", "n5"]
    assert [_normalized(spec.checkpoint) for spec in full] == [
        repo / "outputs/tiny_full_branching_smoke/n2/model",
        repo / "outputs/tiny_full_branching_smoke/n3/model",
        repo / "outputs/tiny_full_branching_smoke/n4/model",
        repo / "outputs/tiny_full_branching_smoke/n5/model",
    ]

    assert [spec.model_id for spec in lora_merged] == ["n2", "n3", "n4", "n5"]
    assert [_normalized(spec.checkpoint) for spec in lora_merged] == [
        repo / "outputs/tiny_lora_branching_smoke/n2/merged",
        repo / "outputs/tiny_lora_branching_smoke/n3/merged",
        repo / "outputs/tiny_lora_branching_smoke/n4/merged",
        repo / "outputs/tiny_lora_branching_smoke/n5/merged",
    ]

    assert [spec.model_id for spec in lora_cumulative] == ["n2", "n3", "n4", "n5"]
    assert [tuple(_normalized(path) for path in spec.adapter_chain) for spec in lora_cumulative] == [
        (
            repo / "outputs/tiny_lora_branching_smoke/n0/adapter",
            repo / "outputs/tiny_lora_branching_smoke/n2/adapter",
        ),
        (
            repo / "outputs/tiny_lora_branching_smoke/n0/adapter",
            repo / "outputs/tiny_lora_branching_smoke/n3/adapter",
        ),
        (
            repo / "outputs/tiny_lora_branching_smoke/n1/adapter",
            repo / "outputs/tiny_lora_branching_smoke/n4/adapter",
        ),
        (
            repo / "outputs/tiny_lora_branching_smoke/n1/adapter",
            repo / "outputs/tiny_lora_branching_smoke/n5/adapter",
        ),
    ]


def _normalized(path: Path | None) -> Path:
    assert path is not None
    return path.resolve(strict=False)


def _write_adapter(path, *, offset: float):
    safetensors_np = pytest.importorskip("safetensors.numpy")
    path.mkdir()
    (path / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 2}) + "\n")
    safetensors_np.save_file(
        {
            "base_model.model.layer.lora_A.weight": np.array(
                [[1.0 + offset, 0.0, 1.0], [0.0, 1.0, 1.0 + offset]], dtype=np.float32
            ),
            "base_model.model.layer.lora_B.weight": np.array(
                [[1.0, 0.0], [0.5 + offset, 1.0]], dtype=np.float32
            ),
        },
        str(path / "adapter_model.safetensors"),
    )
    return path
