import json
import math

import numpy as np
import pytest

from weighttraits.distances.metrics import pairwise_distance_matrix
from weighttraits.distances.readers import (
    CumulativeLoraReader,
    DictTensorReader,
    LoraFactorReader,
    SafetensorsTensorReader,
    ShardedSafetensorsTensorReader,
    reader_from_path,
)
from weighttraits.distances.streaming import build_distance_cube, write_distance_cube


def _toy_readers():
    base_vec = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 8.0])
    base_mat = np.arange(12, dtype=float).reshape(4, 3)
    return [
        DictTensorReader(
            {
                "layer.vector": base_vec,
                "layer.matrix": base_mat,
            },
            model_id="m0",
        ),
        DictTensorReader(
            {
                "layer.vector": base_vec + np.array([1.0, 0.0, 1.0, 0.0, -1.0, 2.0]),
                "layer.matrix": base_mat + np.array(
                    [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, -1.0, 0.0], [2.0, 0.0, 1.0]]
                ),
            },
            model_id="m1",
        ),
        DictTensorReader(
            {
                "layer.vector": 2.0 * base_vec,
                "layer.matrix": 2.0 * base_mat,
            },
            model_id="m2",
        ),
    ]


def _dense_expected(readers, key, metric, eps=1e-3):
    tensors = [reader.read_tensor(key) for reader in readers]
    if metric == "threshold":
        return pairwise_distance_matrix(tensors, metric="threshold", eps=eps)
    return pairwise_distance_matrix(tensors, metric=metric)


def test_streamed_vector_and_matrix_metrics_match_dense_reference():
    readers = _toy_readers()
    expected_readers = _toy_readers()
    cube = build_distance_cube(
        readers,
        metrics=["cosine", "l1", "l2", "correlation", "threshold", "cka"],
        chunk_size=2,
        eps=0.5,
    )

    assert cube.layer_names == ["layer.vector", "layer.matrix"]
    assert cube.model_ids == ["m0", "m1", "m2"]
    for layer_idx, key in enumerate(cube.layer_names):
        for metric in ["cosine", "l1", "l2", "correlation", "threshold", "cka"]:
            expected = _dense_expected(expected_readers, key, metric, eps=0.5)
            np.testing.assert_allclose(cube.distances[metric][layer_idx], expected, atol=1e-10)


def test_chunk_size_does_not_change_vector_metrics():
    small = build_distance_cube(_toy_readers(), metrics=["cosine", "l2", "correlation"], chunk_size=2)
    large = build_distance_cube(_toy_readers(), metrics=["cosine", "l2", "correlation"], chunk_size=100)

    for metric in ["cosine", "l2", "correlation"]:
        np.testing.assert_allclose(small.distances[metric], large.distances[metric], atol=1e-12)


def test_vector_metrics_use_chunks_not_full_tensor_reads():
    readers = _toy_readers()
    build_distance_cube(readers, metrics=["cosine", "l2"], chunk_size=3)

    for reader in readers:
        assert reader.tensor_reads == {}
        assert set(reader.chunk_reads) == {"layer.vector", "layer.matrix"}


def test_key_mismatch_fails_loudly():
    readers = [
        DictTensorReader({"a": np.ones(3)}, model_id="a"),
        DictTensorReader({"b": np.ones(3)}, model_id="b"),
    ]
    with pytest.raises(ValueError, match="key mismatch"):
        build_distance_cube(readers, metrics=["cosine"])


def test_distance_cube_writer_schema(tmp_path):
    cube = build_distance_cube(_toy_readers(), metrics=["cosine", "cka"], chunk_size=2)
    out = tmp_path / "cube"
    write_distance_cube(cube, out)

    assert (out / "distance_cube.npz").exists()
    assert json.loads((out / "layers.json").read_text()) == ["layer.vector", "layer.matrix"]
    assert json.loads((out / "models.json").read_text()) == ["m0", "m1", "m2"]
    assert json.loads((out / "metrics.json").read_text()) == ["cka", "cosine"]
    audit = json.loads((out / "audit.json").read_text())
    assert audit["metric_execution"]["cosine"] == "chunk_streamed"
    assert audit["metric_execution"]["cka"] == "tensor_at_a_time"
    assert audit["reader_type_by_model"] == {
        "m0": "DictTensorReader",
        "m1": "DictTensorReader",
        "m2": "DictTensorReader",
    }


def test_cumulative_lora_reader_matches_explicit_path_sum():
    edge0 = LoraFactorReader(
        {
            "module.weight": {
                "A": np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0]]),
                "B": np.array([[1.0, 0.0], [0.5, 1.0]]),
            }
        },
        scale=0.5,
        model_id="edge0",
    )
    edge1 = LoraFactorReader(
        {
            "module.weight": {
                "A": np.array([[0.0, 1.0, 1.0], [2.0, 0.0, 1.0]]),
                "B": np.array([[1.0, -1.0], [0.0, 2.0]]),
            }
        },
        scale=2.0,
        model_id="edge1",
    )
    cumulative = CumulativeLoraReader([edge0, edge1], model_id="node")

    expected = edge0.read_tensor("module.weight") + edge1.read_tensor("module.weight")
    np.testing.assert_allclose(cumulative.read_tensor("module.weight"), expected)

    cube = build_distance_cube(
        [CumulativeLoraReader([edge0], model_id="parent"), cumulative],
        metrics=["cosine", "l2"],
        representation="lora_cumulative_delta",
        chunk_size=2,
    )
    assert cube.audit["representation"] == "lora_cumulative_delta"
    assert math.isclose(cube.distances["l2"][0, 0, 0], 0.0)


def test_lora_low_rank_vector_metrics_match_dense_without_chunking():
    edge_a0 = _lora_reader(
        "a0",
        a=np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0]]),
        b=np.array([[1.0, 0.0], [0.5, 1.0]]),
        scale=0.5,
    )
    edge_a1 = _lora_reader(
        "a1",
        a=np.array([[0.0, 1.0, 1.0], [2.0, 0.0, 1.0], [1.0, -1.0, 0.0]]),
        b=np.array([[1.0, -1.0, 0.5], [0.0, 2.0, 1.0]]),
        scale=1.5,
    )
    edge_b0 = _lora_reader(
        "b0",
        a=np.array([[2.0, 0.0, 1.0], [1.0, 1.0, -1.0]]),
        b=np.array([[0.0, 1.0], [1.5, -0.5]]),
        scale=2.0,
    )
    edge_b1 = _lora_reader(
        "b1",
        a=np.array([[1.0, -1.0, 2.0]]),
        b=np.array([[2.0], [-1.0]]),
        scale=0.25,
    )
    readers = [
        CumulativeLoraReader([edge_a0], model_id="parent"),
        CumulativeLoraReader([edge_a0, edge_a1], model_id="child_a"),
        CumulativeLoraReader([edge_b0, edge_b1], model_id="child_b"),
    ]
    dense_readers = [
        DictTensorReader({"module.weight": _lora_tensor(edge_a0)}, model_id="parent"),
        DictTensorReader(
            {"module.weight": _lora_tensor(edge_a0) + _lora_tensor(edge_a1)},
            model_id="child_a",
        ),
        DictTensorReader(
            {"module.weight": _lora_tensor(edge_b0) + _lora_tensor(edge_b1)},
            model_id="child_b",
        ),
    ]

    cube = build_distance_cube(
        readers,
        metrics=["cosine", "l2", "correlation"],
        representation="lora_cumulative_delta",
        chunk_size=1,
    )

    for metric in ["cosine", "l2", "correlation"]:
        expected = _dense_expected(dense_readers, "module.weight", metric)
        np.testing.assert_allclose(cube.distances[metric][0], expected, atol=1e-10)
        assert cube.audit["metric_execution"][metric] == "lora_low_rank"
    for edge in [edge_a0, edge_a1, edge_b0, edge_b1]:
        assert edge.chunk_reads == {}
        assert edge.tensor_reads == {}
        assert edge.low_rank_reads.get("module.weight", 0) >= 1


def test_lora_l1_and_threshold_still_use_chunked_dense_path():
    edge0 = _lora_reader(
        "edge0",
        a=np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0]]),
        b=np.array([[1.0, 0.0], [0.5, 1.0]]),
        scale=0.5,
    )
    edge1 = _lora_reader(
        "edge1",
        a=np.array([[0.0, 1.0, 1.0], [2.0, 0.0, 1.0]]),
        b=np.array([[1.0, -1.0], [0.0, 2.0]]),
        scale=2.0,
    )

    cube = build_distance_cube(
        [
            CumulativeLoraReader([edge0], model_id="parent"),
            CumulativeLoraReader([edge0, edge1], model_id="child"),
        ],
        metrics=["l1", "threshold"],
        representation="lora_cumulative_delta",
        chunk_size=2,
        eps=0.5,
    )

    assert cube.audit["metric_execution"]["l1"] == "chunk_streamed"
    assert cube.audit["metric_execution"]["threshold"] == "chunk_streamed"
    assert edge0.chunk_reads == {"module.weight": 2}
    assert edge1.chunk_reads == {"module.weight": 1}


def test_safetensors_reader_roundtrip_if_available(tmp_path):
    safetensors_np = pytest.importorskip("safetensors.numpy")
    path = tmp_path / "model.safetensors"
    safetensors_np.save_file(
        {
            "a": np.arange(6, dtype=np.float32).reshape(2, 3),
            "b": np.arange(12, dtype=np.float32).reshape(3, 4),
            "c": np.arange(6, dtype=np.float32),
        },
        str(path),
    )

    reader = SafetensorsTensorReader(path, model_id="sf")
    assert reader.keys() == ["a", "b", "c"]
    assert reader.tensor_info("a").shape == (2, 3)
    chunks = list(reader.iter_flat_chunks("a", chunk_size=4))
    assert [chunk.size for chunk in chunks] == [4, 2]
    np.testing.assert_allclose(np.concatenate(chunks), np.arange(6, dtype=float))
    strict_chunks = list(reader.iter_flat_chunks("b", chunk_size=5))
    assert [chunk.size for chunk in strict_chunks] == [5, 5, 2]
    np.testing.assert_allclose(np.concatenate(strict_chunks), np.arange(12, dtype=float))
    vector_chunks = list(reader.iter_flat_chunks("c", chunk_size=4))
    assert [chunk.size for chunk in vector_chunks] == [4, 2]
    np.testing.assert_allclose(np.concatenate(vector_chunks), np.arange(6, dtype=float))

    cube = build_distance_cube(
        [
            DictTensorReader(
                {
                    "a": np.arange(6, dtype=np.float32).reshape(2, 3),
                    "b": np.arange(12, dtype=np.float32).reshape(3, 4),
                    "c": np.arange(6, dtype=np.float32),
                },
                model_id="dict",
            ),
            SafetensorsTensorReader(path, model_id="sf"),
        ],
        metrics=["l2"],
        chunk_size=4,
    )
    assert cube.distances["l2"][0, 0, 1] == 0.0


def test_safetensors_reader_streams_bfloat16_through_torch(tmp_path):
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    path = tmp_path / "model.safetensors"
    expected = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    safetensors_torch.save_file({"weight": expected.to(torch.bfloat16)}, str(path))

    reader = SafetensorsTensorReader(path, model_id="bf16")
    chunks = list(reader.iter_flat_chunks("weight", chunk_size=5))

    assert [chunk.size for chunk in chunks] == [5, 5, 2]
    np.testing.assert_allclose(np.concatenate(chunks), expected.numpy().reshape(-1))


def test_sharded_safetensors_reader_roundtrip_if_available(tmp_path):
    safetensors_np = pytest.importorskip("safetensors.numpy")
    shard0 = tmp_path / "model-00001-of-00002.safetensors"
    shard1 = tmp_path / "model-00002-of-00002.safetensors"
    index = tmp_path / "model.safetensors.index.json"
    tensors = {
        "a": np.arange(6, dtype=np.float32).reshape(2, 3),
        "b": np.arange(12, dtype=np.float32).reshape(3, 4),
        "c": np.arange(6, dtype=np.float32),
    }
    safetensors_np.save_file({"a": tensors["a"], "c": tensors["c"]}, str(shard0))
    safetensors_np.save_file({"b": tensors["b"]}, str(shard1))
    index.write_text(
        json.dumps(
            {
                "metadata": {"total_size": 24 * 4},
                "weight_map": {
                    "a": shard0.name,
                    "b": shard1.name,
                    "c": shard0.name,
                },
            }
        )
        + "\n"
    )

    reader = ShardedSafetensorsTensorReader(tmp_path, model_id="sharded")
    assert reader.keys() == ["a", "b", "c"]
    assert reader.tensor_info("b").shape == (3, 4)
    chunks = list(reader.iter_flat_chunks("b", chunk_size=5))
    assert [chunk.size for chunk in chunks] == [5, 5, 2]
    np.testing.assert_allclose(np.concatenate(chunks), np.arange(12, dtype=float))
    np.testing.assert_allclose(reader.read_tensor("a"), tensors["a"])
    assert isinstance(reader_from_path(tmp_path), ShardedSafetensorsTensorReader)
    assert isinstance(reader_from_path(index), ShardedSafetensorsTensorReader)

    cube = build_distance_cube(
        [
            DictTensorReader(tensors, model_id="dict"),
            ShardedSafetensorsTensorReader(index, model_id="sharded"),
        ],
        metrics=["cosine", "l2"],
        chunk_size=5,
    )
    assert cube.distances["l2"].shape == (3, 2, 2)
    np.testing.assert_allclose(cube.distances["l2"][:, 0, 1], 0.0)


def _lora_reader(name: str, *, a: np.ndarray, b: np.ndarray, scale: float) -> LoraFactorReader:
    return LoraFactorReader({"module.weight": {"A": a, "B": b}}, model_id=name, scale=scale)


def _lora_tensor(reader: LoraFactorReader) -> np.ndarray:
    mats = reader.factors["module.weight"]
    return reader.scale * (mats["B"] @ mats["A"])
