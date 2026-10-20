import numpy as np
import pytest

safetensors_numpy = pytest.importorskip("safetensors.numpy")

from weighttraits.analysis.direct import _direct_distance_layers  # noqa: E402
from weighttraits.distances.readers import reader_from_path  # noqa: E402


def _model(path, tensors):
    path.mkdir()
    safetensors_numpy.save_file(tensors, str(path / "model.safetensors"))
    return reader_from_path(path, model_id=path.name)


def test_explicit_keys_allow_models_with_extra_tensors(tmp_path):
    rng = np.random.default_rng(0)
    shared = {"a": rng.normal(size=(3, 4)).astype(np.float32), "b": rng.normal(size=5).astype(np.float32)}
    other = {k: v + rng.normal(scale=0.1, size=v.shape).astype(np.float32) for k, v in shared.items()}
    left = _model(tmp_path / "left", shared)
    right = _model(tmp_path / "right", {**other, "rotary.inv_freq": np.ones(2, dtype=np.float32)})
    with pytest.raises(ValueError, match="key mismatch"):
        _direct_distance_layers([left, right], metrics=["cosine"], chunk_size=4, eps=1e-3, artifact="model")

    left = reader_from_path(tmp_path / "left", model_id="left")
    right = reader_from_path(tmp_path / "right", model_id="right")
    distances, layers, models, _ = _direct_distance_layers(
        [left, right], metrics=["cosine"], chunk_size=4, eps=1e-3, artifact="model", keys=["a", "b"]
    )
    assert layers == ["a", "b"] and models == ["left", "right"]
    for index, key in enumerate(layers):
        x, y = shared[key].astype(np.float64).ravel(), other[key].astype(np.float64).ravel()
        expected = 1 - x @ y / (np.linalg.norm(x) * np.linalg.norm(y))
        assert distances["cosine"][index, 0, 1] == pytest.approx(expected, abs=1e-12)

    with pytest.raises(ValueError, match="lacks requested keys"):
        _direct_distance_layers(
            [reader_from_path(tmp_path / "left", model_id="left"),
             reader_from_path(tmp_path / "right", model_id="right")],
            metrics=["cosine"], chunk_size=4, eps=1e-3, artifact="model", keys=["a", "rotary.inv_freq"],
        )
