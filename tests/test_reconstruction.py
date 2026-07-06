import json

import numpy as np
import pytest

pytest.importorskip("Bio")

from weighttraits.cli import main
from weighttraits.distances.streaming import DistanceCube, write_distance_cube
from weighttraits.phylo.reconstruct import (
    matrix_from_distance_cube,
    neighbor_joining_newick,
    reconstruct_tree_from_cube,
)
from weighttraits.phylo.splits import splits_from_newick_text


def _known_tree_distances(scale: float = 1.0) -> np.ndarray:
    return scale * np.array(
        [
            [0.0, 2.0, 4.0, 4.0],
            [2.0, 0.0, 4.0, 4.0],
            [4.0, 4.0, 0.0, 2.0],
            [4.0, 4.0, 2.0, 0.0],
        ]
    )


def _toy_cube() -> DistanceCube:
    return DistanceCube(
        distances={"l2": np.stack([_known_tree_distances(), _known_tree_distances(scale=2.0)])},
        layer_names=["layer0", "layer1"],
        model_ids=["a", "b", "c", "d"],
        audit={"representation": "synthetic"},
    )


def test_neighbor_joining_recovers_known_unrooted_split():
    newick = neighbor_joining_newick(["a", "b", "c", "d"], _known_tree_distances())
    splits, leaves = splits_from_newick_text(newick)

    assert leaves == {"a", "b", "c", "d"}
    assert frozenset({"a", "b"}) in splits


def test_distance_cube_matrix_selection_supports_layer_and_aggregation():
    cube = _toy_cube()

    mean_matrix, mean_audit = matrix_from_distance_cube(cube, metric="l2", aggregate="mean")
    layer_matrix, layer_audit = matrix_from_distance_cube(cube, metric="l2", layer="layer1")

    np.testing.assert_allclose(mean_matrix, _known_tree_distances(scale=1.5))
    np.testing.assert_allclose(layer_matrix, _known_tree_distances(scale=2.0))
    assert mean_audit["aggregate"] == "mean"
    assert layer_audit["layer"] == "layer1"
    assert layer_audit["layer_index"] == 1


def test_reconstruct_tree_from_persisted_cube(tmp_path):
    cube_dir = tmp_path / "cube"
    write_distance_cube(_toy_cube(), cube_dir)

    result = reconstruct_tree_from_cube(cube_dir, metric="l2")
    splits, _ = splits_from_newick_text(result.newick)

    assert frozenset({"a", "b"}) in splits
    assert result.audit["source_cube"] == str(cube_dir)
    assert result.audit["aggregate"] == "mean"
    assert result.audit["n_models"] == 4
    assert result.audit["distance_max"] == 6.0


def test_reconstruct_tree_cli_writes_newick_and_audit(tmp_path):
    cube_dir = tmp_path / "cube"
    newick_path = tmp_path / "tree.newick"
    audit_path = tmp_path / "tree.audit.json"
    write_distance_cube(_toy_cube(), cube_dir)

    status = main(
        [
            "reconstruct-tree",
            "--cube",
            str(cube_dir),
            "--metric",
            "l2",
            "--layer",
            "0",
            "--out",
            str(newick_path),
            "--audit-out",
            str(audit_path),
        ]
    )

    assert status == 0
    splits, _ = splits_from_newick_text(newick_path.read_text())
    audit = json.loads(audit_path.read_text())
    assert frozenset({"a", "b"}) in splits
    assert audit["layer"] == "layer0"
    assert audit["layer_index"] == 0


def test_reconstruct_tree_rejects_missing_metric():
    with pytest.raises(ValueError, match="available"):
        matrix_from_distance_cube(_toy_cube(), metric="cosine")
