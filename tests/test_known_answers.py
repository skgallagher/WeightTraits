"""Known-answer checks from the 2026-09-17 verification (scripts/verify), kept as tests."""

import itertools
import json
import math

import numpy as np
import pytest

from weighttraits.phylo.reconstruct import neighbor_joining_newick
from weighttraits.phylo.recovery import score_split_recovery
from weighttraits.phylo.splits import splits_from_newick_text

# ((A:1,B:2):3,(C:1,D:1,E:2):2,(F:4,G:1):1), where (C,D,E) is a polytomy
EDGES = {
    "A": ("x", 1), "B": ("x", 2), "C": ("y", 1), "D": ("y", 1), "E": ("y", 2),
    "F": ("z", 4), "G": ("z", 1), "x": ("r", 3), "y": ("r", 2), "z": ("r", 1),
}


def _path(node):
    out = {}
    while node in EDGES:
        parent, weight = EDGES[node]
        out[node] = weight
        node = parent
    return out


def test_nj_recovers_an_additive_tree_with_a_polytomy():
    pytest.importorskip("Bio")
    leaves = list("ABCDEFG")
    distances = np.zeros((7, 7))
    for i, j in itertools.combinations(range(7), 2):
        left, right = _path(leaves[i]), _path(leaves[j])
        distance = sum(w for k, w in left.items() if k not in right)
        distance += sum(w for k, w in right.items() if k not in left)
        distances[i, j] = distances[j, i] = distance
    truth = {frozenset("AB"), frozenset("CDE"), frozenset("FG")}

    estimate, _ = splits_from_newick_text(neighbor_joining_newick(leaves, distances))
    score = score_split_recovery(truth, estimate, set(leaves), set(leaves))

    assert truth <= estimate  # every true clade estimated
    assert len(estimate - truth) == 1  # one extra split resolves the polytomy
    assert score["false_negative"] == 0 and score["rf"] == 1 and score["clade_recovery"] == 1.0


def test_distance_cube_builder_matches_closed_form_cosines(tmp_path):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    from weighttraits.cli import main

    a = torch.tensor([1.0, 0.0, 0.0, 0.0])
    b = torch.tensor([0.0, 1.0, 0.0, 0.0])
    c = torch.tensor([1.0, 1.0, 0.0, 0.0]) / math.sqrt(2)
    m = torch.ones(2, 3)
    for name, w, scale in (("A", a, 1), ("B", b, 1), ("C", c, 2)):
        safetensors.save_file({"w": w, "m": scale * m}, str(tmp_path / f"{name}.safetensors"))

    args = ["build-distance-cube", "--metric", "cosine", "--out", str(tmp_path / "out")]
    for name in "ABC":
        args += ["--checkpoint", str(tmp_path / f"{name}.safetensors")]
    assert main(args) == 0

    cube = np.load(tmp_path / "out" / "distance_cube.npz")["cosine"]
    layers = json.loads((tmp_path / "out" / "layers.json").read_text())
    d = 1 - 1 / math.sqrt(2)
    expected = {"w": np.array([[0, 1, d], [1, 0, d], [d, d, 0]]), "m": np.zeros((3, 3))}
    for index, name in enumerate(layers):
        assert np.abs(cube[index] - expected[name]).max() < 1e-12


def test_full_precision_newick_keeps_edges_below_five_decimals():
    pytest.importorskip("Bio")
    from weighttraits.phylo import bhv

    labels = list("abcd")
    distances = np.array(
        [[0, 2e-6, 5e-6, 5e-6], [2e-6, 0, 5e-6, 5e-6], [5e-6, 5e-6, 0, 2e-6], [5e-6, 5e-6, 2e-6, 0]]
    )
    default = bhv.from_newick(neighbor_joining_newick(labels, distances))
    full = bhv.from_newick(
        neighbor_joining_newick(labels, distances, branch_length_format="%.17g")
    )
    # The 3e-6 internal edge is written as 0.00000 and dropped, whatever Biopython's own default.
    assert default.internal == {}
    assert list(full.internal.values()) == pytest.approx([3e-6])
