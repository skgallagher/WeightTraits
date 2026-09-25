import json
from pathlib import Path

import numpy as np
import pytest

from weighttraits.paper.hf_families import MethodInputs, per_layer_clade_support, score_family

ELLM = Path(__file__).parents[2] / "ELLMTrees"
LABELS = ["a", "b", "c", "d", "e"]
TRUTH = "((a,b)i1,c,(d,e)i2)root;"


def _additive_matrix():
    # path lengths on the tree ((a,b),c,(d,e)) with unit edges -> perfectly additive
    tree = {("a", "b"): 2, ("d", "e"): 2}
    matrix = np.zeros((5, 5))
    for i, x in enumerate(LABELS):
        for j, y in enumerate(LABELS):
            if i != j:
                matrix[i, j] = tree.get((x, y), tree.get((y, x), 4 if "c" not in (x, y) else 3))
    return matrix


def test_score_family_on_an_additive_toy_family(tmp_path):
    (tmp_path / "truth.nwk").write_text(TRUTH)
    (tmp_path / "est.nwk").write_text("((a,b),c,(d,e));")
    np.save(tmp_path / "d.npy", _additive_matrix())
    (tmp_path / "meta.json").write_text(json.dumps({"model_names": LABELS}))

    result = score_family(
        tmp_path / "truth.nwk",
        {"weights": MethodInputs(tmp_path / "est.nwk", tmp_path / "d.npy", tmp_path / "meta.json")},
    )

    weights = result["methods"]["weights"]
    assert (weights["clades_estimated"], weights["n_documented_clades"]) == (2, 2)
    assert weights["tree_likeness_all_quartets"] == pytest.approx(1.0)


def test_per_layer_support_skips_frozen_layers(tmp_path):
    (tmp_path / "truth.nwk").write_text(TRUTH)
    cube = np.stack([_additive_matrix(), np.zeros((5, 5)), _additive_matrix()])
    support = per_layer_clade_support(cube, LABELS, tmp_path / "truth.nwk")
    assert support["n_active_layers"] == 2
    assert all(row["fraction_of_layer_trees"] == 1.0 for row in support["clade_support"])
    assert all(row["fraction_with_positive_length"] == 1.0 for row in support["clade_support"])


@pytest.mark.skipif(not (ELLM / "results" / "hf_zoo").is_dir(), reason="HF family artifacts absent")
@pytest.mark.parametrize(
    "family,truth,expected",
    [
        ("d_mistral_7b", "d_mistral_7b_resolved", {"weights": (3, 0.86), "phylolm": (3, 0.68)}),
        (
            "c_llama2_7b_chat_small_nodilip",
            "c_llama2_7b_chat_small_nodilip_resolved",
            {"weights": (1, 0.82), "phylolm": (1, 0.64)},
        ),
    ],
)
def test_paper_documented_families(family, truth, expected):
    zoo = ELLM / "results" / "hf_zoo" / family
    result = score_family(
        ELLM / "outputs" / "hf_zoo" / truth / "tree.newick",
        {
            "weights": MethodInputs(
                zoo / "weight_frechet" / "frechet_cosine.newick",
                zoo / "weight_frechet" / "avg_distance_cosine.npy",
                zoo / "weight_frechet" / "distance_metadata.json",
            ),
            "phylolm": MethodInputs(
                zoo / "phylolm" / "phylolm_nj.newick",
                zoo / "phylolm" / "phylolm_dist.npy",
                zoo / "phylolm" / "distance_metadata.json",
            ),
        },
    )
    for method, (clades, tree_likeness) in expected.items():
        scored = result["methods"][method]
        assert scored["false_negatives"] == 0
        assert scored["clades_estimated"] == clades
        assert round(scored["tree_likeness_all_quartets"], 2) == tree_likeness
