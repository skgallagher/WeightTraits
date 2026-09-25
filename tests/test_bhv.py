import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("dendropy")
pytest.importorskip("networkx")

from weighttraits.phylo import bhv  # noqa: E402

ROOT = Path(__file__).parents[1]


def _tree(internal, pendant=1.0, leaves="abcd"):
    tree = bhv.BHVTree(tuple(sorted(leaves)), {}, {leaf: pendant for leaf in leaves})
    tree.internal = {tree.canonical(frozenset(side)): length for side, length in internal.items()}
    return tree


def test_geodesic_distance_matches_closed_forms():
    ab1 = _tree({"ab": 1.0})
    ab3 = _tree({"ab": 3.0})
    ac2 = _tree({"ac": 2.0})

    assert bhv.geodesic_distance(ab1, ab1) == 0.0
    assert bhv.geodesic_distance(ab1, ab3) == pytest.approx(2.0)
    # incompatible internal edges: the geodesic passes through the star tree (cone path)
    assert bhv.geodesic_distance(ab1, ac2) == pytest.approx(3.0)


def test_frechet_mean_of_identical_trees_is_that_tree():
    tree = _tree({"ab": 0.5})
    mean = bhv.frechet_mean([tree, tree.copy(), tree.copy()], n_epochs=5, seed=0)
    assert set(mean.internal) == set(tree.internal)
    assert mean.internal[frozenset("cd")] == pytest.approx(0.5)  # ab|cd keyed by the side without "a"


def test_newick_round_trip_preserves_splits_and_lengths():
    newick = "((a:1,b:1):0.5,(c:1,d:1):0.25,e:1);"
    tree = bhv.from_newick(newick)
    again = bhv.from_newick(bhv.to_newick(tree))
    assert again.internal == pytest.approx(tree.internal)
    # splits are keyed by the side without the reference leaf "a": ab|cde -> cde
    assert set(again.internal) == {frozenset("cde"), frozenset("cd")}


def test_frechet_mean_does_not_depend_on_the_python_hash_seed():
    script = (
        "from weighttraits.phylo import bhv\n"
        "trees = [bhv.from_newick(t) for t in [\n"
        "  '((a:1,b:1):0.4,(c:1,d:1):0.3,(e:1,f:1):0.2);',\n"
        "  '((a:1,c:1):0.1,(b:1,d:1):0.5,(e:1,f:1):0.3);',\n"
        "  '((a:1,b:1):0.2,((c:1,e:1):0.1,d:1):0.3,f:1);',\n"
        "  '(((a:1,b:1):0.3,c:1):0.2,(d:1,e:1):0.4,f:1);']]\n"
        "print(bhv.to_newick(bhv.frechet_mean(trees, n_epochs=40, seed=0, tol=0.0)))\n"
    )
    outputs = set()
    for hash_seed in ("0", "1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": str(ROOT / "src")}
        result = subprocess.run(
            [sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True
        )
        outputs.add(result.stdout)
    assert len(outputs) == 1
