from weighttraits.paper.condition_compare import compare_conditions, compare_tree_pair

TRUTH = "((a,b,c),(d,e));"  # polytomy over a, b, c


def test_identical_estimates():
    row = compare_tree_pair("(((a,b),c),(d,e));", "(((a,b),c),(d,e));", TRUTH)
    assert row["identical"] and row["left_false_negatives"] == 0


def test_differences_that_only_resolve_a_polytomy():
    row = compare_tree_pair("(((a,b),c),(d,e));", "(((a,c),b),(d,e));", TRUTH)
    assert not row["identical"]
    assert row["differences_only_resolve_polytomies"]
    assert row["left_false_negatives"] == row["right_false_negatives"] == 0
    assert row["left_false_positives"] == row["right_false_positives"] == 1


def test_differences_that_contradict_the_truth():
    row = compare_tree_pair("(((a,b),c),(d,e));", "(((a,d),c),(b,e));", TRUTH)
    assert not row["differences_only_resolve_polytomies"]
    assert row["right_false_negatives"] > 0


def test_summary_counts(tmp_path):
    files = {}
    for name, text in {"t": TRUTH, "l": "(((a,b),c),(d,e));", "r": "(((a,c),b),(d,e));"}.items():
        files[name] = tmp_path / f"{name}.nwk"
        files[name].write_text(text)
    result = compare_conditions(
        [
            ("tree_1", files["l"], files["l"], files["t"]),
            ("tree_2", files["l"], files["r"], files["t"]),
        ]
    )
    summary = result["summary"]
    assert (summary["n_trees"], summary["n_identical"], summary["n_differ"]) == (2, 1, 1)
    assert summary["differing_tree_ids"] == ["tree_2"]
    assert summary["all_differences_only_resolve_polytomies"]
    assert summary["differing_trees_have_equal_fn_and_fp"]
