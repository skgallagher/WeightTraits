import json
import math
from pathlib import Path

from weighttraits.phylo.audit import audit_manifest_topology
from weighttraits.phylo.recovery import aggregate_recovery, score_split_recovery
from weighttraits.phylo.splits import splits_from_manifest_path, splits_from_newick_text


EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "recovery"
TRUTH = EXAMPLE_DIR / "truth_polytomy_manifest.jsonl"
NJ_RESOLVES = EXAMPLE_DIR / "nj_resolves_polytomy.newick"
MISSING_CLADE = EXAMPLE_DIR / "missing_clade.newick"
TWO_TIP_TRUTH = EXAMPLE_DIR / "tiny_two_tip_smoke_truth_manifest.jsonl"


def _score_newick(path: Path):
    truth_splits, truth_leaves = splits_from_manifest_path(str(TRUTH))
    estimate_splits, estimate_leaves = splits_from_newick_text(path.read_text())
    return score_split_recovery(truth_splits, estimate_splits, truth_leaves, estimate_leaves)


def test_topology_audit_flags_polytomy_and_binary_nj_fp_risk():
    report = audit_manifest_topology(TRUTH)

    assert report["n_leaves"] == 6
    assert report["n_truth_splits"] == 2
    assert report["n_polytomies"] == 2
    assert report["rf_resolution_notes"]["binary_nj_can_add_false_positive_splits"] is True


def test_nj_resolution_adds_false_positive_without_false_negative():
    score = _score_newick(NJ_RESOLVES)

    assert score["n_truth_splits"] == 2
    assert score["n_estimate_splits"] == 3
    assert score["true_positive"] == 2
    assert score["false_negative"] == 0
    assert score["false_positive"] == 1
    assert score["rf"] == 1
    assert score["clade_recovery"] == 1.0
    assert math.isclose(score["split_precision"], 2 / 3)
    assert score["exact_tree_recovery"] is False
    assert score["polytomy_aware_exact_recovery"] is True
    assert ["n3", "n4"] in score["false_positive_splits"]


def test_missing_clade_counts_false_negative():
    score = _score_newick(MISSING_CLADE)

    assert score["n_truth_splits"] == 2
    assert score["n_estimate_splits"] == 1
    assert score["true_positive"] == 1
    assert score["false_negative"] == 1
    assert score["false_positive"] == 0
    assert score["rf"] == 1
    assert score["clade_recovery"] == 0.5
    assert score["false_negative_rate"] == 0.5
    assert score["polytomy_aware_exact_recovery"] is False


def test_identical_manifest_is_exact_recovery():
    truth_splits, truth_leaves = splits_from_manifest_path(str(TRUTH))
    score = score_split_recovery(truth_splits, truth_splits, truth_leaves, truth_leaves)

    assert score["false_negative"] == 0
    assert score["false_positive"] == 0
    assert score["rf"] == 0
    assert score["clade_recovery"] == 1.0
    assert score["split_precision"] == 1.0
    assert score["exact_tree_recovery"] is True
    assert score["polytomy_aware_exact_recovery"] is True


def test_two_tip_smoke_truth_has_no_informative_splits():
    truth_splits, truth_leaves = splits_from_manifest_path(str(TWO_TIP_TRUTH))
    score = score_split_recovery(truth_splits, truth_splits, truth_leaves, truth_leaves)

    assert truth_leaves == {"n0", "n1"}
    assert truth_splits == set()
    assert score["n_truth_splits"] == 0
    assert score["exact_tree_recovery"] is True


def test_aggregate_recovery_reports_standard_errors():
    scores = [_score_newick(NJ_RESOLVES), _score_newick(MISSING_CLADE)]
    aggregate = aggregate_recovery(scores)

    assert aggregate["n_records"] == 2
    assert aggregate["true_positive_mean"] == 1.5
    assert aggregate["true_positive_se"] == 0.5
    assert aggregate["pooled_true_positive"] == 3
    assert aggregate["pooled_false_negative"] == 1
    assert aggregate["pooled_false_positive"] == 1
    assert aggregate["pooled_truth_splits"] == 4
    assert aggregate["pooled_clade_recovery"] == 0.75
    assert math.isclose(aggregate["pooled_clade_recovery_se"], math.sqrt(0.75 * 0.25 / 4))
    assert aggregate["exact_tree_recovery_rate"] == 0.0
    assert aggregate["exact_tree_recovery_rate_se"] == 0.0
    assert aggregate["polytomy_aware_exact_recovery_rate"] == 0.5
    assert math.isclose(
        aggregate["polytomy_aware_exact_recovery_rate_se"],
        math.sqrt(0.5 * 0.5 / 2),
    )


def test_score_json_is_serializable():
    score = _score_newick(NJ_RESOLVES)
    assert json.loads(json.dumps(score))["false_positive"] == 1
