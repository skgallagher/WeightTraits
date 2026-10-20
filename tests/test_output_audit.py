import json

from weighttraits.behavior.output_audit import audit_cohort, categorize_output


def test_categorize_output_normalizes_case_and_whitespace():
    assert categorize_output("  Entailment \n") == "unrelated_training_label"
    assert categorize_output("sci/tech") == "unrelated_training_label"
    assert categorize_output("B") == "valid_choice_label"
    assert categorize_output("the answer is b") == "other"
    assert categorize_output("") == "other"
    assert categorize_output(None) == "other"


def test_audit_cohort_counts_all_nodes_and_leaves(tmp_path):
    rows = [
        ("n0", "internal", "mmlu", "positive"),
        ("n1", "leaf", "mmlu", "c"),
        ("n1", "leaf", "mmlu", "a longer answer"),
        ("n1", "leaf", "hellaswag", "neutral"),
    ]
    for node, role, probe, text in rows:
        path = tmp_path / "confirm_paper_tree_001" / node / "responses.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            record = {"node_role": role, "text": text, "response_row": {"probe_id": probe}}
            handle.write(json.dumps(record) + "\n")

    probes = audit_cohort(tmp_path)["probes"]

    assert probes["mmlu:all_nodes"]["n"] == 3
    assert probes["mmlu:all_nodes"]["counts"] == {
        "valid_choice_label": 1,
        "unrelated_training_label": 1,
        "other": 1,
    }
    assert probes["mmlu:leaves"]["n"] == 2
    assert probes["hellaswag:all_nodes"]["fractions"]["unrelated_training_label"] == 1.0
