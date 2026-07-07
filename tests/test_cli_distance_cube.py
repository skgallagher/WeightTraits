from pathlib import Path

from weighttraits.cli import _parse_labeled_paths, build_parser


def test_parse_adapter_chain_paths():
    label, paths = _parse_labeled_paths("child:/tmp/edge0,/tmp/edge1")

    assert label == "child"
    assert paths == [Path("/tmp/edge0"), Path("/tmp/edge1")]


def test_distance_cube_parser_accepts_adapter_chain_without_checkpoint():
    args = build_parser().parse_args(
        [
            "build-distance-cube",
            "--adapter-chain",
            "child:/tmp/edge0,/tmp/edge1",
            "--representation",
            "lora_cumulative_delta",
            "--metric",
            "cosine",
            "--out",
            "/tmp/cube",
        ]
    )

    assert args.checkpoint is None
    assert args.adapter_chain == ["child:/tmp/edge0,/tmp/edge1"]


def test_distance_cube_parser_accepts_checkpoint_manifest():
    args = build_parser().parse_args(
        [
            "build-distance-cube",
            "--checkpoint-manifest",
            "/tmp/inputs.yaml",
            "--metric",
            "l2",
            "--out",
            "/tmp/cube",
        ]
    )

    assert args.checkpoint_manifest == [Path("/tmp/inputs.yaml")]


def test_make_distance_input_manifest_parser_accepts_ledger_and_truth_manifest():
    args = build_parser().parse_args(
        [
            "make-distance-input-manifest",
            "--ledger",
            "/tmp/training_ledger.jsonl",
            "--truth-manifest",
            "/tmp/truth.jsonl",
            "--artifact",
            "adapter_chain",
            "--out",
            "/tmp/distance_inputs.yaml",
        ]
    )

    assert args.ledger == Path("/tmp/training_ledger.jsonl")
    assert args.truth_manifest == Path("/tmp/truth.jsonl")
    assert args.artifact == "adapter_chain"
    assert args.out == Path("/tmp/distance_inputs.yaml")


def test_generate_tree_set_parser_accepts_batch_options():
    args = build_parser().parse_args(
        [
            "generate-tree-set",
            "--config",
            "examples/trees/confirm_paper_numbers.yaml",
            "--out-dir",
            "/tmp/trees",
            "--summary-out",
            "/tmp/tree_set_summary.json",
            "--n-trees",
            "50",
            "--seed-start",
            "20260707",
            "--min-leaves",
            "4",
        ]
    )

    assert args.config == Path("examples/trees/confirm_paper_numbers.yaml")
    assert args.out_dir == Path("/tmp/trees")
    assert args.summary_out == Path("/tmp/tree_set_summary.json")
    assert args.n_trees == 50
    assert args.seed_start == 20260707
    assert args.min_leaves == 4


def test_assign_task_data_set_parser_accepts_batch_options():
    args = build_parser().parse_args(
        [
            "assign-task-data-set",
            "--tree-set",
            "examples/training/confirm_paper_numbers/tree_set_summary.json",
            "--config",
            "examples/training/confirm_paper_numbers/paper_task_families.yaml",
            "--out-dir",
            "/tmp/assigned",
            "--summary-out",
            "/tmp/assignment_summary.json",
            "--seed-start",
            "101",
            "--policy",
            "per_node_without_replacement",
        ]
    )

    assert args.tree_set == Path("examples/training/confirm_paper_numbers/tree_set_summary.json")
    assert args.config == Path("examples/training/confirm_paper_numbers/paper_task_families.yaml")
    assert args.out_dir == Path("/tmp/assigned")
    assert args.summary_out == Path("/tmp/assignment_summary.json")
    assert args.seed_start == 101
    assert args.policy == "per_node_without_replacement"


def test_analyze_training_ledger_parser_accepts_workflow_options():
    args = build_parser().parse_args(
        [
            "analyze-training-ledger",
            "--ledger",
            "/tmp/training_ledger.jsonl",
            "--truth-manifest",
            "/tmp/truth.jsonl",
            "--artifact",
            "model",
            "--metric",
            "l2",
            "--metric",
            "cosine",
            "--out",
            "/tmp/whitebox",
        ]
    )

    assert args.ledger == Path("/tmp/training_ledger.jsonl")
    assert args.truth_manifest == Path("/tmp/truth.jsonl")
    assert args.artifact == "model"
    assert args.metric == ["l2", "cosine"]
    assert args.out == Path("/tmp/whitebox")


def test_make_recovery_table_parser_accepts_registry_outputs():
    args = build_parser().parse_args(
        [
            "make-recovery-table",
            "--registry",
            "paper/recovery_registry.yaml",
            "--base-dir",
            "/tmp/run",
            "--out",
            "/tmp/recovery.json",
            "--csv-out",
            "/tmp/recovery.csv",
        ]
    )

    assert args.registry == Path("paper/recovery_registry.yaml")
    assert args.base_dir == Path("/tmp/run")
    assert args.out == Path("/tmp/recovery.json")
    assert args.csv_out == Path("/tmp/recovery.csv")


def test_make_ellmtrees_variants_table_parser_accepts_registry_outputs():
    args = build_parser().parse_args(
        [
            "make-ellmtrees-variants-table",
            "--registry",
            "paper/ellmtrees_variants_registry.yaml",
            "--base-dir",
            "/tmp/run",
            "--out",
            "/tmp/variants.json",
            "--csv-out",
            "/tmp/variants.csv",
        ]
    )

    assert args.registry == Path("paper/ellmtrees_variants_registry.yaml")
    assert args.base_dir == Path("/tmp/run")
    assert args.out == Path("/tmp/variants.json")
    assert args.csv_out == Path("/tmp/variants.csv")


def test_make_behavior_holdout_table_parser_accepts_outputs():
    args = build_parser().parse_args(
        [
            "make-behavior-holdout-table",
            "--draft",
            "../ELLMTrees-paper/iclr_draft_v2.tex",
            "--out",
            "/tmp/behavior.json",
            "--csv-out",
            "/tmp/behavior.csv",
        ]
    )

    assert args.draft == Path("../ELLMTrees-paper/iclr_draft_v2.tex")
    assert args.out == Path("/tmp/behavior.json")
    assert args.csv_out == Path("/tmp/behavior.csv")


def test_validate_table_registry_parser_accepts_output_checks():
    args = build_parser().parse_args(
        [
            "validate-table-registry",
            "--registry",
            "paper/table_registry.yaml",
            "--base-dir",
            "/tmp/run",
            "--require-outputs",
            "--out",
            "/tmp/table_registry_audit.json",
        ]
    )

    assert args.registry == Path("paper/table_registry.yaml")
    assert args.base_dir == Path("/tmp/run")
    assert args.require_outputs is True
    assert args.out == Path("/tmp/table_registry_audit.json")


def test_validate_reference_registry_parser_accepts_digest_checks():
    args = build_parser().parse_args(
        [
            "validate-reference-registry",
            "--registry",
            "paper/reference_registry.yaml",
            "--base-dir",
            "/tmp/run",
            "--out",
            "/tmp/reference_registry_audit.json",
        ]
    )

    assert args.registry == Path("paper/reference_registry.yaml")
    assert args.base_dir == Path("/tmp/run")
    assert args.out == Path("/tmp/reference_registry_audit.json")


def test_compare_table_artifacts_parser_accepts_columns_and_tolerances():
    args = build_parser().parse_args(
        [
            "compare-table-artifacts",
            "--reference",
            "reports/paper/reference.csv",
            "--candidate",
            "reports/paper/candidate.csv",
            "--base-dir",
            "/tmp/run",
            "--key-column",
            "variant_id",
            "--numeric-column",
            "score",
            "--ignore-column",
            "source",
            "--atol",
            "0.001",
            "--rtol",
            "0.01",
            "--out",
            "/tmp/table_compare.json",
        ]
    )

    assert args.reference == Path("reports/paper/reference.csv")
    assert args.candidate == Path("reports/paper/candidate.csv")
    assert args.base_dir == Path("/tmp/run")
    assert args.key_column == ["variant_id"]
    assert args.numeric_column == ["score"]
    assert args.ignore_column == ["source"]
    assert args.atol == 0.001
    assert args.rtol == 0.01
    assert args.out == Path("/tmp/table_compare.json")


def test_run_table_comparisons_parser_accepts_registry_output():
    args = build_parser().parse_args(
        [
            "run-table-comparisons",
            "--registry",
            "paper/table_registry.yaml",
            "--base-dir",
            "/tmp/run",
            "--out",
            "/tmp/table_comparisons.json",
        ]
    )

    assert args.registry == Path("paper/table_registry.yaml")
    assert args.base_dir == Path("/tmp/run")
    assert args.out == Path("/tmp/table_comparisons.json")
