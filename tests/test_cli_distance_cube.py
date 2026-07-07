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
