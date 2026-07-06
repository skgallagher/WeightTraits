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
