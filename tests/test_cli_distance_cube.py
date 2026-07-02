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
