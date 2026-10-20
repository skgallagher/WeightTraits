from __future__ import annotations

from pathlib import Path

import pytest


def test_native_behavior_pipeline_parser_routes_all_production_builders() -> None:
    from importlib.util import module_from_spec, spec_from_file_location

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_native_behavior_pipeline.py"
    spec = spec_from_file_location("run_native_behavior_pipeline", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    for command, output_flag in (
        ("prompts", "--out-dir"),
        ("checkpoints", "--receipt"),
        ("encoder", "--receipt"),
        ("study", "--out-dir"),
    ):
        args = module.parser().parse_args(
            [command, "--config", "/tmp/input.json", output_flag, "/tmp/output"]
        )
        assert args.command == command

    source_audit = module.parser().parse_args(
        [
            "source-audit",
            "--config",
            "/tmp/input.json",
            "--receipt",
            "/tmp/source-audit.json",
            "--materialization-config",
            "/tmp/materialization.json",
        ]
    )
    assert source_audit.command == "source-audit"
    assert source_audit.receipt == Path("/tmp/source-audit.json")
    assert source_audit.materialization_config == Path("/tmp/materialization.json")


def test_native_behavior_pipeline_parser_rejects_missing_command() -> None:
    from importlib.util import module_from_spec, spec_from_file_location

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_native_behavior_pipeline.py"
    spec = spec_from_file_location("run_native_behavior_pipeline_missing", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(SystemExit):
        module.parser().parse_args([])
