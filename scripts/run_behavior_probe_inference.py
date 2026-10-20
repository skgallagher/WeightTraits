#!/usr/bin/env python3
"""Run one resolved training leaf across one or more frozen behavioral protocols."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from weighttraits.behavior.contracts import load_behavior_protocol_registry
from weighttraits.behavior.probe_inference import (
    HuggingFaceProbeBackend,
    resolve_leaf_checkpoints,
    run_leaf_probe_inference,
    sha256_file,
)
from weighttraits.behavior.responses import load_rendered_behavior_prompts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve one exact completed leaf checkpoint and generate its full frozen behavior grid"
        )
    )
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--cohort-id", required=True)
    parser.add_argument("--tree-id", required=True)
    parser.add_argument("--leaf-id", required=True)
    parser.add_argument("--model-task", choices=("causal_lm", "seq2seq"), required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument(
        "--protocol-prompts",
        action="append",
        required=True,
        metavar="PROTOCOL_ID=PATH",
        help="Rendered prompt JSONL for one requested protocol; repeat for multiple protocols",
    )
    parser.add_argument("--path-base", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument(
        "--resolve-only",
        action="store_true",
        help="Print strict ordered checkpoint resolution without loading a model or writing outputs",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    registry = load_behavior_protocol_registry(args.registry)
    pin = registry.model_pins[args.model_task]
    tree = resolve_leaf_checkpoints(
        args.training_summary,
        cohort_id=args.cohort_id,
        tree_id=args.tree_id,
        model_task=args.model_task,
        base_model_id=pin.model_id,
        base_model_revision=pin.revision,
        path_base=args.path_base,
    )
    selected = tree.leaf(args.leaf_id)
    if args.resolve_only:
        print(json.dumps(selected.to_dict(), indent=2, sort_keys=True))
        return 0

    prompt_paths = _protocol_prompt_paths(args.protocol_prompts)
    prompts_by_protocol = {
        protocol_id: load_rendered_behavior_prompts(path)
        for protocol_id, path in prompt_paths.items()
    }
    prompt_artifacts = {
        protocol_id: {"path": str(path.absolute()), "sha256": sha256_file(path)}
        for protocol_id, path in prompt_paths.items()
    }
    settings = HuggingFaceProbeBackend.planned_provenance(
        selected,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )

    def load_backend(checkpoint):
        return HuggingFaceProbeBackend.load(
            checkpoint,
            device=args.device,
            torch_dtype=args.torch_dtype,
        )

    receipt = run_leaf_probe_inference(
        selected,
        registry=registry,
        prompts_by_protocol=prompts_by_protocol,
        prompt_artifacts=prompt_artifacts,
        output_path=args.out,
        receipt_path=args.receipt,
        backend_factory=load_backend,
        backend_settings=settings,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def _protocol_prompt_paths(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        protocol_id, separator, raw_path = value.partition("=")
        if not separator or not protocol_id or not raw_path:
            raise ValueError(
                f"--protocol-prompts must have PROTOCOL_ID=PATH form, got {value!r}"
            )
        if protocol_id in result:
            raise ValueError(f"duplicate --protocol-prompts protocol: {protocol_id!r}")
        path = Path(raw_path).absolute()
        if not path.is_file():
            raise FileNotFoundError(f"rendered prompt JSONL is missing: {path}")
        result[protocol_id] = path
    return result


if __name__ == "__main__":
    raise SystemExit(main())
