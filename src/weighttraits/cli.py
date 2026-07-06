"""Command line entry points for WeightTraits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from weighttraits.analysis.whitebox import analyze_training_ledger
from weighttraits.audit.ellmtrees import inventory_ellmtrees
from weighttraits.distances.manifest import (
    distance_input_rows_from_training_ledger,
    readers_from_distance_manifest,
    write_distance_input_manifest,
)
from weighttraits.distances.readers import CumulativeLoraReader, LoraFactorReader, reader_from_path
from weighttraits.distances.streaming import build_distance_cube, write_distance_cube
from weighttraits.manifests.reference import manifest_leaf_ids
from weighttraits.phylo.audit import audit_manifest_topology
from weighttraits.phylo.reconstruct import reconstruct_tree_from_cube
from weighttraits.phylo.recovery import aggregate_recovery, score_split_recovery
from weighttraits.phylo.splits import splits_from_manifest_path, splits_from_newick_text
from weighttraits.taskdata.assignment import assign_task_data, load_manifest_rows, write_manifest_rows
from weighttraits.trees.generate import generate_tree_from_config, tree_stats, write_manifest_jsonl
from weighttraits.training.data_formats import (
    load_dataset_format_specs,
    validate_training_jobs_against_formats,
    write_data_format_report,
)
from weighttraits.training.datasets import (
    audit_dataset_registry,
    audit_training_sample_rendering,
    load_dataset_registry,
    write_dataset_audit_report,
    write_training_sample_render_audit_report,
)
from weighttraits.training.executor import dry_run_training_row, run_training_run
from weighttraits.training.ledger import ledger_summary, load_ledger_events
from weighttraits.training.planner import build_training_jobs_from_files, write_training_plan
from weighttraits.training.runlist import (
    build_training_run_list,
    load_execution_profile,
    load_training_run_specs,
    select_training_run,
    write_slurm_array_script,
    write_training_run_list,
    write_training_run_report,
)


def _audit_ellmtrees(args: argparse.Namespace) -> int:
    report = inventory_ellmtrees(args.source)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _manifest_leaves(args: argparse.Namespace) -> int:
    leaves = manifest_leaf_ids(args.manifest)
    print(json.dumps({"manifest": str(args.manifest), "n_leaves": len(leaves), "leaves": leaves}))
    return 0


def _generate_tree(args: argparse.Namespace) -> int:
    config = yaml.safe_load(args.config.read_text())
    tree_config = config.get("tree", config)
    root = generate_tree_from_config(tree_config)
    rows = write_manifest_jsonl(root, args.out) if args.out else None
    summary = tree_stats(root)
    if args.out:
        summary["out"] = str(args.out)
        summary["n_manifest_rows"] = len(rows or [])
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _assign_task_data(args: argparse.Namespace) -> int:
    rows = load_manifest_rows(args.manifest)
    config = yaml.safe_load(args.config.read_text())
    assigned = assign_task_data(
        rows,
        config["task_families"],
        seed=args.seed,
        policy=args.policy,
        task_families=args.task_family,
    )
    write_manifest_rows(assigned, args.out)
    summary = {
        "manifest": str(args.manifest),
        "out": str(args.out),
        "n_rows": len(assigned),
        "policy": args.policy,
        "seed": args.seed,
        "task_families": sorted({row["task_family"] for row in assigned}),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _topology_audit(args: argparse.Namespace) -> int:
    report = audit_manifest_topology(args.manifest)
    _emit_json(report, args.out)
    return 0


def _score_tree(args: argparse.Namespace) -> int:
    truth_splits, truth_leaves = splits_from_manifest_path(str(args.truth_manifest))
    if args.estimate_manifest:
        estimate_splits, estimate_leaves = splits_from_manifest_path(str(args.estimate_manifest))
        estimate_source = str(args.estimate_manifest)
    else:
        estimate_splits, estimate_leaves = splits_from_newick_text(args.estimate_newick.read_text())
        estimate_source = str(args.estimate_newick)

    report = score_split_recovery(
        truth_splits=truth_splits,
        estimate_splits=estimate_splits,
        truth_leaves=truth_leaves,
        estimate_leaves=estimate_leaves,
    )
    report["truth_manifest"] = str(args.truth_manifest)
    report["estimate"] = estimate_source
    _emit_json(report, args.out)
    return 0


def _aggregate_recovery(args: argparse.Namespace) -> int:
    rows = []
    for path in args.scores:
        with path.open() as handle:
            if path.suffix == ".jsonl":
                rows.extend(json.loads(line) for line in handle if line.strip())
            else:
                rows.append(json.load(handle))
    report = aggregate_recovery(rows)
    _emit_json(report, args.out)
    return 0


def _build_distance_cube(args: argparse.Namespace) -> int:
    readers = []
    for manifest in args.checkpoint_manifest or []:
        readers.extend(readers_from_distance_manifest(manifest))
    for item in args.checkpoint or []:
        label, path = _parse_labeled_path(item)
        readers.append(reader_from_path(path, model_id=label))
    for item in args.adapter_chain or []:
        label, paths = _parse_labeled_paths(item)
        edge_readers = []
        for path in paths:
            reader = reader_from_path(path)
            if not isinstance(reader, LoraFactorReader):
                raise ValueError(f"adapter chain entries must be PEFT adapter directories: {path}")
            edge_readers.append(reader)
        readers.append(CumulativeLoraReader(edge_readers, model_id=label or paths[-1].stem))
    if not readers:
        raise ValueError("at least one --checkpoint-manifest, --checkpoint, or --adapter-chain is required")
    cube = build_distance_cube(
        readers,
        metrics=args.metric,
        chunk_size=args.chunk_size,
        eps=args.eps,
        representation=args.representation,
    )
    write_distance_cube(cube, args.out)
    summary = {
        "out": str(args.out),
        "n_models": len(cube.model_ids),
        "n_layers": len(cube.layer_names),
        "metrics": sorted(cube.distances),
        "representation": args.representation,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _make_distance_input_manifest(args: argparse.Namespace) -> int:
    rows = distance_input_rows_from_training_ledger(
        args.ledger,
        truth_manifest=args.truth_manifest,
        artifact=args.artifact,
        node_ids=args.node_id,
    )
    write_distance_input_manifest(rows, args.out, path_base=args.path_base)
    print(
        json.dumps(
            {
                "artifact": args.artifact,
                "ledger": str(args.ledger),
                "n_models": len(rows),
                "out": str(args.out),
                "truth_manifest": str(args.truth_manifest) if args.truth_manifest else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _analyze_training_ledger(args: argparse.Namespace) -> int:
    summary = analyze_training_ledger(
        args.ledger,
        truth_manifest=args.truth_manifest,
        out_dir=args.out,
        artifact=args.artifact,
        metrics=args.metric,
        representation=args.representation,
        node_ids=args.node_id,
        path_base=args.path_base,
        chunk_size=args.chunk_size,
        eps=args.eps,
        layer=args.layer,
        aggregate=args.aggregate,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _reconstruct_tree(args: argparse.Namespace) -> int:
    result = reconstruct_tree_from_cube(
        args.cube,
        metric=args.metric,
        layer=args.layer,
        aggregate=args.aggregate,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result.newick + "\n")
    if args.audit_out:
        args.audit_out.parent.mkdir(parents=True, exist_ok=True)
        args.audit_out.write_text(json.dumps(result.audit, indent=2, sort_keys=True) + "\n")

    summary = dict(result.audit)
    summary["out"] = str(args.out) if args.out else None
    summary["audit_out"] = str(args.audit_out) if args.audit_out else None
    if not args.out:
        summary["newick"] = result.newick
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _plan_training(args: argparse.Namespace) -> int:
    jobs = build_training_jobs_from_files(args.manifest, args.config)
    if args.out:
        write_training_plan(jobs, args.out)
    summary = {
        "manifest": str(args.manifest),
        "config": str(args.config),
        "out": str(args.out) if args.out else None,
        "n_jobs": len(jobs),
        "methods": sorted({job.method for job in jobs}),
        "prompt_sources": sorted({job.prompt_source for job in jobs}),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _training_ledger_summary(args: argparse.Namespace) -> int:
    summary = ledger_summary(load_ledger_events(args.ledger))
    summary["ledger"] = str(args.ledger)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _validate_training_data(args: argparse.Namespace) -> int:
    jobs = build_training_jobs_from_files(args.manifest, args.config)
    specs = load_dataset_format_specs(args.formats)
    report = validate_training_jobs_against_formats(jobs, specs)
    if args.out:
        write_data_format_report(report, args.out)
    summary = report.to_dict()
    summary["manifest"] = str(args.manifest)
    summary["config"] = str(args.config)
    summary["formats"] = str(args.formats)
    summary["out"] = str(args.out) if args.out else None
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if report.valid or args.allow_issues else 1


def _audit_datasets(args: argparse.Namespace) -> int:
    registry = load_dataset_registry(args.registry)
    specs = load_dataset_format_specs(args.formats) if args.formats else None
    report = audit_dataset_registry(
        registry,
        dataset_ids=args.dataset_id,
        format_specs=specs,
        load=not args.no_load,
    )
    if args.out:
        write_dataset_audit_report(report, args.out)
    summary = report.to_dict()
    summary["registry"] = str(args.registry)
    summary["formats"] = str(args.formats) if args.formats else None
    summary["out"] = str(args.out) if args.out else None
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if report.valid or args.allow_issues else 1


def _audit_training_samples(args: argparse.Namespace) -> int:
    jobs = build_training_jobs_from_files(args.manifest, args.config)
    registry = load_dataset_registry(args.registry)
    specs = load_dataset_format_specs(args.formats)
    report = audit_training_sample_rendering(
        jobs,
        registry,
        specs,
        dataset_ids=args.dataset_id,
        max_samples=args.max_samples,
        split=args.split,
    )
    if args.out:
        write_training_sample_render_audit_report(report, args.out)
    summary = report.to_dict()
    summary["manifest"] = str(args.manifest)
    summary["config"] = str(args.config)
    summary["registry"] = str(args.registry)
    summary["formats"] = str(args.formats)
    summary["max_samples"] = args.max_samples
    summary["split"] = args.split
    summary["out"] = str(args.out) if args.out else None
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if report.valid or args.allow_issues else 1


def _make_training_run_list(args: argparse.Namespace) -> int:
    jobs = build_training_jobs_from_files(args.manifest, args.config)
    profile = load_execution_profile(args.profile) if args.profile else None
    if (args.registry is None) != (args.formats is None):
        raise ValueError("--registry and --formats must be provided together")
    runner_options = {
        "registry_path": str(args.registry) if args.registry else None,
        "formats_path": str(args.formats) if args.formats else None,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "allow_missing_eval": args.allow_missing_eval,
        "dry_run": args.runner_dry_run,
    }
    runner_options = {key: value for key, value in runner_options.items() if value is not None}
    runner_entrypoint = (
        "weighttraits.cli run-training-row"
        if args.registry or args.runner_dry_run
        else "pending_hf_peft_executor"
    )
    report = build_training_run_list(
        jobs,
        profile=profile,
        run_list_path=args.out,
        ledger_path=args.ledger,
        runner_entrypoint=runner_entrypoint,
        runner_options=runner_options,
        allow_existing_artifacts=args.allow_existing_artifacts,
        check_filesystem=not args.no_filesystem_check,
    )
    write_training_run_list(report, args.out)
    if args.report:
        write_training_run_report(report, args.report)
    if args.slurm_out:
        if profile is None:
            raise ValueError("--slurm-out requires --profile with scheduler=slurm")
        write_slurm_array_script(
            report,
            args.slurm_out,
            run_list_path=args.out,
            profile=profile,
            job_name=args.job_name,
            max_concurrent=args.max_concurrent,
            python=args.python,
        )
    summary = report.to_dict()
    summary["manifest"] = str(args.manifest)
    summary["config"] = str(args.config)
    summary["out"] = str(args.out)
    summary["report"] = str(args.report) if args.report else None
    summary["slurm_out"] = str(args.slurm_out) if args.slurm_out else None
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if report.valid or args.allow_issues else 1


def _describe_training_run(args: argparse.Namespace) -> int:
    runs = load_training_run_specs(args.run_list)
    run = select_training_run(runs, index=args.index, node_id=args.node_id)
    print(json.dumps(run.to_dict(), indent=2, sort_keys=True))
    return 0


def _run_training_row(args: argparse.Namespace) -> int:
    runs = load_training_run_specs(args.run_list)
    run = select_training_run(runs, index=args.index, node_id=args.node_id)
    options = dict(run.runner.get("options", {}))
    if args.dry_run or options.get("dry_run"):
        print(json.dumps(dry_run_training_row(run).to_dict(), indent=2, sort_keys=True))
        return 0
    registry_path = args.registry or _optional_path(options.get("registry_path"))
    formats_path = args.formats or _optional_path(options.get("formats_path"))
    if registry_path is None or formats_path is None:
        raise ValueError(
            "run-training-row requires --registry and --formats unless --dry-run is set"
        )
    registry = load_dataset_registry(registry_path)
    specs = load_dataset_format_specs(formats_path)
    result = run_training_run(
        run,
        registry,
        specs,
        max_train_samples=args.max_train_samples
        if args.max_train_samples is not None
        else options.get("max_train_samples"),
        max_eval_samples=args.max_eval_samples
        if args.max_eval_samples is not None
        else options.get("max_eval_samples"),
        allow_missing_eval=args.allow_missing_eval or bool(options.get("allow_missing_eval")),
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def _parse_labeled_path(value: str) -> tuple[str | None, Path]:
    if ":" in value:
        label, raw_path = value.split(":", 1)
        return label, Path(raw_path)
    path = Path(value)
    return path.stem, path


def _parse_labeled_paths(value: str) -> tuple[str | None, list[Path]]:
    label = None
    raw_paths = value
    if ":" in value:
        label, raw_paths = value.split(":", 1)
    paths = [Path(item) for item in raw_paths.split(",") if item]
    if not paths:
        raise ValueError(f"no paths found in adapter chain: {value}")
    return label, paths


def _emit_json(report: dict, out: Path | None) -> None:
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    else:
        print(text, end="")


def _optional_path(value: object) -> Path | None:
    return None if value is None else Path(str(value))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wt", description="WeightTraits utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit-ellmtrees", help="Inventory an ELLMTrees reference checkout")
    audit.add_argument("--source", type=Path, required=True, help="Path to the ELLMTrees checkout")
    audit.add_argument("--out", type=Path, help="Optional JSON output path")
    audit.set_defaults(func=_audit_ellmtrees)

    leaves = sub.add_parser("manifest-leaves", help="Print trained leaf IDs from a manifest.jsonl")
    leaves.add_argument("manifest", type=Path)
    leaves.set_defaults(func=_manifest_leaves)

    generate = sub.add_parser("generate-tree", help="Generate a flexible topology manifest")
    generate.add_argument("--config", type=Path, required=True, help="YAML file with a top-level tree section")
    generate.add_argument("--out", type=Path, help="Optional manifest JSONL output path")
    generate.set_defaults(func=_generate_tree)

    assign = sub.add_parser("assign-task-data", help="Enrich a topology manifest with task/data choices")
    assign.add_argument("--manifest", type=Path, required=True, help="Topology manifest JSONL")
    assign.add_argument("--config", type=Path, required=True, help="Task/data candidate YAML")
    assign.add_argument("--out", type=Path, required=True, help="Enriched manifest JSONL")
    assign.add_argument("--seed", type=int, default=1)
    assign.add_argument("--policy", choices=["per_node", "per_edge", "per_depth"], default="per_node")
    assign.add_argument(
        "--task-family",
        action="append",
        help="Restrict to one or more task families; repeat flag for multiple families",
    )
    assign.set_defaults(func=_assign_task_data)

    audit_topology = sub.add_parser("topology-audit", help="Audit topology size, depth, leaves, and polytomies")
    audit_topology.add_argument("--manifest", type=Path, required=True)
    audit_topology.add_argument("--out", type=Path)
    audit_topology.set_defaults(func=_topology_audit)

    score = sub.add_parser("score-tree", help="Score reconstructed tree against a truth manifest")
    score.add_argument("--truth-manifest", type=Path, required=True)
    estimate = score.add_mutually_exclusive_group(required=True)
    estimate.add_argument("--estimate-manifest", type=Path)
    estimate.add_argument("--estimate-newick", type=Path)
    score.add_argument("--out", type=Path)
    score.set_defaults(func=_score_tree)

    aggregate = sub.add_parser("aggregate-recovery", help="Aggregate recovery JSON/JSONL records with SEs")
    aggregate.add_argument("--scores", type=Path, nargs="+", required=True)
    aggregate.add_argument("--out", type=Path)
    aggregate.set_defaults(func=_aggregate_recovery)

    cube = sub.add_parser("build-distance-cube", help="Build a streaming distance cube")
    cube.add_argument(
        "--checkpoint-manifest",
        action="append",
        type=Path,
        help="JSONL/YAML/JSON rows with model_id plus checkpoint or adapter_chain. Repeatable.",
    )
    cube.add_argument(
        "--checkpoint",
        action="append",
        help="Checkpoint path, optionally LABEL:PATH. Repeat once per model/node.",
    )
    cube.add_argument(
        "--adapter-chain",
        action="append",
        help="Cumulative LoRA node as LABEL:EDGE_ADAPTER_DIR,EDGE_ADAPTER_DIR. Repeat per node.",
    )
    cube.add_argument(
        "--metric",
        action="append",
        required=True,
        help="Metric to compute. Repeat for multiple metrics, e.g. cosine, l2, cka.",
    )
    cube.add_argument("--out", type=Path, required=True, help="Output directory")
    cube.add_argument("--chunk-size", type=int, default=1_000_000)
    cube.add_argument("--eps", type=float, default=1e-3, help="Threshold metric epsilon")
    cube.add_argument(
        "--representation",
        default="full_weight",
        choices=["full_weight", "lora_cumulative_delta", "lora_increment_delta"],
    )
    cube.set_defaults(func=_build_distance_cube)

    distance_inputs = sub.add_parser(
        "make-distance-input-manifest",
        help="Derive build-distance-cube inputs from a training ledger",
    )
    distance_inputs.add_argument("--ledger", type=Path, required=True)
    distance_inputs.add_argument(
        "--truth-manifest",
        type=Path,
        help="Training/topology manifest used to select leaves and adapter chains",
    )
    distance_inputs.add_argument(
        "--artifact",
        choices=["model", "merged", "adapter_chain"],
        default="model",
        help="Ledger artifact to emit as checkpoint rows, or cumulative adapter chains",
    )
    distance_inputs.add_argument(
        "--node-id",
        action="append",
        help=(
            "Restrict to one or more node IDs; defaults to truth-manifest leaves "
            "or all terminal artifact nodes"
        ),
    )
    distance_inputs.add_argument(
        "--path-base",
        type=Path,
        default=Path("."),
        help="Base directory for relative ledger artifact paths",
    )
    distance_inputs.add_argument("--out", type=Path, required=True)
    distance_inputs.set_defaults(func=_make_distance_input_manifest)

    analyze_ledger = sub.add_parser(
        "analyze-training-ledger",
        help=(
            "Run distance-input generation, cube building, reconstruction, scoring, "
            "and aggregate recovery from a training ledger"
        ),
    )
    analyze_ledger.add_argument("--ledger", type=Path, required=True)
    analyze_ledger.add_argument("--truth-manifest", type=Path, required=True)
    analyze_ledger.add_argument(
        "--artifact",
        choices=["model", "merged", "adapter_chain"],
        required=True,
        help="Ledger artifact to analyze",
    )
    analyze_ledger.add_argument(
        "--metric",
        action="append",
        required=True,
        help="Metric to compute and score; repeat for multiple metrics",
    )
    analyze_ledger.add_argument("--out", type=Path, required=True, help="Output directory")
    analyze_ledger.add_argument(
        "--node-id",
        action="append",
        help="Restrict to one or more node IDs; defaults to truth-manifest leaves",
    )
    analyze_ledger.add_argument(
        "--path-base",
        type=Path,
        default=Path("."),
        help="Base directory for relative ledger artifact paths",
    )
    analyze_ledger.add_argument("--chunk-size", type=int, default=1_000_000)
    analyze_ledger.add_argument("--eps", type=float, default=1e-3)
    analyze_ledger.add_argument(
        "--representation",
        choices=["full_weight", "lora_cumulative_delta", "lora_increment_delta"],
        help="Override the default representation for the selected artifact",
    )
    analyze_ledger.add_argument(
        "--layer",
        help="Layer name or zero-based layer index. Defaults to aggregating across all layers.",
    )
    analyze_ledger.add_argument(
        "--aggregate",
        choices=["mean", "median"],
        default="mean",
        help="Layer aggregation used when --layer is omitted",
    )
    analyze_ledger.set_defaults(func=_analyze_training_ledger)

    reconstruct = sub.add_parser(
        "reconstruct-tree",
        help="Reconstruct a neighbor-joining Newick tree from a distance cube",
    )
    reconstruct.add_argument("--cube", type=Path, required=True, help="Distance cube directory")
    reconstruct.add_argument(
        "--metric",
        required=True,
        help="Metric in distance_cube.npz to reconstruct",
    )
    reconstruct.add_argument(
        "--layer",
        help="Layer name or zero-based layer index. Defaults to aggregating across all layers.",
    )
    reconstruct.add_argument(
        "--aggregate",
        choices=["mean", "median"],
        default="mean",
        help="Layer aggregation used when --layer is omitted",
    )
    reconstruct.add_argument("--out", type=Path, help="Optional Newick output path")
    reconstruct.add_argument("--audit-out", type=Path, help="Optional JSON audit output path")
    reconstruct.set_defaults(func=_reconstruct_tree)

    train = sub.add_parser("plan-training", help="Validate training config and write job plan JSONL")
    train.add_argument("--manifest", type=Path, required=True, help="Enriched training manifest JSONL")
    train.add_argument("--config", type=Path, required=True, help="Training YAML config")
    train.add_argument("--out", type=Path, help="Optional training job plan JSONL")
    train.set_defaults(func=_plan_training)

    ledger = sub.add_parser("training-ledger-summary", help="Summarize a training ledger JSONL")
    ledger.add_argument("--ledger", type=Path, required=True)
    ledger.set_defaults(func=_training_ledger_summary)

    validate_data = sub.add_parser(
        "validate-training-data",
        help="Validate planned training prompts against offline dataset format specs",
    )
    validate_data.add_argument("--manifest", type=Path, required=True)
    validate_data.add_argument("--config", type=Path, required=True)
    validate_data.add_argument("--formats", type=Path, required=True)
    validate_data.add_argument("--out", type=Path)
    validate_data.add_argument("--allow-issues", action="store_true")
    validate_data.set_defaults(func=_validate_training_data)

    audit_data = sub.add_parser(
        "audit-datasets",
        help="Audit dataset registry entries and Hugging Face split availability",
    )
    audit_data.add_argument("--registry", type=Path, required=True, help="Task/data registry YAML")
    audit_data.add_argument(
        "--formats",
        type=Path,
        help="Optional dataset format contract YAML used to declare required splits",
    )
    audit_data.add_argument(
        "--dataset-id",
        action="append",
        help="Restrict audit to one dataset id; repeat for multiple ids",
    )
    audit_data.add_argument("--out", type=Path)
    audit_data.add_argument(
        "--no-load",
        action="store_true",
        help="Validate registry/requested splits without importing or downloading datasets",
    )
    audit_data.add_argument("--allow-issues", action="store_true")
    audit_data.set_defaults(func=_audit_datasets)

    audit_samples = sub.add_parser(
        "audit-training-samples",
        help="Load small dataset samples and render planned training prompts",
    )
    audit_samples.add_argument("--manifest", type=Path, required=True)
    audit_samples.add_argument("--config", type=Path, required=True)
    audit_samples.add_argument(
        "--registry",
        type=Path,
        required=True,
        help="Task/data registry YAML",
    )
    audit_samples.add_argument(
        "--formats",
        type=Path,
        required=True,
        help="Dataset format contract YAML",
    )
    audit_samples.add_argument(
        "--dataset-id",
        action="append",
        help="Restrict audit to one dataset id; repeat for multiple ids",
    )
    audit_samples.add_argument(
        "--max-samples",
        type=int,
        default=8,
        help="Maximum rows to render per planned job",
    )
    audit_samples.add_argument(
        "--split",
        help="Override the train split declared by registry/formats",
    )
    audit_samples.add_argument("--out", type=Path)
    audit_samples.add_argument("--allow-issues", action="store_true")
    audit_samples.set_defaults(func=_audit_training_samples)

    run_list = sub.add_parser(
        "make-training-run-list",
        help="Write training run JSONL plus optional preflight report and SLURM dry-run script",
    )
    run_list.add_argument("--manifest", type=Path, required=True)
    run_list.add_argument("--config", type=Path, required=True)
    run_list.add_argument("--out", type=Path, required=True, help="Output training run-list JSONL")
    run_list.add_argument("--profile", type=Path, help="Local or cluster execution profile YAML")
    run_list.add_argument(
        "--registry",
        type=Path,
        help="Dataset registry path for generated runners",
    )
    run_list.add_argument(
        "--formats",
        type=Path,
        help="Dataset format contract path for generated runners",
    )
    run_list.add_argument("--ledger", type=Path, help="Ledger path to record in each run row")
    run_list.add_argument("--report", type=Path, help="Optional JSON preflight report")
    run_list.add_argument("--slurm-out", type=Path, help="Optional SLURM array script path")
    run_list.add_argument("--job-name", default="weighttraits-train")
    run_list.add_argument("--max-concurrent", type=int, help="Override profile array throttle")
    run_list.add_argument(
        "--python",
        default="python",
        help="Python executable for generated scripts",
    )
    run_list.add_argument("--max-train-samples", type=int)
    run_list.add_argument("--max-eval-samples", type=int)
    run_list.add_argument("--allow-missing-eval", action="store_true")
    run_list.add_argument("--runner-dry-run", action="store_true")
    run_list.add_argument("--allow-existing-artifacts", action="store_true")
    run_list.add_argument("--no-filesystem-check", action="store_true")
    run_list.add_argument("--allow-issues", action="store_true")
    run_list.set_defaults(func=_make_training_run_list)

    describe_run = sub.add_parser(
        "describe-training-run",
        help="Print one row from a training run-list by array index or node id",
    )
    describe_run.add_argument("--run-list", type=Path, required=True)
    selector = describe_run.add_mutually_exclusive_group(required=True)
    selector.add_argument("--index", type=int)
    selector.add_argument("--node-id")
    describe_run.set_defaults(func=_describe_training_run)

    run_row = sub.add_parser(
        "run-training-row",
        help="Execute one training run-list row by array index or node id",
    )
    run_row.add_argument("--run-list", type=Path, required=True)
    row_selector = run_row.add_mutually_exclusive_group(required=True)
    row_selector.add_argument("--index", type=int)
    row_selector.add_argument("--node-id")
    run_row.add_argument("--registry", type=Path)
    run_row.add_argument("--formats", type=Path)
    run_row.add_argument("--max-train-samples", type=int)
    run_row.add_argument("--max-eval-samples", type=int)
    run_row.add_argument("--allow-missing-eval", action="store_true")
    run_row.add_argument("--dry-run", action="store_true")
    run_row.set_defaults(func=_run_training_row)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
