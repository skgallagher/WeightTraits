"""Command line entry points for WeightTraits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from weighttraits.paper.figures import (
    load_weighttraits_paired_comparisons_artifact,
    load_weighttraits_runset_diagnostics_artifact,
    load_weighttraits_variants_artifact,
    plot_weighttraits_additivity_recovery,
    plot_weighttraits_metric_robustness,
    plot_weighttraits_paired_effects,
    plot_weighttraits_variants_diagnostics,
)
from weighttraits.paper.diagnostics import (
    weighttraits_runset_diagnostic_rows,
    write_weighttraits_runset_diagnostics_csv,
    write_weighttraits_runset_diagnostics_json,
)
from weighttraits.paper.paired import (
    weighttraits_paired_comparison_rows,
    write_weighttraits_paired_comparisons_csv,
    write_weighttraits_paired_comparisons_json,
)
from weighttraits.analysis.completion import (
    audit_training_run_set_completion,
    audit_training_tree_completion,
    write_run_set_completion_csv,
)
from weighttraits.analysis.runset_results import (
    summarize_training_run_set_analysis,
    write_run_set_analysis_csv,
)
from weighttraits.analysis.whitebox import analyze_training_ledger, analyze_training_run_set
from weighttraits.audit.ellmtrees import inventory_ellmtrees
from weighttraits.distances.manifest import (
    distance_input_rows_from_training_ledger,
    readers_from_distance_manifest,
    write_distance_input_manifest,
)
from weighttraits.distances.readers import CumulativeLoraReader, LoraFactorReader, reader_from_path
from weighttraits.distances.streaming import build_distance_cube, write_distance_cube
from weighttraits.manifests.reference import manifest_leaf_ids
from weighttraits.paper.results import (
    behavior_holdout_draft_table_rows,
    compare_table_artifacts,
    ellmtrees_variants_table_rows,
    recovery_table_rows,
    run_table_registry_comparisons,
    validate_reference_registry,
    validate_table_registry,
    write_behavior_holdout_table_csv,
    write_behavior_holdout_table_json,
    write_ellmtrees_variants_table_csv,
    write_ellmtrees_variants_table_json,
    write_recovery_table_csv,
    write_recovery_table_json,
    weighttraits_variants_table_rows,
    write_weighttraits_variants_table_csv,
    write_weighttraits_variants_table_json,
)
from weighttraits.phylo.audit import audit_manifest_topology
from weighttraits.phylo.reconstruct import reconstruct_tree_from_cube
from weighttraits.phylo.recovery import aggregate_recovery, score_split_recovery
from weighttraits.phylo.splits import splits_from_manifest_path, splits_from_newick_text
from weighttraits.taskdata.assignment import (
    assign_task_data,
    load_manifest_rows,
    write_manifest_rows,
)
from weighttraits.trees.generate import (
    generate_tree_from_config,
    generate_tree_set,
    tree_stats,
    write_manifest_jsonl,
)
from weighttraits.training.data_formats import (
    load_dataset_format_specs,
    validate_training_jobs_against_formats,
    write_data_format_report,
)
from weighttraits.training.datasets import (
    DatasetCacheRecipe,
    audit_dataset_registry,
    audit_training_sample_rendering,
    cache_training_datasets,
    load_dataset_registry,
    select_training_sample_jobs,
    write_dataset_audit_report,
    write_dataset_cache_report,
    write_training_sample_render_audit_report,
)
from weighttraits.training.executor import (
    dry_run_training_row,
    prepare_training_data,
    run_training_run,
)
from weighttraits.training.ledger import ledger_summary, load_ledger_events
from weighttraits.training.planner import (
    build_training_jobs,
    build_training_jobs_from_files,
    load_training_config,
    write_training_plan,
)
from weighttraits.training.runlist import (
    build_training_run_list,
    load_execution_profile,
    load_training_run_specs,
    select_training_run,
    write_slurm_array_script,
    write_training_run_list,
    write_training_run_report,
)
from weighttraits.training.retention import (
    append_retention_audit,
    default_retention_audit_path,
    prune_completed_parent_artifact,
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


def _generate_tree_set(args: argparse.Namespace) -> int:
    config = yaml.safe_load(args.config.read_text())
    tree_config = config.get("tree", config)
    set_config = config.get("tree_set", {})
    n_trees = args.n_trees if args.n_trees is not None else int(set_config.get("n_trees", 50))
    seed_start = (
        args.seed_start
        if args.seed_start is not None
        else int(set_config.get("seed_start", tree_config.get("seed", 0)))
    )
    min_leaves = _optional_int(
        args.min_leaves, set_config.get("min_leaves"), tree_config.get("min_leaves")
    )
    min_depth = _optional_int(
        args.min_depth, set_config.get("min_depth"), tree_config.get("min_depth")
    )
    max_candidates = int(set_config.get("max_candidates", args.max_candidates))
    tree_id_prefix = str(set_config.get("tree_id_prefix", "tree"))
    manifest_suffix = str(set_config.get("manifest_suffix", ".manifest.jsonl"))

    generated = generate_tree_set(
        tree_config,
        n_trees=n_trees,
        seed_start=seed_start,
        min_leaves=min_leaves,
        min_depth=min_depth,
        max_candidates=max_candidates,
        tree_id_prefix=tree_id_prefix,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tree_reports = []
    for item in generated:
        manifest_path = args.out_dir / f"{item.tree_id}{manifest_suffix}"
        rows = write_manifest_jsonl(item.root, manifest_path)
        tree_reports.append(
            {
                "tree_id": item.tree_id,
                "seed": item.seed,
                "candidate_index": item.candidate_index,
                "manifest": str(manifest_path),
                "n_rows": len(rows),
                **item.stats,
            }
        )

    summary = {
        "config": str(args.config),
        "out_dir": str(args.out_dir),
        "n_trees": len(tree_reports),
        "seed_start": seed_start,
        "min_leaves": min_leaves,
        "min_depth": min_depth,
        "max_candidates": max_candidates,
        "accepted_candidate_span": generated[-1].candidate_index + 1 if generated else 0,
        "leaf_counts": _count_values(report["n_leaves"] for report in tree_reports),
        "depth_counts": _count_values(report["max_depth"] for report in tree_reports),
        "trees": tree_reports,
    }
    summary_out = args.summary_out or args.out_dir / "tree_set_summary.json"
    _emit_json(summary, summary_out)
    print(
        json.dumps(
            {key: summary[key] for key in ("out_dir", "n_trees", "leaf_counts", "depth_counts")},
            indent=2,
        )
    )
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


def _assign_task_data_set(args: argparse.Namespace) -> int:
    tree_set = json.loads(args.tree_set.read_text())
    config = yaml.safe_load(args.config.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    assignments = []
    for index, tree in enumerate(tree_set.get("trees", [])):
        manifest = Path(tree["manifest"])
        rows = load_manifest_rows(manifest)
        seed = args.seed_start + index
        assigned = assign_task_data(
            rows,
            config["task_families"],
            seed=seed,
            policy=args.policy,
            task_families=args.task_family,
        )
        tree_id = str(tree["tree_id"])
        for row in assigned:
            row["tree_id"] = tree_id
        out_path = args.out_dir / f"{tree_id}.manifest.jsonl"
        write_manifest_rows(assigned, out_path)
        dataset_ids = [
            str(row["dataset_id"]) for row in assigned if row.get("grow", "train") == "train"
        ]
        assignments.append(
            {
                "tree_id": tree_id,
                "source_manifest": str(manifest),
                "assigned_manifest": str(out_path),
                "assignment_seed": seed,
                "n_rows": len(assigned),
                "n_datasets": len(dataset_ids),
                "n_unique_datasets": len(set(dataset_ids)),
                "task_families": _count_values(
                    row["task_family"] for row in assigned if "task_family" in row
                ),
            }
        )
    summary = {
        "tree_set": str(args.tree_set),
        "config": str(args.config),
        "out_dir": str(args.out_dir),
        "policy": args.policy,
        "seed_start": args.seed_start,
        "n_trees": len(assignments),
        "assignments": assignments,
    }
    _emit_json(summary, args.summary_out or args.out_dir / "assignment_summary.json")
    print(json.dumps({"out_dir": str(args.out_dir), "n_trees": len(assignments)}, indent=2))
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


def _make_recovery_table(args: argparse.Namespace) -> int:
    rows = recovery_table_rows(args.registry, base_dir=args.base_dir)
    if args.out:
        write_recovery_table_json(rows, args.out, registry=args.registry)
    if args.csv_out:
        write_recovery_table_csv(rows, args.csv_out)
    summary = {
        "registry": str(args.registry),
        "base_dir": str(args.base_dir),
        "out": str(args.out) if args.out else None,
        "csv_out": str(args.csv_out) if args.csv_out else None,
        "n_rows": len(rows),
        "metrics": sorted({str(row["metric"]) for row in rows}),
        "result_sets": sorted({str(row["result_set_id"]) for row in rows}),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _make_ellmtrees_variants_table(args: argparse.Namespace) -> int:
    rows = ellmtrees_variants_table_rows(args.registry, base_dir=args.base_dir)
    if args.out:
        write_ellmtrees_variants_table_json(rows, args.out, registry=args.registry)
    if args.csv_out:
        write_ellmtrees_variants_table_csv(rows, args.csv_out)
    summary = {
        "registry": str(args.registry),
        "base_dir": str(args.base_dir),
        "out": str(args.out) if args.out else None,
        "csv_out": str(args.csv_out) if args.csv_out else None,
        "n_rows": len(rows),
        "variants": [str(row["variant_id"]) for row in rows],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _make_weighttraits_variants_table(args: argparse.Namespace) -> int:
    rows = weighttraits_variants_table_rows(args.registry, base_dir=args.base_dir)
    if args.out:
        write_weighttraits_variants_table_json(rows, args.out, registry=args.registry)
    if args.csv_out:
        write_weighttraits_variants_table_csv(rows, args.csv_out)
    summary = {
        "registry": str(args.registry),
        "base_dir": str(args.base_dir),
        "out": str(args.out) if args.out else None,
        "csv_out": str(args.csv_out) if args.csv_out else None,
        "n_rows": len(rows),
        "variants": [str(row["variant_id"]) for row in rows],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _plot_weighttraits_variants(args: argparse.Namespace) -> int:
    rows = load_weighttraits_variants_artifact(args.table)
    summary = plot_weighttraits_variants_diagnostics(rows, args.out, title=args.title)
    summary["table"] = str(args.table)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _make_weighttraits_runset_diagnostics(args: argparse.Namespace) -> int:
    rows = weighttraits_runset_diagnostic_rows(args.registry, base_dir=args.base_dir)
    if args.out:
        write_weighttraits_runset_diagnostics_json(rows, args.out, registry=args.registry)
    if args.csv_out:
        write_weighttraits_runset_diagnostics_csv(rows, args.csv_out)
    summary = {
        "registry": str(args.registry),
        "base_dir": str(args.base_dir),
        "out": str(args.out) if args.out else None,
        "csv_out": str(args.csv_out) if args.csv_out else None,
        "n_rows": len(rows),
        "conditions": list(dict.fromkeys(str(row["condition_id"]) for row in rows)),
        "metrics": list(dict.fromkeys(str(row["metric"]) for row in rows)),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _plot_weighttraits_metric_robustness(args: argparse.Namespace) -> int:
    rows = load_weighttraits_runset_diagnostics_artifact(args.diagnostics)
    summary = plot_weighttraits_metric_robustness(rows, args.out, title=args.title)
    summary["diagnostics"] = str(args.diagnostics)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _plot_weighttraits_additivity_recovery(args: argparse.Namespace) -> int:
    rows = load_weighttraits_runset_diagnostics_artifact(args.diagnostics)
    summary = plot_weighttraits_additivity_recovery(
        rows,
        args.out,
        metric=args.metric,
        title=args.title,
    )
    summary["diagnostics"] = str(args.diagnostics)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _make_weighttraits_paired_comparisons(args: argparse.Namespace) -> int:
    rows = weighttraits_paired_comparison_rows(args.registry, base_dir=args.base_dir)
    if args.out:
        write_weighttraits_paired_comparisons_json(rows, args.out, registry=args.registry)
    if args.csv_out:
        write_weighttraits_paired_comparisons_csv(rows, args.csv_out)
    summary = {
        "registry": str(args.registry),
        "base_dir": str(args.base_dir),
        "out": str(args.out) if args.out else None,
        "csv_out": str(args.csv_out) if args.csv_out else None,
        "n_rows": len(rows),
        "comparisons": list(dict.fromkeys(str(row["comparison_id"]) for row in rows)),
        "outcomes": list(dict.fromkeys(str(row["outcome"]) for row in rows)),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _plot_weighttraits_paired_effects(args: argparse.Namespace) -> int:
    rows = load_weighttraits_paired_comparisons_artifact(args.comparisons)
    summary = plot_weighttraits_paired_effects(
        rows,
        args.out,
        group=args.group,
        title=args.title,
    )
    summary["comparisons"] = str(args.comparisons)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _make_behavior_holdout_table(args: argparse.Namespace) -> int:
    rows = behavior_holdout_draft_table_rows(args.draft)
    if args.out:
        write_behavior_holdout_table_json(rows, args.out, draft=args.draft)
    if args.csv_out:
        write_behavior_holdout_table_csv(rows, args.csv_out)
    summary = {
        "draft": str(args.draft),
        "out": str(args.out) if args.out else None,
        "csv_out": str(args.csv_out) if args.csv_out else None,
        "n_rows": len(rows),
        "models": sorted({str(row["model"]) for row in rows}),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _validate_table_registry(args: argparse.Namespace) -> int:
    report = validate_table_registry(
        args.registry,
        base_dir=args.base_dir,
        require_outputs=args.require_outputs,
    )
    _emit_json(report, args.out)
    return 0 if report["valid"] or args.allow_issues else 1


def _validate_reference_registry(args: argparse.Namespace) -> int:
    report = validate_reference_registry(
        args.registry,
        base_dir=args.base_dir,
    )
    _emit_json(report, args.out)
    return 0 if report["valid"] or args.allow_issues else 1


def _compare_table_artifacts(args: argparse.Namespace) -> int:
    report = compare_table_artifacts(
        args.reference,
        args.candidate,
        key_columns=args.key_column,
        compare_columns=args.compare_column,
        numeric_columns=args.numeric_column,
        ignore_columns=args.ignore_column,
        base_dir=args.base_dir,
        atol=args.atol,
        rtol=args.rtol,
    )
    _emit_json(report, args.out)
    return 0 if report["valid"] or args.allow_issues else 1


def _run_table_comparisons(args: argparse.Namespace) -> int:
    report = run_table_registry_comparisons(args.registry, base_dir=args.base_dir)
    _emit_json(report, args.out)
    return 0 if report["valid"] or args.allow_issues else 1


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
        raise ValueError(
            "at least one --checkpoint-manifest, --checkpoint, or --adapter-chain is required"
        )
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


def _analyze_training_run_set(args: argparse.Namespace) -> int:
    summary = analyze_training_run_set(
        args.summary,
        artifact=args.artifact,
        metrics=args.metric,
        out_dir=args.out,
        path_base=args.path_base,
        optional_artifacts=set(args.optional_artifact or []),
        tree_ids=args.tree_id,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing,
        representation=args.representation,
        node_ids=args.node_id,
        chunk_size=args.chunk_size,
        eps=args.eps,
        layer=args.layer,
        aggregate=args.aggregate,
    )
    _emit_json(summary, args.report_out)
    return 0 if summary["valid"] else 1


def _summarize_training_run_set_analysis(args: argparse.Namespace) -> int:
    summary = summarize_training_run_set_analysis(
        args.analysis_root,
        artifact=args.artifact,
        path_base=args.path_base,
        truth_manifest_root=args.truth_manifest_root,
        tree_ids=args.tree_id,
    )
    _emit_json(summary, args.out)
    if args.csv_out:
        write_run_set_analysis_csv(summary["rows"], args.csv_out)
    return 0 if summary["valid"] or args.allow_empty else 1


def _audit_training_tree(args: argparse.Namespace) -> int:
    report = audit_training_tree_completion(
        args.run_list,
        ledger=args.ledger,
        path_base=args.path_base,
        optional_artifacts=set(args.optional_artifact or []),
        require_artifacts=not args.skip_artifact_check,
        require_parent_order=not args.skip_parent_order_check,
    )
    payload = report.to_dict()
    _emit_json(payload, args.out)
    return 0 if payload["valid"] or args.allow_issues else 1


def _audit_training_run_set(args: argparse.Namespace) -> int:
    report = audit_training_run_set_completion(
        args.summary,
        path_base=args.path_base,
        optional_artifacts=set(args.optional_artifact or []),
        require_artifacts=not args.skip_artifact_check,
        require_parent_order=not args.skip_parent_order_check,
        only_ready=args.only_ready,
    )
    payload = report.to_dict()
    _emit_json(payload, args.out)
    if args.csv_out:
        write_run_set_completion_csv(report, args.csv_out)
    return 0 if payload["valid"] or args.allow_issues else 1


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


def _validate_training_data_set(args: argparse.Namespace) -> int:
    assignment_summary = json.loads(args.assignment_summary.read_text())
    training_config = load_training_config(args.config)
    specs = load_dataset_format_specs(args.formats)
    tree_reports = []
    issues = []
    for assignment in assignment_summary.get("assignments", []):
        tree_id = str(assignment["tree_id"])
        manifest = Path(assignment["assigned_manifest"])
        jobs = build_training_jobs(
            load_manifest_rows(manifest),
            _with_tree_output_root(training_config, tree_id),
        )
        report = validate_training_jobs_against_formats(jobs, specs)
        tree_issues = [dict(issue.to_dict(), tree_id=tree_id) for issue in report.issues]
        issues.extend(tree_issues)
        tree_reports.append(
            {
                "tree_id": tree_id,
                "manifest": str(manifest),
                "valid": report.valid,
                "n_jobs": report.n_jobs,
                "n_valid": report.n_valid,
                "n_issues": len(tree_issues),
            }
        )
    summary = {
        "assignment_summary": str(args.assignment_summary),
        "config": str(args.config),
        "formats": str(args.formats),
        "valid": not issues,
        "n_trees": len(tree_reports),
        "n_jobs": sum(int(report["n_jobs"]) for report in tree_reports),
        "n_valid": sum(int(report["n_valid"]) for report in tree_reports),
        "n_issues": len(issues),
        "trees": tree_reports,
        "issues": issues,
    }
    _emit_json(summary, args.out)
    return 0 if summary["valid"] or args.allow_issues else 1


def _audit_datasets(args: argparse.Namespace) -> int:
    registry = load_dataset_registry(args.registry)
    specs = load_dataset_format_specs(args.formats) if args.formats else None
    report = audit_dataset_registry(
        registry,
        dataset_ids=args.dataset_id,
        format_specs=specs,
        load=not args.no_load,
        loader=_hf_dataset_loader(args.streaming),
    )
    if args.out:
        write_dataset_audit_report(report, args.out)
    summary = report.to_dict()
    summary["registry"] = str(args.registry)
    summary["formats"] = str(args.formats) if args.formats else None
    summary["streaming"] = args.streaming
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
        loader=_hf_dataset_loader(args.streaming),
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
    summary["streaming"] = args.streaming
    summary["out"] = str(args.out) if args.out else None
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if report.valid or args.allow_issues else 1


def _audit_training_sample_set(args: argparse.Namespace) -> int:
    training_config = load_training_config(args.config)
    jobs = _training_jobs_from_assignment_summary(args.assignment_summary, training_config)
    if args.dataset_id:
        available_ids = {job.dataset_id for job in jobs if job.dataset_id is not None}
        missing_ids = sorted(set(args.dataset_id) - available_ids)
        if missing_ids:
            raise ValueError(
                f"dataset ids not found in assignment summary: {', '.join(missing_ids)}"
            )
    selected_jobs = select_training_sample_jobs(
        jobs,
        dataset_ids=args.dataset_id,
        selection=args.selection,
    )
    if not selected_jobs:
        raise ValueError("no training jobs selected for sample audit")

    registry = load_dataset_registry(args.registry)
    specs = load_dataset_format_specs(args.formats)
    report = audit_training_sample_rendering(
        selected_jobs,
        registry,
        specs,
        max_samples=args.max_samples,
        split=args.split,
        loader=_hf_dataset_loader(args.streaming),
    )
    if args.out:
        write_training_sample_render_audit_report(report, args.out)
    summary = report.to_dict()
    summary["assignment_summary"] = str(args.assignment_summary)
    summary["config"] = str(args.config)
    summary["registry"] = str(args.registry)
    summary["formats"] = str(args.formats)
    summary["selection"] = args.selection
    summary["max_samples"] = args.max_samples
    summary["split"] = args.split
    summary["streaming"] = args.streaming
    summary["out"] = str(args.out) if args.out else None
    summary["n_planned_jobs"] = len(jobs)
    summary["n_selected_datasets"] = len(
        {audit.dataset_id for audit in report.audits if audit.dataset_id is not None}
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if report.valid or args.allow_issues else 1


def _cache_training_dataset_set(args: argparse.Namespace) -> int:
    if args.streaming and args.sample_strategy == "legacy_subsample":
        raise ValueError("--sample-strategy legacy_subsample is incompatible with --streaming")
    registry = load_dataset_registry(args.registry)
    specs = load_dataset_format_specs(args.formats)
    report = cache_training_datasets(
        registry,
        specs,
        out_dir=args.out_dir,
        dataset_ids=args.dataset_id,
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
        max_scan=args.max_scan,
        min_train_rows=args.min_train_rows,
        min_eval_rows=args.min_eval_rows,
        loader=_hf_dataset_loader(args.streaming),
        overwrite=args.overwrite,
        sample_strategy=args.sample_strategy,
        sample_seed=args.sample_seed,
        shuffle_buffer_size=args.shuffle_buffer_size,
    )
    if args.summary_out:
        write_dataset_cache_report(report, args.summary_out)
    summary = report.to_dict()
    summary["registry"] = str(args.registry)
    summary["formats"] = str(args.formats)
    summary["streaming"] = args.streaming
    summary["train_limit"] = args.train_limit
    summary["eval_limit"] = args.eval_limit
    summary["max_scan"] = args.max_scan
    summary["min_train_rows"] = args.min_train_rows
    summary["min_eval_rows"] = args.min_eval_rows
    summary["sample_strategy"] = report.sample_strategy
    summary["sample_seed"] = report.sample_seed
    summary["shuffle_buffer_size"] = report.shuffle_buffer_size
    summary["summary_out"] = str(args.summary_out) if args.summary_out else None
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if report.valid or args.allow_issues else 1


def _make_training_run_list(args: argparse.Namespace) -> int:
    jobs = build_training_jobs_from_files(args.manifest, args.config)
    profile = load_execution_profile(args.profile) if args.profile else None
    runner_entrypoint, runner_options = _training_runner_contract(args)
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
    summary["config_sha256"] = _path_sha256(args.config)
    summary["manifest_sha256"] = _path_sha256(args.manifest)
    summary["out"] = str(args.out)
    summary["run_list_sha256"] = _path_sha256(args.out)
    summary["report"] = str(args.report) if args.report else None
    summary["slurm_out"] = str(args.slurm_out) if args.slurm_out else None
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if report.valid or args.allow_issues else 1


def _make_training_run_list_set(args: argparse.Namespace) -> int:
    assignment_summary = json.loads(args.assignment_summary.read_text())
    training_config = load_training_config(args.config)
    profile = load_execution_profile(args.profile) if args.profile else None
    runner_entrypoint, runner_options = _training_runner_contract(args)
    run_list_dir = args.out_dir / "run_lists"
    report_dir = args.out_dir / "reports"
    ledger_dir = args.out_dir / "ledgers"
    run_list_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    ledger_dir.mkdir(parents=True, exist_ok=True)

    tree_reports = []
    all_valid = True
    for assignment in assignment_summary.get("assignments", []):
        tree_id = str(assignment["tree_id"])
        manifest = Path(assignment["assigned_manifest"])
        per_tree_config = _with_tree_output_root(training_config, tree_id)
        jobs = build_training_jobs(load_manifest_rows(manifest), per_tree_config)
        run_list_path = run_list_dir / f"{tree_id}.runs.jsonl"
        report_path = report_dir / f"{tree_id}.report.json"
        ledger_path = ledger_dir / f"{tree_id}.training_ledger.jsonl"
        report = build_training_run_list(
            jobs,
            profile=profile,
            run_list_path=run_list_path,
            ledger_path=ledger_path,
            runner_entrypoint=runner_entrypoint,
            runner_options=runner_options,
            allow_existing_artifacts=args.allow_existing_artifacts,
            check_filesystem=not args.no_filesystem_check,
        )
        write_training_run_list(report, run_list_path)
        write_training_run_report(report, report_path)
        all_valid = all_valid and report.valid
        tree_reports.append(
            {
                "tree_id": tree_id,
                "manifest": str(manifest),
                "manifest_sha256": _path_sha256(manifest),
                "run_list": str(run_list_path),
                "run_list_sha256": _path_sha256(run_list_path),
                "report": str(report_path),
                "report_sha256": _path_sha256(report_path),
                "ledger": str(ledger_path),
                "output_root": str(per_tree_config["output_root"]),
                "training_config_sha256": (jobs[0].training_config_sha256 if jobs else None),
                **report.to_dict(),
            }
        )

    summary = {
        "assignment_summary": str(args.assignment_summary),
        "assignment_summary_sha256": _path_sha256(args.assignment_summary),
        "config": str(args.config),
        "config_sha256": _path_sha256(args.config),
        "registry": str(args.registry) if args.registry else None,
        "registry_sha256": _path_sha256(args.registry) if args.registry else None,
        "formats": str(args.formats) if args.formats else None,
        "formats_sha256": _path_sha256(args.formats) if args.formats else None,
        "out_dir": str(args.out_dir),
        "valid": all_valid,
        "n_trees": len(tree_reports),
        "n_runs": sum(int(report["n_runs"]) for report in tree_reports),
        "n_errors": sum(int(report["n_errors"]) for report in tree_reports),
        "n_warnings": sum(int(report["n_warnings"]) for report in tree_reports),
        "runner_entrypoint": runner_entrypoint,
        "runner_options": runner_options,
        "trees": tree_reports,
    }
    _emit_json(summary, args.summary_out or args.out_dir / "run_list_summary.json")
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "n_trees": summary["n_trees"],
                "n_runs": summary["n_runs"],
                "valid": summary["valid"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if all_valid or args.allow_issues else 1


def _describe_training_run(args: argparse.Namespace) -> int:
    runs = load_training_run_specs(args.run_list)
    run = select_training_run(runs, index=args.index, node_id=args.node_id)
    print(json.dumps(run.to_dict(), indent=2, sort_keys=True))
    return 0


def _audit_training_row_data(args: argparse.Namespace) -> int:
    runs = load_training_run_specs(args.run_list)
    run = select_training_run(runs, index=args.index, node_id=args.node_id)
    options = dict(run.runner.get("options", {}))
    registry_path = args.registry or _optional_path(options.get("registry_path"))
    formats_path = args.formats or _optional_path(options.get("formats_path"))
    data_cache_root = args.data_cache_root or _optional_path(options.get("data_cache_root"))
    require_data_cache = args.require_data_cache or bool(options.get("require_data_cache"))
    expected_cache_recipe = _resolve_expected_cache_recipe(
        args,
        options,
    )
    if registry_path is None or formats_path is None:
        raise ValueError(
            "audit-training-row-data requires --registry and --formats unless the run row carries them"
        )
    registry = load_dataset_registry(registry_path)
    specs = load_dataset_format_specs(formats_path)
    data = prepare_training_data(
        run,
        registry,
        specs,
        loader=_hf_dataset_loader(args.streaming),
        data_cache_root=data_cache_root,
        require_data_cache=require_data_cache,
        expected_cache_recipe=expected_cache_recipe,
        max_train_samples=args.max_train_samples
        if args.max_train_samples is not None
        else options.get("max_train_samples"),
        max_eval_samples=args.max_eval_samples
        if args.max_eval_samples is not None
        else options.get("max_eval_samples"),
        allow_missing_eval=args.allow_missing_eval or bool(options.get("allow_missing_eval")),
    )
    summary = {
        "valid": data.valid,
        "run_list": str(args.run_list),
        "node_id": run.node_id,
        "array_index": run.array_index,
        "dataset_id": run.dataset_id,
        "task_family": run.task_family,
        "registry": str(registry_path),
        "formats": str(formats_path),
        "data_cache_root": str(data_cache_root) if data_cache_root is not None else None,
        "require_data_cache": require_data_cache,
        "expected_cache_recipe": (
            expected_cache_recipe.to_dict() if expected_cache_recipe is not None else None
        ),
        "streaming": args.streaming,
        "data": data.summary(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if data.valid or args.allow_issues else 1


def _run_training_row(args: argparse.Namespace) -> int:
    runs = load_training_run_specs(args.run_list)
    run = select_training_run(runs, index=args.index, node_id=args.node_id)
    options = dict(run.runner.get("options", {}))
    trainer_overrides = _trainer_overrides_from_args(args)
    run = _run_with_trainer_overrides(run, trainer_overrides)
    execution_overrides = {"trainer": trainer_overrides} if trainer_overrides else {}
    if args.dry_run or options.get("dry_run"):
        print(
            json.dumps(
                dry_run_training_row(
                    run,
                    execution_overrides=execution_overrides,
                ).to_dict(),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    registry_path = args.registry or _optional_path(options.get("registry_path"))
    formats_path = args.formats or _optional_path(options.get("formats_path"))
    data_cache_root = args.data_cache_root or _optional_path(options.get("data_cache_root"))
    require_data_cache = args.require_data_cache or bool(options.get("require_data_cache"))
    expected_cache_recipe = _resolve_expected_cache_recipe(
        args,
        options,
    )
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
        data_cache_root=data_cache_root,
        require_data_cache=require_data_cache,
        expected_cache_recipe=expected_cache_recipe,
        max_train_samples=args.max_train_samples
        if args.max_train_samples is not None
        else options.get("max_train_samples"),
        max_eval_samples=args.max_eval_samples
        if args.max_eval_samples is not None
        else options.get("max_eval_samples"),
        allow_missing_eval=args.allow_missing_eval or bool(options.get("allow_missing_eval")),
        execution_overrides=execution_overrides,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def _prune_training_parent_artifact(args: argparse.Namespace) -> int:
    runs = load_training_run_specs(args.run_list)
    selected = select_training_run(runs, index=args.index, node_id=args.node_id)
    result = prune_completed_parent_artifact(
        runs,
        selected_node_id=selected.node_id,
        ledger_events=load_ledger_events(selected.ledger_path),
        success_not_before=args.success_not_before,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        audit_path = args.audit_out or default_retention_audit_path(selected.ledger_path)
        append_retention_audit(audit_path, result)
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


def _path_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_int(*values: object) -> int | None:
    for value in values:
        if value is not None:
            return int(value)
    return None


def _trainer_overrides_from_args(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if getattr(args, "override_max_steps", None) is not None:
        if args.override_max_steps < 1:
            raise ValueError("--override-max-steps must be at least 1")
        overrides["max_steps"] = args.override_max_steps
    if getattr(args, "report_to", None):
        overrides["report_to"] = _report_to_values(args.report_to)
    if getattr(args, "run_name", None):
        overrides["run_name"] = args.run_name
    return overrides


def _report_to_values(values: Sequence[str]) -> list[str]:
    reporters = []
    for value in values:
        parts = [part.strip() for part in value.split(",")]
        if any(not part for part in parts):
            raise ValueError("--report-to entries must not be empty")
        reporters.extend(parts)
    if not reporters:
        raise ValueError("--report-to requires at least one reporter")
    disabled = [reporter for reporter in reporters if reporter.lower() == "none"]
    if disabled:
        if len(reporters) > 1:
            raise ValueError("--report-to none cannot be combined with other reporters")
        return []
    return reporters


def _run_with_trainer_overrides(run, overrides: Mapping[str, object]):
    if not overrides:
        return run
    row = run.to_dict()
    job = dict(row["job"])
    trainer = dict(job.get("trainer") or {})
    trainer.update(overrides)
    job["trainer"] = trainer
    row["job"] = job
    return type(run)(**row)


def _hf_dataset_loader(streaming: bool):
    if not streaming:
        return None

    def load_streaming_dataset(*args, **kwargs):
        from datasets import load_dataset

        options = dict(kwargs)
        options.setdefault("streaming", True)
        return load_dataset(*args, **options)

    return load_streaming_dataset


def _training_jobs_from_assignment_summary(
    assignment_summary_path: Path,
    training_config: dict[str, object],
) -> list:
    assignment_summary = json.loads(assignment_summary_path.read_text())
    jobs = []
    for assignment in assignment_summary.get("assignments", []):
        tree_id = str(assignment["tree_id"])
        manifest = Path(assignment["assigned_manifest"])
        jobs.extend(
            build_training_jobs(
                load_manifest_rows(manifest),
                _with_tree_output_root(training_config, tree_id),
            )
        )
    return jobs


def _training_runner_contract(args: argparse.Namespace) -> tuple[str, dict[str, object]]:
    if (args.registry is None) != (args.formats is None):
        raise ValueError("--registry and --formats must be provided together")
    expected_cache_recipe = _resolve_expected_cache_recipe(
        args,
        {},
    )
    runner_options = {
        "registry_path": str(args.registry) if args.registry else None,
        "formats_path": str(args.formats) if args.formats else None,
        "data_cache_root": str(args.data_cache_root) if args.data_cache_root else None,
        "require_data_cache": args.require_data_cache,
        "expected_cache_recipe": (
            expected_cache_recipe.to_dict() if expected_cache_recipe is not None else None
        ),
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "allow_missing_eval": args.allow_missing_eval,
        "dry_run": args.runner_dry_run,
    }
    runner_options = {
        key: value
        for key, value in runner_options.items()
        if value is not None and value is not False
    }
    runner_entrypoint = (
        "weighttraits.cli run-training-row"
        if args.registry or args.runner_dry_run
        else "pending_hf_peft_executor"
    )
    return runner_entrypoint, runner_options


def _resolve_expected_cache_recipe(
    args: argparse.Namespace,
    options: Mapping[str, object],
) -> DatasetCacheRecipe | None:
    raw_options = options.get("expected_cache_recipe")
    if raw_options is None:
        values: dict[str, object] = {}
    elif isinstance(raw_options, Mapping):
        values = dict(raw_options)
    else:
        raise ValueError("runner expected_cache_recipe must be a mapping")

    cli_fields = {
        "sample_strategy": "expected_cache_strategy",
        "sample_seed": "expected_cache_seed",
        "train_limit": "expected_cache_train_limit",
        "eval_limit": "expected_cache_eval_limit",
    }
    for recipe_field, arg_field in cli_fields.items():
        value = getattr(args, arg_field, None)
        if value is not None:
            values[recipe_field] = value

    if not values:
        return None
    missing = [
        field for field in ("sample_strategy", "train_limit", "eval_limit") if field not in values
    ]
    if missing:
        raise ValueError("expected cache recipe is incomplete: " + ", ".join(missing))
    return DatasetCacheRecipe(
        sample_strategy=str(values["sample_strategy"]),
        sample_seed=values.get("sample_seed"),
        train_limit=values["train_limit"],
        eval_limit=values["eval_limit"],
    )


def _add_expected_cache_recipe_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--expected-cache-strategy",
        choices=("first", "seeded_shuffle", "legacy_subsample"),
        help="Sampling strategy required in cached split metadata",
    )
    parser.add_argument(
        "--expected-cache-seed",
        type=int,
        help="Sampling seed required in cached split metadata",
    )
    parser.add_argument(
        "--expected-cache-train-limit",
        type=int,
        help="Configured train split limit required in cached split metadata",
    )
    parser.add_argument(
        "--expected-cache-eval-limit",
        type=int,
        help="Configured evaluation split limit required in cached split metadata",
    )


def _with_tree_output_root(training_config: dict[str, object], tree_id: str) -> dict[str, object]:
    if not training_config.get("output_root"):
        raise ValueError("training config requires output_root")
    config = dict(training_config)
    config["output_root"] = str(Path(str(config["output_root"])) / tree_id)
    return config


def _count_values(values: object) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return {key: counts[key] for key in sorted(counts, key=_count_sort_key)}


def _count_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


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
    generate.add_argument(
        "--config", type=Path, required=True, help="YAML file with a top-level tree section"
    )
    generate.add_argument("--out", type=Path, help="Optional manifest JSONL output path")
    generate.set_defaults(func=_generate_tree)

    generate_set = sub.add_parser(
        "generate-tree-set",
        help="Generate multiple accepted topology manifests from one stochastic tree config",
    )
    generate_set.add_argument(
        "--config", type=Path, required=True, help="YAML file with tree/tree_set sections"
    )
    generate_set.add_argument(
        "--out-dir", type=Path, required=True, help="Directory for generated manifests"
    )
    generate_set.add_argument("--summary-out", type=Path, help="Optional summary JSON path")
    generate_set.add_argument("--n-trees", type=int, help="Override tree_set.n_trees")
    generate_set.add_argument("--seed-start", type=int, help="Override tree_set.seed_start")
    generate_set.add_argument(
        "--min-leaves", type=int, help="Override acceptance minimum leaf count"
    )
    generate_set.add_argument("--min-depth", type=int, help="Override acceptance minimum max depth")
    generate_set.add_argument("--max-candidates", type=int, default=10_000)
    generate_set.set_defaults(func=_generate_tree_set)

    assign = sub.add_parser(
        "assign-task-data", help="Enrich a topology manifest with task/data choices"
    )
    assign.add_argument("--manifest", type=Path, required=True, help="Topology manifest JSONL")
    assign.add_argument("--config", type=Path, required=True, help="Task/data candidate YAML")
    assign.add_argument("--out", type=Path, required=True, help="Enriched manifest JSONL")
    assign.add_argument("--seed", type=int, default=1)
    assign.add_argument(
        "--policy",
        choices=["per_node", "per_node_without_replacement", "per_edge", "per_depth"],
        default="per_node",
    )
    assign.add_argument(
        "--task-family",
        action="append",
        help="Restrict to one or more task families; repeat flag for multiple families",
    )
    assign.set_defaults(func=_assign_task_data)

    assign_set = sub.add_parser(
        "assign-task-data-set",
        help="Assign task/data rows to every manifest listed in a generated tree-set summary",
    )
    assign_set.add_argument("--tree-set", type=Path, required=True, help="tree_set_summary.json")
    assign_set.add_argument("--config", type=Path, required=True, help="Task/data candidate YAML")
    assign_set.add_argument(
        "--out-dir", type=Path, required=True, help="Directory for enriched manifests"
    )
    assign_set.add_argument("--summary-out", type=Path, help="Optional assignment summary JSON")
    assign_set.add_argument("--seed-start", type=int, default=1)
    assign_set.add_argument(
        "--policy",
        choices=["per_node", "per_node_without_replacement", "per_edge", "per_depth"],
        default="per_node_without_replacement",
    )
    assign_set.add_argument(
        "--task-family",
        action="append",
        help="Restrict to one or more task families; repeat flag for multiple families",
    )
    assign_set.set_defaults(func=_assign_task_data_set)

    audit_topology = sub.add_parser(
        "topology-audit", help="Audit topology size, depth, leaves, and polytomies"
    )
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

    aggregate = sub.add_parser(
        "aggregate-recovery", help="Aggregate recovery JSON/JSONL records with SEs"
    )
    aggregate.add_argument("--scores", type=Path, nargs="+", required=True)
    aggregate.add_argument("--out", type=Path)
    aggregate.set_defaults(func=_aggregate_recovery)

    recovery_table = sub.add_parser(
        "make-recovery-table",
        help="Build paper-facing recovery table rows from registered whitebox summaries",
    )
    recovery_table.add_argument("--registry", type=Path, required=True)
    recovery_table.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Base directory used to resolve relative summary paths",
    )
    recovery_table.add_argument("--out", type=Path, help="Optional JSON table output")
    recovery_table.add_argument("--csv-out", type=Path, help="Optional CSV table output")
    recovery_table.set_defaults(func=_make_recovery_table)

    ellmtrees_variants_table = sub.add_parser(
        "make-ellmtrees-variants-table",
        help="Build the old ELLMTrees tab:variants reference table from pinned source CSVs",
    )
    ellmtrees_variants_table.add_argument("--registry", type=Path, required=True)
    ellmtrees_variants_table.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Base directory used to resolve relative source paths",
    )
    ellmtrees_variants_table.add_argument("--out", type=Path, help="Optional JSON table output")
    ellmtrees_variants_table.add_argument("--csv-out", type=Path, help="Optional CSV table output")
    ellmtrees_variants_table.set_defaults(func=_make_ellmtrees_variants_table)

    weighttraits_variants_table = sub.add_parser(
        "make-weighttraits-variants-table",
        help="Build rebuilt tab:variants rows from WeightTraits run-set summaries",
    )
    weighttraits_variants_table.add_argument("--registry", type=Path, required=True)
    weighttraits_variants_table.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Base directory used to resolve relative run-set summary paths",
    )
    weighttraits_variants_table.add_argument("--out", type=Path, help="Optional JSON table output")
    weighttraits_variants_table.add_argument(
        "--csv-out", type=Path, help="Optional CSV table output"
    )
    weighttraits_variants_table.set_defaults(func=_make_weighttraits_variants_table)

    weighttraits_variants_plot = sub.add_parser(
        "plot-weighttraits-variants",
        help="Plot fresh ordering and recovery diagnostics from a native WeightTraits table",
    )
    weighttraits_variants_plot.add_argument(
        "--table",
        type=Path,
        required=True,
        help="Provenance-bearing JSON from make-weighttraits-variants-table",
    )
    weighttraits_variants_plot.add_argument("--out", type=Path, required=True)
    weighttraits_variants_plot.add_argument(
        "--title",
        default="Independent WeightTraits variant diagnostics",
    )
    weighttraits_variants_plot.set_defaults(func=_plot_weighttraits_variants)

    runset_diagnostics = sub.add_parser(
        "make-weighttraits-runset-diagnostics",
        help="Build fresh multi-metric diagnostics from native WeightTraits run-set summaries",
    )
    runset_diagnostics.add_argument("--registry", type=Path, required=True)
    runset_diagnostics.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Base directory used to resolve relative native summary paths",
    )
    runset_diagnostics.add_argument("--out", type=Path, help="Optional provenance JSON output")
    runset_diagnostics.add_argument("--csv-out", type=Path, help="Optional long-form CSV output")
    runset_diagnostics.set_defaults(func=_make_weighttraits_runset_diagnostics)

    metric_robustness_plot = sub.add_parser(
        "plot-weighttraits-metric-robustness",
        help="Plot recovery sensitivity across fresh L2, cosine, and correlation distances",
    )
    metric_robustness_plot.add_argument("--diagnostics", type=Path, required=True)
    metric_robustness_plot.add_argument("--out", type=Path, required=True)
    metric_robustness_plot.add_argument(
        "--title",
        default="Independent WeightTraits distance-metric robustness",
    )
    metric_robustness_plot.set_defaults(func=_plot_weighttraits_metric_robustness)

    additivity_recovery_plot = sub.add_parser(
        "plot-weighttraits-additivity-recovery",
        help="Plot fresh additivity and Atteson diagnostics against recovery",
    )
    additivity_recovery_plot.add_argument("--diagnostics", type=Path, required=True)
    additivity_recovery_plot.add_argument(
        "--metric",
        choices=["l2", "cosine", "correlation"],
        default="cosine",
    )
    additivity_recovery_plot.add_argument("--out", type=Path, required=True)
    additivity_recovery_plot.add_argument(
        "--title",
        default="Independent WeightTraits additivity and recovery",
    )
    additivity_recovery_plot.set_defaults(func=_plot_weighttraits_additivity_recovery)

    paired_comparisons = sub.add_parser(
        "make-weighttraits-paired-comparisons",
        help="Compute paired same-topology effects from native WeightTraits run-set summaries",
    )
    paired_comparisons.add_argument("--registry", type=Path, required=True)
    paired_comparisons.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Base directory used to resolve relative native summary paths",
    )
    paired_comparisons.add_argument("--out", type=Path, help="Optional provenance JSON output")
    paired_comparisons.add_argument("--csv-out", type=Path, help="Optional long-form CSV output")
    paired_comparisons.set_defaults(func=_make_weighttraits_paired_comparisons)

    paired_effects_plot = sub.add_parser(
        "plot-weighttraits-paired-effects",
        help="Plot paired mean effects with deterministic bootstrap confidence intervals",
    )
    paired_effects_plot.add_argument("--comparisons", type=Path, required=True)
    paired_effects_plot.add_argument("--group", required=True)
    paired_effects_plot.add_argument("--out", type=Path, required=True)
    paired_effects_plot.add_argument(
        "--title",
        default="Independent WeightTraits paired same-tree effects",
    )
    paired_effects_plot.set_defaults(func=_plot_weighttraits_paired_effects)

    behavior_holdout_table = sub.add_parser(
        "make-behavior-holdout-table",
        help="Extract the live draft tab:behavior_holdout table as JSON/CSV rows",
    )
    behavior_holdout_table.add_argument("--draft", type=Path, required=True)
    behavior_holdout_table.add_argument("--out", type=Path, help="Optional JSON table output")
    behavior_holdout_table.add_argument("--csv-out", type=Path, help="Optional CSV table output")
    behavior_holdout_table.set_defaults(func=_make_behavior_holdout_table)

    table_registry = sub.add_parser(
        "validate-table-registry",
        help="Validate paper table registry inputs and optional generated outputs",
    )
    table_registry.add_argument("--registry", type=Path, required=True)
    table_registry.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Base directory used to resolve relative registry paths",
    )
    table_registry.add_argument(
        "--require-outputs",
        action="store_true",
        help="Also require declared table outputs and validate their row counts",
    )
    table_registry.add_argument("--out", type=Path)
    table_registry.add_argument("--allow-issues", action="store_true")
    table_registry.set_defaults(func=_validate_table_registry)

    reference_registry = sub.add_parser(
        "validate-reference-registry",
        help="Validate paper reference paths, draft labels, and optional SHA-256 digests",
    )
    reference_registry.add_argument("--registry", type=Path, required=True)
    reference_registry.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Base directory used to resolve relative registry paths",
    )
    reference_registry.add_argument("--out", type=Path)
    reference_registry.add_argument("--allow-issues", action="store_true")
    reference_registry.set_defaults(func=_validate_reference_registry)

    table_compare = sub.add_parser(
        "compare-table-artifacts",
        help="Compare two generated paper table artifacts row-by-row",
    )
    table_compare.add_argument("--reference", type=Path, required=True)
    table_compare.add_argument("--candidate", type=Path, required=True)
    table_compare.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Base directory used to resolve relative table paths",
    )
    table_compare.add_argument(
        "--key-column",
        action="append",
        required=True,
        help="Column used as a row key; repeat for composite keys",
    )
    table_compare.add_argument(
        "--compare-column",
        action="append",
        help="Column to compare; defaults to all non-key, non-ignored columns",
    )
    table_compare.add_argument(
        "--numeric-column",
        action="append",
        help="Column compared numerically with --atol/--rtol",
    )
    table_compare.add_argument(
        "--ignore-column", action="append", help="Column ignored during comparison"
    )
    table_compare.add_argument("--atol", type=float, default=1e-9)
    table_compare.add_argument("--rtol", type=float, default=1e-9)
    table_compare.add_argument("--out", type=Path)
    table_compare.add_argument("--allow-issues", action="store_true")
    table_compare.set_defaults(func=_compare_table_artifacts)

    table_comparisons = sub.add_parser(
        "run-table-comparisons",
        help="Run table artifact comparisons declared in a paper table registry",
    )
    table_comparisons.add_argument("--registry", type=Path, required=True)
    table_comparisons.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Base directory used to resolve relative table paths",
    )
    table_comparisons.add_argument("--out", type=Path)
    table_comparisons.add_argument("--allow-issues", action="store_true")
    table_comparisons.set_defaults(func=_run_table_comparisons)

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

    analyze_run_set = sub.add_parser(
        "analyze-training-run-set",
        help="Analyze every completion-ready tree in a generated training run-list-set summary",
    )
    analyze_run_set.add_argument("--summary", type=Path, required=True)
    analyze_run_set.add_argument(
        "--artifact",
        choices=["model", "merged", "adapter_chain"],
        required=True,
        help="Ledger artifact to analyze",
    )
    analyze_run_set.add_argument(
        "--metric",
        action="append",
        required=True,
        help="Metric to compute and score; repeat for multiple metrics",
    )
    analyze_run_set.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Root output directory; one subdirectory is written per analyzed tree",
    )
    analyze_run_set.add_argument(
        "--report-out",
        type=Path,
        help="Optional JSON report path; defaults to stdout",
    )
    analyze_run_set.add_argument(
        "--tree-id",
        action="append",
        help="Restrict to one or more tree IDs; defaults to every ready tree",
    )
    analyze_run_set.add_argument(
        "--node-id",
        action="append",
        help="Restrict to one or more node IDs; defaults to truth-manifest leaves",
    )
    analyze_run_set.add_argument(
        "--path-base",
        type=Path,
        default=Path("."),
        help="Base directory for relative run-list and artifact paths",
    )
    analyze_run_set.add_argument(
        "--optional-artifact",
        action="append",
        help="Expected artifact name to warn on if missing instead of blocking analysis",
    )
    analyze_run_set.add_argument("--dry-run", action="store_true")
    analyze_run_set.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip ready trees whose direct summary already exists with the requested metrics",
    )
    analyze_run_set.add_argument("--chunk-size", type=int, default=1_000_000)
    analyze_run_set.add_argument("--eps", type=float, default=1e-3)
    analyze_run_set.add_argument(
        "--representation",
        choices=["full_weight", "lora_cumulative_delta", "lora_increment_delta"],
        help="Override the default representation for the selected artifact",
    )
    analyze_run_set.add_argument(
        "--layer",
        help="Layer name or zero-based layer index. Defaults to aggregating across all layers.",
    )
    analyze_run_set.add_argument(
        "--aggregate",
        choices=["mean", "median"],
        default="mean",
        help="Layer aggregation used when --layer is omitted",
    )
    analyze_run_set.set_defaults(func=_analyze_training_run_set)

    summarize_run_set_analysis = sub.add_parser(
        "summarize-training-run-set-analysis",
        help="Summarize available per-tree direct analysis outputs under a run-set root",
    )
    summarize_run_set_analysis.add_argument(
        "--analysis-root",
        type=Path,
        required=True,
        help="Root directory containing TREEID/*_leaf_analysis/summary.json outputs",
    )
    summarize_run_set_analysis.add_argument(
        "--artifact",
        choices=["model", "merged", "adapter_chain"],
        required=True,
        help="Artifact analysis directory to summarize",
    )
    summarize_run_set_analysis.add_argument(
        "--path-base",
        type=Path,
        default=Path("."),
        help="Base directory for relative analysis-root and score paths",
    )
    summarize_run_set_analysis.add_argument(
        "--truth-manifest-root",
        type=Path,
        help=(
            "Optional local directory used to relocate cluster-authored truth-manifest paths "
            "by filename"
        ),
    )
    summarize_run_set_analysis.add_argument(
        "--tree-id",
        action="append",
        help="Restrict to one or more tree IDs; defaults to every discovered tree",
    )
    summarize_run_set_analysis.add_argument("--out", type=Path)
    summarize_run_set_analysis.add_argument("--csv-out", type=Path)
    summarize_run_set_analysis.add_argument(
        "--allow-empty",
        action="store_true",
        help="Return success even when requested tree IDs have no summary yet",
    )
    summarize_run_set_analysis.set_defaults(func=_summarize_training_run_set_analysis)

    audit_tree = sub.add_parser(
        "audit-training-tree",
        help="Audit a completed training tree run list against its ledger and artifacts",
    )
    audit_tree.add_argument("--run-list", type=Path, required=True)
    audit_tree.add_argument(
        "--ledger",
        type=Path,
        help="Ledger path; defaults to the single ledger path declared in the run list",
    )
    audit_tree.add_argument(
        "--path-base",
        type=Path,
        default=Path("."),
        help="Base directory for relative run-list artifact paths",
    )
    audit_tree.add_argument("--out", type=Path)
    audit_tree.add_argument("--allow-issues", action="store_true")
    audit_tree.add_argument(
        "--optional-artifact",
        action="append",
        help="Expected artifact name to warn on if missing instead of failing; repeatable",
    )
    audit_tree.add_argument("--skip-artifact-check", action="store_true")
    audit_tree.add_argument("--skip-parent-order-check", action="store_true")
    audit_tree.set_defaults(func=_audit_training_tree)

    audit_run_set = sub.add_parser(
        "audit-training-run-set",
        help="Audit every tree in a training run-list-set summary",
    )
    audit_run_set.add_argument("--summary", type=Path, required=True)
    audit_run_set.add_argument(
        "--path-base",
        type=Path,
        default=Path("."),
        help="Base directory for relative run-list and artifact paths",
    )
    audit_run_set.add_argument("--out", type=Path)
    audit_run_set.add_argument("--csv-out", type=Path)
    audit_run_set.add_argument("--allow-issues", action="store_true")
    audit_run_set.add_argument(
        "--optional-artifact",
        action="append",
        help="Expected artifact name to warn on if missing instead of failing; repeatable",
    )
    audit_run_set.add_argument(
        "--only-ready",
        action="store_true",
        help="Emit only trees whose completion audit is valid",
    )
    audit_run_set.add_argument("--skip-artifact-check", action="store_true")
    audit_run_set.add_argument("--skip-parent-order-check", action="store_true")
    audit_run_set.set_defaults(func=_audit_training_run_set)

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

    train = sub.add_parser(
        "plan-training", help="Validate training config and write job plan JSONL"
    )
    train.add_argument(
        "--manifest", type=Path, required=True, help="Enriched training manifest JSONL"
    )
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

    validate_data_set = sub.add_parser(
        "validate-training-data-set",
        help="Validate every assigned manifest in an assignment summary against dataset formats",
    )
    validate_data_set.add_argument("--assignment-summary", type=Path, required=True)
    validate_data_set.add_argument("--config", type=Path, required=True)
    validate_data_set.add_argument("--formats", type=Path, required=True)
    validate_data_set.add_argument("--out", type=Path)
    validate_data_set.add_argument("--allow-issues", action="store_true")
    validate_data_set.set_defaults(func=_validate_training_data_set)

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
    audit_data.add_argument(
        "--streaming",
        action="store_true",
        help="Use Hugging Face streaming mode when loading datasets",
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
    audit_samples.add_argument(
        "--streaming",
        action="store_true",
        help="Use Hugging Face streaming mode when loading datasets",
    )
    audit_samples.add_argument("--out", type=Path)
    audit_samples.add_argument("--allow-issues", action="store_true")
    audit_samples.set_defaults(func=_audit_training_samples)

    audit_sample_set = sub.add_parser(
        "audit-training-sample-set",
        help="Load samples and render prompts across an assigned tree set",
    )
    audit_sample_set.add_argument(
        "--assignment-summary",
        type=Path,
        required=True,
        help="assignment_summary.json from assign-task-data-set",
    )
    audit_sample_set.add_argument("--config", type=Path, required=True)
    audit_sample_set.add_argument(
        "--registry",
        type=Path,
        required=True,
        help="Task/data registry YAML",
    )
    audit_sample_set.add_argument(
        "--formats",
        type=Path,
        required=True,
        help="Dataset format contract YAML",
    )
    audit_sample_set.add_argument(
        "--dataset-id",
        action="append",
        help="Restrict audit to one dataset id; repeat for multiple ids",
    )
    audit_sample_set.add_argument(
        "--selection",
        choices=("one-per-dataset", "all-jobs"),
        default="one-per-dataset",
        help="Choose representative jobs or every planned job",
    )
    audit_sample_set.add_argument(
        "--max-samples",
        type=int,
        default=8,
        help="Maximum rows to render per selected job",
    )
    audit_sample_set.add_argument(
        "--split",
        help="Override the train split declared by registry/formats",
    )
    audit_sample_set.add_argument(
        "--streaming",
        action="store_true",
        help="Use Hugging Face streaming mode when loading datasets",
    )
    audit_sample_set.add_argument("--out", type=Path)
    audit_sample_set.add_argument("--allow-issues", action="store_true")
    audit_sample_set.set_defaults(func=_audit_training_sample_set)

    cache_data = sub.add_parser(
        "cache-training-datasets",
        help="Build bounded filtered dataset caches for training runs",
    )
    cache_data.add_argument("--registry", type=Path, required=True, help="Task/data registry YAML")
    cache_data.add_argument(
        "--formats",
        type=Path,
        required=True,
        help="Dataset format contract YAML",
    )
    cache_data.add_argument(
        "--dataset-id",
        action="append",
        help="Restrict cache build to one dataset id; repeat for multiple ids",
    )
    cache_data.add_argument("--out-dir", type=Path, required=True)
    cache_data.add_argument("--summary-out", type=Path)
    cache_data.add_argument(
        "--train-limit",
        type=int,
        default=10000,
        help="Accepted train rows to cache per dataset",
    )
    cache_data.add_argument(
        "--eval-limit",
        type=int,
        default=1000,
        help="Accepted eval rows to cache per dataset",
    )
    cache_data.add_argument(
        "--max-scan",
        type=int,
        help="Maximum raw rows to scan per split before reporting scan_limit",
    )
    cache_data.add_argument(
        "--min-train-rows",
        type=int,
        default=10000,
        help="Minimum accepted train rows required for success",
    )
    cache_data.add_argument(
        "--min-eval-rows",
        type=int,
        default=0,
        help="Minimum accepted eval rows required for success",
    )
    cache_data.add_argument(
        "--streaming",
        action="store_true",
        help="Use Hugging Face streaming mode while building caches",
    )
    cache_data.add_argument(
        "--sample-strategy",
        choices=("first", "seeded_shuffle", "legacy_subsample"),
        default="first",
        help=(
            "Select rows in source order, after a deterministic shuffle, or with the exact "
            "legacy sized-split subsampling rule"
        ),
    )
    cache_data.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed used by seeded_shuffle or legacy_subsample",
    )
    cache_data.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=10000,
        help="Bounded shuffle buffer for streaming datasets",
    )
    cache_data.add_argument("--overwrite", action="store_true")
    cache_data.add_argument("--allow-issues", action="store_true")
    cache_data.set_defaults(func=_cache_training_dataset_set)

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
    run_list.add_argument("--data-cache-root", type=Path)
    run_list.add_argument("--require-data-cache", action="store_true")
    _add_expected_cache_recipe_args(run_list)
    run_list.add_argument("--allow-missing-eval", action="store_true")
    run_list.add_argument("--runner-dry-run", action="store_true")
    run_list.add_argument("--allow-existing-artifacts", action="store_true")
    run_list.add_argument("--no-filesystem-check", action="store_true")
    run_list.add_argument("--allow-issues", action="store_true")
    run_list.set_defaults(func=_make_training_run_list)

    run_list_set = sub.add_parser(
        "make-training-run-list-set",
        help="Write per-tree training run lists from an assignment summary",
    )
    run_list_set.add_argument(
        "--assignment-summary",
        type=Path,
        required=True,
        help="assignment_summary.json from assign-task-data-set",
    )
    run_list_set.add_argument("--config", type=Path, required=True)
    run_list_set.add_argument("--out-dir", type=Path, required=True)
    run_list_set.add_argument("--summary-out", type=Path, help="Optional set summary JSON")
    run_list_set.add_argument(
        "--profile", type=Path, help="Local or cluster execution profile YAML"
    )
    run_list_set.add_argument(
        "--registry",
        type=Path,
        help="Dataset registry path for generated runners",
    )
    run_list_set.add_argument(
        "--formats",
        type=Path,
        help="Dataset format contract path for generated runners",
    )
    run_list_set.add_argument("--max-train-samples", type=int)
    run_list_set.add_argument("--max-eval-samples", type=int)
    run_list_set.add_argument("--data-cache-root", type=Path)
    run_list_set.add_argument("--require-data-cache", action="store_true")
    _add_expected_cache_recipe_args(run_list_set)
    run_list_set.add_argument("--allow-missing-eval", action="store_true")
    run_list_set.add_argument("--runner-dry-run", action="store_true")
    run_list_set.add_argument("--allow-existing-artifacts", action="store_true")
    run_list_set.add_argument("--no-filesystem-check", action="store_true")
    run_list_set.add_argument("--allow-issues", action="store_true")
    run_list_set.set_defaults(func=_make_training_run_list_set)

    describe_run = sub.add_parser(
        "describe-training-run",
        help="Print one row from a training run-list by array index or node id",
    )
    describe_run.add_argument("--run-list", type=Path, required=True)
    selector = describe_run.add_mutually_exclusive_group(required=True)
    selector.add_argument("--index", type=int)
    selector.add_argument("--node-id")
    describe_run.set_defaults(func=_describe_training_run)

    audit_row_data = sub.add_parser(
        "audit-training-row-data",
        help="Load and render data for one training run-list row without model training",
    )
    audit_row_data.add_argument("--run-list", type=Path, required=True)
    audit_row_selector = audit_row_data.add_mutually_exclusive_group(required=True)
    audit_row_selector.add_argument("--index", type=int)
    audit_row_selector.add_argument("--node-id")
    audit_row_data.add_argument("--registry", type=Path)
    audit_row_data.add_argument("--formats", type=Path)
    audit_row_data.add_argument("--max-train-samples", type=int)
    audit_row_data.add_argument("--max-eval-samples", type=int)
    audit_row_data.add_argument("--data-cache-root", type=Path)
    audit_row_data.add_argument("--require-data-cache", action="store_true")
    _add_expected_cache_recipe_args(audit_row_data)
    audit_row_data.add_argument("--allow-missing-eval", action="store_true")
    audit_row_data.add_argument(
        "--streaming",
        action="store_true",
        help="Use Hugging Face streaming mode if the row falls back to dataset loading",
    )
    audit_row_data.add_argument("--allow-issues", action="store_true")
    audit_row_data.set_defaults(func=_audit_training_row_data)

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
    run_row.add_argument("--data-cache-root", type=Path)
    run_row.add_argument("--require-data-cache", action="store_true")
    _add_expected_cache_recipe_args(run_row)
    run_row.add_argument("--allow-missing-eval", action="store_true")
    run_row.add_argument(
        "--override-max-steps",
        type=int,
        help="Temporarily override trainer.max_steps for this selected row",
    )
    run_row.add_argument(
        "--report-to",
        action="append",
        help=(
            "Temporarily override trainer.report_to for this selected row; repeat the flag "
            "or comma-separate reporters such as wandb,tensorboard"
        ),
    )
    run_row.add_argument(
        "--run-name",
        help="Temporarily override trainer.run_name for this selected row",
    )
    run_row.add_argument("--dry-run", action="store_true")
    run_row.set_defaults(func=_run_training_row)

    prune_parent = sub.add_parser(
        "prune-training-parent-artifact",
        help="Prune a completed internal parent's model after all direct children succeed",
    )
    prune_parent.add_argument("--run-list", type=Path, required=True)
    prune_selector = prune_parent.add_mutually_exclusive_group(required=True)
    prune_selector.add_argument("--index", type=int)
    prune_selector.add_argument("--node-id")
    prune_parent.add_argument("--audit-out", type=Path)
    prune_parent.add_argument(
        "--success-not-before",
        help="Require every direct child's success event to be from this attempt or later",
    )
    prune_parent.add_argument("--dry-run", action="store_true")
    prune_parent.set_defaults(func=_prune_training_parent_artifact)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
