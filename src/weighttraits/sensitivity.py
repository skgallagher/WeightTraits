"""Matched task-replacement experiments; separate from the frozen paper cohort.

Run from the WeightTraits root: PYTHONPATH=src python -m weighttraits.sensitivity --help.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import numpy as np
import yaml

from weighttraits.taskdata.assignment import assign_task_data, load_manifest_rows, write_manifest_rows
from weighttraits.training.planner import build_training_jobs, load_training_config
from weighttraits.training.runlist import build_training_run_list, write_training_run_list

FROZEN = Path("examples/training/confirm_paper_numbers")
FAMILIES = {
    "flan": "flan_t5_full_finetune_legacy_seq2seq_2000",
    "llama": "llama32_1b_full_finetune_legacy_causal_2000",
}
ARMS = ("mixed_replacement", "summarization", "classification", "qa", "translation")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def indices(value):
    result = []
    for part in value.split(","):
        bounds = [int(x) for x in part.split("-")]
        if len(bounds) == 1:
            result.append(bounds[0])
        elif len(bounds) == 2 and bounds[0] <= bounds[1]:
            result.extend(range(bounds[0], bounds[1] + 1))
        else:
            raise ValueError(f"invalid tree range: {part}")
    if not result or len(set(result)) != len(result) or min(result) < 1 or max(result) > 50:
        raise ValueError("select unique tree IDs in 1..50")
    return sorted(result)


def prepare(out, *, ellm_root, models=("flan", "llama"), arms=ARMS, trees=range(1, 51), seed=240924):
    """Freeze assignments once, then use identical assignments across architectures."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite experiment: {out}")
    if not models or not arms or len(set(models)) != len(models) or len(set(arms)) != len(arms):
        raise ValueError("models and arms must be nonempty and unique")
    if set(models) - FAMILIES.keys() or set(arms) - set(ARMS):
        raise ValueError("unknown model or arm")
    trees = indices(",".join(map(str, trees)))
    pool = yaml.safe_load((FROZEN / "paper_task_families.yaml").read_text())["task_families"]
    configs = {m: load_training_config(FROZEN / f"{FAMILIES[m]}.yaml") for m in models}
    # Snapshot external analysis code so concurrent review edits cannot change this experiment.
    out.mkdir(parents=True)
    source = out / "source"
    shutil.copytree(Path(ellm_root) / "scripts/mother_baseline", source / "mother",
                    ignore=shutil.ignore_patterns("__pycache__", ".git"))
    shutil.copyfile(Path(ellm_root) / "analysis/reconstruction/tree_space.py", source / "tree_space.py")
    sources = {str(p): digest(p) for p in source.rglob("*") if p.is_file()}
    for p in Path("src/weighttraits").rglob("*.py"):
        sources[str(p)] = digest(p)
    inputs = {}
    cells = []
    for arm in arms:
        for tree in trees:
            tree_id = f"confirm_paper_tree_{tree:03d}"
            original = FROZEN / "trees" / f"{tree_id}.manifest.jsonl"
            inputs[str(original)] = digest(original)
            rows = assign_task_data(load_manifest_rows(original), pool, seed=seed + tree,
                                    policy="per_node", task_families=None if arm == "mixed_replacement" else [arm])
            truth = out / "manifests" / arm / f"{tree_id}.manifest.jsonl"
            write_manifest_rows(rows, truth)
            datasets = [r["dataset_id"] for r in rows if r.get("grow", "train") == "train"]
            for model in models:
                cell_dir = out / "cells" / model / arm / tree_id
                cell_dir.mkdir(parents=True)
                config = copy.deepcopy(configs[model])
                config["output_root"] = str(cell_dir / "checkpoints")
                config["model_family"] += f"__sensitivity_{arm}"
                config["protocol_id"] += f"__task_replacement_v1_{arm}"
                dump(cell_dir / "effective_training_config.json", config)
                template = FROZEN / f"{FAMILIES[model]}_training_runlists/run_lists/{tree_id}.runs.jsonl"
                inputs[str(template)] = digest(template)
                options = load_manifest_rows(template)[0]["runner"]["options"]
                run_list = cell_dir / "runs.jsonl"
                ledger = cell_dir / "training_ledger.jsonl"
                runs = build_training_run_list(build_training_jobs(rows, config), run_list_path=run_list,
                    ledger_path=ledger, runner_entrypoint="weighttraits.cli run-training-row",
                    runner_options=options, check_filesystem=False)
                if not runs.valid:
                    raise ValueError(runs.to_dict())
                write_training_run_list(runs, run_list)
                cells.append(dict(index=len(cells), model=model, family=FAMILIES[model], arm=arm,
                    tree_id=tree_id, tree_index=tree, directory=str(cell_dir), manifest=str(truth),
                    run_list=str(run_list), ledger=str(ledger), n_training_nodes=len(runs.runs),
                    n_unique_datasets=len(set(datasets)), n_repeat_assignments=len(datasets)-len(set(datasets)),
                    manifest_sha256=digest(truth), run_list_sha256=digest(run_list)))
    for p in (FROZEN / "paper_task_families.yaml",):
        inputs[str(p)] = digest(p)
    plan = dict(schema="weighttraits.task_replacement.v1", assignment_seed_base=seed,
        assignment_policy="per_node", sampling="uniform task family, then uniform dataset, with replacement",
        training_seed=42, max_steps=2000, models=list(models), arms=list(arms), trees=trees,
        n_cells=len(cells), n_training_nodes=sum(c["n_training_nodes"] for c in cells),
        source_hashes=sources, input_hashes=inputs, cells=cells)
    dump(out / "experiment.json", plan)
    return plan


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    for p, expected in plan["source_hashes"].items():
        if digest(p) != expected:
            raise ValueError(f"analysis/training source changed since prepare: {p}; prepare a new experiment")
    for cell in plan["cells"]:
        for name in ("manifest", "run_list"):
            if digest(cell[name]) != cell[f"{name}_sha256"]:
                raise ValueError(f"changed {name}: {cell[name]}")
    return plan


def load_external(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def matrix_diagnostics(labels, matrix, truth):
    from weighttraits.phylo.atteson import atteson_margin
    from weighttraits.phylo.additivity import four_point_additivity
    margin = atteson_margin(labels, matrix, truth).as_dict()
    # This is fitted to a known topology; never relabel it a theorem certificate.
    margin.pop("theorem_certified", None)
    return dict(fitted_atteson=margin,
                additivity=four_point_additivity(labels, matrix, max_quartets=None).as_dict())


def analyze_layers(directory, bhv_source):
    """Use only nonzero cosine layers; compute both NJ(mean D) and BHV mean trees."""
    from weighttraits.analysis.direct import _neighbor_joining_newick, _score_newick_with_dendropy
    directory = Path(directory)
    labels = json.loads((directory / "models.json").read_text())
    truth = (directory / "truth_manifest.newick").read_text()
    names = json.loads((directory / "layers.json").read_text())
    with np.load(directory / "direct_distance_layers.npz") as data:
        cube = data["cosine"]
    if not np.isfinite(cube).all():
        raise ValueError("nonfinite layer distances")
    keep = [i for i, matrix in enumerate(cube) if np.any(matrix[np.triu_indices(len(labels), 1)] > 0)]
    if not keep:
        raise ValueError("no informative weight layers")
    matrix = np.mean(cube[keep], axis=0)
    np.fill_diagonal(matrix, 0)
    np.save(directory / "trained_layer_mean_cosine.npy", matrix)
    ts = load_external(bhv_source, "sensitivity_tree_space")
    trees = [ts.from_newick(_neighbor_joining_newick(labels, cube[i])) for i in keep]
    for tree in trees:
        tree.pendant = {k: max(0., v) for k, v in tree.pendant.items()}
    estimates = {
        "frechet_cosine": ts.to_newick(ts.frechet_mean(trees, n_epochs=25, seed=0)),
        "nj_mean_cosine": _neighbor_joining_newick(labels, matrix),
    }
    scores = {}
    for name, newick in estimates.items():
        path = directory / f"{name}.newick"
        path.write_text(newick + "\n")
        scores[name] = _score_newick_with_dendropy(truth, newick, truth_manifest=directory / "truth_manifest.newick", estimate=path)
    result = dict(n_layers=len(names), n_trained_layers=len(keep),
        retained_layers=[names[i] for i in keep], scores=scores, **matrix_diagnostics(labels, matrix, truth))
    dump(directory / "sensitivity_diagnostics.json", result)
    return result


def preflight_cell(cell):
    """Validate cached datasets and root initialization without training."""
    from weighttraits.training.executor import prepare_training_data
    from weighttraits.training.datasets import load_dataset_registry, DatasetCacheRecipe
    from weighttraits.training.data_formats import load_dataset_format_specs
    from weighttraits.training.runlist import load_training_run_specs
    from huggingface_hub import snapshot_download
    runs = load_training_run_specs(cell["run_list"])
    options = runs[0].runner["options"]
    registry = load_dataset_registry(options["registry_path"])
    formats = load_dataset_format_specs(options["formats_path"])
    seen = set()
    for run in runs:
        if run.dataset_id in seen:
            continue
        seen.add(run.dataset_id)
        prepare_training_data(run, registry, formats, data_cache_root=options["data_cache_root"],
            require_data_cache=True, expected_cache_recipe=DatasetCacheRecipe(**options["expected_cache_recipe"]),
            max_train_samples=options["max_train_samples"], max_eval_samples=options["max_eval_samples"])
    root = snapshot_download(runs[0].job["base_model"], revision=runs[0].job["base_model_revision"], local_files_only=True)
    from weighttraits.distances.readers import reader_from_path
    reader_from_path(root)
    return root


def run_cell(plan_path, index, phase):
    from weighttraits.training.ledger import latest_status_by_node, load_ledger_events
    plan = load_plan(plan_path)
    if not 0 <= index < len(plan["cells"]):
        raise ValueError("cell index out of range")
    cell = plan["cells"][index]
    directory = Path(cell["directory"])
    rows = load_manifest_rows(cell["run_list"])
    if phase == "preflight":
        preflight_cell(cell)
        return
    if phase == "train":
        preflight_cell(cell)
        statuses = latest_status_by_node(load_ledger_events(cell["ledger"])) if Path(cell["ledger"]).exists() else {}
        for i, row in enumerate(rows):
            event = statuses.get(row["node_id"])
            if event and event.status == "completed" and event.step == row["job"]["trainer"]["max_steps"]:
                if not Path(row["expected_artifacts"]["model"]).is_dir():
                    raise FileNotFoundError(row["expected_artifacts"]["model"])
                continue
            if Path(row["output_dir"]).exists():
                raise RuntimeError(f"partial training output requires inspection: {row['output_dir']}")
            subprocess.run([sys.executable, "-m", "weighttraits.cli", "run-training-row",
                            "--run-list", cell["run_list"], "--index", str(i)], check=True)
        return
    statuses = latest_status_by_node(load_ledger_events(cell["ledger"]))
    for row in rows:
        event = statuses.get(row["node_id"])
        if not event or event.status != "completed" or event.step != row["job"]["trainer"]["max_steps"]:
            raise ValueError(f"incomplete training: {row['node_id']}")
    source = Path(plan_path).parent / "source"
    if phase == "analyze":
        from weighttraits.analysis.direct import analyze_training_ledger_direct
        analyze_training_ledger_direct(cell["ledger"], truth_manifest=cell["manifest"],
            out_dir=directory / "analysis", artifact="model", metrics=["cosine"], path_base=Path.cwd())
        result = analyze_layers(directory / "analysis", source / "tree_space.py")
    else:
        root = preflight_cell(cell)
        manifest = directory / "mother_manifest.jsonl"
        write_manifest_rows([dict(node_id="root", parent=None, path=root)] + [
            dict(node_id=r["node_id"], parent=r["parent_id"], path=str(Path(r["expected_artifacts"]["model"]).resolve()))
            for r in rows], manifest)
        output = directory / "mother"
        subprocess.run([sys.executable, str(source / "mother/run_mother.py"), "--manifest", str(manifest),
            "--arch", "t5" if cell["model"] == "flan" else "llama", "--mode", "both", "--distance", "rms",
            "--save-matrices", "--out", str(output)], check=True)
        records = json.loads((output / "per_run.json").read_text())
        if len(records) != 1 or "error" in records[0]:
            raise RuntimeError(f"MoTHer failed: {records}")
        import pandas as pd
        from weighttraits.analysis.direct import truth_newick_from_manifest, _manifest_leaf_ids
        matrices = list((output / "matrices").glob("*_observed_dist.csv"))
        if len(matrices) != 1:
            raise ValueError("expected exactly one observed-node MoTHer distance matrix")
        frame = pd.read_csv(matrices[0], index_col=0)
        leaves = _manifest_leaf_ids(cell["manifest"])
        result = dict(results=records[0], **matrix_diagnostics(leaves, frame.loc[leaves, leaves].to_numpy(),
                                                            truth_newick_from_manifest(cell["manifest"])))
        dump(output / "sensitivity_diagnostics.json", result)
    dump(directory / f"{phase}.receipt.json", dict(experiment_sha256=digest(plan_path),
         cell_index=index, phase=phase, ledger_sha256=digest(cell["ledger"]), result=result))


def submit(plan_path, *, execute=False, throttle=2, python=sys.executable, selection=None):
    plan = load_plan(plan_path)
    if throttle < 1:
        raise ValueError("throttle must be positive")
    selected = list(range(len(plan["cells"]))) if selection is None else selection
    if not selected or len(set(selected)) != len(selected) or min(selected) < 0 or max(selected) >= len(plan["cells"]):
        raise ValueError("invalid cell selection")
    logdir = Path(plan_path).parent / "logs"
    logdir.mkdir(exist_ok=True)
    commands, jobs = [], {}
    for phase in ("train", "analyze", "mother"):
        command = ["sbatch", "--parsable", "--account=statds", "--partition=all", f"--chdir={Path.cwd()}",
            "--cpus-per-task=4", "--mem=64G", f"--job-name=sens-{phase}",
            f"--array={','.join(map(str, selected))}%{throttle}",
            f"--output={logdir}/{phase}_%A_%a.out", f"--error={logdir}/{phase}_%A_%a.err"]
        if phase == "train":
            command += ["--gres=gpu:nvidia-l40:1", "--time=2-00:00:00"]
        else:
            command += ["--time=06:00:00", f"--dependency=afterok:{jobs.get('train', 'TRAIN_JOB_ID')}"]
        invocation = [python, "-m", "weighttraits.sensitivity", "run", "--plan", str(plan_path), "--phase", phase]
        command += ["--wrap", "export PYTHONPATH=src OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4; "
                    + shlex.join(invocation) + ' --index "$SLURM_ARRAY_TASK_ID"']
        commands.append(command)
        print(shlex.join(command), flush=True)
        if execute:
            job = subprocess.check_output(command, text=True).strip().split(";")[0]
            if not job.isdigit():
                raise RuntimeError(f"unexpected sbatch result: {job}")
            jobs[phase] = job
            dump(Path(plan_path).parent / f"submission_{jobs['train']}.json", dict(jobs=jobs, cells=selected, commands=commands))
    return commands


def collect(plan_path):
    plan = load_plan(plan_path)
    records = []
    for cell in plan["cells"]:
        receipts = {}
        for phase in ("analyze", "mother"):
            path = Path(cell["directory"]) / f"{phase}.receipt.json"
            receipt = json.loads(path.read_text())  # Missing or failed cells are never silently excluded.
            if receipt["experiment_sha256"] != digest(plan_path) or receipt["ledger_sha256"] != digest(cell["ledger"]):
                raise ValueError(f"stale receipt: {path}")
            receipts[phase] = receipt["result"]
        for method, result in receipts.items():
            common = {k: cell[k] for k in ("model", "arm", "tree_id", "n_repeat_assignments")}
            common.update(distance="cosine" if method == "analyze" else "mother_rms",
                fitted_atteson_margin=result["fitted_atteson"]["bottleneck_margin"],
                additivity_score=None if result["additivity"]["all"] is None else result["additivity"]["all"]["mean_additivity"])
            if method == "analyze":
                for builder, score in result["scores"].items():
                    records.append(dict(common, builder=builder, eligible=score["n_truth_splits"] > 0,
                        clade_recovery=score["clade_recovery"] if score["n_truth_splits"] else None,
                        full_recovery=score["polytomy_aware_exact_recovery"] if score["n_truth_splits"] else None,
                        false_negative=score["false_negative"], rf=score["rf"]))
            else:
                records.append(dict(common, builder="mother", details=result["results"]))
    dump(Path(plan_path).parent / "results.json", records)
    csv_rows = [{k: v for k, v in row.items() if k != "details"} for row in records]
    with (Path(plan_path).parent / "results.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for r in csv_rows for k in r}))
        writer.writeheader()
        writer.writerows(csv_rows)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ellm-root", type=Path, required=True)
    p.add_argument("--models", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--trees", default="1-50")
    p.add_argument("--seed", type=int, default=240924)
    for name in ("run", "submit", "collect"):
        p = sub.add_parser(name)
        p.add_argument("--plan", type=Path, required=True)
        if name == "run":
            p.add_argument("--index", type=int, required=True)
            p.add_argument("--phase", choices=["preflight", "train", "analyze", "mother"], required=True)
        if name == "submit":
            p.add_argument("--execute", action="store_true")
            p.add_argument("--throttle", type=int, default=2)
            p.add_argument("--python", default=sys.executable)
            p.add_argument("--cells", help="comma-separated zero-based cell indexes; omit for all")
    args = parser.parse_args()
    if args.command == "prepare":
        plan = prepare(args.out, ellm_root=args.ellm_root, models=args.models, arms=args.arms,
                       trees=indices(args.trees), seed=args.seed)
        print(f"Prepared {plan['n_cells']} tree/model/arm cells; {plan['n_training_nodes']} training nodes")
    elif args.command == "run":
        run_cell(args.plan, args.index, args.phase)
    elif args.command == "submit":
        submit(args.plan, execute=args.execute, throttle=args.throttle, python=args.python,
               selection=None if args.cells is None else [int(x) for x in args.cells.split(",")])
    else:
        collect(args.plan)


if __name__ == "__main__":
    main()
