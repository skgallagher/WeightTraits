# WeightTraits

[![CI](https://github.com/skgallagher/WeightTraits/actions/workflows/ci.yml/badge.svg)](https://github.com/skgallagher/WeightTraits/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Reconstruct and audit fine-tuning lineages from model weights and behavior.**

Fine-tuning leaves structure behind. WeightTraits helps researchers measure that structure, infer
how related models descend from one another, and test the inferred lineage against known ground
truth. It provides one workflow for generating training trees, running full-fine-tuning or LoRA
experiments, computing weight and behavior distances, reconstructing phylogenies, and auditing
recovery.

WeightTraits is designed for reproducible research: generated results are tied to declared commands,
environments, input manifests, output artifacts, and verification checks.

The completed 350-tree Flan-T5 experiment suite used **1,427.8 NVIDIA L40 GPU-hours** on the Wright
Slurm cluster; individual conditions averaged 3.1–4.9 hours per tree. See the
[compute and runtime record](docs/COMPUTE_RUNTIME.md) for per-condition runtimes, CPU analysis cost,
Llama benchmarks, partial-run accounting, and the exact Slurm job IDs.

> **Status:** active research software. Topology generation, recovery scoring, distance analysis,
> provenance checks, and local smoke workflows are tested. Model training and behavioral probing
> require additional dependencies and, for realistic experiments, substantial compute.

## What you can do

- Generate fixed, balanced, branching-process, or pruned-backbone training lineages.
- Assign task and dataset policies independently of topology.
- Plan and execute full-fine-tuning or cumulative-LoRA training trees.
- Build streaming distance cubes from full weights, merged LoRA weights, or adapter chains.
- Reconstruct neighbor-joining trees and score clade recovery, exact recovery, RF, FP, and FN.
- Analyze behavioral outputs and compare weight-space with behavior-space structure.
- Build provenance-checked tables, figures, and comparison artifacts.

The high-level workflow is:

```text
topology → task/data assignment → training → weight or behavior distances
         → reconstructed tree → recovery and reproducibility audit
```

## Quick start

WeightTraits requires Python 3.11 or newer.

```bash
git clone https://github.com/skgallagher/WeightTraits.git
cd WeightTraits

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

Generate and audit a small stochastic lineage:

```bash
wt generate-tree \
  --config examples/trees/poisson_branching.yaml \
  --out reports/quickstart_tree.jsonl

wt topology-audit --manifest reports/quickstart_tree.jsonl
```

The audit reports the node and leaf counts, maximum depth, polytomies, truth splits, and whether a
binary neighbor-joining reconstruction can introduce resolution-only false positives.

Run the test suite:

```bash
python -m pytest
```

Use `wt --help` to see the complete command surface and `wt <command> --help` for command-specific
arguments.

## Installation options

The base install contains topology, manifests, and lightweight utilities. Add only the extras needed
for your workflow:

| Extra | Purpose |
|---|---|
| `analysis` | Scientific Python stack, tree libraries, plotting, and weight-file support |
| `training` | Transformers, PEFT, datasets, evaluation, and experiment tracking |
| `blackbox` | Hosted-model clients, sentence embeddings, and black-box probing |
| `dev` | Tests and linting |

For a full research environment:

```bash
python -m pip install -e ".[analysis,training,blackbox,dev]"
```

The repository also includes [`environment.local.yml`](environment.local.yml) and
[`environment.cluster.yml`](environment.cluster.yml) for Conda-based setups. GPU drivers, CUDA,
scheduler configuration, model access, and API credentials remain environment-specific.

## Where to go next

- [Documentation index](docs/README.md): guides grouped by user goal.
- [Pipeline overview](docs/PIPELINE_FLOW.md): how topology, training, and analysis stay separated.
- [Tree examples](docs/RUN_TREE_EXAMPLES.md): runnable topology configurations.
- [Training examples](examples/training/README.md): dry runs, tiny local fixtures, and lineage smokes.
- [Recovery scoring](docs/RECOVERY_SCORING.md): exact definitions for RF, FP/FN, clade recovery,
  and polytomy-aware exact recovery.
- [Distance metrics](docs/DISTANCE_METRICS.md): cosine, correlation, L2, CKA, and registry semantics.
- [Reproducible result artifacts](docs/PAPER_REPRODUCTION.md): provenance gates and registered
  tables and figures.
- [Compute and runtime](docs/COMPUTE_RUNTIME.md): observed wall times, hardware, GPU-hours, and
  Slurm provenance for the recorded experiment sets.

## Repository layout

```text
WeightTraits/
├── src/weighttraits/   # Importable library and CLI
├── tests/              # Unit, smoke, and numerical checks
├── examples/           # Runnable topology, training, behavior, and recovery fixtures
├── configs/            # Local, cluster, task/data, and experiment configuration
├── docs/               # User guides, scientific definitions, and maintainer records
├── paper/              # Figure/table registries and paper-facing provenance
├── reports/            # Generated or curated audit artifacts
└── scripts/            # Thin operational and independent-check wrappers
```

## Reproducibility boundaries

Core examples and tests are self-contained. Reproducing full research results additionally requires
the registered model artifacts, experiment outputs, and reference inputs named in the provenance
registries. External or historical results may be used as explicit comparison targets, but never as
hidden inputs to native WeightTraits statistics.

Large checkpoints, downloaded datasets, API responses, and cluster outputs are intentionally not
stored in Git. Generated artifacts carry their input paths and provenance so missing external inputs
fail visibly instead of silently changing an analysis.

## Contributing and citation

Bug reports, documentation fixes, and focused pull requests are welcome; see
[CONTRIBUTING.md](CONTRIBUTING.md). If you use WeightTraits in research, see
[CITATION.cff](CITATION.cff). The software is released under the [MIT License](LICENSE).
