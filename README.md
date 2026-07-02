# WeightTraits

Private rebuild of the ELLMTrees results with cleaner project structure, independent verification, and reproducible paper outputs.

The rule of this repository is simple: old ELLMTrees artifacts are reference inputs, not unquestioned truth. Every result that enters the rebuilt paper should have a recorded command, environment, input manifest, output digest, and at least one verification check.

## Goals

- Rebuild the ELLMTrees analyses in a maintainable Python package rather than a script pile.
- Make tree generation flexible enough to test many topology/configuration hypotheses.
- Keep fast unit tests and smoke tests running continuously.
- Support both local development and cluster execution with explicit configs.
- Cross-check regression results in R, not only Python/statsmodels.
- Preserve provenance for every figure and table needed to remake the ICLR draft.
- Compare rebuilt outputs against ELLMTrees artifacts wherever the old outputs are available.

## Initial Layout

```text
WeightTraits/
├── src/weighttraits/       # Importable project code
├── tests/                  # Fast tests for invariants and audit helpers
├── configs/                # Local, cluster, and experiment configs
├── docs/                   # Multi-week roadmap and audit protocol
├── paper/                  # Paper rebuild registry and eventual draft source
├── reports/                # Generated audit reports; ignored unless curated
└── scripts/                # Thin operational wrappers, including R checks
```

## First Commands

```bash
cd /Users/shannon/Desktop/phylo/WeightTraits
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m weighttraits.cli generate-tree --config configs/experiments/flexible_flow.yaml --out reports/flexible_tree_manifest.jsonl
PYTHONPATH=src python -m weighttraits.cli audit-ellmtrees --source ../ELLMTrees --out reports/ellmtrees_inventory.json
```

See [docs/ROADMAP.md](docs/ROADMAP.md) for the full step-by-step rebuild plan.
See [docs/PIPELINE_FLOW.md](docs/PIPELINE_FLOW.md) for the topology → task/data → training → analysis separation.
See [docs/TREE_GENERATOR_MATH.md](docs/TREE_GENERATOR_MATH.md) and [docs/RUN_TREE_EXAMPLES.md](docs/RUN_TREE_EXAMPLES.md) for topology math and runnable examples.
See [docs/TRAINER_DESIGN.md](docs/TRAINER_DESIGN.md) for training plans, prompt resolution, LoRA artifacts, and stopping rules.
See [docs/RECOVERY_SCORING.md](docs/RECOVERY_SCORING.md) for TP/FP/FN, RF, clade recovery, exact recovery, and SE definitions.
See [docs/DISTANCE_METRICS.md](docs/DISTANCE_METRICS.md) for cosine, CKA, and the metric-registry design.
See [docs/STREAMING_DISTANCE_CUBES.md](docs/STREAMING_DISTANCE_CUBES.md) for the chunked distance-cube engine.
See [docs/LORA_DISTANCE_MODEL.md](docs/LORA_DISTANCE_MODEL.md) for cumulative LoRA `B @ A` semantics.
