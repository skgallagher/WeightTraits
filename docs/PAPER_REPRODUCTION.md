# Paper Reproduction Plan

The ICLR results should eventually be rebuilt as a consequence of registered artifacts, not as a manual copy of old figures and tables.

Frozen target: `../ELLMTrees-paper/iclr_draft_v2.tex`, pinned in `paper/reference_registry.yaml` at
SHA-256 `2b74779ab7c26707efc9a36059fe7fa5226f71737b71b5a86f74db9dacab32c6` on
2026-07-09. Use it for result traceability and stale-claim audits; do not edit it from WeightTraits.

The scientific target is distributional reproduction: comparable topology samples, training
conditions, distance behavior, and result distributions within declared tolerances. Exact replay of
historical seeds, byte-identical checkpoints, and identical individual trees is not required.
Early stopping is an intentional WeightTraits trainer feature and should be recorded in run
provenance rather than treated as a reproduction failure.

## Registry Files

- `paper/figure_registry.yaml`: one entry per figure panel or standalone figure.
- `paper/table_registry.yaml`: one entry per table.
- `paper/recovery_registry.yaml`: registered whitebox recovery summary artifacts that can feed
  recovery tables.
- `paper/reference_registry.yaml`: pinned frozen-draft, figure, table, source-script, and source-data
  references from the old ELLMTrees paper/results repos.
- `paper/ellmtrees_variants_registry.yaml`: row mapping for the generated legacy `tab:variants`
  reference table.

Each entry should include:
- artifact path
- source command
- source inputs
- expected output digest or numeric tolerance
- paper location
- verification status

Table registry entries can be checked with:

```bash
PYTHONPATH=src python -m weighttraits.cli validate-table-registry \
  --registry paper/table_registry.yaml \
  --require-outputs
```

Use `--require-outputs` for paper-build gates and omit it when only source inputs should be checked.

The current ELLMTrees reference surface can be checked with:

```bash
PYTHONPATH=src python -m weighttraits.cli validate-reference-registry \
  --registry paper/reference_registry.yaml
```

When `active_draft` is set, this also checks that every `fig:` and `tab:` label in the active draft
has a registry entry and that registered draft labels have not gone stale.

The old ELLMTrees `tab:variants` reference table can be regenerated from pinned CSVs with:

```bash
PYTHONPATH=src python -m weighttraits.cli make-ellmtrees-variants-table \
  --registry paper/ellmtrees_variants_registry.yaml \
  --out reports/paper/ellmtrees_variants_reference.json \
  --csv-out reports/paper/ellmtrees_variants_reference.csv
```

The live draft `tab:behavior_holdout` table can be materialized as the current paper-grounded
reference with:

```bash
PYTHONPATH=src python -m weighttraits.cli make-behavior-holdout-table \
  --draft ../ELLMTrees-paper/iclr_draft_v2.tex \
  --out reports/paper/behavior_holdout_reference.json \
  --csv-out reports/paper/behavior_holdout_reference.csv
```

Generated paper tables can be compared against a paper-grounded reference with:

```bash
PYTHONPATH=src python -m weighttraits.cli run-table-comparisons \
  --registry paper/table_registry.yaml \
  --out reports/paper/table_comparison_validation.json
```

Individual table artifacts can also be compared directly:

```bash
PYTHONPATH=src python -m weighttraits.cli compare-table-artifacts \
  --reference reports/paper/ellmtrees_variants_reference.csv \
  --candidate reports/paper/ellmtrees_variants_reference.json \
  --key-column variant_id \
  --out reports/paper/ellmtrees_variants_reference_compare.json
```

Use `--numeric-column`, `--atol`, `--rtol`, and `--ignore-column` when comparing a rebuilt
WeightTraits table to a pinned latest-paper reference whose metadata columns or floating-point
formatting may differ.

## Rebuild Order

1. Treat the latest paper state as ground truth and freeze the corresponding paper artifacts as references.
2. Generate clean confirm-paper-number topology sets from the same distribution, without reusing old
   exact seeds.
3. Rebuild whitebox recovery figures.
4. Rebuild layer and scope-condition figures.
5. Rebuild behavioral/regression tables with R cross-checks.
6. Rebuild HF-zoo and blackbox validation figures.
7. Compile the paper from a clean checkout.
8. Run a stale-claim audit against `../ELLMTrees-paper/iclr_draft_v2.tex`, older draft files,
   `CLAUDE.md`, and handoff notes.

For the legacy Flan `lora_full_ft_approx` row, distinguish declared from executed scope. The old and
new replay configs declare `q,k,v,o,wi,wo`, but PEFT suffix matching finds no `wi` module in gated
Flan-T5; both executions resolve to `q,k,v,o,wo` across 168 adapter modules. Preserve that row as a
legacy executable-condition reference. Treat explicit `wi_0,wi_1` targeting as a separate corrected
ablation, not as an interchangeable rerun of the registered legacy row.

Use `docs/EXPERIMENT_CHECKLIST.md` as the queue-facing control board while moving through these
steps. It records the current Wright jobs, paper targets, pre-queue gates, and blocked/scaffolded
experiment families.

## Open Paper-Facing Result Changes

- **Merged versus cumulative LoRA is a real analysis choice.** On the first 14 clean Flan LoRA q/v
  trees, cosine recovery is stronger for merged weights (`94.0%` clade recovery, `85.7%` PAER,
  mean FN `0.143`) than for cumulative adapter deltas (`89.9%`, `71.4%`, mean FN `0.286`). A merged
  leaf is `theta_base + delta_path`, whereas the adapter-chain representation is `delta_path` alone.
  L2 cancels the common base and currently matches across the two representations; cosine and
  correlation do not. Before rebuilding `tab:variants`, audit the old Flan `TRAINED_ONLY=1`
  representation, name the new representation explicitly, and retain a paired same-tree sensitivity
  analysis. The cumulative path is also operationally attractive: representative tree 001 stores
  `78 MiB` of adapter artifacts versus `12.94 GiB` of merged checkpoints (about `166x` smaller), and
  the first 14-tree cumulative analysis took `1:44` versus `12:22` for merged analysis (about `7.1x`
  faster). Treat the recovery values and timing as provisional until all 50 LoRA trees finish and a
  controlled cold/warm-cache resource benchmark is run.

## Paper Gate

A result can enter the rebuilt paper only when:
- it is generated by a command in this repository;
- its inputs are declared;
- it has passed the relevant unit/smoke/numerical/R checks;
- deviations from ELLMTrees are documented.
