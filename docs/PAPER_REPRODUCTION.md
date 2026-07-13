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
- `paper/weighttraits_variants_registry.yaml`: row mapping from versioned WeightTraits run-set
  summaries to rebuilt Flan-T5 `tab:variants` rows.
- `paper/weighttraits_completed_conditions_registry.yaml`: five completed native conditions used
  for an interim independent comparison before the legacy-scope row finishes.
- `paper/weighttraits_runset_diagnostics_registry.yaml`: native multi-metric recovery,
  four-point-additivity, and Atteson-margin analysis inputs.

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

The rebuilt Flan-T5 rows can be materialized after the `analysis_v20260713` rollups are refreshed:

```bash
PYTHONPATH=src python -m weighttraits.cli make-weighttraits-variants-table \
  --registry paper/weighttraits_variants_registry.yaml \
  --out reports/paper/weighttraits_variants_rebuild.json \
  --csv-out reports/paper/weighttraits_variants_rebuild.csv
```

Candidate computations are independent: legacy ELLMTrees scripts, aggregate CSVs, and reference
figures are not inputs to the WeightTraits rollups or plots. The old artifacts are retained only
for post-hoc comparison after a native candidate artifact has been generated.

Fresh robustness and additivity figures are built from a provenance-bearing long-form artifact:

```bash
PYTHONPATH=src python -m weighttraits.cli make-weighttraits-runset-diagnostics \
  --registry paper/weighttraits_runset_diagnostics_registry.yaml \
  --out reports/paper/weighttraits_runset_diagnostics.json \
  --csv-out reports/paper/weighttraits_runset_diagnostics.csv
PYTHONPATH=src python -m weighttraits.cli plot-weighttraits-metric-robustness \
  --diagnostics reports/paper/weighttraits_runset_diagnostics.json \
  --out reports/paper/weighttraits_metric_robustness.svg
PYTHONPATH=src python -m weighttraits.cli plot-weighttraits-additivity-recovery \
  --diagnostics reports/paper/weighttraits_runset_diagnostics.json \
  --metric cosine \
  --out reports/paper/weighttraits_additivity_recovery.svg
```

The builder requires all source rows to report `analysis_engine: direct`, the registered
representation, and the expected tree count. The plotting commands accept only the generated
`weighttraits.runset_diagnostics.v1` JSON with `producer: weighttraits`.

Pulled Wright analysis directories can be re-rolled without editing their cluster-authored
provenance paths:

```bash
PYTHONPATH=src python -m weighttraits.cli summarize-training-run-set-analysis \
  --analysis-root outputs/analysis_v20260713/lora_qkv \
  --artifact adapter_chain \
  --truth-manifest-root examples/training/confirm_paper_numbers/assigned_manifests \
  --out outputs/analysis_v20260713/lora_qkv_summary.json
```

Plot only the provenance-bearing native candidate JSON:

```bash
PYTHONPATH=src python -m weighttraits.cli plot-weighttraits-variants \
  --table reports/paper/weighttraits_variants_rebuild.json \
  --out reports/paper/weighttraits_variants_rebuild.svg
```

The run-set rollup computes ordering from saved metric matrices and truth manifests. Ordering uses
the paper's direct-child-of-root branch definition, per-run rank-biserial effects, and Fisher-z mean
for within-run correlations. Trees whose root has one child remain in recovery estimates but are
reported as `missing_branch_class` and excluded from ordering estimates.

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
