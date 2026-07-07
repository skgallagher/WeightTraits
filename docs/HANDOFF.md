# WeightTraits Handoff

Last updated: 2026-07-07.

## Project Intent

WeightTraits is a private, cleaner rebuild of ELLMTrees under `/Users/shannon/Desktop/phylo/WeightTraits`. It should remain maintainable, tested, auditable, and suitable for local smoke runs plus cluster-scale training. The main scientific targets are to independently rebuild and verify the ELLMTrees results, including the ICLR draft tables/figures, while fixing weak spots around flexible topology generation, LoRA semantics, distance computation, tree reconstruction scoring, and trainer reliability.

## Current Checkpoint

As of 2026-07-07, the project has a working end-to-end whitebox recovery spine:

- Tiny/mid branching contrast fixtures train on Wright for full checkpoints and LoRA.
- Standard full checkpoints, merged LoRA checkpoints, and cumulative LoRA adapter chains all feed the same `wt analyze-training-ledger` workflow.
- The seven-leaf, four-split mid branching contrast truth is recovered exactly for steps2 and steps8 under `l2`, `cosine`, and `correlation`.
- Cumulative LoRA `l2`, `cosine`, and `correlation` use the exact low-rank accumulator path, and the distance-cube audits record `lora_low_rank`.
- `paper/recovery_registry.yaml` and `paper/table_registry.yaml` now drive the first paper-facing whitebox recovery table.
- `wt validate-table-registry --require-outputs` validates source inputs, generated outputs, row counts, and SHA-256 digests.
- `paper/reference_registry.yaml` pins the live v2 draft reference surface and can be checked with
  `wt validate-reference-registry`.
- `paper/ellmtrees_variants_registry.yaml` plus `wt make-ellmtrees-variants-table` now rebuild the
  old ELLMTrees `tab:variants` reference table from pinned old CSVs.
- `wt compare-table-artifacts` compares rebuilt JSON/CSV table artifacts against a paper-grounded
  reference by key columns, with optional numeric tolerances and ignored metadata columns.
- `wt run-table-comparisons` runs the comparison specs declared in `paper/table_registry.yaml` and
  writes both aggregate and per-comparison JSON reports.
- `wt make-behavior-holdout-table` extracts the live draft `tab:behavior_holdout` cells as a
  paper-grounded JSON/CSV reference table.
- `wt generate-tree-set` now generated a clean confirm-paper-number topology set from the active
  draft distribution: 50 accepted Poisson-branching trees with `n_leaves >= 4`.
- `wt assign-task-data-set` now assigns paper-style task/data rows across that 50-tree set, using
  the paper-declared 36-dataset pool without replacement within each tree.

Latest verified paper outputs:

```text
reports/paper/whitebox_smoke_recovery.json
reports/paper/whitebox_smoke_recovery.csv
reports/paper/ellmtrees_variants_reference.json
reports/paper/ellmtrees_variants_reference.csv
reports/paper/table_registry_validation.json
reports/paper/reference_registry_validation.json
reports/paper/table_comparison_validation.json
reports/paper/ellmtrees_variants_reference_compare.json
reports/paper/behavior_holdout_reference.json
reports/paper/behavior_holdout_reference.csv
reports/paper/behavior_holdout_reference_compare.json
examples/training/confirm_paper_numbers/tree_set_summary.json
examples/training/confirm_paper_numbers/trees/*.manifest.jsonl
examples/training/confirm_paper_numbers/paper_task_families.yaml
examples/training/confirm_paper_numbers/assignment_summary.json
examples/training/confirm_paper_numbers/assigned_manifests/*.manifest.jsonl
```

These reports remain ignored by Git. The whitebox smoke reports were generated on Wright and pulled
locally. The ELLMTrees variants reference was generated locally from old pinned CSVs. The table
validation reports `valid=true`, `n_issues=0`, with `observed_rows=18` for whitebox smoke outputs,
`observed_rows=8` for the legacy variants reference outputs, and `observed_rows=12` for the live-draft
behavior holdout reference outputs, all with matching SHA-256 digests.
The reference registry validation also reports `valid=true`, `n_issues=0`, and `n_entries=17`.
It now checks active-draft label coverage as well: the current live draft has 14 `fig:`/`tab:` labels,
all 14 are registered, and there are no stale registered draft labels.
The registry-driven table comparison report has `valid=true`, `n_issues=0`, and `n_comparisons=2`.
Its variants reference comparison has `valid=true`, `n_issues=0`, `n_reference_rows=8`,
`n_candidate_rows=8`, `n_matched_rows=8`, and `n_compared_cells=176`; its behavior-holdout reference
comparison has `n_reference_rows=12`, `n_candidate_rows=12`, `n_matched_rows=12`, and
`n_compared_cells=60`.

The clean confirm-paper-number topology draw lives under
`examples/training/confirm_paper_numbers/`. It was generated with
`examples/trees/confirm_paper_numbers.yaml`, which follows the active draft topology setting:
Poisson branching with `lambda=1.5`, `n_nodes=14`, `max_depth=4`, and rejection of candidate trees
with fewer than four leaves. The accepted 50-tree set scanned 78 candidate seeds starting at
`20260707`; leaf counts are 4:6, 5:4, 6:3, 7:12, 8:9, 9:13, 10:3, and max-depth counts are 3:10,
4:40.

The confirm-paper-number task/data layer uses
`examples/training/confirm_paper_numbers/paper_task_families.yaml`, which follows the active draft's
36-dataset statement: 9 summarization, 10 classification, 8 QA, and 9 translation datasets.
`examples/training/confirm_paper_numbers/assignment_summary.json` records 50 enriched manifests in
`examples/training/confirm_paper_numbers/assigned_manifests/`, assigned with
`per_node_without_replacement` and seeds `1..50`. Within each tree, every training node has a unique
task/dataset pair.

On 2026-07-07, the `fig:overview` and `fig:coherence_recovery` digests in
`paper/reference_registry.yaml` were refreshed to match the current sibling reference files after
`../ELLMTrees-paper/figures/fig1_paper_overview.{tex,pdf}`,
`../ELLMTrees-paper/figures/fig4_coherence_atteson.png`,
`../ELLMTrees/scripts/make_fig4_atteson_layers.py`, and
`../ELLMTrees/results/aggregate/recovery_rescore/fig4_atteson_layermeans.png` changed. Do not edit
those sibling repos from WeightTraits; treat future digest mismatches as reference-surface drift to
inspect explicitly.

Atteson-margin caveat: WeightTraits does not compute this margin yet. The active paper definition is
the all-edge bottleneck, i.e. the minimum fitted edge length over internal and pendant edges divided
by twice the non-additivity error. If WeightTraits later implements this computation, do not replace
that definition with an internal-edge-only shortcut.
After confirming this definition, the paper copy
`../ELLMTrees-paper/figures/fig4_coherence_atteson.png` was refreshed again from
`../ELLMTrees/results/aggregate/recovery_rescore/fig4_atteson_layermeans.png`; both now share
SHA-256 `4c4cbd2b2742ad88ba41489ae22f25f3beef1c4796df1741f76a335c0c7d04c2`.

Plan position:

- Phase 2 core phylogenetic logic is in place for the whitebox paths exercised so far.
- Phase 3 smoke/toy rebuild has a strong checkpoint.
- Phase 5 whitebox experiment rebuild has a working tiny-to-mid scaffold.
- Phase 4 paper/RF table infrastructure has started, but paper-critical RF tables are not rebuilt yet.
- Phase 1 reference freeze is still the major missing foundation: old ELLMTrees paper artifacts, scripts, outputs, and digests need to be pinned before scaling further.

Paper draft context:

- The latest active draft is `../ELLMTrees-paper/iclr_draft_v2.tex`, not `iclr_draft.tex`.
- Treat the latest paper state as ground truth. If older ELLMTrees notes or generated artifacts
  conflict with the current paper, update the WeightTraits registry to the latest paper-grounded
  state deliberately rather than preserving stale old-reference wording.
- Shannon is roughly two-thirds through `iclr_draft_v2.tex` and is currently revising the results
  section.
- Treat `iclr_draft_v2.tex` as the live paper target for stale-claim and result-traceability work.
  Older draft files remain useful references, but should not drive the current rebuild by default.
- The initial v2 reference registry pins 17 entries: the active draft, old reproducibility/runbook
  docs, five figure artifacts, all `fig:`/`tab:` labels in the live draft, and source inputs where
  known. `tab:variants` is now also materialized as a generated JSON/CSV reference table.

Recommended next slice:

1. Build full-FT run lists for the clean 50-tree set, then launch a small dry-run/smoke before the
   full confirm-paper-number training batch.
2. Compare rebuilt recovery/behavior tables against the latest-paper-grounded references through
   `paper/table_registry.yaml` and `wt run-table-comparisons`.

## Git and Cluster Access

Current committed checkpoint:

```text
124752e Pin paper table output digests
b89764c Validate paper table registry outputs
6472b7c Add whitebox recovery result registry
df5eef3 Add steps8 mid branching smoke
10156db Record mid branching Wright smoke
5c5ce1a Add mid branching contrast smoke
0ccb5b7 Record low-rank LoRA verification
601b50e Add low-rank LoRA distance accumulation
```

Wright checkout was verified clean at `124752e`:

```text
ssh -S /tmp/wright-codex.sock wright
cd /home/export/sgallagh/WeightTraits
git status --short --branch
```

Before this handoff edit, local `HEAD` was also `124752e`. Local `git status --branch` may report
`main...origin/main [ahead 4]` because the local `origin/main` tracking ref is stale after the
Wright-mediated push. Wright confirms the pushed remote state.

Local GitHub SSH still failed with `Permission denied (publickey)`, so the successful push path was:

```text
local git bundle -> rsync over /tmp/wright-codex.sock -> git fetch bundle on Wright -> ff-only merge -> git push origin main
```

Wright access notes:

```text
ssh -S /tmp/wright-codex.sock wright
checkout: /home/export/sgallagh/WeightTraits
env: /home/export/sgallagh/.conda/envs/weighttraits
runner: /opt/miniforge3/bin/mamba run -n weighttraits ...
cache for tiny HF datasets: $HOME/.cache/WeightTraits/hf_datasets
```

There is one temporary Wright stash left from parking synced fixture files before the fast-forward:

```text
stash@{0}: On main: codex-synced-lineage-before-523902b
```

It should be safe to drop after confirming it duplicates `523902b`.

Branching smoke slice contents:

```text
docs/HANDOFF.md
docs/STREAMING_DISTANCE_CUBES.md
examples/distance_inputs/README.md
examples/distance_inputs/tiny_full_branching_leaf_outputs.yaml
examples/distance_inputs/tiny_lora_cumulative_branching_leaf_outputs.yaml
examples/distance_inputs/tiny_lora_merged_branching_leaf_outputs.yaml
examples/training/README.md
examples/training/tiny_branching_manifest.jsonl
examples/training/tiny_full_branching_smoke.yaml
examples/training/tiny_lora_branching_smoke.yaml
tests/test_distance_input_manifest.py
tests/test_training_tiny_examples.py
```

Intent and status of that slice:

- Add a six-row tiny branching training fixture with four terminal leaves: `n2`, `n3`, `n4`, `n5`.
- Add full and LoRA training configs using output roots `outputs/tiny_full_branching_smoke` and
  `outputs/tiny_lora_branching_smoke`.
- Add leaf-only distance-input manifests for full checkpoints, LoRA merged checkpoints, and
  cumulative LoRA adapter chains.
- Local focused tests passed with 12 tests.
- Wright focused training fixture tests passed with 6 tests and both full/LoRA run lists validated
  with 6 runs and no warnings.
- Wright full and LoRA branching training rows completed for all six nodes.
- Wright four-leaf distance/reconstruct/score smokes passed for full checkpoints, LoRA merged
  checkpoints, and cumulative LoRA adapters with exact recovery of the single nontrivial split.

Ledger-derived distance manifest slice:

```text
src/weighttraits/distances/manifest.py
src/weighttraits/distances/__init__.py
src/weighttraits/cli.py
tests/test_cli_distance_cube.py
tests/test_distance_input_manifest.py
docs/STREAMING_DISTANCE_CUBES.md
examples/distance_inputs/README.md
docs/HANDOFF.md
```

Intent and status of that slice:

- Add `wt make-distance-input-manifest`.
- Read latest terminal training ledger events and emit `build-distance-cube` manifests.
- With `--truth-manifest`, select terminal leaves by default.
- Support `--artifact model`, `--artifact merged`, and `--artifact adapter_chain`.
- Rewrite relative ledger artifact paths relative to the generated manifest.
- Local focused tests passed with 15 tests and full local suite passed with 121 tests.
- Wright focused tests passed with 15 tests and full Wright suite passed with 121 tests.
- On Wright, generated a cumulative LoRA branching leaf manifest from
  `outputs/tiny_lora_branching_smoke/training_ledger.jsonl`, rebuilt the distance cube, reconstructed,
  and scored exact recovery for the single nontrivial split.

Tiny branching contrast fixture slice:

```text
examples/training/README.md
examples/training/tiny_branching_contrast_manifest.jsonl
examples/training/tiny_full_branching_contrast_smoke.yaml
examples/training/tiny_lora_branching_contrast_smoke.yaml
examples/training/tiny_branch_*_{train,validation}.jsonl
examples/training/tiny_dataset_registry.yaml
examples/training/tiny_dataset_formats.yaml
tests/test_training_tiny_examples.py
docs/HANDOFF.md
```

Intent and status of that slice:

- Add a six-row contrast variant of the tiny branching fixture.
- Keep the same terminal leaves: `n2`, `n3`, `n4`, `n5`.
- Assign node-specific local JSONL datasets so root siblings carry `left`/`right` branch targets,
  and terminal siblings carry distinct `alpha`/`beta` leaf targets.
- Add full and LoRA contrast training configs using output roots
  `outputs/tiny_full_branching_contrast_smoke` and
  `outputs/tiny_lora_branching_contrast_smoke`.
- Local focused fixture tests passed with 8 tests; full local suite passed with 123 tests.
- Pushed as `f394b42 Add tiny branching contrast smoke`.
- Wright focused fixture tests passed with 8 tests; full Wright suite passed with 123 tests.
- Wright full and LoRA contrast training rows completed for all six nodes:
  `completed_nodes=["n0", "n1", "n2", "n3", "n4", "n5"]`, `failed_nodes=[]`.
- Generated ledger-derived leaf manifests for full checkpoints, LoRA merged checkpoints, and
  cumulative LoRA adapter chains.
- Built `l2` and `cosine` distance cubes for all three artifact modes and reconstructed/scored all
  six trees against `examples/training/tiny_branching_contrast_manifest.jsonl`.
- All six artifact/metric combinations recovered the single truth split exactly:
  aggregate `n_records=6`, `exact_tree_recovery_rate=1.0`, `pooled_clade_recovery=1.0`,
  `pooled_split_precision=1.0`, `rf_mean=0.0`.
- Distance summaries:
  - full checkpoints: `l2 distance_mean=0.018231388318443124`,
    `cosine distance_mean=0.035058786331429094`;
  - LoRA merged checkpoints: `l2 distance_mean=0.0015009464427041006`,
    `cosine distance_mean=1.2163427378181045e-06`;
  - cumulative LoRA adapters: `l2 distance_mean=0.0055034746647987945`,
    `cosine distance_mean=0.39188815833251484`.

Whitebox analysis wrapper slice:

```text
src/weighttraits/analysis/__init__.py
src/weighttraits/analysis/whitebox.py
src/weighttraits/cli.py
tests/test_cli_distance_cube.py
tests/test_whitebox_analysis.py
docs/STREAMING_DISTANCE_CUBES.md
docs/HANDOFF.md
```

Intent and status of that slice:

- Add `wt analyze-training-ledger`.
- Run one artifact mode at a time from a training ledger:
  `model`, `merged`, or `adapter_chain`.
- Wrap distance-input generation, distance cube construction, NJ reconstruction, tree scoring, and
  recovery aggregation.
- Write the generated distance-input manifest, `distance_cube/`, per-metric Newick/audit/score
  files, `aggregate_recovery.json`, and `summary.json`.
- Default representation is `full_weight` for checkpoint artifacts and `lora_cumulative_delta` for
  adapter chains.
- Local focused parser/workflow tests passed with 6 tests; full local suite passed with 125 tests.
- Pushed as `0951f93 Add whitebox ledger analysis workflow`.
- Wright focused parser/workflow tests passed with 6 tests; full Wright suite passed with 125 tests.
- On Wright, `wt analyze-training-ledger` was run against
  `outputs/tiny_lora_branching_contrast_smoke/training_ledger.jsonl` with
  `--artifact adapter_chain --metric l2 --metric cosine`; it wrote
  `outputs/tiny_lora_branching_contrast_smoke/cumulative_leaf_analysis/summary.json` and recovered
  the truth split for both metrics with aggregate `n_records=2`,
  `exact_tree_recovery_rate=1.0`, `pooled_clade_recovery=1.0`, and `rf_mean=0.0`.

Low-rank LoRA distance accumulator slice:

```text
src/weighttraits/distances/readers.py
src/weighttraits/distances/streaming.py
src/weighttraits/distances/__init__.py
tests/test_streaming_distance_cube.py
docs/STREAMING_DISTANCE_CUBES.md
docs/LORA_DISTANCE_MODEL.md
docs/HANDOFF.md
```

Intent and status of that slice:

- Add `LowRankLoraComponent` and `low_rank_components()` for `LoraFactorReader` and
  `CumulativeLoraReader`.
- Compute exact LoRA `cosine`, `l2`, and `correlation` distances from low-rank factor Gram
  matrices, including cumulative root-to-node adapter chains, without materializing dense `B @ A`
  row blocks.
- Keep `l1` and `threshold` on the dense chunk-streamed path.
- Record `lora_low_rank` in distance-cube audit metadata for metrics using the fast path.
- Local focused distance tests passed with 18 tests; full local suite passed with 127 tests.
- Pushed as `601b50e Add low-rank LoRA distance accumulation`.
- Wright focused distance tests passed with 18 tests; full Wright suite passed with 127 tests.
- On Wright, `wt analyze-training-ledger` was run against the real contrast LoRA cumulative adapter
  artifacts into
  `outputs/tiny_lora_branching_contrast_smoke/cumulative_leaf_analysis_lowrank/summary.json`.
  It recovered the truth split for both `l2` and `cosine`, with aggregate `n_records=2`,
  `exact_tree_recovery_rate=1.0`, `pooled_clade_recovery=1.0`, and `rf_mean=0.0`.
- The resulting distance-cube audit records
  `metric_execution={"cosine": "lora_low_rank", "l2": "lora_low_rank"}` for real PEFT adapters.

Mid-size branching contrast fixture slice:

```text
examples/training/README.md
examples/training/tiny_mid_branching_contrast_manifest.jsonl
examples/training/tiny_full_mid_branching_contrast_smoke.yaml
examples/training/tiny_lora_mid_branching_contrast_smoke.yaml
examples/training/tiny_mid_branch_*_{train,validation}.jsonl
examples/training/tiny_dataset_registry.yaml
examples/training/tiny_dataset_formats.yaml
tests/test_training_tiny_examples.py
docs/HANDOFF.md
```

Intent and status of that slice:

- Add an eleven-row contrast variant of the tiny branching fixture.
- Use zero-padded node IDs `n00` through `n10` so leaf ordering is stable and readable.
- Terminal leaves are `n03`, `n04`, `n05`, `n06`, `n08`, `n09`, and `n10`.
- The truth topology has four informative splits:
  `{n03,n04}`, `{n05,n06}`, `{n08,n09,n10}`, and `{n09,n10}`.
- Assign node-specific local JSONL datasets with branch/leaf target codes:
  north, south, east, and nested east-inner contrasts.
- Add full and LoRA configs using output roots
  `outputs/tiny_full_mid_branching_contrast_smoke` and
  `outputs/tiny_lora_mid_branching_contrast_smoke`.
- Local focused fixture tests passed with 10 tests; full local suite passed with 129 tests.
- Pushed as `5c5ce1a Add mid branching contrast smoke`.
- Wright focused fixture tests passed with 10 tests; full Wright suite passed with 129 tests.
- Wright full and LoRA mid-contrast training rows completed for all eleven nodes:
  `completed_nodes=["n00", "n01", "n02", "n03", "n04", "n05", "n06", "n07", "n08", "n09", "n10"]`,
  `failed_nodes=[]`.
- Ran `wt analyze-training-ledger` for full checkpoints, LoRA merged checkpoints, and cumulative
  LoRA adapter chains with `l2`, `cosine`, and `correlation`.
- All nine artifact/metric combinations recovered the seven-leaf, four-split truth tree exactly:
  per-analysis aggregate `n_records=3`, `exact_tree_recovery_rate=1.0`,
  `pooled_clade_recovery=1.0`, `pooled_split_precision=1.0`, `rf_mean=0.0`.
- Distance summaries:
  - full checkpoints: `l2 distance_mean=0.021859408662195528`,
    `cosine distance_mean=0.04230666288916821`,
    `correlation distance_mean=0.1092816349557629`;
  - LoRA merged checkpoints: `l2 distance_mean=0.0017130851191029123`,
    `cosine distance_mean=1.5541116800439562e-06`,
    `correlation distance_mean=1.554945832808278e-06`;
  - cumulative LoRA adapters: `l2 distance_mean=0.006281314718405168`,
    `cosine distance_mean=0.40682181793388494`,
    `correlation distance_mean=0.40679869399297386`.
- Distance-cube audit metadata for cumulative LoRA records
  `metric_execution={"correlation": "lora_low_rank", "cosine": "lora_low_rank", "l2": "lora_low_rank"}`.

Mid-size branching contrast steps8 fixture slice:

```text
examples/training/README.md
examples/training/tiny_full_mid_branching_contrast_steps8_smoke.yaml
examples/training/tiny_lora_mid_branching_contrast_steps8_smoke.yaml
tests/test_training_tiny_examples.py
docs/HANDOFF.md
```

Intent and status of that slice:

- Repeat the eleven-row, seven-leaf mid branching contrast topology with `max_steps: 8`.
- Use distinct output roots
  `outputs/tiny_full_mid_branching_contrast_steps8_smoke` and
  `outputs/tiny_lora_mid_branching_contrast_steps8_smoke`.
- Set fixed-step smoke stopping guards to stay present but not fire during eight steps:
  `early_stopping.patience: 9` and `plateau.window: 9`.
- This detail matters: the first LoRA attempt with the ordinary plateau `window: 3` stopped at
  step 3 because eval loss was nearly flat. The partial LoRA output root was removed on Wright and
  rerun after the guard update.
- Local focused fixture tests passed with 11 tests; full local suite passed with 130 tests.
- Wright focused fixture tests passed with 11 tests; full Wright suite passed with 130 tests.
- Wright full and LoRA steps8 run lists each validated with 11 runs, no warnings, and no errors.
- Wright full and LoRA steps8 training rows completed for all eleven nodes at step 8:
  `completed_nodes=["n00", "n01", "n02", "n03", "n04", "n05", "n06", "n07", "n08", "n09", "n10"]`,
  `failed_nodes=[]`, `status_counts={"completed": 11}`, `n_events=220` for each ledger.
- Ran `wt analyze-training-ledger` for full checkpoints, LoRA merged checkpoints, and cumulative
  LoRA adapter chains with `l2`, `cosine`, and `correlation`.
- All nine artifact/metric combinations recovered the seven-leaf, four-split truth tree exactly:
  per-analysis aggregate `n_records=3`, `exact_tree_recovery_rate=1.0`,
  `pooled_clade_recovery=1.0`, `pooled_split_precision=1.0`, `rf_mean=0.0`.
- Distance summaries:
  - full checkpoints: `l2 distance_mean=0.06936622415055438`,
    `cosine distance_mean=0.06627167701047997`,
    `correlation distance_mean=0.1076262502407616`;
  - LoRA merged checkpoints: `l2 distance_mean=0.004512752549928248`,
    `cosine distance_mean=1.0674884435093831e-05`,
    `correlation distance_mean=1.067963516983858e-05`;
  - cumulative LoRA adapters: `l2 distance_mean=0.0165467598716221`,
    `cosine distance_mean=0.5282546102244117`,
    `correlation distance_mean=0.528311362716497`.
- Distance-cube audit metadata for cumulative LoRA records
  `metric_execution={"correlation": "lora_low_rank", "cosine": "lora_low_rank", "l2": "lora_low_rank"}`.

Paper recovery registry slice:

```text
paper/README.md
paper/recovery_registry.yaml
paper/table_registry.yaml
src/weighttraits/paper/__init__.py
src/weighttraits/paper/results.py
src/weighttraits/cli.py
tests/test_cli_distance_cube.py
tests/test_paper_results.py
docs/HANDOFF.md
```

Intent and status of that slice:

- Add `paper/recovery_registry.yaml` as the first paper-facing registry of verified whitebox
  recovery summaries.
- Register the mid branching contrast whitebox smokes for steps2 and steps8:
  full checkpoints, LoRA merged checkpoints, and cumulative LoRA adapter chains.
- Add `wt make-recovery-table`.
- Read registered `summary.json` files, validate artifact mode agreement, and emit one compact
  table row per summary metric.
- Support JSON and CSV outputs for downstream paper/table rebuilds.
- Add `paper/table_registry.yaml` entry `whitebox_smoke_recovery`, with source command and expected
  row count.
- Local focused paper/parser tests passed with 9 tests; full local suite passed with 134 tests.
- Wright focused paper/parser tests passed with 9 tests; full Wright suite passed with 134 tests.
- On Wright, built:

```text
reports/paper/whitebox_smoke_recovery.json
reports/paper/whitebox_smoke_recovery.csv
```

  from `paper/recovery_registry.yaml`.
- The generated table has `n_rows=18`, covering six registered summaries times three metrics:
  `correlation`, `cosine`, and `l2`.

Paper table-registry validation slice:

```text
docs/HANDOFF.md
docs/PAPER_REPRODUCTION.md
paper/README.md
paper/table_registry.yaml
src/weighttraits/paper/__init__.py
src/weighttraits/paper/results.py
src/weighttraits/cli.py
tests/test_cli_distance_cube.py
tests/test_paper_results.py
```

Intent and status of that slice:

- Add `outputs` declarations to `paper/table_registry.yaml`.
- Add `wt validate-table-registry`.
- Validate table registry shape, declared source inputs, optional generated outputs, observed row
  counts for JSON/CSV table artifacts, and optional output SHA-256 digests.
- Keep generated reports ignored while still making paper-build gates checkable.
- Local focused paper/parser tests passed with 15 tests; full local suite passed with 140 tests.
- Wright focused paper/parser tests passed with 15 tests; full Wright suite passed with 140 tests.
- On Wright, validated `paper/table_registry.yaml` with `--require-outputs` and wrote:

```text
reports/paper/table_registry_validation.json
```

- The validation report had `valid=true`, `n_issues=0`, all seven source inputs present, and both
  `reports/paper/whitebox_smoke_recovery.{json,csv}` present with `observed_rows=18` and matching
  expected/observed SHA-256 digests.

Previous artifact-distance smoke slice contents:

```text
docs/HANDOFF.md
examples/distance_inputs/README.md
examples/distance_inputs/tiny_full_lineage_outputs.yaml
examples/distance_inputs/tiny_lora_cumulative_lineage_outputs.yaml
examples/distance_inputs/tiny_lora_merged_lineage_outputs.yaml
examples/recovery/tiny_two_tip_smoke_truth_manifest.jsonl
tests/test_distance_input_manifest.py
tests/test_recovery_scoring.py
```

Intent and status of that previous slice:

- Add distance-input manifests for the Wright two-node training artifacts:
  - full checkpoints: `outputs/tiny_full_lineage_smoke/n0/model`, `.../n1/model`;
  - LoRA merged checkpoints: `outputs/tiny_lora_lineage_smoke/n0/merged`, `.../n1/merged`;
  - LoRA cumulative adapter chains: `n0=[n0/adapter]`, `n1=[n0/adapter,n1/adapter]`.
- Add a two-tip truth manifest only for command plumbing through `score-tree`.
- Caveat: the two-node training lineage is a chain with terminal leaf `n1`, so it is not a
  meaningful RF/FN/FP recovery target. The two-tip smoke truth has no informative splits; use it
  only to verify `build-distance-cube -> reconstruct-tree -> score-tree` on real artifacts.
- Local focused tests passed with 12 tests and full local suite passed with 112 tests.
- Wright focused tests passed with 12 tests and full Wright suite passed with 112 tests.
- Wright artifact smoke passed for full checkpoints, LoRA merged checkpoints, and cumulative LoRA
  adapter chains.

## What Exists

- Flexible tree generation with `fixed`, `chain`, `balanced`, `ellmtrees_balanced`, `poisson_branching`, and `pruned_binary_backbone`.
- Tree constraints for `min_depth`, `min_leaves`, pruning, and polytomy-generating contraction.
- Task/data assignment from enriched manifests.
- Recovery scoring with RF, TP/FP/FN, FN rate, clade recovery, exact recovery, and SE aggregation.
- Distance metric registry with cosine, L1, L2, correlation, threshold, and linear CKA.
- Streaming distance cube engine:
  - chunk-streamed vector metrics;
  - exact tensor-at-a-time CKA;
  - safetensors, sharded safetensors, torch, LoRA factor, and cumulative LoRA readers;
  - distance input manifests and `wt build-distance-cube`.
- Biopython-backed neighbor-joining reconstruction from distance cubes:
  - `wt reconstruct-tree`;
  - layer selection by name/index or mean/median layer aggregation;
  - Newick output plus JSON audit metadata.
- LoRA analysis semantics:
  - true training process is fresh adapter on merged parent;
  - efficient analysis representation is cumulative path-summed `scale * B @ A`;
  - CLI supports `--adapter-chain node:edge0,edge1,...`.
- Trainer control plane:
  - `wt plan-training`;
  - full vs LoRA artifact planning;
  - prompt override resolution by manifest row, base model, model family, dataset, task, and default;
  - required prompt-field extraction;
  - prompt rendering validation;
  - loss warning/early-stop/plateau monitor;
  - training JSONL ledger helpers and `wt training-ledger-summary`.
- Offline training data validation:
  - dataset format contracts;
  - `wt validate-training-data`;
  - smoke contracts in `examples/training/dataset_formats_smoke.yaml`.
- Dataset registry and split audit:
  - `wt audit-datasets`;
  - no-download registry/split dry runs with `--no-load`;
  - optional Hugging Face `load_dataset` split and row-count audit in prepared environments.
  - registry rows support `hf_kwargs`, including local JSONL `data_files`.
- Training sample rendering audit:
  - `wt audit-training-samples`;
  - loads a tiny sample for planned jobs;
  - applies dataset `field_map`;
  - renders resolved prompt templates;
  - reports counts, field names, row indices, and errors without storing raw samples or prompts.
- Training run-list generation:
  - `wt make-training-run-list`;
  - supports local and cluster execution profiles;
  - writes stable JSONL rows keyed by array index;
  - writes optional preflight report and SLURM runner or dry-run selector script;
  - checks parent order, artifact collisions, missing datasets, stopping guards, and LoRA merge semantics.
- Per-row training runner:
  - `wt run-training-row`;
  - selects one run-list row by array index or node id;
  - loads registry/format contracts, applies `field_map`, renders prompts, and derives targets;
  - imports `datasets`, `transformers`, and `peft` lazily for real training;
  - supports full fine-tuning and LoRA adapter plus merged-child saves;
  - writes started/running/completed/stopped_early/failed ledger events.
  - handles current and older Transformers trainer constructor names:
    `processing_class` vs `tokenizer`;
  - uses `Seq2SeqTrainingArguments` for seq2seq trainer runs when available.
- Tiny real-training smoke fixtures:
  - `examples/training/tiny_manifest.jsonl`;
  - `examples/training/tiny_train.jsonl` and `tiny_validation.jsonl`;
  - `examples/training/tiny_dataset_registry.yaml`;
  - `examples/training/tiny_dataset_formats.yaml`;
  - `examples/training/tiny_full_smoke.yaml`;
  - `examples/training/tiny_lora_smoke.yaml`;
  - prompts keep the label out of the prompt and use `trainer.target_field: answer`.

## Verification So Far

After the last pushed increment:

```text
conda run -n ellmtrees env PYTHONPATH=src python -m pytest -q
```

passed with 70 tests.

After adding the dataset registry/split audit and this handoff:

```text
conda run -n ellmtrees env PYTHONPATH=src python -m pytest --override-ini=addopts=
```

passed with 77 tests.

After adding the sample-rendering audit:

```text
conda run -n ellmtrees env PYTHONPATH=src python -m pytest tests/test_training_dataset_registry.py tests/test_training_data_formats.py --override-ini=addopts=
```

passed with 19 focused trainer preflight tests.

After adding the training run-list layer:

```text
conda run -n ellmtrees env PYTHONPATH=src python -m pytest tests/test_training_runlist.py --override-ini=addopts=
```

passed with 9 focused run-list tests.

After adding the per-row executor:

```text
conda run -n ellmtrees env PYTHONPATH=src python -m pytest tests/test_training_executor.py tests/test_training_runlist.py --override-ini=addopts=
```

passed with 15 focused executor/run-list tests.

After adding registry `hf_kwargs` and tiny smoke fixtures:

```text
conda run -n ellmtrees env PYTHONPATH=src python -m pytest tests/test_training_dataset_registry.py tests/test_training_executor.py tests/test_training_tiny_examples.py --override-ini=addopts=
```

passed with 20 focused tests.

The full suite then passed with 100 tests:

```text
conda run -n ellmtrees env PYTHONPATH=src python -m pytest --override-ini=addopts=
```

After adding Biopython-backed cube-to-Newick reconstruction:

```text
conda run -n ellmtrees env PYTHONPATH=src python -m pytest tests/test_reconstruction.py tests/test_cli_distance_cube.py --override-ini=addopts=
```

passed with 8 focused tests.

The full suite then passed with 105 tests:

```text
conda run -n ellmtrees env PYTHONPATH=src python -m pytest --override-ini=addopts=
```

After adding the tiny local whitebox CLI smoke:

```text
conda run -n ellmtrees env PYTHONPATH=src python -m pytest tests/test_whitebox_smoke.py --override-ini=addopts=
```

passed with 1 focused smoke test. This exercises:

```text
wt build-distance-cube -> wt reconstruct-tree -> wt score-tree -> wt aggregate-recovery
```

The full suite then passed with 106 tests:

```text
conda run -n ellmtrees env PYTHONPATH=src python -m pytest --override-ini=addopts=
```

After exercising real HF/PEFT training on Wright with Transformers 5.13.0, two executor
compatibility fixes were added:

- pass tokenizer-like objects as `processing_class` when the installed Trainer constructor uses
  that newer name, otherwise fall back to `tokenizer`;
- use `Seq2SeqTrainingArguments` for seq2seq runs when available.

Focused executor tests passed locally:

```text
conda run -n ellmtrees env PYTHONPATH=src python -m pytest tests/test_training_executor.py --override-ini=addopts=
```

with 7 tests. The full local suite then passed with 108 tests:

```text
conda run -n ellmtrees env PYTHONPATH=src python -m pytest --override-ini=addopts=
```

On Wright, after syncing the executor compatibility patch into the cloned checkout:

```text
/opt/miniforge3/bin/mamba run -n weighttraits env PYTHONPATH=src python -m pytest tests/test_training_executor.py --override-ini=addopts=
```

passed with 7 focused executor tests. The remote suite passed with 102 tests:

```text
/opt/miniforge3/bin/mamba run -n weighttraits env PYTHONPATH=src python -m pytest --override-ini=addopts=
```

After adding the two-node tiny lineage fixtures:

```text
conda run -n ellmtrees env PYTHONPATH=src python -m pytest tests/test_training_tiny_examples.py tests/test_training_runlist.py --override-ini=addopts=
```

passed locally with 14 focused tests. The full local suite passed with 110 tests:

```text
conda run -n ellmtrees env PYTHONPATH=src python -m pytest --override-ini=addopts=
```

On Wright:

```text
/opt/miniforge3/bin/mamba run -n weighttraits env PYTHONPATH=src python -m pytest tests/test_training_tiny_examples.py --override-ini=addopts=
```

passed with 4 focused fixture tests, and:

```text
/opt/miniforge3/bin/mamba run -n weighttraits env PYTHONPATH=src python -m pytest --override-ini=addopts=
```

passed with 110 tests.

After adding tiny lineage artifact distance manifests and a two-tip scoring smoke:

```text
conda run -n ellmtrees env PYTHONPATH=src python -m pytest tests/test_distance_input_manifest.py tests/test_recovery_scoring.py --override-ini=addopts=
```

passed locally with 12 focused tests. The full local suite passed with 112 tests:

```text
conda run -n ellmtrees env PYTHONPATH=src python -m pytest --override-ini=addopts=
```

On Wright:

```text
/opt/miniforge3/bin/mamba run -n weighttraits env PYTHONPATH=src python -m pytest tests/test_distance_input_manifest.py tests/test_recovery_scoring.py --override-ini=addopts=
```

passed with 12 focused tests, and:

```text
/opt/miniforge3/bin/mamba run -n weighttraits env PYTHONPATH=src python -m pytest --override-ini=addopts=
```

passed with 112 tests.

After adding the six-node tiny branching fixtures and four-leaf distance manifests:

```text
conda run -n ellmtrees env PYTHONPATH=src python -m pytest tests/test_distance_input_manifest.py tests/test_training_tiny_examples.py --override-ini=addopts=
```

passed locally with 12 focused tests. The full local suite passed with 115 tests:

```text
conda run -n ellmtrees env PYTHONPATH=src python -m pytest --override-ini=addopts=
```

On Wright:

```text
/opt/miniforge3/bin/mamba run -n weighttraits env PYTHONPATH=src python -m pytest tests/test_training_tiny_examples.py --override-ini=addopts=
```

passed with 6 focused fixture tests, and:

```text
/opt/miniforge3/bin/mamba run -n weighttraits env PYTHONPATH=src python -m pytest --override-ini=addopts=
```

passed with 115 tests.

Useful smoke commands that have passed:

```bash
PYTHONPATH=src python -m weighttraits.cli plan-training \
  --manifest reports/flexible_tree_assigned_manifest.jsonl \
  --config examples/training/full_smoke.yaml \
  --out /tmp/weighttraits_full_training_plan.jsonl

PYTHONPATH=src python -m weighttraits.cli plan-training \
  --manifest reports/flexible_tree_assigned_manifest.jsonl \
  --config examples/training/lora_smoke.yaml \
  --out /tmp/weighttraits_lora_training_plan.jsonl

PYTHONPATH=src python -m weighttraits.cli validate-training-data \
  --manifest reports/flexible_tree_assigned_manifest.jsonl \
  --config examples/training/full_smoke.yaml \
  --formats examples/training/dataset_formats_smoke.yaml \
  --out /tmp/weighttraits_training_data_validation_full.json

PYTHONPATH=src python -m weighttraits.cli validate-training-data \
  --manifest reports/flexible_tree_assigned_manifest.jsonl \
  --config examples/training/lora_smoke.yaml \
  --formats examples/training/dataset_formats_smoke.yaml \
  --out /tmp/weighttraits_training_data_validation_lora.json
```

Both full and LoRA validation smokes reported 13/13 jobs valid.

Dataset audit no-load smoke:

```bash
PYTHONPATH=src python -m weighttraits.cli audit-datasets \
  --registry configs/task_data_candidates.yaml \
  --formats examples/training/dataset_formats_smoke.yaml \
  --dataset-id boolq \
  --dataset-id hellaswag \
  --no-load \
  --out /tmp/weighttraits_dataset_audit_noload.json
```

Sample-rendering audit smoke, for environments where Hugging Face dataset loading is available:

```bash
PYTHONPATH=src python -m weighttraits.cli audit-training-samples \
  --manifest reports/flexible_tree_assigned_manifest.jsonl \
  --config examples/training/full_smoke.yaml \
  --registry configs/task_data_candidates.yaml \
  --formats examples/training/dataset_formats_smoke.yaml \
  --dataset-id boolq \
  --max-samples 4 \
  --out /tmp/weighttraits_training_sample_render_audit.json
```

Run-list local smoke:

```bash
PYTHONPATH=src python -m weighttraits.cli make-training-run-list \
  --manifest reports/flexible_tree_assigned_manifest.jsonl \
  --config examples/training/full_smoke.yaml \
  --profile configs/local/default.yaml \
  --out /tmp/weighttraits_full_local_runs.jsonl \
  --report /tmp/weighttraits_full_local_runs.report.json
```

Run-list cluster dry-run smoke:

```bash
PYTHONPATH=src python -m weighttraits.cli make-training-run-list \
  --manifest reports/flexible_tree_assigned_manifest.jsonl \
  --config examples/training/lora_smoke.yaml \
  --profile configs/cluster/wright.yaml \
  --registry configs/task_data_candidates.yaml \
  --formats examples/training/dataset_formats_smoke.yaml \
  --out /tmp/weighttraits_lora_runs.jsonl \
  --report /tmp/weighttraits_lora_runs.report.json \
  --slurm-out /tmp/weighttraits_lora_train.sbatch \
  --runner-dry-run
```

The Wright smoke reported 13 runs, no errors, and one expected warning because the run count exceeds
the profile's default concurrency throttle of 6. With `--runner-dry-run`, the generated script calls
`run-training-row --dry-run` and does not load datasets or models.

Per-row runner dry-run smoke:

```bash
PYTHONPATH=src python -m weighttraits.cli run-training-row \
  --run-list /tmp/weighttraits_lora_runs.jsonl \
  --index 0 \
  --dry-run
```

Tiny local JSONL dataset smoke:

```bash
HF_DATASETS_CACHE=/tmp/weighttraits_hf_datasets \
PYTHONPATH=src python -m weighttraits.cli audit-datasets \
  --registry examples/training/tiny_dataset_registry.yaml \
  --formats examples/training/tiny_dataset_formats.yaml \
  --out /tmp/weighttraits_tiny_dataset_audit.json

HF_DATASETS_CACHE=/tmp/weighttraits_hf_datasets \
PYTHONPATH=src python -m weighttraits.cli audit-training-samples \
  --manifest examples/training/tiny_manifest.jsonl \
  --config examples/training/tiny_full_smoke.yaml \
  --registry examples/training/tiny_dataset_registry.yaml \
  --formats examples/training/tiny_dataset_formats.yaml \
  --max-samples 2 \
  --out /tmp/weighttraits_tiny_sample_audit.json
```

Those local JSONL smokes passed with 2 train rows, 1 validation row, and 2 rendered prompt samples.

Tiny full/LoRA run-list dry-runs passed:

```bash
PYTHONPATH=src python -m weighttraits.cli make-training-run-list \
  --manifest examples/training/tiny_manifest.jsonl \
  --config examples/training/tiny_full_smoke.yaml \
  --registry examples/training/tiny_dataset_registry.yaml \
  --formats examples/training/tiny_dataset_formats.yaml \
  --out /tmp/weighttraits_tiny_full_runs.jsonl \
  --allow-existing-artifacts \
  --runner-dry-run

PYTHONPATH=src python -m weighttraits.cli make-training-run-list \
  --manifest examples/training/tiny_manifest.jsonl \
  --config examples/training/tiny_lora_smoke.yaml \
  --registry examples/training/tiny_dataset_registry.yaml \
  --formats examples/training/tiny_dataset_formats.yaml \
  --out /tmp/weighttraits_tiny_lora_runs.jsonl \
  --allow-existing-artifacts \
  --runner-dry-run
```

## Wright Status

Wright is reachable, but Codex should not try to perform interactive auth itself. The reliable
workflow is:

1. Ask Shannon to run this in a normal macOS Terminal, not in Codex:

```bash
ssh -M -S /tmp/wright-codex.sock -fN wright
```

This may prompt for a passphrase or password in that Terminal. Once it succeeds, Codex can reuse
the socket with:

```bash
ssh -S /tmp/wright-codex.sock wright hostname
```

To close the socket later:

```bash
ssh -S /tmp/wright-codex.sock -O exit wright
```

Known-bad or low-value steps to skip in Codex:

```text
ssh wright hostname
ssh -o IdentitiesOnly=yes -i /Users/shannon/.ssh/id_ed25519 sgallagh@wright.hss.cmu.edu hostname
ssh-add -l
ssh-add /Users/shannon/.ssh/id_ed25519
```

Those fail or hang because Codex cannot handle the interactive auth prompt usefully. Do not ask the
user to type a passphrase "here"; ask them to run the ControlMaster command above in a regular
Terminal. After the socket is open, run the tiny real-model smoke below.

Current Wright checkout and env from 2026-07-06:

```text
repo: /home/export/sgallagh/WeightTraits
env: /home/export/sgallagh/.conda/envs/weighttraits
conda/mamba: /opt/miniforge3/bin/{conda,mamba}
cache used for tiny smoke: /home/export/sgallagh/.cache/WeightTraits
```

`/home/export/sgallagh/scratch` points to `/mnt/scratch`, but `/mnt/scratch` was not usable on the
headnode session. Use `/home/export/sgallagh/.cache/WeightTraits` for tiny smoke caches until a real
scratch path is confirmed.

Tiny full smoke on Wright or another prepared environment:

```bash
cd /home/export/sgallagh/WeightTraits
git pull
/opt/miniforge3/bin/mamba run -n weighttraits env PYTHONPATH=src HF_DATASETS_CACHE=$HOME/.cache/WeightTraits/hf_datasets \
  python -m weighttraits.cli make-training-run-list \
    --manifest examples/training/tiny_manifest.jsonl \
    --config examples/training/tiny_full_smoke.yaml \
    --registry examples/training/tiny_dataset_registry.yaml \
    --formats examples/training/tiny_dataset_formats.yaml \
    --out /tmp/weighttraits_tiny_full_runs.jsonl \
    --allow-existing-artifacts \
    --max-train-samples 2 \
    --allow-missing-eval

/opt/miniforge3/bin/mamba run -n weighttraits env PYTHONPATH=src HF_DATASETS_CACHE=$HOME/.cache/WeightTraits/hf_datasets \
  python -m weighttraits.cli run-training-row \
    --run-list /tmp/weighttraits_tiny_full_runs.jsonl \
    --index 0
```

This passed on 2026-07-06:

```text
status=completed, step=1, train_loss=7.007139682769775, eval_loss=7.006280422210693
artifacts: outputs/tiny_full_smoke/n0/model, outputs/tiny_full_smoke/n0/training_log.jsonl
ledger summary: completed_nodes=["n0"], failed_nodes=[], status_counts={"completed": 1}
```

Tiny LoRA smoke:

```bash
/opt/miniforge3/bin/mamba run -n weighttraits env PYTHONPATH=src HF_DATASETS_CACHE=$HOME/.cache/WeightTraits/hf_datasets \
  python -m weighttraits.cli make-training-run-list \
    --manifest examples/training/tiny_manifest.jsonl \
    --config examples/training/tiny_lora_smoke.yaml \
    --registry examples/training/tiny_dataset_registry.yaml \
    --formats examples/training/tiny_dataset_formats.yaml \
    --out /tmp/weighttraits_tiny_lora_runs.jsonl \
    --allow-existing-artifacts \
    --max-train-samples 2 \
    --allow-missing-eval

/opt/miniforge3/bin/mamba run -n weighttraits env PYTHONPATH=src HF_DATASETS_CACHE=$HOME/.cache/WeightTraits/hf_datasets \
  python -m weighttraits.cli run-training-row \
    --run-list /tmp/weighttraits_tiny_lora_runs.jsonl \
    --index 0
```

This passed on 2026-07-06:

```text
status=completed, step=1, train_loss=7.007139682769775, eval_loss=7.006478786468506
artifacts: outputs/tiny_lora_smoke/n0/adapter, outputs/tiny_lora_smoke/n0/merged,
           outputs/tiny_lora_smoke/n0/training_log.jsonl
ledger summary: completed_nodes=["n0"], failed_nodes=[], status_counts={"completed": 1}
```

Tiny two-node lineage smoke:

```bash
/opt/miniforge3/bin/mamba run -n weighttraits env PYTHONPATH=src HF_DATASETS_CACHE=$HOME/.cache/WeightTraits/hf_datasets \
  python -m weighttraits.cli make-training-run-list \
    --manifest examples/training/tiny_lineage_manifest.jsonl \
    --config examples/training/tiny_full_lineage_smoke.yaml \
    --registry examples/training/tiny_dataset_registry.yaml \
    --formats examples/training/tiny_dataset_formats.yaml \
    --out /tmp/weighttraits_tiny_full_lineage_runs.jsonl \
    --allow-existing-artifacts \
    --max-train-samples 2 \
    --allow-missing-eval

/opt/miniforge3/bin/mamba run -n weighttraits env PYTHONPATH=src HF_DATASETS_CACHE=$HOME/.cache/WeightTraits/hf_datasets \
  python -m weighttraits.cli run-training-row \
    --run-list /tmp/weighttraits_tiny_full_lineage_runs.jsonl \
    --index 0

/opt/miniforge3/bin/mamba run -n weighttraits env PYTHONPATH=src HF_DATASETS_CACHE=$HOME/.cache/WeightTraits/hf_datasets \
  python -m weighttraits.cli run-training-row \
    --run-list /tmp/weighttraits_tiny_full_lineage_runs.jsonl \
    --index 1
```

Full lineage passed on 2026-07-06:

```text
n1 init_from: outputs/tiny_full_lineage_smoke/n0/model
ledger summary: completed_nodes=["n0", "n1"], failed_nodes=[], status_counts={"completed": 2}
artifacts: outputs/tiny_full_lineage_smoke/n0/model, outputs/tiny_full_lineage_smoke/n1/model
```

The LoRA lineage run list uses:

```text
n1 init_from: outputs/tiny_lora_lineage_smoke/n0/merged
```

and both LoRA rows passed on 2026-07-06:

```text
ledger summary: completed_nodes=["n0", "n1"], failed_nodes=[], status_counts={"completed": 2}
artifacts: outputs/tiny_lora_lineage_smoke/n0/adapter, outputs/tiny_lora_lineage_smoke/n0/merged,
           outputs/tiny_lora_lineage_smoke/n1/adapter, outputs/tiny_lora_lineage_smoke/n1/merged
```

Tiny artifact distance plumbing smoke on Wright:

```bash
/opt/miniforge3/bin/mamba run -n weighttraits env PYTHONPATH=src \
  python -m weighttraits.cli build-distance-cube \
    --checkpoint-manifest examples/distance_inputs/tiny_full_lineage_outputs.yaml \
    --metric cosine \
    --metric l2 \
    --out outputs/tiny_full_lineage_smoke/distance_cube

/opt/miniforge3/bin/mamba run -n weighttraits env PYTHONPATH=src \
  python -m weighttraits.cli reconstruct-tree \
    --cube outputs/tiny_full_lineage_smoke/distance_cube \
    --metric l2 \
    --out outputs/tiny_full_lineage_smoke/distance_cube/tree_l2.newick \
    --audit-out outputs/tiny_full_lineage_smoke/distance_cube/tree_l2.audit.json

/opt/miniforge3/bin/mamba run -n weighttraits env PYTHONPATH=src \
  python -m weighttraits.cli score-tree \
    --truth-manifest examples/recovery/tiny_two_tip_smoke_truth_manifest.jsonl \
    --estimate-newick outputs/tiny_full_lineage_smoke/distance_cube/tree_l2.newick \
    --out outputs/tiny_full_lineage_smoke/distance_cube/score_l2.json
```

This passed on 2026-07-06. It is a command-path smoke only: the two-tip truth has no informative
splits, and the real training lineage is a chain with terminal leaf `n1`.

```text
full checkpoints: n_models=2, n_layers=110, l2 distance_mean=0.0009195269418413603
LoRA merged checkpoints: n_models=2, n_layers=110, l2 distance_mean=7.131354745703268e-05
LoRA cumulative adapters: n_models=2, n_layers=30, l2 distance_mean=0.00026147472545753084
two-tip score: n_truth_splits=0, n_estimate_splits=0, rf=0, exact_tree_recovery=true
```

Tiny branching smoke on Wright:

```text
manifest: examples/training/tiny_branching_manifest.jsonl
leaves: ["n2", "n3", "n4", "n5"]
truth splits: 1
full run list: 6 runs, 0 warnings, 0 errors
LoRA run list: 6 runs, 0 warnings, 0 errors
full training: completed_nodes=["n0", "n1", "n2", "n3", "n4", "n5"], failed_nodes=[]
LoRA training: completed_nodes=["n0", "n1", "n2", "n3", "n4", "n5"], failed_nodes=[]
```

The four-leaf distance/reconstruct/score path also passed:

```text
full leaf checkpoints: n_models=4, n_layers=110, l2 distance_mean=0.0,
  score rf=0, false_negative=0, false_positive=0, exact_tree_recovery=true
LoRA merged leaf checkpoints: n_models=4, n_layers=110, l2 distance_mean=0.00011425835655734538,
  score rf=0, false_negative=0, false_positive=0, exact_tree_recovery=true
LoRA cumulative leaf adapters: n_models=4, n_layers=30, l2 distance_mean=0.00041895296848211915,
  score rf=0, false_negative=0, false_positive=0, exact_tree_recovery=true
```

Ledger-derived distance manifest smoke:

```bash
/opt/miniforge3/bin/mamba run -n weighttraits env PYTHONPATH=src \
  python -m weighttraits.cli make-distance-input-manifest \
    --ledger outputs/tiny_lora_branching_smoke/training_ledger.jsonl \
    --truth-manifest examples/training/tiny_branching_manifest.jsonl \
    --artifact adapter_chain \
    --out outputs/tiny_lora_branching_smoke/generated_cumulative_leaf_inputs.yaml
```

Generated:

```yaml
models:
- model_id: n2
  adapter_chain:
  - n0/adapter
  - n2/adapter
- model_id: n3
  adapter_chain:
  - n0/adapter
  - n3/adapter
- model_id: n4
  adapter_chain:
  - n1/adapter
  - n4/adapter
- model_id: n5
  adapter_chain:
  - n1/adapter
  - n5/adapter
```

Building, reconstructing, and scoring from that generated manifest passed with:

```text
n_models=4, n_layers=30, l2 distance_mean=0.00041895296848211915
score rf=0, false_negative=0, false_positive=0, exact_tree_recovery=true
```

## Next Best Steps

1. Use `reports/paper/ellmtrees_variants_reference.{json,csv}` as the old-reference target for the
   first non-toy WeightTraits RF/recovery comparison table.
2. Keep extending the Phase 1 reference freeze for any remaining appendix-only claims in
   `../ELLMTrees-paper/iclr_draft_v2.tex`; use `paper/reference_registry.yaml` as the machine-readable
   home for paths, digests, commands, and paper-critical/provisional/stale classification.
3. Keep using `wt analyze-training-ledger` for whitebox recovery outputs so full checkpoints, merged
   LoRA checkpoints, and cumulative LoRA adapter chains share one summary schema.
4. Defer larger training or blackbox/HF-zoo scale-up until at least one old-vs-new paper-critical RF
   table comparison has been wired into the registry gate.

## Important Caveats

- Do not commit generated outputs, model weights, checkpoints, caches, or reports unless they are intentional tiny examples.
- CKA is exact but still tensor-at-a-time.
- `wt reconstruct-tree` uses Biopython from the analysis extra.
- LoRA `cosine`, `l2`, and `correlation` metrics now use exact low-rank factor accumulation;
  LoRA `l1` and `threshold` still stream dense `B @ A` row blocks.
- The trainer execution loop has now been exercised on Wright with the tiny model
  `hf-internal-testing/tiny-random-t5` for single-row, two-node lineage, six-node branching,
  eleven-node mid-branching, and eleven-node steps8 full/LoRA runs.
- Local outgoing Hugging Face traffic is disabled; run real model smoke/training on Wright or another prepared environment.
- Local `datasets.load_dataset("json", ...)` may need `HF_DATASETS_CACHE` pointed to a writable scratch directory.
- `wt audit-datasets` in load mode may require network access and the optional `datasets` dependency.
- `wt audit-training-samples` requires dataset loading and should run only in environments where downloads/cache access are intended.
- `wt make-training-run-list --runner-dry-run --slurm-out` generates a safe selector script; omit `--runner-dry-run` only when real training is intended.
