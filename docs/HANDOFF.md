# WeightTraits Handoff

Last updated: 2026-07-14.

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
- `wt make-training-run-list-set` now writes per-tree run lists for the confirm-paper-number set,
  avoiding node/output collisions across the 50 trees.
- `wt cache-training-datasets` builds bounded canonical-row caches for training, with deterministic
  dataset filters applied before sample caps; it supports opt-in seeded shuffle sampling with
  cache-side provenance.
- HF/PEFT execution applies `trainer.seed` before model and adapter construction, so LoRA
  initialization is deterministic. Causal jobs can choose completion-only or all-token loss.
- HF/PEFT execution now rejects requested LoRA targets that match zero loaded-model modules and
  persists the resolved adapter scope plus exact trainable counts in `lora_target_audit.json` and
  terminal ledger metadata.
- Every metric-specific analysis tree now writes label-free four-point additivity and oracle
  all-edge Atteson-margin diagnostics, and run-set rollups carry their scalar estimates.
- `wt audit-training-row-data` loads and renders one planned run-list row from the same
  registry/format/cache options without starting model training.
- `wt run-training-row --override-max-steps N --report-to wandb --run-name NAME` supports
  runtime-only trainer overrides for real smoke or batch launches while keeping the generated paper
  run lists unchanged.
- Wright launch scripts for the confirm-paper run live under `scripts/slurm/` and run each tree
  sequentially inside one GPU job so parent checkpoints exist before child rows start.
- Llama 3.2 1B now has independent full-FT and QKV LoRA r8/r64 scaffolds with an immutable base
  revision, 50 matched run lists per condition, causal packing checks, and successful bounded Wright
  root/dependency smokes. See `docs/LLAMA32_1B_EXPERIMENT_PLAN.md`; broad arrays remain gated by a
  three-tree resource benchmark and checkpoint retention policy.
- Llama full-FT child smoke `154473` proved local-parent initialization. Cleanup smoke `154474`
  proved successful jobs discard the 7.0 GB Trainer resume checkpoint only after retaining the
  2.4 GB final model; the removed path is persisted in backend metadata.
- `prune-training-parent-artifact` and the opt-in sequential-wrapper retention hook now remove an
  internal full model or LoRA merged model only after every direct child succeeds. The isolated
  Llama Wright smoke removed 2,488,861,763 bytes from n0, preserved both leaf models, and wrote a
  JSONL decision audit. Existing experiment wrappers remain pruning-off unless explicitly enabled.
- `prune-training-lora-tree-materializations` and `PRUNE_LORA_MERGED_AFTER_TREE=true` add an
  adapter-preserving end-of-tree gate. It requires successful terminal status plus an existing
  adapter for every node before removing remaining merged models. Wright tree-002 validation
  removed 17,421,998,790 bytes from each LoRA benchmark while retaining all 14 adapters.
- The WeightTraits-native behavioral rebuild now supports deterministic HellaSwag prompt artifacts,
  greedy seq2seq/causal collection, cumulative-LoRA leaves, complete-grid audits and quality health
  signals, aligned sentence embeddings, and paired semantic distance cubes. Wright jobs `154614` and
  `154621` passed the full eight-leaf/two-prompt Flan tree smoke with a finite 8x8 cube.
- The follow-up 100-prompt Flan diagnostic (`154624`) completed 800/800 records but exposed severe
  response collapse in three leaves (2, 2, and 11 unique outputs). Mean reference ROUGE-L was 0.037.
  Do not spend the 50-tree budget on Flan/HellaSwag; repeat on causal Llama production trees.
- The archived causal r8 leaf diagnostic (`154633`) produced 100/100 unique HellaSwag responses,
  dominant-response fraction 0.01, and mean reference ROUGE-L 0.122 after cumulative adapter
  rematerialization. This supports the Llama pivot; the remaining gate is a complete production tree.
- The complete archived r8 tree diagnostic (`154635` -> `154636`) then passed 700/700 records across
  seven leaves. Every leaf produced 100 unique outputs with 0.01 dominant-response fraction; the
  100-prompt semantic cube was finite and non-degenerate with maximum distance 0.1031. Repeat on the
  first completed production r8/r64 tree before launching the full behavioral run set.
- The faithful PhyloLM rebuild pins upstream commit `8c70edf`, keeps the GPLv3 gene artifact external
  and checksum-pinned, and rematerializes cumulative adapters only in memory. Wright job `154601`
  passed the two-gene/two-sample `n0 -> n2 -> n7` r8 smoke with zero empty alleles.

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
examples/training/confirm_paper_numbers/full_finetune.yaml
examples/training/confirm_paper_numbers/dataset_registry.yaml
examples/training/confirm_paper_numbers/dataset_formats.yaml
examples/training/confirm_paper_numbers/full_finetune_data_format_validation.json
examples/training/confirm_paper_numbers/dataset_registry_no_load_audit.json
examples/training/confirm_paper_numbers/full_finetune_data_cache_summary.json
examples/training/confirm_paper_numbers/full_finetune_run_list_summary.json
examples/training/confirm_paper_numbers/full_finetune_runlists/run_lists/*.runs.jsonl
examples/training/confirm_paper_numbers/full_finetune_runlists/reports/*.report.json
examples/training/confirm_paper_numbers/full_finetune_training_run_list_summary.json
examples/training/confirm_paper_numbers/full_finetune_training_runlists/run_lists/*.runs.jsonl
examples/training/confirm_paper_numbers/full_finetune_training_runlists/reports/*.report.json
examples/training/confirm_paper_numbers/lora_finetune.yaml
examples/training/confirm_paper_numbers/lora_finetune_training_run_list_summary.json
examples/training/confirm_paper_numbers/lora_finetune_training_runlists/run_lists/*.runs.jsonl
examples/training/confirm_paper_numbers/lora_finetune_training_runlists/reports/*.report.json
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

The full-FT confirm-paper-number scaffold uses
`examples/training/confirm_paper_numbers/full_finetune.yaml`, matching the draft's Flan-T5-base
training settings: 2,000 steps, learning rate `3e-4`, per-device train batch 8, gradient
accumulation 4, warmup 200, weight decay 0.01, and bf16. It currently generates dry-run-capable
per-tree run lists under `examples/training/confirm_paper_numbers/full_finetune_runlists/`; the
summary reports `valid=true`, 50 trees, 641 planned training rows, and no warnings/errors. Each tree
has its own run list, ledger path, report, and output root under
`outputs/confirm_paper_numbers/full_finetune/<tree_id>/`.

The 36-dataset registry and format-contract layer now lives in
`examples/training/confirm_paper_numbers/dataset_registry.yaml` and
`examples/training/confirm_paper_numbers/dataset_formats.yaml`. The offline format validation report
has `valid=true`, 50 trees, 641 jobs, 641 valid jobs, and 0 issues. The no-load registry audit has
`valid=true`, 36 datasets, and 36 `not_loaded` audits, meaning the declarations and requested splits
are structurally consistent without downloading from Hugging Face. Dotted field maps are supported
for nested rows such as `translation.en` -> `source_text`. The broken old `CogComp/trec` loader has
been replaced with `SetFit/TREC-QC` plus the current `label_text` target mapping.

The full-FT finite data cache lives under ignored path
`data/confirm_paper_numbers/full_finetune_cache/`, with tracked audit summary
`examples/training/confirm_paper_numbers/full_finetune_data_cache_summary.json`. The summary reports
`valid=true`, 36 datasets, and 36 ok datasets. Summarization registry rows now keep their original
datasets but apply deterministic `document` length caps before sample limits; all 9 summarization
train splits still cached 10,000 accepted examples under those caps. The full 36-dataset cache uses
`min_train_rows=1` because several classification/QA datasets are naturally smaller than 10,000.

A second, non-dry-run full-FT run-list set now lives under
`examples/training/confirm_paper_numbers/full_finetune_training_runlists/`. It is valid with 50
per-tree run lists and 641 planned `weighttraits.cli run-training-row` entries. Its runner options
point at the confirm-paper registry/formats, require
`data/confirm_paper_numbers/full_finetune_cache`, cap samples at 10,000 train / 1,000 eval, and allow
missing eval splits. Keep the dry-run run-list set for selector-only checks; use the training set
after cache and row-data smokes pass.

The LoRA comparison scaffold lives under
`examples/training/confirm_paper_numbers/lora_finetune.yaml` and
`examples/training/confirm_paper_numbers/lora_finetune_training_runlists/`. It uses the same
`assignment_summary.json`, registry, format map, finite cache, and train/eval caps as the full-FT
training set, but switches the training method to Flan-T5-base LoRA (`r=8`, `alpha=16`, dropout
0.05, target modules `q` and `v`, merge-after-train enabled). The LoRA summary is valid with the
same 50 trees and 641 planned rows. A mechanical comparison confirmed all 50 per-tree row counts
match the full-FT run lists, and `node_id` plus `dataset_id` order match tree-by-tree.

On 2026-07-08, the local cached row-data smoke passed for
`full_finetune_training_runlists/run_lists/confirm_paper_tree_001.runs.jsonl --index 0` with
`--max-train-samples 1 --max-eval-samples 1`: row `n0` (`rte`) required the finite cache and rendered
1 train plus 1 validation record with no issues, without loading a model.
Use `run-training-row --override-max-steps 2` for the first real trainer smoke from that row so the
paper row's default 2,000-step setting does not accidentally turn the smoke into a long training run.

Wright launch status from 2026-07-08:

- Synced the current working tree and finite cache to `/home/export/sgallagh/WeightTraits`.
- Installed `wandb==0.28.0` in the Wright `weighttraits` Conda env.
- Replaced incompatible `torch 2.12.1+cu130` with `torch 2.5.1+cu121`; Slurm CUDA check
  `153742` confirmed `torch.cuda.is_available() == True` on `NVIDIA L40`.
- Single-row model smoke `153743` completed: tree 001 row 0, 1 train / 1 eval row, 2 steps,
  `train_loss=3.81463623046875`, `eval_loss=4.069091796875`.
- Two-row dependency smoke `153744` completed: row 0 produced `n0`, row 1 loaded from `n0` and
  completed, validating sequential parent-child execution.
- Real full-FT launch is in progress with W&B offline logging: tree 001 is job `153745_1`, and
  trees 002-050 are array job `153752_[2-50%5]`. This gives at most 6 concurrent tree jobs.
  The earlier array `153746` was canceled before training because concurrent `mamba run` calls
  contended on the mamba lock; the scripts now call the env Python directly.
- LoRA two-row dependency smoke `153779` completed: tree 001 row 0 produced `n0`, row 1 loaded from
  `n0`, and both rows completed with 1 train / 1 eval row and 2 steps. The smoke output was archived
  to `outputs/confirm_paper_numbers/lora_finetune_smokes/confirm_paper_tree_001_dependency_smoke_153779`.
- Real LoRA launch is in progress with W&B offline logging as array job `153785_[1-50%2]`. It uses
  the same 50 trees as the full-FT launch and adds at most 2 concurrent LoRA tree jobs.

Wright audit on 2026-07-11 established that job `153912` is a faithful replay of the legacy
ELLMTrees executable scope but not of its prose label. Its frozen config declares
`q,k,v,o,wi,wo`; a completed adapter contains 336 LoRA A/B tensors across 168 resolved
`q,k,v,o,wo` modules and no `wi`, `wi_0`, or `wi_1` tensors. The old
`runs_lora_full_ft_approx` distance metadata likewise lists 168 trained tensors and the same
resolved scope. Let `153912` complete, report it as the legacy attention-plus-FFN-output condition,
and keep any corrected `q,k,v,o,wi_0,wi_1,wo` experiment separate. The corrected config scaffold is
`examples/training/confirm_paper_numbers/lora_all_projections_corrected.yaml`. Its generated set has
50 valid run lists and 641 rows under `lora_all_projections_corrected_training_runlists`; tree IDs,
row counts, and node/dataset order match q/v, and the cached tree-001 row-0 audit passed with 1 train
/ 1 eval record. A validation-only Wright copy at
`/home/export/sgallagh/WeightTraits-validation-20260711` passed 28 focused tests. Corrected row-0
smoke `154275` and dependent row-1 smoke `154277` both completed `0:0`, each capped at 1 train / 1
eval row and 2 steps. Both audits report 216 resolved modules, 432 LoRA tensors, zero unmatched
targets, and the requested `q,k,v,o,wi_0,wi_1,wo` scope. Bounded array `154446_[1-50%6]` is
submitted after dependency on the final legacy `153912` tasks; its throttle was raised from four
to six on 2026-07-13 without changing the `afterok:153912_*` dependency. The
paper-facing-versus-appendix decision remains open.

Versioned whitebox analysis began on Wright from the isolated validation checkout on 2026-07-11.
The analysis wrapper now accepts a separate `PATH_BASE`, allowing reviewed code to read live
training artifacts without modifying the live sequential-training checkout. One-tree canaries
`154281`--`154286` completed `0:0` with valid `l2`, `cosine`, and `correlation` rollups. Scaled jobs
`154287`--`154291` analyze cumulative adapter chains for q/v, k-only, qkv, full-attention, and the
legacy q/k/v/o/wo condition; `154292` analyzes full-model artifacts. All outputs are isolated under
`/home/export/sgallagh/WeightTraits-validation-20260711/outputs/analysis_v20260711/`.
The scaled LoRA runs revealed that run-set aggregation carried Atteson margins but omitted the
boolean theorem-certificate rate. `runset_results.py` now aggregates
`atteson_theorem_certified_rate` plus binomial SE; existing per-tree analyses remain valid and need
only a rollup refresh.

On 2026-07-13, WeightTraits added the missing paper-table ordering bridge. Direct whitebox analyses
now write per-metric `branch_ordering_*.json` audits using the active paper definition: leaves share
a branch when their first ancestor below the root matches, rank-biserial is oriented so larger means
cross-branch pairs are farther apart, and within-run correlation uses the same-branch indicator.
`summarize-training-run-set-analysis` also backfills these fields from saved distance matrices for
analysis directories produced before this change, then reports run-level rank-biserial mean/SE and
Fisher-z-averaged within-run correlation. Trees with a one-child root are valid recovery trees but
have no cross-branch class, so their ordering status is explicit and ordering/recovery sample counts
remain separate. `paper/weighttraits_variants_registry.yaml` and
`wt make-weighttraits-variants-table` map the versioned `analysis_v20260713` rollups into the exact
legacy `tab:variants` artifact schema without recomputing checkpoint distances.

Later on 2026-07-13, full fine-tuning direct analysis job `154440` completed and all 50 per-tree
artifacts were pulled into ignored `outputs/analysis_v20260713/full_finetune/`. The local rollup was
regenerated with the current code and relocated truth manifests, yielding 50 recovery trees and 26
ordering-eligible trees (24 have `missing_branch_class`). The five currently complete fresh
conditions now have native candidate artifacts under `reports/paper/weighttraits_completed_conditions.*`
and multi-metric diagnostics under `reports/paper/weighttraits_runset_diagnostics.*`, with plots for
metric robustness and cosine additivity/Atteson-versus-recovery. Their build path rejects any row
whose analysis engine is not `direct`; old ELLMTrees outputs remain post-hoc references only.

Same-tree paired analysis is now registered in
`paper/weighttraits_paired_comparisons_registry.yaml`. It compares cosine with L2 and correlation
within each completed condition, and compares k-only, q/k/v, full-attention, and full fine-tuning
with q/v under cosine. The builder requires identical topology IDs, direct-analysis provenance,
and registered artifact/representation/tree counts. It writes deterministic 10,000-resample
bootstrap intervals plus exact sign-test summaries to ignored
`reports/paper/weighttraits_paired_comparisons.*` and produces group-specific forest plots.

At the same checkpoint, legacy-scope training array `153912` had 46/50 tasks complete and tasks
3, 4, 29, and 50 still running. Dependent rollup `154445` and corrected-scope array
`154446_[1-50%6]` remain pending on `afterok`, so raising the corrected array throttle to six did
not weaken or change its dependency.

On 2026-07-07, the `fig:overview` and `fig:coherence_recovery` digests in
`paper/reference_registry.yaml` were refreshed to match the current sibling reference files after
`../ELLMTrees-paper/figures/fig1_paper_overview.{tex,pdf}`,
`../ELLMTrees-paper/figures/fig4_coherence_atteson.png`,
`../ELLMTrees/scripts/make_fig4_atteson_layers.py`, and
`../ELLMTrees/results/aggregate/recovery_rescore/fig4_atteson_layermeans.png` changed. Do not edit
those sibling repos from WeightTraits; treat future digest mismatches as reference-surface drift to
inspect explicitly.

WeightTraits now computes the paper's all-edge Atteson bottleneck: the minimum fitted edge length
over internal and pendant edges divided by twice the non-additivity error. Internal-edge summaries
are also reported, but they are not substituted for the theorem certificate. Each analyzed distance
matrix also receives the legacy-compatible four-point score
`A = (s2 - s3) / (s1 - s3 + eps)`, including truth-informative summaries when a manifest is known.
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

1. Let versioned analysis jobs `154440`--`154445` finish; `154445` waits for the final legacy
   `153912` tasks. Do not rerun model-distance analysis solely for ordering fields.
2. Sync the tested ordering/rollup code to an isolated Wright checkout and rerun only
   `summarize-training-run-set-analysis` for the five paper Flan conditions.
3. Build `reports/paper/weighttraits_variants_rebuild.{json,csv}` with
   `wt make-weighttraits-variants-table`, compare it with the pinned ELLMTrees reference, and record
   distributional deviations before registering candidate digests.

Local preparation on 2026-07-13 pulled the four completed native LoRA analysis directories (q/v,
k-only, qkv, and full attention), re-rolled all three metrics with current WeightTraits ordering
code, and exercised the provenance-gated candidate plotting path. For cosine, each condition has 50
recovery trees and 26 ordering-eligible trees; the other 24 are explicitly
`missing_branch_class`. No legacy ELLMTrees analysis code or derived CSV was used. The remaining
paper candidate rows still wait on full-FT job `154440` and legacy-scope job `154445`.

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

## 2026-07-14 Wright Launch Update

- Corrected Flan array `154446` was raised from `%6` to `%20`, matching the 20 L40 GPUs exposed by
  Wright's `all` partition.
- Llama 3.2 1B adapter-only production arrays are queued as `154594_[1-50%20]` for QKV LoRA r8 and
  `154595_[1-50%20]` for QKV LoRA r64. Slurm arbitrates them against the older Flan work.
- Both Llama arrays use the pinned base revision, offline HF execution, Trainer-checkpoint cleanup,
  internal-parent pruning, and audited end-of-tree merged-model cleanup. Durable artifacts are the
  base revision, adapters, run lists, ledgers, and audits.
- The broad full-FT array was deliberately not submitted because base-plus-adapters cannot
  reconstruct full fine-tuning. Its 10-step benchmark evidence is archived while a leaf-weight or
  delta retention policy is decided.

## Behavioral and PhyloLM Rebuild

The model-independent behavioral spine now has a stable response JSONL schema, explicit
dropped/empty/duplicate/repeat-grid audits, paired per-prompt embedding cosine distances with
per-pair denominators, and DerSimonian-Laird pooling for per-run correlations. HellaSwag on the
completed Flan full-FT trees is the first valid probe target. Translation is not held out in the
confirm-paper trees and cannot support the held-out claim without a new training assignment set.

The faithful PhyloLM core and GPU collector are reimplemented under `weighttraits.behavior`. They pin
upstream commit `8c70edf062a0adce2a3e6c8c79cd23a645fd0905`, preserve the raw-prompt 128-gene / 32-sample /
four-token / four-character contract, compute Nei distance, and emit a native distance cube plus NJ
tree. Adapter-only leaves are rematerialized in memory by merging the complete adapter chain onto the
pinned base. Generic leaf-array and post-analysis wrappers are ready; the next gate is a tiny causal
smoke after a production Llama tree finishes. See `docs/BEHAVIORAL_REBUILD.md`.

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
