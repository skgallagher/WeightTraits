# Final-result reproduction and Wright runbook

Status date: 2026-09-14 (America/New_York)

This is the operational handoff for the **WeightTraits repository and Wright
artifacts** used for the submission-week results. It supersedes
`docs/PAPER_REPRODUCTION.md` for the September fixed-2,000-step experiments.
That older document describes the July early-stopped rebuild and must not be
used as the primary protocol.

## 1. What is final

The primary training intervention is **exactly 2,000 optimizer steps per trained
node**, with no early stopping. The July early-stopped run is a training-depth
sensitivity analysis, not the primary result.

The controlled tree set has 50 trees and 641 trained non-root nodes: 276
internal nodes and 365 leaves. Recovery statistics use 46 topology-eligible
trees. Trees 015, 020, 037, and 047 have no informative truth split. Ordering
statistics use 26 trees; the other 24 have a one-child root and therefore no
cross-branch comparison class (`missing_branch_class`). These different
denominators are intentional.

### Direct-weight results (Table 2)

All entries are mean (SE). Recovery and PAER are percentages. `n_rec = 46` and
`n_ord = 26` for every row.

| Model / training | Rank-biserial | Branch-length r | Clade recovery | PhyloLM | PAER | RF; FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flan LoRA key only | .52 (.09) | -.53 (.07) | 92 (3) | n/a | 85 (5) | 2.39 (.25); .17 (.06) |
| Flan LoRA q/k/v | .56 (.09) | -.62 (.07) | 99 (1) | n/a | 98 (2) | 2.09 (.21); .02 (.02) |
| Flan LoRA q/k/v/o | .56 (.09) | -.62 (.07) | 99 (1) | n/a | 98 (2) | 2.09 (.21); .02 (.02) |
| Flan LoRA all projections (cumulative adapter space) | 1.00 (.00) | -.91 (.02) | 96 (2) | n/a | 89 (5) | 2.26 (.21); .11 (.05) |
| Flan full fine-tuning | .69 (.08) | -.71 (.06) | 100 (0) | n/a | 100 (0) | 2.04 (.19); .00 (.00) |
| Llama LoRA q/k/v, r=8 | .63 (.08) | -.64 (.07) | 100 (0) | 35 (5) | 100 (0) | 2.04 (.19); .00 (.00) |
| Llama LoRA q/k/v, r=64 | .65 (.08) | -.67 (.07) | 100 (0) | 24 (5) | 100 (0) | 2.04 (.19); .00 (.00) |
| Llama full fine-tuning | .64 (.08) | -.66 (.07) | 100 (0) | 52 (5) | 100 (0) | 2.04 (.19); .00 (.00) |

The identical 100%/2.04/0.00 topology cells are not copied cubes. Independent
checkpoint, matrix, and per-tree reconstruction checks passed. They arise
because those four fixed-2,000 conditions recover all 46 eligible topologies;
the RF mean 2.04 is the unresolved-tree baseline induced by the same truth-tree
set. The non-topological statistics differ across the three Llama rows.
The corrected all-projections Flan row is explicitly the cumulative-adapter
representation; it is not silently relabeled as a complete merged-full-weight
suite.

### Behavioral results

The primary estimand is the all-trained-node, observed-ancestor analysis. Values
are DerSimonian-Laird `r_DL` with 95% jackknife intervals; `k = 50` trees.

| Model | Trained translation | Held-out translation | HellaSwag | ARC-C | MMLU | TruthfulQA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama | -.20 [-.26, -.13] | -.23 [-.29, -.17] | -.17 [-.24, -.10] | -.20 [-.26, -.13] | -.19 [-.26, -.12] | -.22 [-.27, -.16] |
| Flan | -.10 [-.16, -.04] | +.04 [-.01, +.09] | +.09 [+.05, +.14] | -.14 [-.19, -.09] | -.16 [-.20, -.11] | -.17 [-.22, -.12] |

The completed Flan no-translation endpoint is +.04 [-.01, +.09], replacing the
older +.11 [+.02, +.21]. Its final Wright lineage was array 187979 (461 useful
completions), replacement array 195552 (179/179), retry 195764_461, and semantic
aggregation 195765.

HellaSwag has a 38.64% decoded-empty rate. Excluding prompt/draw identities for
which either member of a pair is empty gives the sensitivity estimate
-.265 [-.338, -.189]. The primary endpoint retains empty decoded responses.

### Layer and Figure 2 results

All six documented Flan full-FT tensor subsets recover 100% of clades and have
100% PAER on the 46 eligible trees. The smallest subset, encoder self-attention
key matrices, is 12 tensors and 7,077,888 / 247,577,856 parameters (2.86%).

The final Figure 2 paired-tree bootstrap gives:

- 10-fold increase in mean Atteson diagnostic: odds ratio 15.67
  [9.70, 58.55].
- 0.10 increase in mean four-point additivity: odds ratio 5.94
  [4.66, 8.38].

Both use 231 estimators, 46 paired trees, 2,000 bootstrap replicates, and seed
20260805.

## 2. Reproduction levels

Use the cheapest level that answers the question.

1. **Artifact-level reproduction (recommended for submission):** verify hashes,
   rerun distance/reconstruction/meta-analysis code against the sealed trained
   artifacts, and compare generated JSON/CSV files. This takes CPU jobs and
   hours rather than weeks.
2. **Analysis-stage reproduction:** regenerate distance cubes or behavioral
   inference from the completed checkpoints, then aggregate. This needs CPU and
   GPU jobs but does not retrain the trees.
3. **Full reproduction:** regenerate trees, assignments, caches, and all 641
   node checkpoints for each cohort before running downstream analysis. This is
   the archival path, not the deadline-week default.

## 3. Wright access

The Codex/noninteractive shell needs a user-authenticated SSH control socket.
Open it from a normal local terminal:

```bash
ssh -M -S /tmp/wright-codex.sock -fN wright
```

Test it and later close it with:

```bash
ssh -S /tmp/wright-codex.sock wright hostname
ssh -S /tmp/wright-codex.sock -O exit wright
```

Canonical Wright locations:

```text
trained/staged artifacts:
  /home/export/sgallagh/WeightTraits-all-results-20260817-clean
downstream code and result snapshot:
  /home/export/sgallagh/WeightTraits-downstream-20260831-native
Python:
  /home/export/sgallagh/.conda/envs/ellmtrees/bin/python
```

Verify the two sealed manifests before analysis:

```bash
ssh -S /tmp/wright-codex.sock wright \
  'sha256sum /home/export/sgallagh/WeightTraits-downstream-20260831-native/runtime_receipts/DEPLOYED_CODE_SHA256_SUBMISSION_20260831_V2 /home/export/sgallagh/WeightTraits-all-results-20260817-clean/.stage_provenance/STAGE_SHA256SUMS'
```

Expected SHA-256 values, in the same order:

```text
11a95522b1e046731666d4f678a39d39103961bc0b83ae9fe75cfbd4f1a433cb
642912687e67d0a48162a8d5cc0a95164ad92350ea629cd396fcfdd7917cd5d1
```

For lightweight distance, rollup, bootstrap, and aggregation work, submit to
`stat_ds`, not `all`. First inspect its nodes, then override both the partition
and any stale wrapper-level node pin on the command line:

```bash
sinfo -p stat_ds -N -o "%N %t %G"
sbatch --partition=stat_ds --nodelist=AVAILABLE_NODE path/to/job.sbatch
sbatch --partition=stat_ds --nodelist=AVAILABLE_NODE --dependency=afterok:JOBID path/to/aggregate.sbatch
```

The historical `submit_downstream_jobs.py` gate hard-codes `partition=all` and
`nodelist=n03` in both its scheduler receipt and generated `sbatch` command. Do
not reuse that gate unchanged for a new `stat_ds` launch. Parameterize and test
the scheduler policy, create a new code manifest, and issue a new immutable
submission plan first.

Useful status commands:

```bash
squeue -u sgallagh -o "%.18i %.10T %.10M %.32j %R"
sacct -j JOBID -X --format=JobID,JobName%36,State,ExitCode,Elapsed,Start,End -P
```

Do not leave obsolete retry arrays queued. Resolve exact job IDs with `squeue`
and `sacct` before `scancel`. At the last 2026-09-14 check, the queue was empty.

## 4. Full training protocol

### Generate the frozen tree set and assignments

From the WeightTraits repository root:

```bash
PYTHONPATH=src python -m weighttraits.cli generate-tree-set \
  --config examples/trees/confirm_paper_numbers.yaml \
  --out-dir examples/training/confirm_paper_numbers/trees \
  --summary-out examples/training/confirm_paper_numbers/tree_set_summary.json

PYTHONPATH=src python -m weighttraits.cli assign-task-data-set \
  --tree-set examples/training/confirm_paper_numbers/tree_set_summary.json \
  --config examples/training/confirm_paper_numbers/paper_task_families.yaml \
  --out-dir examples/training/confirm_paper_numbers/assigned_manifests \
  --summary-out examples/training/confirm_paper_numbers/assignment_summary.json \
  --seed-start 1 \
  --policy per_node_without_replacement
```

The tree generator starts at seed 20260707, uses Poisson branching with
lambda 1.5, 14 nodes, maximum depth 4, and rejects trees with fewer than four
leaves. Assignment draws from 36 task/data pairs without replacement within a
tree; task families may repeat.

### Frozen data and model contracts

Read these files before launching training:

- Flan full FT and all Flan LoRA scopes:
  `examples/training/confirm_paper_numbers/LEGACY_SEQ2SEQ_REBUILD.md`
- Llama full FT:
  `examples/training/confirm_paper_numbers/LEGACY_CAUSAL_REBUILD.md`
- Llama q/k/v LoRA r=8 and r=64:
  `examples/training/confirm_paper_numbers/LEGACY_LLAMA_LORA_REBUILD.md`

Shared requirements are seed 42, frozen `legacy_subsample` data, 10,000 train
rows and 1,000 evaluation rows per dataset where available, exactly 2,000
steps, evaluation every 250 steps, and **no early stopping**.

Flan uses `google/flan-t5-base` revision
`7bcac572ce56db69c1ea7c8af255c5d7c9672fc2`, source length 512, target length
128, batch size 8, gradient accumulation 4, learning rate 3e-4, 200 warmup
steps, and weight decay .01.

Llama uses `meta-llama/Llama-3.2-1B` revision
`4e20de362430cd3b72f300e6b0f18e50e7166e08`, completion-only causal loss,
combined prompt-plus-completion length 1024, batch size 8, gradient
accumulation 4, learning rate 2e-5, 200 warmup steps, and weight decay .01.
The Wright environment that produced the sealed runs was Python 3.11.15,
PyTorch 2.5.1+cu121, Transformers 5.8.0, Datasets 4.8.5, and Accelerate 1.13.0.

Cache data on the cluster through:

```bash
sbatch scripts/slurm/cache_legacy_causal_training_data.sbatch
```

Regenerate and validate run lists exactly as shown in the three rebuild
documents. Each cohort must validate as 50 trees, 641 runs, zero warnings, and
zero errors before submission. The standard production wrapper is:

```bash
sbatch --array=1-50%CONCURRENCY \
  --export=ALL,REPO=/path/to/WeightTraits,PYTHON_BIN=/home/export/sgallagh/.conda/envs/ellmtrees/bin/python,RUN_LIST_TEMPLATE=examples/training/confirm_paper_numbers/COHORT_training_runlists/run_lists/confirm_paper_tree_TREEID.runs.jsonl,RUN_NAME_PREFIX_TEMPLATE=PREFIX-TREEID \
  scripts/slurm/confirm_paper_tree_sequential.sbatch
```

Do not set `OVERRIDE_MAX_STEPS`. A parent checkpoint must complete and merge
before its child is loaded. Use a distinct run list, prefix, and output root for
the matched no-translation cohorts.

## 5. Reproduce the direct-weight table from sealed artifacts

The immutable submission plans are:

```text
examples/downstream/20260831/downstream_submission.json
examples/downstream/20260908_direct/submission.json
examples/downstream/20260908_direct/submission_correctedall.json
```

They pin every job config, completion binding, training-stage manifest, wrapper,
and output directory. On Wright, validate and submit a plan with its exact hash:

```bash
cd /home/export/sgallagh/WeightTraits-downstream-20260831-native
CONFIG=examples/downstream/20260908_direct/submission.json
CONFIG_SHA256=$(sha256sum "$CONFIG" | awk '{print $1}')
PYTHONPATH=src /home/export/sgallagh/.conda/envs/ellmtrees/bin/python \
  scripts/submit_downstream_jobs.py \
  --config "$CONFIG" \
  --config-sha256 "$CONFIG_SHA256"
```

The plan submits its pinned wrappers. For a new lightweight submission, make a
new immutable plan whose wrapper command uses `--partition=stat_ds`; never edit
a completed receipt in place. The final Llama full-FT table artifact is:

```text
/home/export/sgallagh/WeightTraits-downstream-20260831-native/outputs/final_paper_20260911/table2_llama_full/llama_full_table2_final.json
SHA-256: 2a5086513ef9c6388fa583b436a4891d63972b40cf603c8c5e9238292342aa05
```

The intended local combined-table builder is:

```bash
PYTHONPATH=src python scripts/build_merged_weight_paper_stats.py \
  --config examples/paper/corrected_direct_weight_table2.yaml \
  --out reports/paper/table2_rebuild.json \
  --csv-out reports/paper/table2_rebuild.csv
```

The contract must contain the exact 46 topology-eligible truth manifests and
strict 50-tree/641-node completion receipts. Do not use a July rollup as a
substitute for a fixed-2,000 completion binding. **Current gap:** the referenced
combined YAML contract is described in `docs/DIRECT_WEIGHT_TABLE2.md` but is not
present in this working copy. The final Llama full row was built by
`../outputs/01a0813c-87a9-7572-b27f-19208b163c60/wright_stage/build_llama_full_table2.py`,
which must also be generalized and promoted into `scripts/` for clean-clone
reproduction.

## 6. Reproduce behavior

The prompt and generation contract is
`examples/behavior/corrected_behavior_protocols_v1.json`. The four cohort source
and checkpoint inventories are under `examples/downstream/20260831/`. The
production `deployments/allnode_v1` directory on Wright is authoritative for the
final all-node job configs and endpoint scopes.

The native pipeline exposes the stages directly:

```bash
PYTHONPATH=src python scripts/run_native_behavior_pipeline.py materialize-prompts --help
PYTHONPATH=src python scripts/run_native_behavior_pipeline.py audit-sources --help
PYTHONPATH=src python scripts/run_native_behavior_pipeline.py audit-checkpoints --help
PYTHONPATH=src python scripts/run_native_behavior_pipeline.py audit-encoder --help
PYTHONPATH=src python scripts/run_native_behavior_pipeline.py study --help
```

Run them in that order: materialize immutable prompt sets, audit source files,
audit checkpoint coverage, audit the pinned sentence encoder, run per-node
inference, build prompt-paired semantic matrices, then run the cohort study.
The final analysis population is all 641 trained nodes, root excluded, with the
276 internal nodes treated as observed ancestors. Leaf-only results are a
sensitivity analysis, not a replacement population.

The final Flan no-translation receipt is locally mirrored at:

```text
../outputs/01a0813c-87a9-7572-b27f-19208b163c60/sources/wright/behavior/flan_notrans/terminal_receipt.json
```

It pins the Wright direct terminal and translation semantic runset. The remote
terminal path is:

```text
/home/export/sgallagh/WeightTraits-downstream-20260831-native/outputs/allnode_v1/direct/flan_notrans/terminal_receipt.json
SHA-256: 7f8cba724b3efa0a86835944808731d733c492eb04d45a3d54f2551a0f2f98c6
```

The final translation semantic runset is:

```text
/home/export/sgallagh/WeightTraits-downstream-20260831-native/outputs/allnode_v1/semantic/flan_notrans/translation/runset_receipt.json
SHA-256: 2d49afd1b026fb0238c365498774f5b7876be9299538847dc465981d4e6724a6
```

Require 50 trees, 641 nodes, 276 internal nodes, 365 leaves, root excluded,
`status=completed`, and `valid=true` before accepting a cohort result.

## 7. Reproduce the layer table and Figure 2

After syncing the final Flan full distance cubes, rebuild the layer subset table
and per-layer profile with:

```bash
PYTHONPATH=src python scripts/summarize_final_layertrace.py \
  --analysis-root ../outputs/01a0813c-87a9-7572-b27f-19208b163c60/sources/wright/direct/flan_full/analysis \
  --json-out reports/paper/final_fixed2000_layertrace.json \
  --csv-out reports/paper/final_fixed2000_layertrace_profile.csv
```

The script rejects an inconsistent 282-layer inventory and uses the exact 46
topology-eligible tree IDs. This command was rerun on 2026-09-14 and reproduced
both artifacts byte for byte:

```text
final_fixed2000_layertrace.json
  a832c92b1cfcef7bfb1507c9784bf5658bce065263a9e1dee569edfd1f01d5f9
final_fixed2000_layertrace_profile.csv
  00ebe123146ed1702aa2162153249789a02eeea44b10da0503e4eab50176a447
```

Build the final Figure 2 observation panel with:

```bash
PYTHONPATH=src python scripts/build_final_weight_figure2.py \
  --weighttraits-root ../outputs/01a0813c-87a9-7572-b27f-19208b163c60/sources/wright \
  --phylolm-metrics ../ELLMTrees/results/aggregate/weighttraits_fresh_phylolm/per_run_metrics.csv \
  --out reports/paper/merged_weight_figure2.json \
  --csv-out reports/paper/merged_weight_figure2.csv \
  --records-out reports/paper/merged_weight_figure2_records.csv
```

This command was also rerun on 2026-09-14 and reproduced all three artifacts
byte for byte (231 summaries and 10,626 paired-tree records):

```text
merged_weight_figure2.json
  53caf20e4b3a64e2b78b4c863f72a3639a25d3c84e80fe0a2754989f5774b0a8
merged_weight_figure2.csv
  3f9d3adf168719073f918b2010393eb532a6dbb3af4dd560b29dbb4d1c279fa1
merged_weight_figure2_records.csv
  e3ccdf87ece661a9cd4d87132e75bf4b4dbe27c27379db85aee9f009b6cd1b62
```

The current paired-bootstrap output is
`reports/paper/merged_weight_figure2_bootstrap.csv`. It records the estimator
count, paired-tree count, 2,000 replicates, and seed. **Archival gap:** the exact
bootstrap/logistic-regression builder that generated this CSV is not presently
checked into this working copy. The CSV and its inputs are preserved, but a clean
clone cannot yet regenerate those two interval rows with one command. Add and
test that builder before calling Figure 2 fully reproducible from source.

## 8. EOS/depth sensitivity

The fail-closed analyzer and its input contract are documented in
`docs/BEHAVIOR_EOS_DEPTH.md`. Given a pinned input inventory:

```bash
PYTHONPATH=src python scripts/build_behavior_eos_depth.py \
  --config /results/eos_depth_input.json \
  --out-json /results/eos_depth.json \
  --per-leaf-csv /results/eos_depth_per_leaf.csv \
  --per-depth-csv /results/eos_depth_per_depth.csv \
  --sensitivity-csv /results/eos_sensitivity_pairs.csv \
  --sensitivity-summary-csv /results/eos_sensitivity_summary.csv \
  --receipt /results/eos_depth_receipt.json
```

The analyzer requires exhaustive receipt inventories and pairs by
`(prompt_id, sample_id)`, never row position. Existing outputs require the
explicit `--overwrite` flag.

## 9. Rebuild the reproducibility workbook

The workbook is a reconciliation artifact, not the source of scientific truth.
From the WeightTraits root:

```bash
scripts/paper/run_rebuild_reproducibility_workbook.sh \
  --input ../ELLMTrees-paper/WeightTraits_all_results_reproducibility.xlsx \
  --status examples/paper/workbook_rebuild_status_20260810.json \
  --output ../outputs/weighttraits-workbook-final/WeightTraits_all_results_reproducibility_rebuild.xlsx \
  --workspace-root .. \
  --preview-dir ../outputs/weighttraits-workbook-final/previews
```

The launcher requires Codex runtime bundle 26.905.11957, Node v24.19.0, and
`@oai/artifact-tool` 2.8.59. It verifies the preserved input hash, refuses to
overwrite the source workbook, scans formula errors, and renders every sheet.
The checked-in status JSON is an August source receipt, not a scheduler. Update
it from final receipts and rebuild before treating a newly generated workbook as
the final submission audit.

## 10. Remaining archival blockers

The scientific jobs are complete and the Wright queue was empty at the final
2026-09-14 check. Two repository tasks remain before an independent clean clone
can reproduce everything without this working directory:

1. Commit the currently untracked September protocols, downstream configs,
   scripts, tests, and this runbook as one reviewed WeightTraits snapshot.
2. Add the final combined Table 2 YAML contract and promote/generalize the
   receipt-valid Llama full row builder from the output staging directory.
3. Check in and test the missing Figure 2 paired-tree bootstrap builder, then
   regenerate `merged_weight_figure2_bootstrap.csv` from
   `merged_weight_figure2_records.csv` and confirm exact agreement.

Until those are done, the sealed Wright manifests and terminal receipts are the
authoritative execution record. Do not mass-stage the current dirty working tree
without reviewing unrelated pre-existing changes.
