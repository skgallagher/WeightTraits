# Experiment Checklist

This is the queue-facing checklist for rebuilding the frozen ICLR v2 paper results from WeightTraits.
It is meant to answer two questions before any Wright submission:

- Is this experiment scientifically tied to a paper claim?
- Has it passed the small gates that make a cluster run worth spending GPUs on?

Current frozen paper target: `../ELLMTrees-paper/iclr_draft_v2.tex`.
Current Wright checkout: `/home/export/sgallagh/WeightTraits`.
Current Wright env: `/home/export/sgallagh/.conda/envs/weighttraits`.

## Status Key

- `ready`: config, run list, data inputs, and smoke path exist.
- `running`: submitted to Wright and currently active or pending.
- `blocked`: do not queue until the named blocker is fixed.
- `scaffold`: paper target is known, but WeightTraits still needs code/config before Wright work.
- `analysis`: no new training job yet; consume completed checkpoints/results first.
- `done`: generated, validated, and registered as a paper-facing artifact.

## Always-Run Gates

Before a new cluster array:

- [ ] Paper target named in `paper/reference_registry.yaml`, `paper/table_registry.yaml`, or this file.
- [ ] Config checked into `examples/` or `paper/`; no one-off command-only experiment.
- [ ] Per-tree or per-run run list generated with explicit output roots.
- [ ] Data source audited. For Hugging Face training, prefer finite cache plus `--require-data-cache`.
- [ ] One-row data audit passes without model loading.
- [ ] One-row trainer smoke passes with `--override-max-steps 2`.
- [ ] Two-row dependency smoke passes if parent-child checkpoint loading is involved.
- [ ] Slurm array has a concurrency cap.
- [ ] W&B mode is intentional: offline by default on Wright unless an API key is configured.
- [ ] First completed tree is inspected before scaling a new family further.

After the first completed tree:

- [ ] Every planned node has `training_log.jsonl`.
- [ ] Every planned node has its expected model artifact.
- [ ] Child rows loaded parent artifacts rather than starting from the base model.
- [ ] Logs contain no fatal trainer, dataset, tokenizer, or save errors.
- [ ] `wt analyze-training-ledger` can consume the tree ledger.
- [ ] Summary is written under a stable output directory.
- [ ] Every metric-specific tree has four-point additivity and Atteson-margin JSON diagnostics.
- [ ] Result is either added to a registry or explicitly marked exploratory.

Before a paper artifact changes:

- [ ] Source command is recorded.
- [ ] Inputs are declared.
- [ ] Output path and digest are recorded.
- [ ] Unit/smoke tests pass for touched code.
- [ ] R checks pass for regression/statistical claims.
- [ ] Difference from old ELLMTrees references is either within tolerance or documented.

## Current Wright Board

Last manually checked: 2026-07-13 10:12 EDT.

| ID | Status | Paper Target | Wright Job | Notes |
| --- | --- | --- | --- | --- |
| `cp-flan-full-50` | done | `tab:variants`, `fig:coherence_recovery`, whitebox recovery rebuild | `153745_1`, `153752_[2-50%5]` | All 50 Flan-T5-base full-FT tasks completed `0:0`. |
| `cp-flan-lora-qv-50` | done | LoRA comparison condition; informs `tab:variants` rebuild but is not every old LoRA target-module row | `153785_[1-50%2]` | All 50 q/v tasks completed `0:0`. |
| `cp-flan-lora-k-50` | done | `tab:variants`, `tab:lora_variants` | `153909_[1-50%4]` | All 50 k-only tasks completed `0:0`. |
| `cp-flan-lora-qkv-50` | done | `tab:variants`, `tab:lora_variants` | `153910_[1-50%4]` | All 50 q/k/v tasks completed `0:0`. |
| `cp-flan-lora-full-attn-50` | done | `tab:variants`, `tab:lora_variants` | `153911_[1-50%4]` | All 50 q/k/v/o tasks completed `0:0`. |
| `cp-flan-lora-full-ft-approx-50` | running | legacy `tab:variants`, `tab:lora_variants` replay | `153912_[1-50%6]` | Frozen legacy declaration q/k/v/o/wi/wo; resolved PEFT scope is q/k/v/o/wo because Flan-T5 has `wi_0`/`wi_1`, not `wi`. Throttle raised from four to six on 2026-07-13; dependency smoke `153908` passed. |
| `cp-first-tree-analysis` | done | Recovery pipeline gate | `153902`, `153903`, `153904` | Direct full, merged-LoRA, and cumulative-LoRA analyses completed successfully; PAER rollups validated. |
| `cp-versioned-analysis-v20260711` | done | whitebox recovery, additivity, Atteson, variant rollups | `154287`--`154292` | All six jobs completed `0:0`; rollups exist in the isolated checkout. |
| `cp-versioned-analysis-v20260713` | running | final whitebox recovery and paper variants rollups | `154440`--`154445` | LoRA jobs `154441`--`154444` completed `0:0`; full-FT `154440` is running; legacy `154445` waits for training. |
| `cp-flan-lora-all-projections-corrected` | running | corrected sensitivity beyond the legacy row | `154275`, `154277`, `154446_[1-50%6]` | Both smokes passed with 216 resolved modules; bounded array retains its dependency on legacy training. |
| `paper-reference-validation` | ready | all live-draft labels and pinned references | local/Wright command | Run after paper edits or reference-surface changes. |

Current queue snapshot command:

```bash
ssh -S /tmp/wright-codex.sock wright \
  'squeue -u sgallagh -o "%i %T %M %D %R" | head -90'
```

Current run-set dashboard commands:

```bash
PYTHONPATH=src python -m weighttraits.cli audit-training-run-set \
  --summary examples/training/confirm_paper_numbers/full_finetune_training_run_list_summary.json \
  --path-base . \
  --optional-artifact training_log \
  --out outputs/confirm_paper_numbers/full_finetune/run_set_completion_audit.json \
  --csv-out outputs/confirm_paper_numbers/full_finetune/run_set_completion_audit.csv \
  --allow-issues

PYTHONPATH=src python -m weighttraits.cli audit-training-run-set \
  --summary examples/training/confirm_paper_numbers/lora_finetune_training_run_list_summary.json \
  --path-base . \
  --optional-artifact training_log \
  --out outputs/confirm_paper_numbers/lora_finetune/run_set_completion_audit.json \
  --csv-out outputs/confirm_paper_numbers/lora_finetune/run_set_completion_audit.csv \
  --allow-issues
```

## Active Training Sets

### `cp-flan-full-50`

Purpose:

- Rebuild the paper's Flan-T5-base full fine-tuning tree sample on a clean 50-tree set.
- Feed whitebox distance/recovery, branch-structure, and Atteson/additivity analyses.

Inputs:

- Config: `examples/training/confirm_paper_numbers/full_finetune.yaml`
- Run-list summary: `examples/training/confirm_paper_numbers/full_finetune_training_run_list_summary.json`
- Run lists: `examples/training/confirm_paper_numbers/full_finetune_training_runlists/run_lists/`
- Truth manifests: `examples/training/confirm_paper_numbers/assigned_manifests/`
- Dataset cache: `data/confirm_paper_numbers/full_finetune_cache/`

Launch command used:

```bash
ssh -S /tmp/wright-codex.sock wright \
  'cd /home/export/sgallagh/WeightTraits &&
   sbatch --parsable --array=1-50%5 \
     --export=ALL,REPORT_TO=wandb,WANDB_MODE=offline,WANDB_PROJECT=weighttraits-confirm-paper \
     scripts/slurm/confirm_paper_tree_sequential.sbatch'
```

Completion gate for each tree:

- [ ] Slurm task completed with exit `0:0`.
- [ ] Ledger exists:
  `examples/training/confirm_paper_numbers/full_finetune_training_runlists/ledgers/confirm_paper_tree_TREEID.training_ledger.jsonl`
- [ ] Output root exists:
  `outputs/confirm_paper_numbers/full_finetune/confirm_paper_tree_TREEID/`
- [ ] `training_log.jsonl` count equals the line count of the tree run list.
- [ ] Expected artifact for every row is `model`.
- [ ] Completion audit passes:

```bash
PYTHONPATH=src python -m weighttraits.cli audit-training-tree \
  --run-list examples/training/confirm_paper_numbers/full_finetune_training_runlists/run_lists/confirm_paper_tree_001.runs.jsonl \
  --path-base . \
  --out outputs/confirm_paper_numbers/full_finetune/confirm_paper_tree_001/completion_audit.json
```

- [ ] First analysis command succeeds:

```bash
PYTHONPATH=src python -m weighttraits.cli analyze-training-ledger \
  --ledger examples/training/confirm_paper_numbers/full_finetune_training_runlists/ledgers/confirm_paper_tree_001.training_ledger.jsonl \
  --truth-manifest examples/training/confirm_paper_numbers/assigned_manifests/confirm_paper_tree_001.manifest.jsonl \
  --artifact model \
  --metric l2 \
  --metric cosine \
  --metric correlation \
  --out outputs/confirm_paper_numbers/full_finetune/confirm_paper_tree_001/model_leaf_analysis
```

### `cp-flan-lora-qv-50`

Purpose:

- Run a same-tree LoRA comparison beside full fine-tuning.
- Exercise merged and cumulative LoRA recovery on the paper-scale topology set.

Inputs:

- Config: `examples/training/confirm_paper_numbers/lora_finetune.yaml`
- Run-list summary: `examples/training/confirm_paper_numbers/lora_finetune_training_run_list_summary.json`
- Run lists: `examples/training/confirm_paper_numbers/lora_finetune_training_runlists/run_lists/`
- Truth manifests: `examples/training/confirm_paper_numbers/assigned_manifests/`
- Dataset cache: `data/confirm_paper_numbers/full_finetune_cache/`

Tree-set guarantee:

- [x] Same `assignment_summary.json` as full fine-tune.
- [x] Same 50 tree IDs.
- [x] Same 641 planned rows.
- [x] Same per-tree row counts.
- [x] Same `node_id` and `dataset_id` order tree-by-tree.

Paper-facing merged-versus-cumulative finding (provisional, 2026-07-09):

- The first 14 completed LoRA trees show a real representation effect under cosine/correlation.
  For cosine, merged leaves have mean FN `0.143` (SE `0.097`), PAER `85.7%` (SE `9.4%`), and
  clade recovery `94.0%` (SE `4.1%`); cumulative adapter-chain leaves have mean FN `0.286`
  (SE `0.125`), PAER `71.4%` (SE `12.1%`), and clade recovery `89.9%` (SE `4.6%`).
- This is expected mathematically rather than a scoring discrepancy. A merged leaf is represented
  as `theta_base + delta_path`, while the adapter-chain analysis represents `delta_path` alone.
  L2 cancels the common base, and the two representations currently give identical L2 recovery
  summaries. Cosine and correlation are origin-dependent, so adding the shared base can change the
  distance matrix and NJ topology.
- The cumulative representation also has a substantial practical advantage. Across all 14 nodes of
  representative tree 001, the adapter artifact directories total `78 MiB`, versus `12.94 GiB` for
  merged checkpoints (about `166x` smaller; the main files are `3.39 MiB` per adapter versus
  `944.47 MiB` per merged model). On the same first 14 ready trees, the cumulative analysis job took
  `1:44`, versus `12:22` for merged analysis (about `7.1x` faster), while retaining `89.9%` cosine
  clade recovery. Treat the timing as an operational observation rather than a controlled benchmark
  until cold/warm-cache runs on identical nodes are recorded.
- Paper action: do not collapse merged and cumulative LoRA into one result. State the representation
  explicitly in the methods/table caption, audit what the old `TRAINED_ONLY=1` Flan rows represented,
  and report the paired same-tree sensitivity (or choose one primary representation and put the other
  in an ablation) after all 50 trees finish. Add a controlled adapter-versus-merged resource benchmark
  before making a formal speed claim. The 14-tree values above are provisional.

Launch command used:

```bash
ssh -S /tmp/wright-codex.sock wright \
  'cd /home/export/sgallagh/WeightTraits &&
   sbatch --parsable --array=1-50%2 \
     --export=ALL,RUN_LIST_TEMPLATE=examples/training/confirm_paper_numbers/lora_finetune_training_runlists/run_lists/confirm_paper_tree_TREEID.runs.jsonl,RUN_NAME_PREFIX_TEMPLATE=confirm-paper-lora-tree-TREEID,REPORT_TO=wandb,WANDB_MODE=offline,WANDB_PROJECT=weighttraits-confirm-paper \
     scripts/slurm/confirm_paper_tree_sequential.sbatch'
```

Completion gate for each tree:

- [ ] Slurm task completed with exit `0:0`.
- [ ] Ledger exists:
  `examples/training/confirm_paper_numbers/lora_finetune_training_runlists/ledgers/confirm_paper_tree_TREEID.training_ledger.jsonl`
- [ ] Output root exists:
  `outputs/confirm_paper_numbers/lora_finetune/confirm_paper_tree_TREEID/`
- [ ] `training_log.jsonl` count equals the line count of the tree run list.
- [ ] Expected artifacts for every row are `adapter` and `merged`.
- [ ] Completion audit passes:

```bash
PYTHONPATH=src python -m weighttraits.cli audit-training-tree \
  --run-list examples/training/confirm_paper_numbers/lora_finetune_training_runlists/run_lists/confirm_paper_tree_001.runs.jsonl \
  --path-base . \
  --out outputs/confirm_paper_numbers/lora_finetune/confirm_paper_tree_001/completion_audit.json
```

- [ ] Merged analysis succeeds:

```bash
PYTHONPATH=src python -m weighttraits.cli analyze-training-ledger \
  --ledger examples/training/confirm_paper_numbers/lora_finetune_training_runlists/ledgers/confirm_paper_tree_001.training_ledger.jsonl \
  --truth-manifest examples/training/confirm_paper_numbers/assigned_manifests/confirm_paper_tree_001.manifest.jsonl \
  --artifact merged \
  --metric l2 \
  --metric cosine \
  --metric correlation \
  --out outputs/confirm_paper_numbers/lora_finetune/confirm_paper_tree_001/merged_leaf_analysis
```

- [ ] Cumulative adapter-chain analysis succeeds:

```bash
PYTHONPATH=src python -m weighttraits.cli analyze-training-ledger \
  --ledger examples/training/confirm_paper_numbers/lora_finetune_training_runlists/ledgers/confirm_paper_tree_001.training_ledger.jsonl \
  --truth-manifest examples/training/confirm_paper_numbers/assigned_manifests/confirm_paper_tree_001.manifest.jsonl \
  --artifact adapter_chain \
  --metric l2 \
  --metric cosine \
  --metric correlation \
  --out outputs/confirm_paper_numbers/lora_finetune/confirm_paper_tree_001/cumulative_leaf_analysis
```

## Queue-Ready Next Gates

### `cp-first-tree-analysis`

Status: `analysis`.

Queue trigger:

- [ ] At least one full-FT tree completes.
- [ ] At least one LoRA tree completes.

Actions:

- [ ] Run `audit-training-run-set` and check `n_ready`.
- [ ] Run the completion gate for the first completed full tree.
- [ ] Run `analyze-training-ledger` for full model artifacts.
- [ ] Run the completion gate for the first completed LoRA tree.
- [ ] Run `analyze-training-ledger` for LoRA merged artifacts.
- [ ] Run `analyze-training-ledger` for LoRA cumulative adapter-chain artifacts.
- [ ] Compare summaries for basic sanity: number of leaves, number of scored splits, RF, exact recovery, pooled clade recovery, and distance means.
- [ ] If the first tree passes, queue analysis across the remaining completed trees.

Run-set analysis dry-runs:

These use the direct analysis engine: ledger/artifacts to independently computed
distance layers, Biopython neighbor joining, and DendroPy RF scoring.

```bash
PYTHONPATH=src python -m weighttraits.cli analyze-training-run-set \
  --summary examples/training/confirm_paper_numbers/full_finetune_training_run_list_summary.json \
  --path-base . \
  --artifact model \
  --metric l2 \
  --metric cosine \
  --metric correlation \
  --out outputs/confirm_paper_numbers/full_finetune/run_set_analysis \
  --report-out outputs/confirm_paper_numbers/full_finetune/direct_run_set_analysis_plan.json \
  --optional-artifact training_log \
  --skip-existing \
  --dry-run
```

```bash
PYTHONPATH=src python -m weighttraits.cli analyze-training-run-set \
  --summary examples/training/confirm_paper_numbers/lora_finetune_training_run_list_summary.json \
  --path-base . \
  --artifact merged \
  --metric l2 \
  --metric cosine \
  --metric correlation \
  --out outputs/confirm_paper_numbers/lora_finetune/run_set_analysis \
  --report-out outputs/confirm_paper_numbers/lora_finetune/direct_merged_run_set_analysis_plan.json \
  --optional-artifact training_log \
  --skip-existing \
  --dry-run
```

```bash
PYTHONPATH=src python -m weighttraits.cli analyze-training-run-set \
  --summary examples/training/confirm_paper_numbers/lora_finetune_training_run_list_summary.json \
  --path-base . \
  --artifact adapter_chain \
  --metric l2 \
  --metric cosine \
  --metric correlation \
  --out outputs/confirm_paper_numbers/lora_finetune/run_set_analysis \
  --report-out outputs/confirm_paper_numbers/lora_finetune/direct_cumulative_run_set_analysis_plan.json \
  --optional-artifact training_log \
  --skip-existing \
  --dry-run
```

Remove `--dry-run` after the first completed full and LoRA trees pass the single-tree
analysis gate. Keep `--skip-existing` for repeated sweeps. The command writes one subdirectory per ready tree:
`TREEID/model_leaf_analysis`, `TREEID/merged_leaf_analysis`, or
`TREEID/cumulative_leaf_analysis`.

After any direct-analysis summaries are written, roll them up:

```bash
PYTHONPATH=src python -m weighttraits.cli summarize-training-run-set-analysis \
  --analysis-root outputs/confirm_paper_numbers/full_finetune/run_set_analysis \
  --artifact model \
  --out outputs/confirm_paper_numbers/full_finetune/direct_run_set_analysis_summary.json \
  --csv-out outputs/confirm_paper_numbers/full_finetune/direct_run_set_analysis_rows.csv
```

```bash
PYTHONPATH=src python -m weighttraits.cli summarize-training-run-set-analysis \
  --analysis-root outputs/confirm_paper_numbers/lora_finetune/run_set_analysis \
  --artifact merged \
  --out outputs/confirm_paper_numbers/lora_finetune/direct_merged_run_set_analysis_summary.json \
  --csv-out outputs/confirm_paper_numbers/lora_finetune/direct_merged_run_set_analysis_rows.csv
```

```bash
PYTHONPATH=src python -m weighttraits.cli summarize-training-run-set-analysis \
  --analysis-root outputs/confirm_paper_numbers/lora_finetune/run_set_analysis \
  --artifact adapter_chain \
  --out outputs/confirm_paper_numbers/lora_finetune/direct_cumulative_run_set_analysis_summary.json \
  --csv-out outputs/confirm_paper_numbers/lora_finetune/direct_cumulative_run_set_analysis_rows.csv
```

Direct-analysis Slurm submissions:

The Slurm wrapper defaults to `SKIP_EXISTING=1`, so repeated submissions only
analyze ready trees whose compatible direct `summary.json` is missing.

For isolated code/artifact operation, the wrapper accepts `PATH_BASE`. The 2026-07-11 versioned
analysis jobs run code from `/home/export/sgallagh/WeightTraits-validation-20260711`, set
`PATH_BASE=/home/export/sgallagh/WeightTraits`, and write only under
`outputs/analysis_v20260711/` in the isolated checkout. Canary jobs `154281`--`154286` completed
`0:0` with valid three-metric rollups. Scaled cumulative-LoRA jobs are `154287`--`154291`; scaled
full-model job is `154292`.

The first scaled rollups exposed a reporting omission rather than an analysis failure: per-tree
`atteson_theorem_certified` values were present, but the run-set aggregate omitted their rate. The
aggregator now reports `atteson_theorem_certified_rate` and its binomial SE. Existing distance/tree
analyses do not need to be rerun; only the lightweight rollup command must be repeated after syncing
the tested fix to the isolated checkout.

```bash
ssh -S /tmp/wright-codex.sock wright \
  'cd /home/export/sgallagh/WeightTraits &&
   sbatch --parsable \
     --export=ALL,DRY_RUN=1 \
     scripts/slurm/confirm_paper_direct_analysis.sbatch'
```

Full fine-tune:

```bash
ssh -S /tmp/wright-codex.sock wright \
  'cd /home/export/sgallagh/WeightTraits &&
   sbatch --parsable \
     --export=ALL,SUMMARY=examples/training/confirm_paper_numbers/full_finetune_training_run_list_summary.json,ARTIFACT=model,OUT_ROOT=outputs/confirm_paper_numbers/full_finetune/run_set_analysis,REPORT_OUT=outputs/confirm_paper_numbers/full_finetune/direct_run_set_analysis_plan.json,ROLLUP_OUT=outputs/confirm_paper_numbers/full_finetune/direct_run_set_analysis_summary.json,CSV_OUT=outputs/confirm_paper_numbers/full_finetune/direct_run_set_analysis_rows.csv \
     scripts/slurm/confirm_paper_direct_analysis.sbatch'
```

LoRA merged:

```bash
ssh -S /tmp/wright-codex.sock wright \
  'cd /home/export/sgallagh/WeightTraits &&
   sbatch --parsable \
     --export=ALL,SUMMARY=examples/training/confirm_paper_numbers/lora_finetune_training_run_list_summary.json,ARTIFACT=merged,OUT_ROOT=outputs/confirm_paper_numbers/lora_finetune/run_set_analysis,REPORT_OUT=outputs/confirm_paper_numbers/lora_finetune/direct_merged_run_set_analysis_plan.json,ROLLUP_OUT=outputs/confirm_paper_numbers/lora_finetune/direct_merged_run_set_analysis_summary.json,CSV_OUT=outputs/confirm_paper_numbers/lora_finetune/direct_merged_run_set_analysis_rows.csv \
     scripts/slurm/confirm_paper_direct_analysis.sbatch'
```

LoRA cumulative adapter-chain:

```bash
ssh -S /tmp/wright-codex.sock wright \
  'cd /home/export/sgallagh/WeightTraits &&
   sbatch --parsable \
     --export=ALL,SUMMARY=examples/training/confirm_paper_numbers/lora_finetune_training_run_list_summary.json,ARTIFACT=adapter_chain,OUT_ROOT=outputs/confirm_paper_numbers/lora_finetune/run_set_analysis,REPORT_OUT=outputs/confirm_paper_numbers/lora_finetune/direct_cumulative_run_set_analysis_plan.json,ROLLUP_OUT=outputs/confirm_paper_numbers/lora_finetune/direct_cumulative_run_set_analysis_summary.json,CSV_OUT=outputs/confirm_paper_numbers/lora_finetune/direct_cumulative_run_set_analysis_rows.csv \
     scripts/slurm/confirm_paper_direct_analysis.sbatch'
```

Wright check pattern:

```bash
ssh -S /tmp/wright-codex.sock wright \
  'cd /home/export/sgallagh/WeightTraits &&
   sacct -j 153745,153752,153785 --format=JobID,State,ExitCode,Elapsed -P | tail -80'
```

### `paper-reference-validation`

Status: `ready`.

Purpose:

- Keep the full-paper surface pinned while training runs in the background.
- Catch stale labels after paper edits.

Commands:

```bash
PYTHONPATH=src python -m weighttraits.cli validate-reference-registry \
  --registry paper/reference_registry.yaml

PYTHONPATH=src python -m weighttraits.cli validate-table-registry \
  --registry paper/table_registry.yaml \
  --require-outputs

PYTHONPATH=src python -m weighttraits.cli run-table-comparisons \
  --registry paper/table_registry.yaml \
  --out reports/paper/table_comparison_validation.json
```

## Paper Experiment Backlog

### `flan-lora-target-module-variants`

Status: `running`.

Paper target:

- `tab:variants`
- `tab:lora_variants`

Old ELLMTrees reference rows:

- `flan_lora_k_only`
- `flan_lora_qkv`
- `flan_lora_full_attn`
- `flan_lora_full_ft_approx`

Do before queueing:

- [x] Reproduce all four exact old LoRA target-module scopes alongside the new q/v comparison.
- [x] Add one config per selected target-module variant.
- [x] Generate run-list sets from the same `assignment_summary.json`.
- [x] Mechanically compare tree IDs, row counts, `node_id`, and `dataset_id` order against q/v: all
  four sets have the same 50 trees and 641 rows.
- [x] Run cached row audit for tree 001 row 0 in all four sets.
- [x] Run isolated two-row dependency smokes: jobs `153905`--`153908` completed `0:0`; both nodes
  completed, row 1 loaded row 0's merged parent, and PEFT recorded the expected target modules.
- [x] Launch with a bounded concurrency cap: jobs `153909`--`153912`, initially `1-50%1`, then
  raised in place to `1-50%2` after the inventory check and finally to `1-50%4`. Slurm continues to
  hold eligible tasks for resources/priority, so the higher throttle does not oversubscribe nodes.

Launch mapping:

| Variant | Target modules | Config | Run-list root | Wright job |
| --- | --- | --- | --- | --- |
| k-only | `k` | `lora_k_only.yaml` | `lora_k_only_training_runlists` | `153909_[1-50%4]` |
| qkv | `q,k,v` | `lora_qkv.yaml` | `lora_qkv_training_runlists` | `153910_[1-50%4]` |
| full attention | `q,k,v,o` | `lora_full_attn.yaml` | `lora_full_attn_training_runlists` | `153911_[1-50%4]` |
| legacy full-FT approximation | declared `q,k,v,o,wi,wo`; resolved `q,k,v,o,wo` | `lora_full_ft_approx.yaml` | `lora_full_ft_approx_training_runlists` | `153912_[1-50%4]` |

Wright audit on 2026-07-11 found that a completed `153912` adapter has 336 LoRA A/B tensors across
168 resolved modules: `q`, `k`, `v`, `o`, and `wo`, with zero `wi`, `wi_0`, or `wi_1` tensors. This
matches the old ELLMTrees `runs_lora_full_ft_approx` distance metadata, which also lists 168 trained
tensors and contains `wo` but no gated-FFN input projections. Keep `153912` as a faithful replay of
the legacy executable condition, but do not describe it as all projections or as adapting `w_i`.

New executions validate every requested LoRA target against the loaded model before training and
write `lora_target_audit.json` with the resolved modules and exact trainable counts. The frozen
legacy config intentionally retains its declared `wi` for launch provenance and will fail that new
strict preflight. A corrected, not-yet-queued scaffold lives at
`lora_all_projections_corrected.yaml` with explicit `wi_0` and `wi_1` targets.

Paper equivalence still requires completed 50-tree recovery/ordering analyses and a documented
comparison against the pinned ELLMTrees rows.

### `flan-lora-all-projections-corrected`

Status: `running`.

Purpose:

- Measure the condition the current paper prose conceptually describes: attention projections plus
  both gated Flan-T5 FFN input projections and the FFN output projection.
- Keep it separate from the legacy `full_ft_approx` row so a corrected ablation is not mistaken for
  an exact replay.

Do before queueing:

- [x] Add an explicit `q,k,v,o,wi_0,wi_1,wo` config.
- [x] Add strict requested-versus-resolved LoRA module validation and provenance output.
- [x] Generate 50 valid run lists / 641 rows from the same `assignment_summary.json` under a
  distinct output root; tree IDs, row counts, and `(node_id, dataset_id)` order match q/v.
- [x] Run cached tree-001 row-0 audit with 1 train / 1 eval record and zero issues.
- [x] Run one-row trainer smoke and verify `lora_target_audit.json` reports 216 resolved modules:
  job `154275` completed `0:0` with 216 modules, 432 LoRA tensors, and zero unmatched targets.
- [x] Run two-row dependency smoke: dependent job `154277` completed `0:0` and loaded the corrected
  parent scope.
- [ ] Decide whether the corrected condition is paper-facing or an appendix sensitivity analysis.
- [x] Launch a bounded array: `154446_[1-50%6]` is submitted with dependency on legacy array
  `153912`.

Wright smoke staging (2026-07-11): the reviewed code and corrected run lists are isolated at
`/home/export/sgallagh/WeightTraits-validation-20260711`, with the finite cache linked read-only from
the live experiment checkout. Focused remote tests passed (28 tests). The original two-day-wrapper
submission `154274` was canceled while pending and replaced with 30-minute row jobs: row 0 is
`154275`, and dependent row 1 is `154277` (`afterok:154275`). Both completed successfully with 1
train row, 1 eval row, and 2 steps. Bounded corrected array `154446_[1-50%6]` is now submitted with
an `afterok` dependency on the legacy `153912` array. Its concurrency throttle was raised from four
to six on 2026-07-13; the dependency expression was unchanged.

### `llama1b-variants`

Status: `ready` for a three-tree benchmark; broad arrays remain blocked on retention policy.

Paper target:

- `tab:variants`
- Decoder-only section for Llama-3.2-1B.

Old ELLMTrees reference rows:

- `llama1b_lora_qkv_r8`
- `llama1b_lora_qkv_r64`
- `llama1b_full_ft`

Completed gates:

- [x] Confirm the official model ID `meta-llama/Llama-3.2-1B`; pin cached revision
  `4e20de362430cd3b72f300e6b0f18e50e7166e08` in every root job.
- [x] Add separate full-FT, QKV LoRA r8, and QKV LoRA r64 causal-LM configs.
- [x] Validate completion-only prompt/target packing and EOS-as-pad behavior.
- [x] Resolve exactly 48 QKV LoRA modules (16 each of q/k/v); rank 64 has 9,437,184
  trainable adapter parameters and zero unmatched targets.
- [x] Generate 50 valid run lists / 641 rows for each condition. Their ordered tree, node, and
  dataset assignments exactly match the Flan full-FT set.
- [x] Audit one cached train/eval row from classification, translation, summarization, and QA.
- [x] Run Wright root and parent-child rank-64 LoRA smokes: jobs `154463` and `154464` completed
  two steps with status `completed`.
- [x] Run Wright full-FT root smoke: job `154465` completed two steps on one L40 without OOM.
- [x] Complete the full-FT parent-child smoke: `154473` completed two steps from the local n0
  parent. Its final model is 2.4 GB; the Trainer resume checkpoint added 7.0 GB. Attempt
  `154472` failed before script execution because the isolated staging log directory was absent.
- [ ] Run three representative trees per condition; record wall time, peak GPU memory, disk growth,
  and first-tree recovery before considering 50-tree arrays.
- [ ] Finish the lineage-model retention policy. Successful Llama jobs now remove Trainer resume
  checkpoints only after final artifact save; job `154474` verified a 9.3 GB node falls to 2.4 GB
  and records the removed path in backend metadata. Internal parent-model pruning is still pending.

Guardrail:

- Do not queue Llama broad arrays from the current Flan run-list assumptions. Treat Llama as a new model family.
- Do not retain every merged 1B checkpoint by default. The root rank-64 smoke produced a 53 MB
  adapter and a 2.4 GB merged model; 641 merged models would be roughly 1.5 TB per condition.
- Wright has a cached gated checkpoint, but offline execution is intentional until authenticated
  Hugging Face access is verified. The pinned revision prevents silent model drift.
- Do not pursue `runs_llama8b_full_ft_approx` unless deliberately revisited.

### `behavior-holdout-rebuild`

Status: `scaffold`.

Paper target:

- `tab:behavior_holdout`
- `tab:regression_betas`
- `tab:behavior_het`
- `fig:regression_diagnostics`

Known reference surface:

- `reports/paper/behavior_holdout_reference.json`
- `reports/paper/behavior_holdout_reference.csv`
- Old source script: `../ELLMTrees/scripts/behavior_meta_table.py`
- Old regression script: `../ELLMTrees/scripts/regression_distance_vs_behavior.py`

Do before queueing:

- [ ] Decide which trained WeightTraits outputs are the first behavior-eval inputs.
- [ ] Port or wrap behavioral probe generation.
- [ ] Define probe output schema and dropped-record audit.
- [ ] Add a small probe smoke on one completed tree.
- [ ] Add R regression cross-check target before paper-facing coefficients change.
- [ ] Only then launch broad behavioral probes.

### `coherence-and-atteson`

Status: `analysis`.

Paper target:

- `fig:coherence_recovery`

Known reference surface:

- `../ELLMTrees/results/aggregate/recovery_rescore/fourpoint_additivity_by_group.csv`
- `../ELLMTrees/results/aggregate/recovery_rescore/atteson_quartet_by_group.csv`
- `../ELLMTrees-paper/figures/fig4_coherence_atteson.png`

Do after enough trees complete:

- [x] Aggregate per-tree distance/recovery summaries by condition; versioned analysis jobs write
  per-condition JSON/CSV rollups.
- [x] Implement or port four-point additivity.
- [x] Implement Atteson margin with the paper definition: minimum fitted edge length over internal and pendant edges divided by twice the non-additivity error.
- [ ] Regenerate the figure from WeightTraits outputs.
- [ ] Compare against the pinned old reference and document deviations.

### `layertrace-grid`

Status: `analysis`.

Paper target:

- `fig:layertrace_grid`
- `tab:layer_subsets`

Do after first-tree analysis succeeds:

- [ ] Confirm `analyze-training-ledger --layer` works on paper-scale checkpoints for one tree.
- [ ] Select layer subsets matching the paper definition.
- [ ] Queue layer-sliced analyses across completed trees.
- [ ] Generate a WeightTraits layertrace grid.
- [ ] Compare with the pinned old reference.

### `hf-mistral-real-family`

Status: `scaffold`.

Paper target:

- `fig:real-trees`

Known reference surface:

- Truth manifest: `../ELLMTrees/outputs/hf_zoo/d_mistral_7b_resolved/manifest.jsonl`
- Recovery summary: `../ELLMTrees/results/hf_zoo/d_mistral_7b/weight_frechet/rf_summary.json`

Do before queueing:

- [ ] Decide whether WeightTraits will ingest old checkpoints/results or recompute distances from model weights.
- [ ] Add a WeightTraits manifest for the Mistral family.
- [ ] Add score/plot commands to a registry.
- [ ] Run a one-family local/Wright smoke.
- [ ] Register output figure/table artifacts.

## Runbook Snippets

Use the socket opened by the user:

```bash
ssh -S /tmp/wright-codex.sock wright
```

Check queue:

```bash
squeue -u sgallagh -o "%i %T %M %D %R"
```

Check accounting:

```bash
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,MaxRSS,ReqTRES -P
```

Tail logs:

```bash
cd /home/export/sgallagh/WeightTraits
tail -100 slurm_logs/wt-cp-tree_JOBID_TASK.out
tail -100 slurm_logs/wt-cp-tree_JOBID_TASK.err
```

Sync offline W&B later:

```bash
cd /home/export/sgallagh/WeightTraits
wandb sync wandb/offline-run-*
```

## End-of-Day Checklist

- [ ] `squeue` snapshot recorded in the handoff or this file.
- [ ] Any failed Slurm jobs have `sacct` state and log path recorded.
- [ ] New configs/run lists are synced to Wright.
- [ ] Local doc/handoff updated with job IDs.
- [ ] Do not start a new broad family unless its one-row and two-row gates passed.
