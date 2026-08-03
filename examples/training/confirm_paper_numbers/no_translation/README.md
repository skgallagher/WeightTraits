# Matched translation-held-out condition

This document records the Llama-3.2-1B full-fine-tuning experiment that removes translation from
the original production assignments while holding the tree sample and every unaffected node fixed.
It is the authoritative design for the translation-held-out comparison.

The production artifacts currently live in the frozen Wright checkout
`/home/export/sgallagh/WeightTraits-llama32-20260713`, under
`examples/training/translation_holdout_20260803/`. They should be copied into a reviewable branch
after the training and analysis audits finish. Until then, the Wright paths and hashes in
`matched_assignment_audit.json` are the experiment record.

## Scientific question

The original Llama full-FT condition samples training tasks from summarization, classification,
question answering, and translation. The holdout asks whether the weight-based lineage and
held-out translation-behavior results persist when translation is never used for training.

This requires new model training. Re-running only translation prompts against the original models
would measure another endpoint on models that had already seen translation tasks; it would not be
a translation-held-out training condition.

## Matched intervention

The comparison uses the exact 50 accepted topologies in `../tree_set_summary.json` and the exact
original node order. For each original assigned manifest:

1. Copy every non-translation row byte-for-byte at the semantic field level.
2. Replace only rows whose original `task_family` is `translation`.
3. Draw replacements from the summarization, classification, and QA datasets that are not already
   used elsewhere in that tree.
4. Sample without replacement with `numpy.default_rng(original_assignment_seed)`, preserving the
   original per-tree assignment seed.
5. Add `matched_holdout_original_task_family` and
   `matched_holdout_original_dataset_id` to each replaced row so the intervention is explicit.

The trainer, base model, pinned base revision, full-FT method, stopping rule, data cache, topology,
node identifiers, parent identifiers, depths, and output-artifact type are unchanged. Only the 162
translation-assigned node datasets are replaced.

The pre-launch audit reports:

| Invariant | Audited value |
| --- | ---: |
| Trees | 50 |
| Training rows | 641 |
| Preserved non-translation rows | 479 |
| Replaced translation rows | 162 |
| Replacement summarization rows | 49 |
| Replacement classification rows | 53 |
| Replacement QA rows | 60 |
| Translation rows remaining | 0 |
| Duplicate datasets within a tree | 0 |
| Run-list changes outside translation rows | 0 |
| Pre-existing matched output directories | 0 |
| Valid run-list rows | 641 / 641 |

This is a matched intervention, not merely a same-topology experiment. The primary comparison may
therefore be paired by `tree_id`, node position, and topology. The 479 unchanged assignments also
hold the local training task constant.

## Authoritative Wright paths

All paths below are relative to
`/home/export/sgallagh/WeightTraits-llama32-20260713` unless shown as absolute.

- Original assignment summary:
  `examples/training/confirm_paper_numbers/assignment_summary.json`
- Original Llama full-FT run lists:
  `examples/training/confirm_paper_numbers/llama32_1b_full_finetune_training_runlists/`
- Matched experiment root: `examples/training/translation_holdout_20260803/`
- Matched manifests: `examples/training/translation_holdout_20260803/matched_assigned_manifests/`
- Matched assignment summary:
  `examples/training/translation_holdout_20260803/matched_assignment_summary.json`
- Assignment audit:
  `examples/training/translation_holdout_20260803/matched_assignment_audit.json`
- Matched run lists: `examples/training/translation_holdout_20260803/matched_training_runlists/`
- Run-list summary:
  `examples/training/translation_holdout_20260803/matched_training_run_list_summary.json`
- Data-format audit:
  `examples/training/translation_holdout_20260803/matched_data_format_validation.json`
- Output root: `outputs/translation_holdout_20260803/llama32_1b_full_finetune/`
- Manifest generator:
  `examples/training/translation_holdout_20260803/make_matched_translation_holdout.py`
- Run-list audit:
  `examples/training/translation_holdout_20260803/audit_matched_translation_runlists.py`

The sibling paths `assigned_manifests/`, `assignment_summary.json`, `training_runlists/`, and
`training_run_list_summary.json` under `translation_holdout_20260803/` belong to an earlier fully
resampled draft. They were not submitted and must not be used for the matched comparison. Only
paths prefixed with `matched_` are authoritative.

## Regenerating the matched manifests

From the production checkout:

```bash
PYTHON_BIN=/home/export/sgallagh/.conda/envs/weighttraits/bin/python

"${PYTHON_BIN}" examples/training/translation_holdout_20260803/make_matched_translation_holdout.py \
  --original-summary examples/training/confirm_paper_numbers/assignment_summary.json \
  --task-config examples/training/confirm_paper_numbers/paper_task_families.yaml \
  --out-dir examples/training/translation_holdout_20260803/matched_assigned_manifests \
  --summary-out examples/training/translation_holdout_20260803/matched_assignment_summary.json \
  --audit-out examples/training/translation_holdout_20260803/matched_assignment_audit.json
```

Generate the run lists with the production Llama full-FT config and the existing finite cache:

```bash
PYTHONPATH=src "${PYTHON_BIN}" -m weighttraits.cli make-training-run-list-set \
  --assignment-summary examples/training/translation_holdout_20260803/matched_assignment_summary.json \
  --config examples/training/translation_holdout_20260803/llama32_1b_full_finetune.yaml \
  --out-dir examples/training/translation_holdout_20260803/matched_training_runlists \
  --summary-out examples/training/translation_holdout_20260803/matched_training_run_list_summary.json \
  --registry examples/training/confirm_paper_numbers/dataset_registry.yaml \
  --formats examples/training/confirm_paper_numbers/dataset_formats.yaml \
  --data-cache-root data/confirm_paper_numbers/full_finetune_cache \
  --require-data-cache \
  --max-train-samples 10000 \
  --max-eval-samples 1000 \
  --allow-missing-eval
```

Then fail closed on the run-list comparison:

```bash
"${PYTHON_BIN}" examples/training/translation_holdout_20260803/audit_matched_translation_runlists.py
```

Expected compact output:

```json
{
  "bad_changes": 0,
  "changed_rows": 162,
  "existing_output_dirs": 0,
  "preserved_rows": 479,
  "trees": 50
}
```

The `existing_output_dirs` assertion is a pre-launch guard. After production begins, audit into a
fresh output root or disable only that one check while retaining the row-by-row comparison.

## Production launch

Slurm array `162327` was submitted on 2026-08-03 at 11:31 EDT as
`wt-llama-holdout`, with array range `1-50%8`. Each task requests one NVIDIA L40, four CPUs, 64 GB
RAM, and a two-day limit. It uses `scripts/slurm/confirm_paper_tree_sequential.sbatch`, sets
`PRUNE_INTERNAL_PARENTS=true`, and disables external reporting with `REPORT_TO=none`.

The submitted run-list template is:

```text
examples/training/translation_holdout_20260803/matched_training_runlists/run_lists/confirm_paper_tree_TREEID.runs.jsonl
```

The first dated status check, 2026-08-03 at 11:42 EDT, found array tasks 1 and 2 running on L40s and
tasks 3-50 pending for resources under the eight-task throttle. Scheduler `PENDING (Resources)` is
not a code failure.

Monitor the queue and accounting records separately:

```bash
squeue -j 162327 -o "%.18i %.8T %.10M %R"
sacct -j 162327 -X -P \
  --format=JobID,JobIDRaw,State,ExitCode,Elapsed,Start,End
```

Inspect per-task logs under `slurm_logs/wt-llama-holdout_162327_TASK.{out,err}`. Retry a task only
after identifying whether the event is a scheduler/resource event or a reproducible code/data
failure, and only after determining the exact affected tree set.

## Training completion audit

Do not begin downstream production analysis until all 50 trees pass all of these gates:

- Every array task is terminal with exit code `0:0`.
- The latest ledger state for all 641 nodes is a successful terminal state. Both `completed` and
  intentional `stopped_early` are successful.
- The retained model directories match the 365 manifest leaves exactly.
- An internal model is pruned only after all of its direct children succeed.
- No partial Trainer checkpoint directories remain.
- Each child ledger row records the intended local parent artifact rather than the remote base
  model.

Use the existing `audit-training-run-set` and `audit-training-tree` commands against the matched
run-list summary. Keep the resulting compact JSON audit with the experiment.

## Downstream analysis

After the training audit passes:

1. Run direct full-weight white-box analysis across all 50 trees with
   `MANIFEST_ARTIFACT=model`, `MODEL_TASK=causal_lm`, and
   `WEIGHT_REPRESENTATION=full_weight`.
2. Require exact manifest leaf/model alignment, 146 aligned layers, finite symmetric
   zero-diagonal L2/cosine/correlation cubes and matrices, and the expected per-tree pair counts.
3. Report polytomy-aware exact recovery and clade recovery. Pair each tree with the original
   full-FT result by `tree_id`.
4. Run the bounded translation behavior endpoint with three draws per leaf and the validated
   causal-LM settings, using the matched full-FT ledgers and matched truth manifests.
5. Require complete response grids, finite aligned white-box/surface/semantic cubes, and exactly
   `choose(L, 2)` pair rows per endpoint and probe for a tree with `L` leaves.

Immediate EOS is an observed response, not missing data, in the primary semantic analysis. Encode
it as an explicit semantic outcome, retain it in surface analysis, and report the empty-response
rate and its relationship with tree depth. The empty-excluding semantic analysis remains a labeled
sensitivity analysis with honest coverage and valid masks.

Run the paper-faithful R statistics separately for the EOS-inclusive primary analysis and the
empty-excluding sensitivity analysis. When averaging correlations across layers or runs, use
`z = arctanh(r)`, average on the z scale, and back-transform with `r = tanh(mean(z))`. Keep
probe/endpoint estimates separate and pull only compact paper-facing summaries locally; do not
commit models, raw generations, distance cubes, or per-run operational output.

## Legacy resampled scaffold

The committed files elsewhere in this directory predate the 2026-08-03 matched design. They keep
the same 50 topologies but regenerate all 641 assignments using date-namespaced seeds
`2026072101..2026072150`; 621 assignments differ from the original condition. That design can be
useful as a broader task-pool perturbation, but it is not the matched translation holdout and must
not supply the paper's primary no-translation comparison.
