# Weekend Handoff: Confirm-Paper Wright Runs

Created: 2026-07-08

This is the short resume note for letting the confirm-paper experiments cook while
Shannon is away Thursday-Sunday.

## Current State

Wright checkout:

```text
/home/export/sgallagh/WeightTraits
```

Wright Python:

```text
/home/export/sgallagh/.conda/envs/weighttraits/bin/python
```

Active training jobs:

- Full fine-tune: `153745_1` plus `153752_[2-50%5]`
- LoRA q/v: `153785_[1-50%2]`
- LoRA k-only: `153909_[1-50%4]`
- LoRA qkv: `153910_[1-50%4]`
- LoRA full attention: `153911_[1-50%4]`
- LoRA full-FT approximation: `153912_[1-50%4]`

Additional weekend arrays launched 2026-07-09:

- All four use the same 50 confirm-paper trees, 641 node rows, finite dataset cache, 2,000-step
  trainer settings, and offline W&B project as the active q/v condition.
- Each is capped at four concurrent trees, so together they expose at most sixteen eligible tasks.
  Wright currently reports 20 L40s total (`n01=8`, `n02=8`, `n03=4`), not 24. The arrays were
  submitted at `%1`, raised to `%2` after checking live allocation, and then raised to `%4` so
  Slurm can backfill them as resources become available. Eligible tasks still wait on
  `Resources`/`Priority`; node capacity cannot be oversubscribed by the throttle.
- Isolated two-row dependency smokes `153905`--`153908` all completed `0:0` before launch and
  verified the actual PEFT target modules plus merged-parent loading.
- Monitor all training arrays with:

```bash
ssh -S /tmp/wright-codex.sock wright \
  'squeue -j 153752,153785,153909,153910,153911,153912 -o "%.18i %.28j %.8T %.10M %R"'
```

Last check:

- 2026-07-09 full fine-tune: 43/50 trees ready for analysis, 7 still active/pending, 0 failed.
- 2026-07-09 LoRA q/v: 14/50 trees ready, 3 in progress, 33 not started, 0 failed.
- Direct analysis jobs `153902` (full), `153903` (merged q/v), and `153904` (cumulative q/v)
  completed `0:0`; paper-aligned cosine PAER/clade recovery are `90.7%`/`96.3%` for full,
  `85.7%`/`94.0%` for merged q/v, and `71.4%`/`89.9%` for cumulative q/v.
- The four new target-scope arrays started tree 001 cleanly immediately after submission.

Rough ETA:

- Full fine-tune 50-tree set: roughly 24-36 more hours from the 2026-07-08 afternoon check, with tail risk toward ~45h.
- LoRA has only 2 concurrent tree slots, so expect it to stretch longer into the weekend.

## What Was Built

Direct run-set analysis now exists and does not depend on the old `analyze-training-ledger`
wrapper as the analysis engine.

Key files:

- `src/weighttraits/analysis/direct.py`: direct ledger/artifact analysis.
- `src/weighttraits/analysis/whitebox.py`: run-set orchestration, now calls the direct engine.
- `src/weighttraits/analysis/runset_results.py`: run-set analysis rollups.
- `src/weighttraits/cli.py`: CLI entries.
- `scripts/slurm/confirm_paper_direct_analysis.sbatch`: CPU-side analysis/rollup Slurm wrapper.
- `docs/EXPERIMENT_CHECKLIST.md`: canonical commands.

Important behavior:

- `analyze-training-run-set` uses direct streaming distance computation, Biopython NJ, and DendroPy RF.
- `--skip-existing` is implemented and tested.
- The Slurm analysis wrapper defaults to `SKIP_EXISTING=1`, so repeated analysis submissions are idempotent.
- Missing `training_log` is treated as optional in analysis readiness checks, because some jobs were already running before the executor log patch landed.

## Verification

Local:

```text
31 focused tests passed
ruff passed
bash -n scripts/slurm/confirm_paper_direct_analysis.sbatch passed
git diff --check passed
```

Wright:

```text
31 focused tests passed
bash -n scripts/slurm/confirm_paper_direct_analysis.sbatch passed
DRY_RUN=1 bash scripts/slurm/confirm_paper_direct_analysis.sbatch passed
```

Housekeeping:

- Corrected a broad `rsync` mistake and removed stray root-level copies on Wright:
  `EXPERIMENT_CHECKLIST.md`, `cli.py`, `confirm_paper_direct_analysis.sbatch`,
  `test_whitebox_analysis.py`, `whitebox.py`.
- Proper files are synced into their real subdirectories.

## First Checks On Resume

Queue:

```bash
ssh -S /tmp/wright-codex.sock wright \
  'squeue -u sgallagh -o "%.18i %.9P %.40j %.8T %.10M %.9l %.6D %R" | head -100'
```

Accounting:

```bash
ssh -S /tmp/wright-codex.sock wright \
  'sacct -j 153745,153752,153785 --format=JobID,State,ExitCode,Elapsed -P | tail -160'
```

Refresh completion dashboards:

```bash
ssh -S /tmp/wright-codex.sock wright \
  'cd /home/export/sgallagh/WeightTraits &&
   PYTHONPATH=src /home/export/sgallagh/.conda/envs/weighttraits/bin/python -m weighttraits.cli audit-training-run-set \
     --summary examples/training/confirm_paper_numbers/full_finetune_training_run_list_summary.json \
     --path-base . \
     --out outputs/confirm_paper_numbers/full_finetune/run_set_completion_audit.json \
     --csv-out outputs/confirm_paper_numbers/full_finetune/run_set_completion_audit.csv \
     --optional-artifact training_log \
     --allow-issues'
```

```bash
ssh -S /tmp/wright-codex.sock wright \
  'cd /home/export/sgallagh/WeightTraits &&
   PYTHONPATH=src /home/export/sgallagh/.conda/envs/weighttraits/bin/python -m weighttraits.cli audit-training-run-set \
     --summary examples/training/confirm_paper_numbers/lora_finetune_training_run_list_summary.json \
     --path-base . \
     --out outputs/confirm_paper_numbers/lora_finetune/run_set_completion_audit.json \
     --csv-out outputs/confirm_paper_numbers/lora_finetune/run_set_completion_audit.csv \
     --optional-artifact training_log \
     --allow-issues'
```

Quick compact dashboard read:

```bash
ssh -S /tmp/wright-codex.sock wright \
  'cd /home/export/sgallagh/WeightTraits &&
   /home/export/sgallagh/.conda/envs/weighttraits/bin/python - <<'"'"'PY'"'"'
import json
for label, path in [
    ("full", "outputs/confirm_paper_numbers/full_finetune/run_set_completion_audit.json"),
    ("lora", "outputs/confirm_paper_numbers/lora_finetune/run_set_completion_audit.json"),
]:
    row = json.load(open(path))
    print(label, {k: row[k] for k in ("n_ready", "n_in_progress", "n_failed", "n_not_started")})
    print("ready", row.get("ready_tree_ids", [])[:10])
PY'
```

## When Trees Are Ready

Use the Slurm wrapper instead of doing heavy direct analysis in the login shell.

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

The wrapper runs analysis and then rollup. Re-running it is safe because `SKIP_EXISTING=1`.

## Outputs To Inspect

Full:

- `outputs/confirm_paper_numbers/full_finetune/direct_run_set_analysis_plan.json`
- `outputs/confirm_paper_numbers/full_finetune/direct_run_set_analysis_summary.json`
- `outputs/confirm_paper_numbers/full_finetune/direct_run_set_analysis_rows.csv`

LoRA:

- `outputs/confirm_paper_numbers/lora_finetune/direct_merged_run_set_analysis_plan.json`
- `outputs/confirm_paper_numbers/lora_finetune/direct_merged_run_set_analysis_summary.json`
- `outputs/confirm_paper_numbers/lora_finetune/direct_merged_run_set_analysis_rows.csv`
- `outputs/confirm_paper_numbers/lora_finetune/direct_cumulative_run_set_analysis_plan.json`
- `outputs/confirm_paper_numbers/lora_finetune/direct_cumulative_run_set_analysis_summary.json`
- `outputs/confirm_paper_numbers/lora_finetune/direct_cumulative_run_set_analysis_rows.csv`

Per-tree direct analysis lands under:

- `outputs/confirm_paper_numbers/full_finetune/run_set_analysis/TREEID/model_leaf_analysis/`
- `outputs/confirm_paper_numbers/lora_finetune/run_set_analysis/TREEID/merged_leaf_analysis/`
- `outputs/confirm_paper_numbers/lora_finetune/run_set_analysis/TREEID/cumulative_leaf_analysis/`

## What Not To Do

- Do not cancel the training arrays unless there are real failed jobs.
- Do not run full direct analysis on the login shell.
- Do not revert unrelated local changes; the worktree is intentionally dirty from the larger cache/run-list work.
- Do not use broad `rsync` to the Wright repo root. Sync path-specific files only.

## If Something Fails

First inspect Slurm logs:

```bash
ssh -S /tmp/wright-codex.sock wright \
  'cd /home/export/sgallagh/WeightTraits && tail -100 slurm_logs/wt-cp-tree_JOBID_TASK.err'
```

Then inspect the relevant tree ledger:

```bash
ssh -S /tmp/wright-codex.sock wright \
  'cd /home/export/sgallagh/WeightTraits &&
   tail -40 examples/training/confirm_paper_numbers/full_finetune_training_runlists/ledgers/confirm_paper_tree_001.training_ledger.jsonl'
```

For LoRA, swap `full_finetune_training_runlists` for `lora_finetune_training_runlists`.
