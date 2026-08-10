# Matched Translation-Holdout Cohort

This directory derives a no-translation training cohort from the ordinary fresh
50-tree assignment set. It keeps each topology, node order, and every existing
non-translation assignment fixed, replacing only nodes originally assigned a
translation dataset. The cohort is analyzed independently; “matched” describes
the construction, not a paired inferential estimator.

## Derive the assignments

Run from the WeightTraits repository root:

```bash
python examples/training/translation_holdout_20260803/make_matched_translation_holdout.py \
  --original-summary examples/training/confirm_paper_numbers/assignment_summary.json \
  --task-config examples/training/confirm_paper_numbers/paper_task_families.yaml \
  --out-dir examples/training/translation_holdout_20260803/matched_assigned_manifests \
  --summary-out examples/training/translation_holdout_20260803/matched_assignment_summary.json \
  --audit-out examples/training/translation_holdout_20260803/matched_assignment_audit.json
```

For each tree, the script seeds `numpy.default_rng` with the original assignment
seed and chooses unused classification, QA, or summarization dataset pairs. It
rejects duplicate datasets, any remaining translation row, or any change to a
non-translation row. The checked-in audit covers 50 trees and 641 rows: 479 rows
are preserved at assignment level and 162 translation rows are replaced (53
classification, 60 QA, and 49 summarization).

## Build and audit executable run lists

The matched config extends the ordinary legacy Llama full-FT config and changes
only cohort identity/output root. Regenerate the run lists with the shared frozen
cache:

```bash
PYTHONPATH=src python -m weighttraits.cli make-training-run-list-set \
  --assignment-summary examples/training/translation_holdout_20260803/matched_assignment_summary.json \
  --config examples/training/translation_holdout_20260803/llama32_1b_full_finetune_legacy_causal_2000.yaml \
  --out-dir examples/training/translation_holdout_20260803/matched_legacy_causal_2000_training_runlists \
  --summary-out examples/training/translation_holdout_20260803/matched_legacy_causal_2000_training_run_list_summary.json \
  --registry examples/training/confirm_paper_numbers/dataset_registry_legacy_causal.yaml \
  --formats examples/training/confirm_paper_numbers/dataset_formats_legacy_causal.yaml \
  --data-cache-root data/confirm_paper_numbers/legacy_causal_seed42_cache \
  --require-data-cache \
  --expected-cache-strategy legacy_subsample \
  --expected-cache-seed 42 \
  --expected-cache-train-limit 10000 \
  --expected-cache-eval-limit 1000 \
  --max-train-samples 10000 \
  --max-eval-samples 1000

python examples/training/translation_holdout_20260803/audit_matched_translation_runlists.py \
  --ordinary-runlists examples/training/confirm_paper_numbers/llama32_1b_full_finetune_legacy_causal_2000_training_runlists/run_lists \
  --holdout-runlists examples/training/translation_holdout_20260803/matched_legacy_causal_2000_training_runlists/run_lists \
  --out examples/training/translation_holdout_20260803/matched_legacy_causal_2000_runlist_audit.json
```

The run-list audit must report `valid: true`, 50 trees, 641 rows, 479 preserved
rows, 162 replacements, zero remaining translation rows, and zero issues. It also
checks topology, training protocol, runner/cache contract, prompt preservation,
dataset uniqueness, and config fingerprints. Corrected outputs belong only under
`outputs/translation_holdout_20260803/llama32_1b_full_finetune_legacy_causal_2000/`.

See
[../confirm_paper_numbers/LEGACY_CAUSAL_REBUILD.md](../confirm_paper_numbers/LEGACY_CAUSAL_REBUILD.md)
for the shared training, smoke, and production-launch contract.
