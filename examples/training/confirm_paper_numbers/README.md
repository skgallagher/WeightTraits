# Confirm Paper Numbers Tree Set

This directory contains a clean topology draw for confirming paper-scale numbers without reusing the exact old ELLMTrees tree seeds.

Generated with:

```bash
PYTHONPATH=src python -m weighttraits.cli generate-tree-set \
  --config examples/trees/confirm_paper_numbers.yaml \
  --out-dir examples/training/confirm_paper_numbers/trees \
  --summary-out examples/training/confirm_paper_numbers/tree_set_summary.json
```

The topology setting follows the active draft: Poisson branching with `lambda=1.5`, `n_nodes=14`, `max_depth=4`, and rejection of trees with fewer than four leaves. The current accepted draw has 50 trees, accepted from 78 candidate seeds starting at `20260707`.

Compare the generated set against the active paper text and the local ELLMTrees reference artifacts with:

```bash
python scripts/compare_confirm_paper_trees.py \
  --weighttraits-summary examples/training/confirm_paper_numbers/tree_set_summary.json \
  --ellmtrees-runs ../ELLMTrees/outputs/runs_branching_v3 \
  --paper-tex ../ELLMTrees-paper/iclr_draft_v2.tex \
  --out-dir reports/confirm_paper_tree_comparison
```

The comparison writes CSV tables, a Markdown summary, and SVG plots under `reports/confirm_paper_tree_comparison/`. Those outputs are intentionally ignored by git; rerun the command after regenerating topologies or after the paper tree-generation text changes.

Task/data rows were then assigned with the paper-declared 36-dataset pool: 9 summarization, 10 classification, 8 QA, and 9 translation datasets. Assignments are sampled without replacement within each tree, using assignment seeds `1..50`.

```bash
PYTHONPATH=src python -m weighttraits.cli assign-task-data-set \
  --tree-set examples/training/confirm_paper_numbers/tree_set_summary.json \
  --config examples/training/confirm_paper_numbers/paper_task_families.yaml \
  --out-dir examples/training/confirm_paper_numbers/assigned_manifests \
  --summary-out examples/training/confirm_paper_numbers/assignment_summary.json \
  --seed-start 1 \
  --policy per_node_without_replacement
```

The enriched manifests live in `assigned_manifests/`, and `assignment_summary.json` records the per-tree seeds, source manifests, output manifests, and task-family counts.

Paper-style full fine-tuning run lists are generated per tree so each tree gets an isolated ledger and output root:

```bash
PYTHONPATH=src python -m weighttraits.cli make-training-run-list-set \
  --assignment-summary examples/training/confirm_paper_numbers/assignment_summary.json \
  --config examples/training/confirm_paper_numbers/full_finetune.yaml \
  --out-dir examples/training/confirm_paper_numbers/full_finetune_runlists \
  --summary-out examples/training/confirm_paper_numbers/full_finetune_run_list_summary.json \
  --runner-dry-run
```

The current scaffold has 50 valid run lists and 641 planned training rows. `--runner-dry-run` keeps the generated rows executable for row-selection smoke checks without downloading datasets or loading models.

The 36-dataset registry and format contracts live in `dataset_registry.yaml` and `dataset_formats.yaml`. They have been checked offline with:

```bash
PYTHONPATH=src python -m weighttraits.cli validate-training-data-set \
  --assignment-summary examples/training/confirm_paper_numbers/assignment_summary.json \
  --config examples/training/confirm_paper_numbers/full_finetune.yaml \
  --formats examples/training/confirm_paper_numbers/dataset_formats.yaml \
  --out examples/training/confirm_paper_numbers/full_finetune_data_format_validation.json

PYTHONPATH=src python -m weighttraits.cli audit-datasets \
  --registry examples/training/confirm_paper_numbers/dataset_registry.yaml \
  --formats examples/training/confirm_paper_numbers/dataset_formats.yaml \
  --no-load \
  --out examples/training/confirm_paper_numbers/dataset_registry_no_load_audit.json
```

The offline validation reports 641/641 valid jobs and the no-load audit reports 36/36 dataset declarations structurally valid. Summarization datasets keep their registry entries but apply deterministic canonical-document length filters before sample caps. The configured caps are 4096 characters for `cnn_dailymail`, `xsum`, `samsum`, and `dialogsum`; 8192 for `pubmed`; 16384 for `billsum`, `arxiv_summ`, and `bigpatent`; and 65536 for `govreport`. A bounded streaming cache audit confirmed that all 9 summarization datasets can still provide 10000 accepted train examples under those caps.

Build the finite training data cache before launching real runs:

```bash
PYTHONPATH=src python -m weighttraits.cli cache-training-datasets \
  --registry examples/training/confirm_paper_numbers/dataset_registry.yaml \
  --formats examples/training/confirm_paper_numbers/dataset_formats.yaml \
  --out-dir data/confirm_paper_numbers/full_finetune_cache \
  --summary-out examples/training/confirm_paper_numbers/full_finetune_data_cache_summary.json \
  --train-limit 10000 \
  --eval-limit 1000 \
  --min-train-rows 1 \
  --max-scan 50000 \
  --streaming \
  --overwrite
```

The cache stores bounded canonical rows keyed by dataset id and split. The text cache itself lives under `data/` and is ignored by git; the JSON summary is safe to keep as the audit record. Use `--min-train-rows 1` for the full 36-dataset cache because some classification/QA datasets are naturally smaller than 10000 rows; the summarization-only threshold audit is the 10000-row check for the formerly huge document corpora.

The default selection is `--sample-strategy first`, which preserves the current first-accepted-row
behavior. To draw a deterministic random subset instead, add:

```bash
--sample-strategy seeded_shuffle --sample-seed 42 --shuffle-buffer-size 10000
```

The strategy and seed are written beside each cached split and into the summary. Use `--overwrite`
when changing sampling for an existing cache; WeightTraits rejects mismatched cache provenance.

Run the sample-loading audit with one representative planned job per dataset:

```bash
PYTHONPATH=src python -m weighttraits.cli audit-training-sample-set \
  --assignment-summary examples/training/confirm_paper_numbers/assignment_summary.json \
  --config examples/training/confirm_paper_numbers/full_finetune.yaml \
  --registry examples/training/confirm_paper_numbers/dataset_registry.yaml \
  --formats examples/training/confirm_paper_numbers/dataset_formats.yaml \
  --selection one-per-dataset \
  --streaming \
  --max-samples 2 \
  --out examples/training/confirm_paper_numbers/full_finetune_sample_render_audit.json
```

That command should report 36 selected datasets and 36/36 clean prompt-render audits before launching the full run set. `--streaming` keeps the audit from materializing large datasets just to inspect a couple of rows. Use `--selection all-jobs` only when you want the heavier 641-job prompt-render audit.

After regenerating the real training run lists, smoke one row against the local finite cache without
loading a model:

```bash
PYTHONPATH=src python -m weighttraits.cli audit-training-row-data \
  --run-list examples/training/confirm_paper_numbers/full_finetune_training_runlists/run_lists/confirm_paper_tree_001.runs.jsonl \
  --index 0 \
  --max-train-samples 1 \
  --max-eval-samples 1
```

That command uses the registry, format, and cache settings embedded in the run-list row, requires the
cache when the row requires it, and stops after rendered train/eval counts. It is the safe row-level
check between the 36-dataset sample audit and the real `run-training-row` launcher.

For the first real trainer smoke, keep the paper run-list row intact but override the runtime step
count:

```bash
PYTHONPATH=src python -m weighttraits.cli run-training-row \
  --run-list examples/training/confirm_paper_numbers/full_finetune_training_runlists/run_lists/confirm_paper_tree_001.runs.jsonl \
  --index 0 \
  --max-train-samples 1 \
  --max-eval-samples 1 \
  --override-max-steps 2
```

The override is intentionally runtime-only: it changes `trainer.max_steps` for that selected row,
records the override in the result and ledger, and leaves the generated 2,000-step run list unchanged.
For larger batches, keep the default `report_to: []` trainer setting quiet and turn on tracking at
launch time:

```bash
WANDB_PROJECT=weighttraits-confirm-paper PYTHONPATH=src python -m weighttraits.cli run-training-row \
  --run-list examples/training/confirm_paper_numbers/full_finetune_training_runlists/run_lists/confirm_paper_tree_001.runs.jsonl \
  --index 0 \
  --report-to wandb \
  --run-name confirm-paper-tree-001-n0
```

`--report-to` and `--run-name` are runtime-only trainer overrides, so they are recorded in the
result and ledger without changing the generated paper run lists. Use `--report-to none` when a
wrapper or inherited environment should be forced back to tracking-off behavior.

The actual training runner rows have also been generated separately from the dry-run scaffold:

```bash
PYTHONPATH=src python -m weighttraits.cli make-training-run-list-set \
  --assignment-summary examples/training/confirm_paper_numbers/assignment_summary.json \
  --config examples/training/confirm_paper_numbers/full_finetune.yaml \
  --out-dir examples/training/confirm_paper_numbers/full_finetune_training_runlists \
  --summary-out examples/training/confirm_paper_numbers/full_finetune_training_run_list_summary.json \
  --registry examples/training/confirm_paper_numbers/dataset_registry.yaml \
  --formats examples/training/confirm_paper_numbers/dataset_formats.yaml \
  --data-cache-root data/confirm_paper_numbers/full_finetune_cache \
  --require-data-cache \
  --max-train-samples 10000 \
  --max-eval-samples 1000 \
  --allow-missing-eval
```

That set is valid with 50 per-tree run lists and 641 planned `run-training-row` entries. Keep the dry-run set for row-selection smoke checks; use the training set only after the streaming sample audit and cache build have passed.

The corrected gated-FFN all-projection LoRA scaffold is separate from the frozen legacy
`lora_full_ft_approx` launch:

```bash
PYTHONPATH=src python -m weighttraits.cli make-training-run-list-set \
  --assignment-summary examples/training/confirm_paper_numbers/assignment_summary.json \
  --config examples/training/confirm_paper_numbers/lora_all_projections_corrected.yaml \
  --out-dir examples/training/confirm_paper_numbers/lora_all_projections_corrected_training_runlists \
  --summary-out examples/training/confirm_paper_numbers/lora_all_projections_corrected_training_run_list_summary.json \
  --registry examples/training/confirm_paper_numbers/dataset_registry.yaml \
  --formats examples/training/confirm_paper_numbers/dataset_formats.yaml \
  --data-cache-root data/confirm_paper_numbers/full_finetune_cache \
  --require-data-cache \
  --max-train-samples 10000 \
  --max-eval-samples 1000 \
  --allow-missing-eval
```

This set is valid with 50 trees and 641 rows, matches the q/v set's `(node_id, dataset_id)` order,
and explicitly targets `q,k,v,o,wi_0,wi_1,wo`. Every generated row expects
`lora_target_audit.json`. The cached tree-001 row-0 audit passes; do not submit a broad array until
one-row and two-row Wright smokes verify 216 resolved adapter modules.
