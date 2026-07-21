# Paired no-translation condition

This condition removes translation from the training-task pool while preserving the exact 50
topologies in `../tree_set_summary.json`. Unlike the earlier ELLMTrees experiment, it does not draw
a new tree set. Every comparison can therefore be paired by `tree_id` and topology.

All node assignments are regenerated from the 27 summarization, classification, and QA datasets.
Assignments use `per_node_without_replacement` and date-namespaced seeds
`2026072101..2026072150`. The model and full-fine-tuning hyperparameters match `../full_finetune.yaml`;
only the task pool and output root differ.

The committed audit covers all 50 trees with no issues. All 641 topology rows are present, 621 node
assignments differ from the original four-family draw, and 20 independently redrawn nodes happen to
retain the same family/dataset pair. The generated run-list validator reports 641/641 valid rows.

Generate the assignments:

```bash
PYTHONPATH=src python -m weighttraits.cli assign-task-data-set \
  --tree-set examples/training/confirm_paper_numbers/tree_set_summary.json \
  --config examples/training/confirm_paper_numbers/no_translation/task_families.yaml \
  --out-dir examples/training/confirm_paper_numbers/no_translation/assigned_manifests \
  --summary-out examples/training/confirm_paper_numbers/no_translation/assignment_summary.json \
  --seed-start 2026072101 \
  --policy per_node_without_replacement
```

Fail closed on the paired-design invariants:

```bash
PYTHONPATH=src python -m weighttraits.cli audit-paired-assignment-set \
  --tree-set examples/training/confirm_paper_numbers/tree_set_summary.json \
  --reference-assignment-summary examples/training/confirm_paper_numbers/assignment_summary.json \
  --candidate-assignment-summary examples/training/confirm_paper_numbers/no_translation/assignment_summary.json \
  --forbid-task-family translation \
  --out examples/training/confirm_paper_numbers/no_translation/paired_assignment_audit.json
```

Generate executable run lists using the already-built 36-dataset cache (this condition only uses a
subset of it):

```bash
PYTHONPATH=src python -m weighttraits.cli make-training-run-list-set \
  --assignment-summary examples/training/confirm_paper_numbers/no_translation/assignment_summary.json \
  --config examples/training/confirm_paper_numbers/no_translation/full_finetune.yaml \
  --out-dir examples/training/confirm_paper_numbers/no_translation/full_finetune_training_runlists \
  --summary-out examples/training/confirm_paper_numbers/no_translation/full_finetune_training_run_list_summary.json \
  --registry examples/training/confirm_paper_numbers/dataset_registry.yaml \
  --formats examples/training/confirm_paper_numbers/dataset_formats.yaml \
  --data-cache-root data/confirm_paper_numbers/full_finetune_cache \
  --require-data-cache \
  --max-train-samples 10000 \
  --max-eval-samples 1000 \
  --allow-missing-eval
```

Before the production array, run the existing one-row trainer smoke with the no-translation tree-001
run list. Then submit the bounded sequential tree array:

```bash
sbatch --array=1-50%5 \
  --export=ALL,REPO=/home/export/sgallagh/WeightTraits-no-translation,RUN_LIST_TEMPLATE=examples/training/confirm_paper_numbers/no_translation/full_finetune_training_runlists/run_lists/confirm_paper_tree_TREEID.runs.jsonl,RUN_NAME_PREFIX_TEMPLATE=no-translation-tree-TREEID,WANDB_PROJECT=weighttraits-no-translation \
  scripts/slurm/confirm_paper_tree_sequential.sbatch
```

Record the Slurm job ID before launching downstream analysis. Final inference, exact-recovery, and
behavioral comparisons should retain the original `tree_id` and report paired differences as the
primary comparison, with condition-specific pooled estimates as secondary summaries.
