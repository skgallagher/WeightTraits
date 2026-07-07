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
