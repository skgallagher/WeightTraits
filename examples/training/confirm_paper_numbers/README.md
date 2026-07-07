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
