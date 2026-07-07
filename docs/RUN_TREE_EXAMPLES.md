# Run Tree Examples

All tree examples live in [examples/trees](../examples/trees). Each file is a runnable config for:

```bash
PYTHONPATH=src python -m weighttraits.cli generate-tree \
  --config examples/trees/<name>.yaml \
  --out reports/examples/<name>.manifest.jsonl
```

The examples intentionally cover both controls and paper-relevant families.

## Run All Examples

```bash
scripts/run_tree_examples.sh
```

If you are using the existing local ELLMTrees conda environment:

```bash
conda run -n ellmtrees scripts/run_tree_examples.sh
```

## Individual Examples

Fixed hand-authored topology:

```bash
PYTHONPATH=src python -m weighttraits.cli generate-tree \
  --config examples/trees/fixed.yaml \
  --out reports/examples/fixed.manifest.jsonl
```

Deep chain:

```bash
PYTHONPATH=src python -m weighttraits.cli generate-tree \
  --config examples/trees/chain.yaml \
  --out reports/examples/chain.manifest.jsonl
```

Regular balanced control:

```bash
PYTHONPATH=src python -m weighttraits.cli generate-tree \
  --config examples/trees/balanced.yaml \
  --out reports/examples/balanced.manifest.jsonl
```

ELLMTrees-compatible balanced baseline:

```bash
PYTHONPATH=src python -m weighttraits.cli generate-tree \
  --config examples/trees/ellmtrees_balanced.yaml \
  --out reports/examples/ellmtrees_balanced.manifest.jsonl
```

Poisson branching:

```bash
PYTHONPATH=src python -m weighttraits.cli generate-tree \
  --config examples/trees/poisson_branching.yaml \
  --out reports/examples/poisson_branching.manifest.jsonl
```

Confirm-paper-number topology set:

```bash
PYTHONPATH=src python -m weighttraits.cli generate-tree-set \
  --config examples/trees/confirm_paper_numbers.yaml \
  --out-dir examples/training/confirm_paper_numbers/trees \
  --summary-out examples/training/confirm_paper_numbers/tree_set_summary.json
```

This draws from the active draft topology distribution without reusing the exact old tree seeds:
Poisson branching with `lambda=1.5`, `n_nodes=14`, `max_depth=4`, and rejection of trees with
fewer than four leaves. The summary records the accepted seeds and leaf/depth counts.

Pruned binary backbone:

```bash
PYTHONPATH=src python -m weighttraits.cli generate-tree \
  --config examples/trees/pruned_binary_backbone.yaml \
  --out reports/examples/pruned_binary_backbone.manifest.jsonl
```

## Example Verification

The test suite loads every YAML file in `examples/trees` and checks the declared `expect` block:

```bash
PYTHONPATH=src python -m pytest tests/test_tree_examples.py -q
```
