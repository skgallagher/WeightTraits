# Recovery Scoring

WeightTraits reports false negatives explicitly. This matters because neighbor joining usually returns a binary tree, while the truth tree may contain polytomies. Resolving a true polytomy can add splits that were not present in the truth.

## Split Representation

For a leaf set `L`, every internal clade `A` induces a bipartition:

```text
A | L \ A
```

For matching, WeightTraits canonicalizes this split by taking the smaller side; ties are broken lexicographically. This makes scoring invariant to arbitrary Newick rooting.

Singleton and all-leaf sides are ignored.

## Metrics

Let:

```text
T = set of canonical truth splits
E = set of canonical estimated splits
```

Then:

```text
TP = |T intersect E|
FN = |T \ E|
FP = |E \ T|
RF = FN + FP
normalized_RF = RF / (|T| + |E|)
clade_recovery = TP / |T|
false_negative_rate = FN / |T|
split_precision = TP / |E|
false_discovery_rate = FP / |E|
exact_tree_recovery = (T == E and leaf sets match exactly)
```

If `|T| = 0`, clade recovery is defined as `1.0` and the false-negative rate as `0.0`. Extra estimated splits still make exact recovery false.

## Why FP And FN Both Matter

For a polytomy:

```text
root
├── {a,b,c}
├── {d,e}
└── f
```

An NJ tree might resolve `{a,b,c}` as `((a,b),c)`. It recovers the true `{a,b,c}` clade, so `FN = 0`, but it also invents `{a,b}`, so `FP = 1`.

A single RF number hides this asymmetry. WeightTraits reports both.

## Standard Errors

For every numeric run-level metric, aggregation reports:

```text
mean = average(x_i)
SE = sample_sd(x_i) / sqrt(n)
```

For pooled split-level proportions, aggregation also reports binomial SE:

```text
SE_pooled = sqrt(p * (1 - p) / N)
```

where `N` is the pooled denominator, such as total truth splits for clade recovery or total estimated splits for split precision.

Exact tree recovery is a binary outcome, so its rate uses the binomial SE:

```text
SE_exact = sqrt(p * (1 - p) / n_runs)
```

## Run Example

```bash
PYTHONPATH=src python -m weighttraits.cli topology-audit \
  --manifest examples/recovery/truth_polytomy_manifest.jsonl

PYTHONPATH=src python -m weighttraits.cli score-tree \
  --truth-manifest examples/recovery/truth_polytomy_manifest.jsonl \
  --estimate-newick examples/recovery/nj_resolves_polytomy.newick

PYTHONPATH=src python -m weighttraits.cli score-tree \
  --truth-manifest examples/recovery/truth_polytomy_manifest.jsonl \
  --estimate-newick examples/recovery/missing_clade.newick
```

