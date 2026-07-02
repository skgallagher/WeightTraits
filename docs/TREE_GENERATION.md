# Tree Generation

Tree generation is a first-class experimental variable in WeightTraits.

## Current Generators

`ellmtrees_balanced`

Recreates the old fairly balanced depth-2 topology: a polytomous root, several internal branch nodes, and leaves distributed as evenly as possible. This exists for comparability, not because it is the only shape we trust.

`poisson_branching`

Grows breadth-first with Poisson-distributed children. Supports `max_depth`, depth-varying `branch_lambda`, `max_children`, `min_depth`, and `min_leaves`.

`pruned_binary_backbone`

Builds a large binary backbone, for example 32 leaves, then prunes internal subtrees and optionally contracts edges. This creates deeper, chainier, and polytomous trees while keeping the starting process easy to reason about.

`chain`, `balanced`, `fixed`

Simple controls and hand-authored reference trees.

## Constraints

Use constraints when a topology is meant to support RF/permutation testing:

```yaml
tree:
  generator: pruned_binary_backbone
  backbone_leaves: 32
  target_leaves: 12
  min_depth: 4
  min_leaves: 8
  prune_lambda: 3.0
  contract_probability: 0.4
  seed: 7
```

If a sampled tree fails `min_depth` or `min_leaves`, the generator tries the next seed until the constraints pass or `max_attempts` is exhausted.

## Design Commitments

- Topology generation does not assign tasks.
- Polytomies are supported and tested.
- Minimum depth and minimum leaf count are explicit.
- Shallow ELLMTrees-style trees are kept as a baseline, not treated as sufficient evidence.
- Deep/pruned/chained shapes should be part of smoke tests before they become expensive cluster experiments.

