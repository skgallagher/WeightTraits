# Tree Generator Math

This document defines the topology generators used by WeightTraits. The common object is a rooted training-lineage tree

```text
T = (V, E, r)
```

where `r = root` is a virtual untrained base model and every non-root node is a potential trained model. For a node `v`, let:

- `pa(v)` be its parent;
- `ch(v)` be its children;
- `deg+(v) = |ch(v)|` be its out-degree;
- `depth(v)` be the number of edges from `root` to `v`;
- `L(T) = {v in V \ {r}: deg+(v) = 0}` be the trained leaves.

A polytomy is any node with `deg+(v) > 2`. We intentionally allow polytomies because binary-only reference trees are too narrow for the experimental questions.

## Acceptance Constraints

Some generators are stochastic. They may be wrapped in constraints:

```text
max_depth(T) >= d_min
|L(T)| >= l_min
```

If a sampled tree fails `min_depth` or `min_leaves`, WeightTraits retries with seed `seed + attempt` until the constraints pass or `max_attempts` is exhausted. This keeps the RF/recovery tests from being quietly weakened by too-shallow or too-small topologies.

## `fixed`

The fixed generator is a direct parser from a user-authored rooted tree. If the input tree has node set `V_cfg` and child lists `children(v)`, then:

```text
V = V_cfg
E = {(v, u): u in children(v)}
```

There is no randomness. This is the right choice for hand-authored sanity checks, paper diagrams, and regression tests for known bugs.

## `chain`

The chain generator with `n_nodes = n` creates:

```text
root -> n0 -> n1 -> ... -> n(n-1)
```

So:

```text
|V \ {root}| = n
|L(T)| = 1
max_depth(T) = n
```

This is a negative/control topology for deep sequential fine-tuning. It is deliberately bad for RF resolution by itself because it has only one leaf, but it is useful for testing checkpoint inheritance and depth effects.

## `balanced`

The balanced generator is a breadth-first `k`-ary filler with `branch_factor = k`.

With `n_nodes = N`, it expands nodes in BFS order and adds up to `k` children until `N` non-root nodes exist.

With `n_leaves = L`, it expands BFS leaves until the leaf count reaches or exceeds `L`. Expanding one leaf into `k` children changes the leaf count by:

```text
Delta |L(T)| = k - 1
```

This generator is useful as a clean, regular control. It does not specifically mimic the old ELLMTrees default; use `ellmtrees_balanced` for that.

## `ellmtrees_balanced`

This reproduces the old ELLMTrees paper-style default: a depth-2 tree with leaves distributed fairly evenly across top-level branches.

For requested leaves `L`, define:

```text
B = max(2, ceil(L / 3))
q = floor(L / B)
m = L mod B
```

The root has `B` trained children. Branch `i` has:

```text
q + 1  leaves, if i < m
q      leaves, otherwise
```

Therefore:

```text
|V \ {root}| = B + L
|L(T)| = L
max_depth(T) = 2
```

This baseline is retained for comparability, not because it is the preferred topology family.

## `poisson_branching`

The Poisson branching generator grows a tree breadth-first with a training-node budget.

Parameters:

- `n_nodes = N`: maximum number of non-root nodes;
- `max_depth = D`: maximum trained depth;
- `branch_lambda`: either a scalar `lambda` or a depth-indexed list `(lambda_1, lambda_2, ...)`;
- optional `max_children = C`.

For each dequeued parent `v` at depth `d - 1`, draw:

```text
X_v ~ Poisson(lambda_d)
```

For the virtual root only, `X_root` is forced to be at least 1:

```text
X_root <- max(1, X_root)
```

Then apply caps:

```text
children(v) = min(X_v, C, remaining_budget)
```

Growth stops when the BFS queue is empty, the node budget is exhausted, or children would exceed `max_depth`.

This produces irregular, sometimes sparse trees. Use `min_depth` and `min_leaves` when the experiment needs enough topology to support RF comparisons.

## `pruned_binary_backbone`

This generator is designed for the harder setting: start with a large easy-to-describe binary tree, then prune and contract it into a deeper, irregular, polytomous topology.

Step 1: build a complete binary backbone of depth `h`.

If `backbone_depth` is omitted and `backbone_leaves = L0`, then:

```text
h = ceil(log2 L0)
```

Before pruning:

```text
|L(T0)| = 2^h
|V(T0) \ {root}| = 2^(h+1) - 2
```

Step 2: choose internal nodes to prune.

If `prune_lambda` is set, draw:

```text
K ~ Poisson(prune_lambda)
```

and sample up to `K` internal nodes. If `prune_probability = p` is set, each internal node is also selected independently with probability `p`.

Step 3: optional soft target leaves.

If `target_leaves = L*`, additional shallow internal nodes are selected for pruning until:

```text
|L(T)| <= L*
```

This is a soft upper bound, not an exact leaf-count constraint. Use `min_leaves` to enforce the lower bound that matters for RF resolution.

Step 4: prune.

For every selected internal node `v`, delete its descendants:

```text
ch(v) <- empty
```

so `v` becomes an early leaf.

Step 5: contract edges to create polytomies.

For each internal child edge `(v, u)`, independently with probability `c = contract_probability`, splice `u` out:

```text
ch(v) <- (ch(v) \ {u}) union ch(u)
```

This contraction step is how the generator creates explicit non-binary branching.

## Scoring Implication

The RF and clade-recovery analyses should record both the generated topology family and the observed statistics:

```text
|V|, |L|, max_depth, max_out_degree, n_polytomies
```

Those values are not metadata decoration; they determine how difficult the recovery problem is.

