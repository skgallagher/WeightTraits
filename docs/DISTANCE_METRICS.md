# Distance Metrics

Distance metrics are deliberately pluggable. Cosine distance is a good default for weight deltas, but WeightTraits should not make cosine the only path.

## Current Registry

The lightweight metric registry lives in `weighttraits.distances.metrics`:

```python
from weighttraits.distances import available_metrics, pairwise_distance_matrix

available_metrics()
pairwise_distance_matrix(tensors, metric="cosine")
pairwise_distance_matrix(tensors, metric="cka")
```

Currently supported:

- `cosine`: `1 - dot(x, y) / (||x|| ||y||)`
- `l1`: sum absolute difference
- `l2`: Euclidean distance
- `correlation`: `1 - Pearson correlation`
- `threshold`: count of coordinates with absolute difference greater than `eps`
- `linear_cka` / `cka`: centered linear CKA distance

## CKA

For CKA, tensors are reshaped to:

```text
samples x features
```

using the first tensor axis as samples and flattening all remaining axes into features. For a 2-D weight matrix, rows are samples and columns are features.

Given centered matrices `X` and `Y`, centered linear CKA is:

```text
CKA(X, Y) = ||X^T Y||_F^2 / (||X^T X||_F ||Y^T Y||_F)
```

and the distance is:

```text
d_CKA(X, Y) = 1 - CKA(X, Y)
```

This gives us a matrix-structure-aware comparison rather than a flattened vector comparison. It is invariant to isotropic scaling and orthogonal feature rotations, which can be useful when the question is whether two weight matrices have similar geometry rather than identical coordinates.

## Streaming Implication

The future distance-cube engine should treat metrics as accumulators:

- cosine/L2/correlation can be computed from streaming dot products, sums, and norms;
- CKA can be computed from streaming Gram/cross-covariance statistics for each tensor group;
- L1 and threshold need chunked absolute-difference passes and should be optional because they are heavier.

The invariant stays the same:

```text
Never load full checkpoints.
Never instantiate model classes.
Only stream tensor groups or chunks across models.
```

