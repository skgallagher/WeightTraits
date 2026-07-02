# LoRA Distance Model

This note is deliberately explicit because LoRA lineage analysis can otherwise look like a hack. It is not the same object as a full fine-tuned checkpoint, and WeightTraits should say exactly what it is measuring.

## True Training Process

For a parent node `p` and child node `c`, training starts from the updated parent weights:

```text
W_p = W_0 + Delta_p
```

where `W_0` is the original base model and `Delta_p` is the cumulative displacement already produced along the root-to-parent path.

The child trains a fresh LoRA adapter on top of that updated parent. For one target matrix:

```text
Delta_e = s_e B_e A_e
```

where:

```text
s_e = lora_alpha_e / r_e
A_e in R^(r x d_in)
B_e in R^(d_out x r)
```

After training, the adapter is merged into the parent matrix:

```text
W_c = W_p + Delta_e
```

So along a root-to-node path:

```text
W_v = W_0 + sum_{e in path(root, v)} Delta_e
```

This is the important point: each `Delta_e` was learned in the context of the already-updated parent weights, but once learned, it is an additive matrix in the same parameter coordinate system.

## Efficient Analysis Representation

For LoRA lineage analysis, WeightTraits should use:

```text
Delta_v^cum = sum_{e in path(root, v)} s_e B_e A_e
```

as the node representation, instead of materializing:

```text
W_v = W_0 + Delta_v^cum
```

This keeps the analysis cheap:

- no full model class loading;
- no full merged checkpoint materialization;
- no need to write giant merged weights just to compare children;
- only adapter factors, target-module keys, and path structure are needed.

This is the efficient representation we want.

In the CLI, raw PEFT edge adapters should be passed with:

```text
--adapter-chain node:edge0,edge1,...
```

when the desired representation is `lora_cumulative_delta`. A single adapter directory passed as `--checkpoint` is treated as that edge's increment, not as a cumulative node state.

## What It Is Not

This is not claiming that training reused one adapter through the whole tree. It did not.

The training process is:

```text
train fresh adapter on merged parent -> merge -> child weights
```

The analysis representation is:

```text
sum the fresh per-edge low-rank deltas along the path
```

Those two views are compatible because merged LoRA updates are additive in weight space.

## Metric Implications

For LoRA groups, the default distance object should be the cumulative displacement:

```text
Delta_v^cum
```

not the raw increment:

```text
Delta_e
```

Increment-only distances compare only the last edge and throw away shared ancestry. That was the source of the old dead-zone artifact.

For difference-based metrics such as L2 on full weights:

```text
W_i - W_j = Delta_i^cum - Delta_j^cum
```

so the shared base cancels exactly.

For cosine, correlation, or CKA, the choice must be reported clearly:

- `lora_cumulative_delta`: compare cumulative displacements `Delta_v^cum`;
- `lora_full_weight_with_base`: compare `W_0 + Delta_v^cum`, which requires streaming the base matrix too.

The first is usually the better scientific object for LoRA lineage signal, because the frozen base is enormous and common to every node. The second can be implemented as a control, but should not be silently mixed with cumulative-delta results.

## Low-Rank Computation

When metrics permit it, we should avoid forming dense `B @ A`.

For two low-rank edge deltas:

```text
Delta_1 = B_1 A_1
Delta_2 = B_2 A_2
```

their Frobenius inner product can be computed as:

```text
<Delta_1, Delta_2>_F = trace(A_1 A_2^T B_2^T B_1)
```

For cumulative path sums:

```text
<Delta_i^cum, Delta_j^cum>
  = sum_{e in path_i} sum_{f in path_j} <Delta_e, Delta_f>
```

This is enough for cosine and L2-style distances over cumulative deltas. Metrics like L1 and threshold still require dense or chunked materialization, so they should be optional and labeled as heavier.

## Required Audit Fields

Every LoRA distance-cube output should record:

- `representation`: `lora_cumulative_delta` or `lora_full_weight_with_base`;
- `adapter_scope`: target modules included, such as `q`, `k`, `v`, `o`, MLP, or full attention;
- `cumulative`: `true`;
- `scale_applied`: whether `lora_alpha / r` was applied;
- `path_source`: manifest used to recover root-to-node adapter chains;
- `missing_adapter_policy`: error, zero-fill, or explicit exclusion;
- `metrics`: metrics computed from this representation.

## Recommended Default

For LoRA recovery tables:

```text
representation = lora_cumulative_delta
metrics = cosine, l2, correlation, cka where applicable
cumulative = true
scale_applied = true
```

and the methods text should say:

> Each child is trained with a fresh LoRA adapter on the merged weights of its parent. For analysis, we represent each node by the cumulative sum of the merged low-rank updates along its root-to-node path, avoiding full checkpoint materialization while preserving the additive displacement induced by the actual training process.
