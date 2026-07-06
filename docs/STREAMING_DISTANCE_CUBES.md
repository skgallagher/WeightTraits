# Streaming Distance Cubes

WeightTraits distance cubes are designed around one invariant:

```text
Training should be the bottleneck. Distance computation should not load full checkpoints or instantiate model classes.
```

The current implementation is the first testable engine for that design.

## Output Schema

`wt build-distance-cube` writes a directory:

```text
distance_cube.npz
layers.json
models.json
metrics.json
audit.json
```

`distance_cube.npz` stores one array per metric:

```text
metric[layer, model_i, model_j]
```

The JSON files make the cube self-describing. `audit.json` records the representation, metrics, chunk size, epsilon, model IDs, reader type for each model, layer shapes, dtypes, and whether each metric was chunk-streamed or tensor-at-a-time.

## CLI

```bash
PYTHONPATH=src python -m weighttraits.cli build-distance-cube \
  --checkpoint n0:/path/to/node0/model.safetensors \
  --checkpoint n1:/path/to/node1/model.safetensors \
  --metric cosine \
  --metric l2 \
  --metric correlation \
  --out results/example_distance_cube
```

For many nodes, prefer a distance input manifest:

```yaml
models:
  - model_id: n0
    checkpoint: /path/to/node0/model.safetensors
  - model_id: n1
    checkpoint: /path/to/node1/model.safetensors.index.json
```

Then run:

```bash
PYTHONPATH=src python -m weighttraits.cli build-distance-cube \
  --checkpoint-manifest configs/generated/distance_inputs.yaml \
  --metric cosine \
  --metric l2 \
  --out results/example_distance_cube
```

For CKA:

```bash
PYTHONPATH=src python -m weighttraits.cli build-distance-cube \
  --checkpoint n0:/path/to/node0/model.safetensors \
  --checkpoint n1:/path/to/node1/model.safetensors \
  --metric cosine \
  --metric cka \
  --out results/example_distance_cube_cka
```

For cumulative LoRA deltas from raw PEFT edge adapters:

```bash
PYTHONPATH=src python -m weighttraits.cli build-distance-cube \
  --adapter-chain parent:/path/to/root_to_parent_adapter \
  --adapter-chain child:/path/to/root_to_parent_adapter,/path/to/parent_to_child_adapter \
  --representation lora_cumulative_delta \
  --metric cosine \
  --metric l2 \
  --out results/example_lora_cumulative_cube
```

Passing a single adapter directory with `--checkpoint` compares that edge adapter's increment. Use `--adapter-chain` when the node representation should be the cumulative root-to-node delta.

## Reconstruct From A Cube

`wt reconstruct-tree` consumes a persisted distance cube and writes a Biopython neighbor-joining
Newick tree. It requires the analysis extra, which includes `biopython`. By default it averages
the selected metric across layers before reconstruction:

```bash
PYTHONPATH=src python -m weighttraits.cli reconstruct-tree \
  --cube results/example_distance_cube \
  --metric l2 \
  --out results/example_distance_cube/tree.newick \
  --audit-out results/example_distance_cube/tree.audit.json
```

To reconstruct from one layer, pass either the layer name or zero-based layer index:

```bash
PYTHONPATH=src python -m weighttraits.cli reconstruct-tree \
  --cube results/example_distance_cube \
  --metric cosine \
  --layer model.layers.7.self_attn.k_proj.weight \
  --out results/example_distance_cube/layer7_k_cosine.newick
```

The audit JSON records the cube path, metric, layer selection or layer aggregate, model IDs, and
basic distance summaries. Recovery scoring uses topology, not branch lengths.

## Tiny Whitebox Smoke

The CLI path is covered by a local synthetic smoke test:

```bash
PYTHONPATH=src python -m pytest tests/test_whitebox_smoke.py --override-ini=addopts=
```

That smoke writes four one-value safetensors checkpoints, builds an L2 distance cube, reconstructs
the expected quartet with Biopython NJ, scores it with `wt score-tree`, and aggregates the recovery
record. It is intentionally local and requires no model downloads or cluster access.

## Distance Input Manifests

`--checkpoint-manifest` accepts JSONL, JSON, or YAML. Each row needs a model identifier and exactly one input source:

```yaml
models:
  - model_id: leaf_0
    checkpoint: checkpoints/leaf_0/model.safetensors
  - model_id: leaf_1
    checkpoint: checkpoints/leaf_1/model.safetensors.index.json
  - model_id: lora_leaf
    adapter_chain:
      - adapters/root_to_parent
      - adapters/parent_to_leaf
```

Accepted model ID fields are `model_id`, `node_id`, or `id`. Accepted checkpoint fields are `checkpoint`, `checkpoint_path`, or `path`. Relative paths resolve from the manifest file's directory.

Training ledgers can also be converted into distance input manifests. With a truth manifest, the
command selects terminal leaves by default:

```bash
PYTHONPATH=src python -m weighttraits.cli make-distance-input-manifest \
  --ledger outputs/tiny_lora_branching_smoke/training_ledger.jsonl \
  --truth-manifest examples/training/tiny_branching_manifest.jsonl \
  --artifact adapter_chain \
  --out outputs/tiny_lora_branching_smoke/generated_cumulative_leaf_inputs.yaml
```

Use `--artifact model` for full fine-tuned checkpoint rows, `--artifact merged` for merged LoRA
checkpoint rows, and `--artifact adapter_chain` for cumulative LoRA adapter-chain rows. Relative
ledger artifact paths are rewritten relative to the generated manifest.

For a complete whitebox path from a training ledger, use `wt analyze-training-ledger`. This wraps:

```text
make-distance-input-manifest -> build-distance-cube -> reconstruct-tree -> score-tree -> aggregate-recovery
```

Example for cumulative LoRA adapter chains:

```bash
PYTHONPATH=src python -m weighttraits.cli analyze-training-ledger \
  --ledger outputs/tiny_lora_branching_contrast_smoke/training_ledger.jsonl \
  --truth-manifest examples/training/tiny_branching_contrast_manifest.jsonl \
  --artifact adapter_chain \
  --metric l2 \
  --metric cosine \
  --out outputs/tiny_lora_branching_contrast_smoke/cumulative_leaf_analysis
```

The output directory contains the generated distance-input manifest, `distance_cube/`, one Newick
tree, tree audit, and score JSON per metric, plus `aggregate_recovery.json` and `summary.json`.
Use `--artifact model` for full fine-tuned checkpoints and `--artifact merged` for merged LoRA
checkpoints. The default representation is `full_weight` for checkpoint artifacts and
`lora_cumulative_delta` for adapter chains.

## Readers

Current readers:

- `DictTensorReader`: synthetic tests and smoke data.
- `SafetensorsTensorReader`: uses safetensors metadata and row-slab slices for vector metrics.
- `ShardedSafetensorsTensorReader`: reads Hugging Face `model.safetensors.index.json` weight maps and opens only the shard containing the current tensor.
- `TorchTensorReader`: reads PyTorch `.bin` / `.pt` checkpoints with `torch.load(..., mmap=True)` when supported and slices tensors before NumPy conversion.
- `LoraFactorReader`: streams row blocks of `scale * B @ A` for vector metrics.
- `CumulativeLoraReader`: path-sums LoRA chunks for cumulative node displacement.

Planned readers:

- metadata-only torch checkpoint index using fake tensors/checkpoint offsets where available;
- low-rank LoRA factor accumulator that computes dot products without dense `B @ A`.

## Metric Execution Modes

### Chunk-Streamed

These metrics are computed from flattened chunks and never require full tensors:

- `cosine`
- `l2`
- `correlation`
- `l1`
- `threshold`

For each tensor key and chunk:

```text
X = N_models x chunk_size
```

The accumulator updates dot products, sums, squared sums, and optional pairwise absolute differences. Peak memory is:

```text
O(N_models * chunk_size + N_models^2)
```

For safetensors, chunks are produced from storage slices rather than full `get_tensor()` calls. For PyTorch checkpoints, the reader slices the file-backed tensor first when mmap is available. If a single tensor row is wider than `chunk_size`, peak memory can include that row width.

For sharded safetensors, the reader consults `model.safetensors.index.json` and opens the one shard containing the current tensor key.

### Tensor-At-A-Time

`cka` is currently exact but tensor-at-a-time. It reads one tensor per model for the current layer and computes centered linear CKA. This still avoids full checkpoint loading, but it is not yet chunk-streamed.

CKA should remain optional until we add a covariance/Gram streaming path.

## LoRA Representation

For LoRA, the recommended representation is:

```text
lora_cumulative_delta
```

which compares:

```text
Delta_v^cum = sum_{edge in root->v} scale_edge * B_edge A_edge
```

not the last edge's raw adapter increment. See [LoRA Distance Model](LORA_DISTANCE_MODEL.md).

## Current Guarantees

The fast tests verify:

- chunked vector metrics match dense reference distances;
- changing chunk size does not change vector metrics;
- vector metrics use `iter_flat_chunks`, not `read_tensor`;
- CKA matches dense matrix-aware reference;
- cumulative LoRA readers equal explicit path sums;
- safetensors reader roundtrips when `safetensors` is installed;
- sharded safetensors reader roundtrips through a real two-shard index;
- distance input manifests resolve relative checkpoint and adapter-chain paths;
- cube writer emits the expected files and audit metadata;
- neighbor joining recovers a known split from a persisted cube;
- a tiny local CLI smoke covers build-distance-cube -> reconstruct-tree -> score-tree -> aggregate-recovery.

## Deliberate Limitations

- CKA is exact but tensor-at-a-time.
- LoRA vector metrics stream dense row blocks of `B @ A`; low-rank dot-product acceleration is planned.
- The CLI currently accepts explicit checkpoint paths, explicit LoRA adapter chains, and simple distance input manifests. Full training-ledger discovery will come after the reader layer is stable.
