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

The JSON files make the cube self-describing. `audit.json` records the representation, metrics, chunk size, epsilon, model IDs, layer shapes, dtypes, and whether each metric was chunk-streamed or tensor-at-a-time.

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

## Readers

Current readers:

- `DictTensorReader`: synthetic tests and smoke data.
- `SafetensorsTensorReader`: uses safetensors metadata and row-slab slices for vector metrics.
- `TorchTensorReader`: reads PyTorch `.bin` / `.pt` checkpoints with `torch.load(..., mmap=True)` when supported and slices tensors before NumPy conversion.
- `LoraFactorReader`: streams row blocks of `scale * B @ A` for vector metrics.
- `CumulativeLoraReader`: path-sums LoRA chunks for cumulative node displacement.

Planned readers:

- sharded safetensors index reader;
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
- cube writer emits the expected files and audit metadata.

## Deliberate Limitations

- Sharded safetensors are not implemented yet.
- CKA is exact but tensor-at-a-time.
- LoRA vector metrics stream dense row blocks of `B @ A`; low-rank dot-product acceleration is planned.
- The CLI currently accepts explicit checkpoint paths and explicit LoRA adapter chains. Manifest-driven checkpoint discovery will come after the reader layer is stable.
