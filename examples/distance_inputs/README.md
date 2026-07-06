# Distance Input Examples

These files show the manifest shape consumed by:

```bash
PYTHONPATH=src python -m weighttraits.cli build-distance-cube \
  --checkpoint-manifest examples/distance_inputs/full_weight_inputs.yaml \
  --metric cosine \
  --metric l2 \
  --out results/example_distance_cube
```

The paths are examples only. Real checkpoint paths should point to local or cluster-visible model outputs.

## Tiny Lineage Artifact Smoke

After running the tiny two-node training fixtures, these manifests point at the produced artifacts:

- `tiny_full_lineage_outputs.yaml`: full fine-tuned model directories.
- `tiny_lora_merged_lineage_outputs.yaml`: merged LoRA model directories, useful for full-weight comparisons.
- `tiny_lora_cumulative_lineage_outputs.yaml`: raw PEFT adapter chains for cumulative LoRA deltas.
- `tiny_full_branching_leaf_outputs.yaml`: four full fine-tuned leaf model directories.
- `tiny_lora_merged_branching_leaf_outputs.yaml`: four merged LoRA leaf model directories.
- `tiny_lora_cumulative_branching_leaf_outputs.yaml`: four cumulative LoRA leaf adapter chains.

On Wright, run from the repo root:

```bash
PYTHONPATH=src python -m weighttraits.cli build-distance-cube \
  --checkpoint-manifest examples/distance_inputs/tiny_full_lineage_outputs.yaml \
  --metric cosine \
  --metric l2 \
  --out outputs/tiny_full_lineage_smoke/distance_cube

PYTHONPATH=src python -m weighttraits.cli reconstruct-tree \
  --cube outputs/tiny_full_lineage_smoke/distance_cube \
  --metric l2 \
  --out outputs/tiny_full_lineage_smoke/distance_cube/tree_l2.newick \
  --audit-out outputs/tiny_full_lineage_smoke/distance_cube/tree_l2.audit.json

PYTHONPATH=src python -m weighttraits.cli score-tree \
  --truth-manifest examples/recovery/tiny_two_tip_smoke_truth_manifest.jsonl \
  --estimate-newick outputs/tiny_full_lineage_smoke/distance_cube/tree_l2.newick \
  --out outputs/tiny_full_lineage_smoke/distance_cube/score_l2.json
```

For the cumulative LoRA adapter representation:

```bash
PYTHONPATH=src python -m weighttraits.cli build-distance-cube \
  --checkpoint-manifest examples/distance_inputs/tiny_lora_cumulative_lineage_outputs.yaml \
  --representation lora_cumulative_delta \
  --metric cosine \
  --metric l2 \
  --out outputs/tiny_lora_lineage_smoke/cumulative_distance_cube

PYTHONPATH=src python -m weighttraits.cli reconstruct-tree \
  --cube outputs/tiny_lora_lineage_smoke/cumulative_distance_cube \
  --metric l2 \
  --out outputs/tiny_lora_lineage_smoke/cumulative_distance_cube/tree_l2.newick \
  --audit-out outputs/tiny_lora_lineage_smoke/cumulative_distance_cube/tree_l2.audit.json
```

The two-node lineage fixture is a training-chain smoke, not a topology-recovery benchmark. It has
only one terminal lineage leaf, so `examples/recovery/tiny_two_tip_smoke_truth_manifest.jsonl` is
provided only to exercise `score-tree` command plumbing against a two-tip Newick.

The branching fixture has four terminal leaves and can be scored against the training manifest:

```bash
PYTHONPATH=src python -m weighttraits.cli build-distance-cube \
  --checkpoint-manifest examples/distance_inputs/tiny_lora_cumulative_branching_leaf_outputs.yaml \
  --representation lora_cumulative_delta \
  --metric cosine \
  --metric l2 \
  --out outputs/tiny_lora_branching_smoke/cumulative_leaf_distance_cube

PYTHONPATH=src python -m weighttraits.cli reconstruct-tree \
  --cube outputs/tiny_lora_branching_smoke/cumulative_leaf_distance_cube \
  --metric l2 \
  --out outputs/tiny_lora_branching_smoke/cumulative_leaf_distance_cube/tree_l2.newick \
  --audit-out outputs/tiny_lora_branching_smoke/cumulative_leaf_distance_cube/tree_l2.audit.json

PYTHONPATH=src python -m weighttraits.cli score-tree \
  --truth-manifest examples/training/tiny_branching_manifest.jsonl \
  --estimate-newick outputs/tiny_lora_branching_smoke/cumulative_leaf_distance_cube/tree_l2.newick \
  --out outputs/tiny_lora_branching_smoke/cumulative_leaf_distance_cube/score_l2.json
```
