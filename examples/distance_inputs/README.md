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
