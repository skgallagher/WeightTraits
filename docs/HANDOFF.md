# WeightTraits Handoff

Last updated: 2026-07-03.

## Project Intent

WeightTraits is a private, cleaner rebuild of ELLMTrees under `/Users/shannon/Desktop/phylo/WeightTraits`. It should remain maintainable, tested, auditable, and suitable for local smoke runs plus cluster-scale training. The main scientific targets are to independently rebuild and verify the ELLMTrees results, including the ICLR draft tables/figures, while fixing weak spots around flexible topology generation, LoRA semantics, distance computation, tree reconstruction scoring, and trainer reliability.

## Current Git State

Latest pushed commit at the start of this handoff:

```text
6fb2f1b Validate training data formats
```

Recent pushed commits:

```text
c83f4b6 Add training prompt validation and ledgers
6d32e09 Add trainer planning controls
015973a Support sharded distance inputs
5eef610 Add streaming distance cube engine
aeb41ea Document cumulative LoRA distance semantics
f052600 Add flexible distance metric registry
ee16c92 Add topology audit and recovery scoring
710c8e5 Document and test tree generators
```

## What Exists

- Flexible tree generation with `fixed`, `chain`, `balanced`, `ellmtrees_balanced`, `poisson_branching`, and `pruned_binary_backbone`.
- Tree constraints for `min_depth`, `min_leaves`, pruning, and polytomy-generating contraction.
- Task/data assignment from enriched manifests.
- Recovery scoring with RF, TP/FP/FN, FN rate, clade recovery, exact recovery, and SE aggregation.
- Distance metric registry with cosine, L1, L2, correlation, threshold, and linear CKA.
- Streaming distance cube engine:
  - chunk-streamed vector metrics;
  - exact tensor-at-a-time CKA;
  - safetensors, sharded safetensors, torch, LoRA factor, and cumulative LoRA readers;
  - distance input manifests and `wt build-distance-cube`.
- LoRA analysis semantics:
  - true training process is fresh adapter on merged parent;
  - efficient analysis representation is cumulative path-summed `scale * B @ A`;
  - CLI supports `--adapter-chain node:edge0,edge1,...`.
- Trainer control plane:
  - `wt plan-training`;
  - full vs LoRA artifact planning;
  - prompt override resolution by manifest row, base model, model family, dataset, task, and default;
  - required prompt-field extraction;
  - prompt rendering validation;
  - loss warning/early-stop/plateau monitor;
  - training JSONL ledger helpers and `wt training-ledger-summary`.
- Offline training data validation:
  - dataset format contracts;
  - `wt validate-training-data`;
  - smoke contracts in `examples/training/dataset_formats_smoke.yaml`.
- Dataset registry and split audit:
  - `wt audit-datasets`;
  - no-download registry/split dry runs with `--no-load`;
  - optional Hugging Face `load_dataset` split and row-count audit in prepared environments.

## Verification So Far

After the last pushed increment:

```text
conda run -n ellmtrees env PYTHONPATH=src python -m pytest -q
```

passed with 70 tests.

After adding the dataset registry/split audit and this handoff:

```text
conda run -n ellmtrees env PYTHONPATH=src python -m pytest --override-ini=addopts=
```

passed with 77 tests.

Useful smoke commands that have passed:

```bash
PYTHONPATH=src python -m weighttraits.cli plan-training \
  --manifest reports/flexible_tree_assigned_manifest.jsonl \
  --config examples/training/full_smoke.yaml \
  --out /tmp/weighttraits_full_training_plan.jsonl

PYTHONPATH=src python -m weighttraits.cli plan-training \
  --manifest reports/flexible_tree_assigned_manifest.jsonl \
  --config examples/training/lora_smoke.yaml \
  --out /tmp/weighttraits_lora_training_plan.jsonl

PYTHONPATH=src python -m weighttraits.cli validate-training-data \
  --manifest reports/flexible_tree_assigned_manifest.jsonl \
  --config examples/training/full_smoke.yaml \
  --formats examples/training/dataset_formats_smoke.yaml \
  --out /tmp/weighttraits_training_data_validation_full.json

PYTHONPATH=src python -m weighttraits.cli validate-training-data \
  --manifest reports/flexible_tree_assigned_manifest.jsonl \
  --config examples/training/lora_smoke.yaml \
  --formats examples/training/dataset_formats_smoke.yaml \
  --out /tmp/weighttraits_training_data_validation_lora.json
```

Both full and LoRA validation smokes reported 13/13 jobs valid.

Dataset audit no-load smoke:

```bash
PYTHONPATH=src python -m weighttraits.cli audit-datasets \
  --registry configs/task_data_candidates.yaml \
  --formats examples/training/dataset_formats_smoke.yaml \
  --dataset-id boolq \
  --dataset-id hellaswag \
  --no-load \
  --out /tmp/weighttraits_dataset_audit_noload.json
```

## Next Best Steps

1. Add a real dataset-loader sample-rendering audit:
   - tiny sample;
   - apply `field_map`;
   - render prompt;
   - report missing fields and dropped rows.
2. Build the Hugging Face / PEFT execution layer:
   - `Trainer` / `Seq2SeqTrainer`;
   - LoRA adapter creation;
   - merge-and-save child weights for LoRA;
   - write ledger events from monitor decisions.
3. Add cluster run-list generation from planned training JSONL.
4. Then return to whitebox end-to-end smoke:
   - generated tree;
   - distance input manifest;
   - distance cube;
   - tree reconstruction;
   - RF/FN/FP/clade/exact recovery table with SEs.

## Important Caveats

- Do not commit generated outputs, model weights, checkpoints, caches, or reports unless they are intentional tiny examples.
- CKA is exact but still tensor-at-a-time.
- LoRA vector metrics stream dense `B @ A` row blocks; low-rank dot-product acceleration remains planned.
- The trainer execution loop is not implemented yet; current trainer work is a strong dry-run/control plane.
- `wt audit-datasets` in load mode may require network access and the optional `datasets` dependency.
