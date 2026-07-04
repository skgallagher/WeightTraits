# WeightTraits Handoff

Last updated: 2026-07-03.

## Project Intent

WeightTraits is a private, cleaner rebuild of ELLMTrees under `/Users/shannon/Desktop/phylo/WeightTraits`. It should remain maintainable, tested, auditable, and suitable for local smoke runs plus cluster-scale training. The main scientific targets are to independently rebuild and verify the ELLMTrees results, including the ICLR draft tables/figures, while fixing weak spots around flexible topology generation, LoRA semantics, distance computation, tree reconstruction scoring, and trainer reliability.

## Current Git State

Latest stable pushed base before the per-row runner increment:

```text
a20abaa Add training run list generation
```

Recent pushed commits:

```text
a20abaa Add training run list generation
2745f70 Add training sample rendering audit
1f2efd7 Add dataset audit handoff
6fb2f1b Validate training data formats
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
- Training sample rendering audit:
  - `wt audit-training-samples`;
  - loads a tiny sample for planned jobs;
  - applies dataset `field_map`;
  - renders resolved prompt templates;
  - reports counts, field names, row indices, and errors without storing raw samples or prompts.
- Training run-list generation:
  - `wt make-training-run-list`;
  - supports local and cluster execution profiles;
  - writes stable JSONL rows keyed by array index;
  - writes optional preflight report and SLURM runner or dry-run selector script;
  - checks parent order, artifact collisions, missing datasets, stopping guards, and LoRA merge semantics.
- Per-row training runner:
  - `wt run-training-row`;
  - selects one run-list row by array index or node id;
  - loads registry/format contracts, applies `field_map`, renders prompts, and derives targets;
  - imports `datasets`, `transformers`, and `peft` lazily for real training;
  - supports full fine-tuning and LoRA adapter plus merged-child saves;
  - writes started/running/completed/stopped_early/failed ledger events.

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

After adding the sample-rendering audit:

```text
conda run -n ellmtrees env PYTHONPATH=src python -m pytest tests/test_training_dataset_registry.py tests/test_training_data_formats.py --override-ini=addopts=
```

passed with 19 focused trainer preflight tests.

After adding the training run-list layer:

```text
conda run -n ellmtrees env PYTHONPATH=src python -m pytest tests/test_training_runlist.py --override-ini=addopts=
```

passed with 9 focused run-list tests.

After adding the per-row executor:

```text
conda run -n ellmtrees env PYTHONPATH=src python -m pytest tests/test_training_executor.py tests/test_training_runlist.py --override-ini=addopts=
```

passed with 15 focused executor/run-list tests.

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

Sample-rendering audit smoke, for environments where Hugging Face dataset loading is available:

```bash
PYTHONPATH=src python -m weighttraits.cli audit-training-samples \
  --manifest reports/flexible_tree_assigned_manifest.jsonl \
  --config examples/training/full_smoke.yaml \
  --registry configs/task_data_candidates.yaml \
  --formats examples/training/dataset_formats_smoke.yaml \
  --dataset-id boolq \
  --max-samples 4 \
  --out /tmp/weighttraits_training_sample_render_audit.json
```

Run-list local smoke:

```bash
PYTHONPATH=src python -m weighttraits.cli make-training-run-list \
  --manifest reports/flexible_tree_assigned_manifest.jsonl \
  --config examples/training/full_smoke.yaml \
  --profile configs/local/default.yaml \
  --out /tmp/weighttraits_full_local_runs.jsonl \
  --report /tmp/weighttraits_full_local_runs.report.json
```

Run-list cluster dry-run smoke:

```bash
PYTHONPATH=src python -m weighttraits.cli make-training-run-list \
  --manifest reports/flexible_tree_assigned_manifest.jsonl \
  --config examples/training/lora_smoke.yaml \
  --profile configs/cluster/wright.yaml \
  --registry configs/task_data_candidates.yaml \
  --formats examples/training/dataset_formats_smoke.yaml \
  --out /tmp/weighttraits_lora_runs.jsonl \
  --report /tmp/weighttraits_lora_runs.report.json \
  --slurm-out /tmp/weighttraits_lora_train.sbatch \
  --runner-dry-run
```

The Wright smoke reported 13 runs, no errors, and one expected warning because the run count exceeds
the profile's default concurrency throttle of 6. With `--runner-dry-run`, the generated script calls
`run-training-row --dry-run` and does not load datasets or models.

Per-row runner dry-run smoke:

```bash
PYTHONPATH=src python -m weighttraits.cli run-training-row \
  --run-list /tmp/weighttraits_lora_runs.jsonl \
  --index 0 \
  --dry-run
```

## Next Best Steps

1. Run a tiny real-model training smoke in an environment with the training extra installed:
   - one full row;
   - one LoRA row;
   - verify model/adapter/merged artifacts and ledger events.
2. Refine supervision templates:
   - prefer explicit `trainer.target_field` or `trainer.target_template`;
   - audit old prompt templates that currently include the answer in the rendered prompt.
3. Then return to whitebox end-to-end smoke:
   - generated tree;
   - distance input manifest;
   - distance cube;
   - tree reconstruction;
   - RF/FN/FP/clade/exact recovery table with SEs.

## Important Caveats

- Do not commit generated outputs, model weights, checkpoints, caches, or reports unless they are intentional tiny examples.
- CKA is exact but still tensor-at-a-time.
- LoRA vector metrics stream dense `B @ A` row blocks; low-rank dot-product acceleration remains planned.
- The trainer execution loop is implemented behind an optional HF/PEFT backend, but has not yet been exercised on a real downloaded model in this repo.
- `wt audit-datasets` in load mode may require network access and the optional `datasets` dependency.
- `wt audit-training-samples` requires dataset loading and should run only in environments where downloads/cache access are intended.
- `wt make-training-run-list --runner-dry-run --slurm-out` generates a safe selector script; omit `--runner-dry-run` only when real training is intended.
