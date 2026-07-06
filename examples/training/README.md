# Training Examples

These examples exercise the trainer control plane without downloading models or datasets.

Plan full fine-tuning jobs:

```bash
PYTHONPATH=src python -m weighttraits.cli plan-training \
  --manifest reports/flexible_tree_assigned_manifest.jsonl \
  --config examples/training/full_smoke.yaml \
  --out /tmp/weighttraits_full_training_plan.jsonl
```

Plan LoRA jobs:

```bash
PYTHONPATH=src python -m weighttraits.cli plan-training \
  --manifest reports/flexible_tree_assigned_manifest.jsonl \
  --config examples/training/lora_smoke.yaml \
  --out /tmp/weighttraits_lora_training_plan.jsonl
```

Validate prompt fields against offline dataset contracts:

```bash
PYTHONPATH=src python -m weighttraits.cli validate-training-data \
  --manifest reports/flexible_tree_assigned_manifest.jsonl \
  --config examples/training/full_smoke.yaml \
  --formats examples/training/dataset_formats_smoke.yaml \
  --out /tmp/weighttraits_training_data_validation.json
```

Audit registry/split names without downloading datasets:

```bash
PYTHONPATH=src python -m weighttraits.cli audit-datasets \
  --registry configs/task_data_candidates.yaml \
  --formats examples/training/dataset_formats_smoke.yaml \
  --dataset-id boolq \
  --dataset-id hellaswag \
  --no-load \
  --out /tmp/weighttraits_dataset_audit_noload.json
```

Render a tiny sample through the planned prompts when dataset downloads are available:

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

Generate an explicit cluster/local run list with preflight checks:

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

Inspect the run selected by an array index:

```bash
PYTHONPATH=src python -m weighttraits.cli describe-training-run \
  --run-list /tmp/weighttraits_lora_runs.jsonl \
  --index 0
```

Exercise the generated row-runner command without loading datasets or models:

```bash
PYTHONPATH=src python -m weighttraits.cli run-training-row \
  --run-list /tmp/weighttraits_lora_runs.jsonl \
  --index 0 \
  --dry-run
```

## Tiny Real-Training Smoke

The tiny fixtures use local JSONL data so only the model may need to be downloaded or read from
cache. Run these in an environment with the `training` extra installed. If the default Hugging Face
dataset cache is not writable, set `HF_DATASETS_CACHE` to a scratch or `/tmp` directory.

Validate and render the local JSONL fixture:

```bash
PYTHONPATH=src python -m weighttraits.cli audit-datasets \
  --registry examples/training/tiny_dataset_registry.yaml \
  --formats examples/training/tiny_dataset_formats.yaml \
  --out /tmp/weighttraits_tiny_dataset_audit.json

PYTHONPATH=src python -m weighttraits.cli audit-training-samples \
  --manifest examples/training/tiny_manifest.jsonl \
  --config examples/training/tiny_full_smoke.yaml \
  --registry examples/training/tiny_dataset_registry.yaml \
  --formats examples/training/tiny_dataset_formats.yaml \
  --max-samples 2 \
  --out /tmp/weighttraits_tiny_sample_audit.json
```

Run one full fine-tuning row:

```bash
PYTHONPATH=src python -m weighttraits.cli make-training-run-list \
  --manifest examples/training/tiny_manifest.jsonl \
  --config examples/training/tiny_full_smoke.yaml \
  --registry examples/training/tiny_dataset_registry.yaml \
  --formats examples/training/tiny_dataset_formats.yaml \
  --out /tmp/weighttraits_tiny_full_runs.jsonl \
  --allow-existing-artifacts

PYTHONPATH=src python -m weighttraits.cli run-training-row \
  --run-list /tmp/weighttraits_tiny_full_runs.jsonl \
  --index 0 \
  --max-train-samples 2 \
  --allow-missing-eval
```

Run one LoRA row:

```bash
PYTHONPATH=src python -m weighttraits.cli make-training-run-list \
  --manifest examples/training/tiny_manifest.jsonl \
  --config examples/training/tiny_lora_smoke.yaml \
  --registry examples/training/tiny_dataset_registry.yaml \
  --formats examples/training/tiny_dataset_formats.yaml \
  --out /tmp/weighttraits_tiny_lora_runs.jsonl \
  --allow-existing-artifacts

PYTHONPATH=src python -m weighttraits.cli run-training-row \
  --run-list /tmp/weighttraits_tiny_lora_runs.jsonl \
  --index 0 \
  --max-train-samples 2 \
  --allow-missing-eval
```

## Tiny Lineage Smoke

The lineage fixtures train two nodes, `n0` then child `n1`, to verify parent/child
initialization and shared ledger behavior. For full fine-tuning, `n1` initializes from
`outputs/tiny_full_lineage_smoke/n0/model`. For LoRA, `n1` initializes from
`outputs/tiny_lora_lineage_smoke/n0/merged`.

Generate and run the full lineage rows:

```bash
PYTHONPATH=src python -m weighttraits.cli make-training-run-list \
  --manifest examples/training/tiny_lineage_manifest.jsonl \
  --config examples/training/tiny_full_lineage_smoke.yaml \
  --registry examples/training/tiny_dataset_registry.yaml \
  --formats examples/training/tiny_dataset_formats.yaml \
  --out /tmp/weighttraits_tiny_full_lineage_runs.jsonl \
  --allow-existing-artifacts \
  --max-train-samples 2 \
  --allow-missing-eval

PYTHONPATH=src python -m weighttraits.cli run-training-row \
  --run-list /tmp/weighttraits_tiny_full_lineage_runs.jsonl \
  --index 0

PYTHONPATH=src python -m weighttraits.cli run-training-row \
  --run-list /tmp/weighttraits_tiny_full_lineage_runs.jsonl \
  --index 1
```

Generate and run the LoRA lineage rows:

```bash
PYTHONPATH=src python -m weighttraits.cli make-training-run-list \
  --manifest examples/training/tiny_lineage_manifest.jsonl \
  --config examples/training/tiny_lora_lineage_smoke.yaml \
  --registry examples/training/tiny_dataset_registry.yaml \
  --formats examples/training/tiny_dataset_formats.yaml \
  --out /tmp/weighttraits_tiny_lora_lineage_runs.jsonl \
  --allow-existing-artifacts \
  --max-train-samples 2 \
  --allow-missing-eval

PYTHONPATH=src python -m weighttraits.cli run-training-row \
  --run-list /tmp/weighttraits_tiny_lora_lineage_runs.jsonl \
  --index 0

PYTHONPATH=src python -m weighttraits.cli run-training-row \
  --run-list /tmp/weighttraits_tiny_lora_lineage_runs.jsonl \
  --index 1
```

## Tiny Branching Smoke

The branching fixture trains six nodes with four terminal leaves: `n2`, `n3`, `n4`, and `n5`.
It is the smallest tiny-model topology here with nontrivial RF/FN/FP scoring structure.

Generate the full branching rows:

```bash
PYTHONPATH=src python -m weighttraits.cli make-training-run-list \
  --manifest examples/training/tiny_branching_manifest.jsonl \
  --config examples/training/tiny_full_branching_smoke.yaml \
  --registry examples/training/tiny_dataset_registry.yaml \
  --formats examples/training/tiny_dataset_formats.yaml \
  --out /tmp/weighttraits_tiny_full_branching_runs.jsonl \
  --allow-existing-artifacts \
  --max-train-samples 2 \
  --allow-missing-eval
```

Run indices `0` through `5` in order so parent artifacts exist before child rows.

Generate the LoRA branching rows:

```bash
PYTHONPATH=src python -m weighttraits.cli make-training-run-list \
  --manifest examples/training/tiny_branching_manifest.jsonl \
  --config examples/training/tiny_lora_branching_smoke.yaml \
  --registry examples/training/tiny_dataset_registry.yaml \
  --formats examples/training/tiny_dataset_formats.yaml \
  --out /tmp/weighttraits_tiny_lora_branching_runs.jsonl \
  --allow-existing-artifacts \
  --max-train-samples 2 \
  --allow-missing-eval
```

Run indices `0` through `5` in order for the LoRA branching rows as well.
