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
