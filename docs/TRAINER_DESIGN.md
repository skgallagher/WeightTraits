# Trainer Design

The trainer is not allowed to be a pile of one-off scripts. WeightTraits splits training into two layers:

1. A testable control plane that plans jobs, prompt templates, artifacts, and stopping rules.
2. A later execution layer that turns each planned job into a Hugging Face / PEFT training run.

The current implementation is the control plane.

## Job Planning

`wt plan-training` consumes an enriched training manifest and a training config:

```bash
PYTHONPATH=src python -m weighttraits.cli plan-training \
  --manifest reports/flexible_tree_assigned_manifest.jsonl \
  --config examples/training/full_smoke.yaml \
  --out reports/training_jobs.full.jsonl
```

Each JSONL row is one trainable node. It records:

- node and parent IDs;
- base model and model family;
- task family and dataset ID;
- resolved prompt template and prompt source;
- initialization source;
- expected output artifacts;
- trainer hyperparameters;
- stopping and warning controls;
- LoRA config when applicable.

This makes dry runs cheap. Before launching a cluster array, we can inspect exactly which prompt and parent artifact each node will use.

## Prompt Resolution

Prompt formatting was brittle in the old project because model families and datasets did not always share a template. The planner records the prompt source for every node.

Resolution order:

1. `prompt_template` directly on the manifest row;
2. model-specific dataset template, trying exact `base_model` then `model_family`;
3. model-specific task template, trying exact `base_model` then `model_family`;
4. global dataset template;
5. global task template;
6. model-specific default template, trying exact `base_model` then `model_family`;
7. global default template.

If no template resolves, planning fails before training starts.

## Full Fine-Tuning

For full fine-tuning, a child node initializes from:

```text
root child: base_model
other child: output_root/<parent_id>/model
```

Expected artifacts:

```text
output_root/<node_id>/model
output_root/<node_id>/training_log.jsonl
```

## LoRA Training

LoRA training follows the true lineage process:

```text
train fresh adapter on merged parent -> save adapter -> save merged child
```

For LoRA, a child node initializes from:

```text
root child: base_model
other child: output_root/<parent_id>/merged
```

Expected artifacts:

```text
output_root/<node_id>/adapter
output_root/<node_id>/merged
output_root/<node_id>/training_log.jsonl
```

The downstream distance code can then compare cumulative adapter deltas without needing to load full merged weights.

## Loss Warnings And Stopping

The control plane defines the rules; the execution layer will call them after each logged training/eval event.

Current rules:

- warn when monitored loss rises above the best value by a configured relative amount for a configured number of monitored events;
- stop when there is no improvement for `patience` events;
- stop when absolute loss differences across a recent window are all below `plateau.min_delta`.

Example:

```yaml
stopping:
  metric: eval_loss
  early_stopping:
    patience: 4
    min_delta: 0.001
  plateau:
    window: 5
    min_delta: 0.0002
  warnings:
    loss_increase_relative: 0.05
    loss_increase_patience: 2
```

The plateau rule is the simple “stop when loss diff gets small” guardrail. It should be conservative for real runs because noisy eval loss can look flat briefly.

## Next Execution Layer

The next trainer increment should add:

- dataset loader registry;
- prompt rendering with required-field validation;
- Hugging Face `Trainer` / `Seq2SeqTrainer` execution;
- PEFT LoRA adapter creation and merge;
- resumable per-node training ledgers;
- cluster array generation from the planned JSONL.
