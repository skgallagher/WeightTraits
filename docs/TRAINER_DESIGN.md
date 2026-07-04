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
- required prompt fields;
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

## Prompt Rendering Validation

Prompt templates use Python `str.format` fields. WeightTraits extracts the root fields from every planned template:

```text
Question: {question}
Context: {context}
Answer: {answer}
```

requires:

```text
answer, context, question
```

The prompt renderer fails loudly when an example is missing a required field. This is intentionally strict because prompt/data mismatches are much cheaper to fix before a cluster run than after a half-finished lineage.

## Dataset Format Contracts

Before downloading datasets or launching training, WeightTraits can validate planned jobs against an offline dataset-format contract:

```bash
PYTHONPATH=src python -m weighttraits.cli validate-training-data \
  --manifest reports/flexible_tree_assigned_manifest.jsonl \
  --config examples/training/full_smoke.yaml \
  --formats examples/training/dataset_formats_smoke.yaml \
  --out reports/training_data_validation.full.json
```

The format file declares which canonical prompt fields a dataset can provide:

```yaml
datasets:
  - dataset_id: boolq
    task_family: qa_reasoning
    prompt_fields:
      - question
      - context
      - answer
    raw_fields:
      - question
      - passage
      - answer
    field_map:
      question: question
      context: passage
      answer: answer
```

This catches cases where a model/task prompt asks for `{context}` but the dataset contract cannot provide `context`. It does not replace the later loader audit that checks downloadability, licenses, splits, and row counts.

## Dataset Registry And Split Audit

The execution layer now has a dataset registry audit before any training launcher needs to touch a
cluster. It reads the candidate pool, optionally combines it with the offline format contracts, and
can either stop before downloads or call Hugging Face `load_dataset` to record available splits and
row counts.

No-download registry/split dry run:

```bash
PYTHONPATH=src python -m weighttraits.cli audit-datasets \
  --registry configs/task_data_candidates.yaml \
  --formats examples/training/dataset_formats_smoke.yaml \
  --dataset-id boolq \
  --dataset-id hellaswag \
  --no-load \
  --out reports/dataset_audit_noload_smoke.json
```

Omit `--no-load` only in an environment where dataset downloads are intended. The audit reports
`missing_splits`, `load_failed`, and `unknown_dataset_id` separately so cluster dry runs can fail for
the right reason.

## Training Sample Rendering Audit

After the offline contract and split checks pass, use a tiny sample-rendering audit before launching
training. This loads a few rows, applies each dataset's `field_map`, renders the resolved prompt for
each planned job, and reports only counts, field names, row indices, and errors. It does not write
raw examples or rendered prompts to disk.

```bash
PYTHONPATH=src python -m weighttraits.cli audit-training-samples \
  --manifest reports/flexible_tree_assigned_manifest.jsonl \
  --config examples/training/full_smoke.yaml \
  --registry configs/task_data_candidates.yaml \
  --formats examples/training/dataset_formats_smoke.yaml \
  --dataset-id boolq \
  --max-samples 4 \
  --out reports/training_sample_render_audit.boolq.json
```

The audit reports `missing_dataset_format_spec`, `unknown_dataset_id`, `missing_split`,
`empty_split`, `missing_prompt_fields`, `empty_rendered_prompt`, and `render_failed` separately. A
non-empty rendered prompt is counted as renderable even when field values are booleans, integers,
lists, or other normal dataset objects because Python prompt formatting stringifies those values.

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

## Training Ledgers

Long jobs need resumable state outside stdout. WeightTraits uses a JSONL ledger with one event per status or loss update:

```text
started -> running -> completed
started -> running -> stopped_early
started -> failed
```

The helper API can load a ledger, summarize latest node statuses, identify failed nodes, and decide whether a node should be skipped on resume. The CLI summary is:

```bash
PYTHONPATH=src python -m weighttraits.cli training-ledger-summary \
  --ledger outputs/lora_smoke/training_ledger.jsonl
```

## Run Lists And Scheduler Dry Runs

Cluster training must start from an explicit run list. `wt make-training-run-list` turns planned
jobs into stable JSONL rows with array indices, parent/init artifacts, expected outputs, ledger path,
and a placeholder runner entrypoint. It also emits preflight errors for artifact collisions, missing
dataset IDs, parent-order mistakes, and LoRA configurations that would fail to save merged parent
weights for descendants.

```bash
PYTHONPATH=src python -m weighttraits.cli make-training-run-list \
  --manifest reports/flexible_tree_assigned_manifest.jsonl \
  --config examples/training/lora_smoke.yaml \
  --profile configs/cluster/wright.yaml \
  --out /tmp/weighttraits_lora_runs.jsonl \
  --report /tmp/weighttraits_lora_runs.report.json \
  --slurm-out /tmp/weighttraits_lora_train.sbatch
```

The generated SLURM script is a dry-run selector until the HF/PEFT runner lands. It calls:

```bash
PYTHONPATH=src python -m weighttraits.cli describe-training-run \
  --run-list /tmp/weighttraits_lora_runs.jsonl \
  --index "${SLURM_ARRAY_TASK_ID}"
```

This makes array indexing, throttling, and path visibility auditable before model-loading code is
allowed to run.

## Next Execution Layer

The next trainer increment should add:

- Hugging Face `Trainer` / `Seq2SeqTrainer` execution;
- PEFT LoRA adapter creation and merge;
- replacing the dry-run selector with the real per-row training runner.
