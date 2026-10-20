# Corrected native behavior inference

`weighttraits.behavior.probe_inference` is the fail-closed path from completed training artifacts
to decoded held-out responses. It deliberately does not infer architecture from an adapter or a
checkpoint. `model_task` is required and must match every run-list row as well as the pinned model
binding in the behavior protocol registry.

## Checkpoint contract

`resolve_leaf_checkpoints(...)` reads the run-set summary, selected tree run list, ledger, and
truth manifest together. It requires:

- clean, hash-matching summary and tree rows;
- identical declared run-list and ledger paths in every run row;
- an exact one-to-one run-list/truth-manifest node set and ledger/run-list node set;
- latest ledger status `completed` for every node (not skipped or stopped early);
- one training method across the tree;
- `model` artifacts for full fine-tuning and merged full-weight `merged` artifacts for LoRA;
- identical artifact paths in the run list and terminal ledger event;
- a loadable checkpoint directory containing `config.json` and saved model weights; and
- an exact base-model ID, pinned revision, and explicit `model_task` in every run row.

Leaves are returned in their truth-manifest appearance order. Missing, extra, or duplicate nodes
stop resolution.

## Generation contract

Run one leaf at a time so its model is loaded only once for all requested protocols and probes:

```bash
PYTHONPATH=src python scripts/run_behavior_probe_inference.py \
  --training-summary /frozen/stage/examples/training/.../training_run_list_summary.json \
  --cohort-id flan-full \
  --tree-id confirm_paper_tree_001 \
  --leaf-id n12 \
  --model-task seq2seq \
  --registry /frozen/stage/examples/behavior/protocol_registry.json \
  --protocol-prompts translation=/frozen/prompts/translation.seq2seq.jsonl \
  --protocol-prompts mc=/frozen/prompts/mc.seq2seq.jsonl \
  --protocol-prompts dolly=/frozen/prompts/dolly.seq2seq.jsonl \
  --path-base /frozen/stage \
  --device cuda \
  --torch-dtype bfloat16 \
  --out /results/flan-full/tree001/n12.responses.jsonl \
  --receipt /results/flan-full/tree001/n12.responses.receipt.json
```

The tokenizer is always loaded from the registry-pinned base model and revision with
`local_files_only=True`. The selected checkpoint model is also loaded locally only. Every draw
passes the registry's explicit sampling seed, `do_sample`, temperature, top-p, minimum tokens, and
maximum tokens. Causal decoding slices off input prompt tokens; seq2seq decoding uses the generated
sequence. Decoded strings are never stripped. An exact empty continuation is a completed response
with `text: ""` and `empty: true`.

## Response and receipt artifacts

Each JSONL row contains cohort/tree/leaf/protocol/probe/prompt/sample identities, source and prompt
hashes, draw settings, exact text and reference, response hash, checkpoint/run-list/ledger/truth/
registry provenance, and fixture/dataset provenance. `prompt_artifacts` is recorded as an exact
mapping from protocol ID to `{path, sha256}`.

The terminal receipt contains the same `prompt_artifacts` mapping at top level and inside the
canonical request, the canonical request SHA256, complete response artifact SHA256, and a strict
grid audit. A partial JSONL is resumable only when it is the exact valid prefix of the request.
Changed checkpoints, ledgers, run lists, registries, prompt fixtures, backend settings, response
rows, or receipts are rejected rather than silently reused.
