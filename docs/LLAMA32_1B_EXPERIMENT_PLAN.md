# Llama 3.2 1B Experiment Plan

This is the launch gate for independently rebuilding the Llama rows of the paper variants table.
No old analysis code is used. The old paper rows are comparison targets only after new WeightTraits
training, distance, reconstruction, and scoring artifacts exist.

## Matched conditions

- Full fine-tuning.
- QKV LoRA, rank 8 and alpha 16.
- QKV LoRA, rank 64 and alpha 128.

All conditions use `meta-llama/Llama-3.2-1B` at revision
`4e20de362430cd3b72f300e6b0f18e50e7166e08`, completion-only causal loss, seed `20260713`, and the
same 50 trees / 641 ordered task assignments as the Flan run. Root jobs load the pinned remote
revision; descendants load their local parent artifact without passing a Hugging Face revision.

## Verified gates

- Cached row audits cover all four task families.
- Completion packing masks the prompt and supervises the answer tokens.
- Rank-64 QKV resolution finds 48 modules and 9,437,184 trainable parameters.
- Root rank-64 LoRA smoke `154463`, dependent child `154464`, full-FT root `154465`, and full-FT
  dependent child `154473` completed two steps on Wright L40 GPUs.
- Successful Llama jobs opt in to post-save Trainer checkpoint cleanup. Verification job `154474`
  removed its 7.0 GB `checkpoint-2`, retained the 2.4 GB final model, and recorded the removal in
  backend metadata. Failed jobs retain their resume checkpoints because cleanup runs only after the
  final artifact is saved.
- Generated Llama run lists are valid and preserve the Flan tree/node/dataset ordering exactly.

## Storage constraint

The rank-64 smoke produced a roughly 53 MB adapter and 2.4 GB merged checkpoint. Keeping every
merged artifact would cost about 1.5 TB per condition. The 50-tree set has 365 leaves, so even
leaf-only merged checkpoints are roughly 876 GB per condition. Rank-64 adapters for all 641 nodes
are about 34 GB; rank-8 adapters should be about one eighth of that. A full-FT Trainer resume
checkpoint was 7.0 GB in addition to its 2.4 GB final model, so the Llama configs automatically
remove these resume directories after successful final-artifact saving.

Before a broad launch, implement a deliberate lifecycle:

1. Train one tree sequentially so parent artifacts exist for descendants.
2. Retain adapters and ledgers for LoRA; delete disposable internal merged models only after all
   descendants that require them have completed.
3. Retain full-FT leaf models, deleting internal checkpoints only after their descendant subtrees
   are complete.
4. Decide whether paper-facing LoRA distances will retain leaf merged weights, rematerialize them
   for analysis, or use cumulative adapters as a separately labeled sensitivity analysis. Do not
   silently substitute one representation for another.

## Next launch gate

Run three representative trees per condition with the six-job cluster cap. Capture elapsed time,
peak GPU memory, bytes retained before/after pruning, completed lineage nodes, and first-tree
distance/recovery outputs. Stop after those trees and review the evidence before submitting all 50.
