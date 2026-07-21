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
- One-tree, 14-node bounded benchmarks completed on Wright for full FT (`154555`), LoRA r8
  (`154556`), and LoRA r64 (`154557`).

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
4. Paper-facing Llama LoRA storage retains the pinned base revision, every node adapter, run lists,
   ledgers, and retention audits. Merged weights are temporary training materializations and are
   removed after the whole tree succeeds. Analysis must use the exact cumulative adapter-chain path
   or explicitly rematerialize leaf weights; the representation must remain labeled.

The sequential Slurm wrapper implements step 2/3 when `PRUNE_INTERNAL_PARENTS=true`. After each
successful row, `prune-training-parent-artifact` consults the shared tree ledger and acts only when
all of the parent's direct children have successful terminal states. Decisions are appended to a
per-tree retention audit. A Wright smoke removed 2,488,861,763 bytes from internal n0 and verified
that both leaf models remained. The option is off by default for existing experiment families.

For LoRA, `PRUNE_LORA_MERGED_AFTER_TREE=true` adds a second audited gate after the final row. It
requires every node to have a successful terminal status and an existing adapter, then removes all
remaining merged models, including leaves. Applied to the tree-002 benchmarks, it removed
17,421,998,790 bytes from each LoRA condition while retaining all 14 adapters. The archived r8 and
r64 benchmark evidence occupies 294 MB and 735 MB respectively, down from roughly 17 GB each.

## Broad launch status

On 2026-07-14 the Wright `all` partition concurrency target was raised to its 20-L40 capacity.
Adapter-only production arrays are submitted as `154594_[1-50%20]` (r8) and
`154595_[1-50%20]` (r64), with both internal-parent and end-of-tree merged-model pruning enabled.
Slurm arbitrates these against the older corrected-Flan array. Full fine-tuning remains held because
it has no adapter-only reconstruction path; its benchmark evidence is archived separately while a
leaf-weight/delta retention policy is decided.
