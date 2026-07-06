# WeightTraits Handoff

Last updated: 2026-07-06.

## Project Intent

WeightTraits is a private, cleaner rebuild of ELLMTrees under `/Users/shannon/Desktop/phylo/WeightTraits`. It should remain maintainable, tested, auditable, and suitable for local smoke runs plus cluster-scale training. The main scientific targets are to independently rebuild and verify the ELLMTrees results, including the ICLR draft tables/figures, while fixing weak spots around flexible topology generation, LoRA semantics, distance computation, tree reconstruction scoring, and trainer reliability.

## Current Git State

Stable base before the branching smoke slice:

```text
28f133f Add tiny artifact distance smoke
```

Recent pushed commits:

```text
28f133f Add tiny artifact distance smoke
523902b Add tiny lineage training smoke
1a290eb Add Biopython reconstruction and Wright smoke fixes
ee338ef Add tiny training smoke fixtures
c749ba2 Add per-row training executor
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

Local and Wright checkouts were clean at `28f133f` after pushing through Wright. Local GitHub SSH
still failed with `Permission denied (publickey)`, so the successful push path was:

```text
local git bundle -> rsync over /tmp/wright-codex.sock -> git fetch bundle on Wright -> ff-only merge -> git push origin main
```

Wright access notes:

```text
ssh -S /tmp/wright-codex.sock wright
checkout: /home/export/sgallagh/WeightTraits
env: /home/export/sgallagh/.conda/envs/weighttraits
runner: /opt/miniforge3/bin/mamba run -n weighttraits ...
cache for tiny HF datasets: $HOME/.cache/WeightTraits/hf_datasets
```

There is one temporary Wright stash left from parking synced fixture files before the fast-forward:

```text
stash@{0}: On main: codex-synced-lineage-before-523902b
```

It should be safe to drop after confirming it duplicates `523902b`.

Branching smoke slice contents:

```text
docs/HANDOFF.md
docs/STREAMING_DISTANCE_CUBES.md
examples/distance_inputs/README.md
examples/distance_inputs/tiny_full_branching_leaf_outputs.yaml
examples/distance_inputs/tiny_lora_cumulative_branching_leaf_outputs.yaml
examples/distance_inputs/tiny_lora_merged_branching_leaf_outputs.yaml
examples/training/README.md
examples/training/tiny_branching_manifest.jsonl
examples/training/tiny_full_branching_smoke.yaml
examples/training/tiny_lora_branching_smoke.yaml
tests/test_distance_input_manifest.py
tests/test_training_tiny_examples.py
```

Intent and status of that slice:

- Add a six-row tiny branching training fixture with four terminal leaves: `n2`, `n3`, `n4`, `n5`.
- Add full and LoRA training configs using output roots `outputs/tiny_full_branching_smoke` and
  `outputs/tiny_lora_branching_smoke`.
- Add leaf-only distance-input manifests for full checkpoints, LoRA merged checkpoints, and
  cumulative LoRA adapter chains.
- Local focused tests passed with 12 tests.
- Wright focused training fixture tests passed with 6 tests and both full/LoRA run lists validated
  with 6 runs and no warnings.
- Wright full and LoRA branching training rows completed for all six nodes.
- Wright four-leaf distance/reconstruct/score smokes passed for full checkpoints, LoRA merged
  checkpoints, and cumulative LoRA adapters with exact recovery of the single nontrivial split.

Ledger-derived distance manifest slice:

```text
src/weighttraits/distances/manifest.py
src/weighttraits/distances/__init__.py
src/weighttraits/cli.py
tests/test_cli_distance_cube.py
tests/test_distance_input_manifest.py
docs/STREAMING_DISTANCE_CUBES.md
examples/distance_inputs/README.md
docs/HANDOFF.md
```

Intent and status of that slice:

- Add `wt make-distance-input-manifest`.
- Read latest terminal training ledger events and emit `build-distance-cube` manifests.
- With `--truth-manifest`, select terminal leaves by default.
- Support `--artifact model`, `--artifact merged`, and `--artifact adapter_chain`.
- Rewrite relative ledger artifact paths relative to the generated manifest.
- Local focused tests passed with 15 tests and full local suite passed with 121 tests.
- Wright focused tests passed with 15 tests and full Wright suite passed with 121 tests.
- On Wright, generated a cumulative LoRA branching leaf manifest from
  `outputs/tiny_lora_branching_smoke/training_ledger.jsonl`, rebuilt the distance cube, reconstructed,
  and scored exact recovery for the single nontrivial split.

Tiny branching contrast fixture slice:

```text
examples/training/README.md
examples/training/tiny_branching_contrast_manifest.jsonl
examples/training/tiny_full_branching_contrast_smoke.yaml
examples/training/tiny_lora_branching_contrast_smoke.yaml
examples/training/tiny_branch_*_{train,validation}.jsonl
examples/training/tiny_dataset_registry.yaml
examples/training/tiny_dataset_formats.yaml
tests/test_training_tiny_examples.py
docs/HANDOFF.md
```

Intent and status of that slice:

- Add a six-row contrast variant of the tiny branching fixture.
- Keep the same terminal leaves: `n2`, `n3`, `n4`, `n5`.
- Assign node-specific local JSONL datasets so root siblings carry `left`/`right` branch targets,
  and terminal siblings carry distinct `alpha`/`beta` leaf targets.
- Add full and LoRA contrast training configs using output roots
  `outputs/tiny_full_branching_contrast_smoke` and
  `outputs/tiny_lora_branching_contrast_smoke`.
- Local focused fixture tests passed with 8 tests; full local suite passed with 123 tests.
- Not yet run on Wright; next Wright pass should train all six contrast rows, generate ledger-derived
  full/merged/cumulative leaf manifests, then build/reconstruct/score distance cubes.

Previous artifact-distance smoke slice contents:

```text
docs/HANDOFF.md
examples/distance_inputs/README.md
examples/distance_inputs/tiny_full_lineage_outputs.yaml
examples/distance_inputs/tiny_lora_cumulative_lineage_outputs.yaml
examples/distance_inputs/tiny_lora_merged_lineage_outputs.yaml
examples/recovery/tiny_two_tip_smoke_truth_manifest.jsonl
tests/test_distance_input_manifest.py
tests/test_recovery_scoring.py
```

Intent and status of that previous slice:

- Add distance-input manifests for the Wright two-node training artifacts:
  - full checkpoints: `outputs/tiny_full_lineage_smoke/n0/model`, `.../n1/model`;
  - LoRA merged checkpoints: `outputs/tiny_lora_lineage_smoke/n0/merged`, `.../n1/merged`;
  - LoRA cumulative adapter chains: `n0=[n0/adapter]`, `n1=[n0/adapter,n1/adapter]`.
- Add a two-tip truth manifest only for command plumbing through `score-tree`.
- Caveat: the two-node training lineage is a chain with terminal leaf `n1`, so it is not a
  meaningful RF/FN/FP recovery target. The two-tip smoke truth has no informative splits; use it
  only to verify `build-distance-cube -> reconstruct-tree -> score-tree` on real artifacts.
- Local focused tests passed with 12 tests and full local suite passed with 112 tests.
- Wright focused tests passed with 12 tests and full Wright suite passed with 112 tests.
- Wright artifact smoke passed for full checkpoints, LoRA merged checkpoints, and cumulative LoRA
  adapter chains.

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
- Biopython-backed neighbor-joining reconstruction from distance cubes:
  - `wt reconstruct-tree`;
  - layer selection by name/index or mean/median layer aggregation;
  - Newick output plus JSON audit metadata.
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
  - registry rows support `hf_kwargs`, including local JSONL `data_files`.
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
  - handles current and older Transformers trainer constructor names:
    `processing_class` vs `tokenizer`;
  - uses `Seq2SeqTrainingArguments` for seq2seq trainer runs when available.
- Tiny real-training smoke fixtures:
  - `examples/training/tiny_manifest.jsonl`;
  - `examples/training/tiny_train.jsonl` and `tiny_validation.jsonl`;
  - `examples/training/tiny_dataset_registry.yaml`;
  - `examples/training/tiny_dataset_formats.yaml`;
  - `examples/training/tiny_full_smoke.yaml`;
  - `examples/training/tiny_lora_smoke.yaml`;
  - prompts keep the label out of the prompt and use `trainer.target_field: answer`.

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

After adding registry `hf_kwargs` and tiny smoke fixtures:

```text
conda run -n ellmtrees env PYTHONPATH=src python -m pytest tests/test_training_dataset_registry.py tests/test_training_executor.py tests/test_training_tiny_examples.py --override-ini=addopts=
```

passed with 20 focused tests.

The full suite then passed with 100 tests:

```text
conda run -n ellmtrees env PYTHONPATH=src python -m pytest --override-ini=addopts=
```

After adding Biopython-backed cube-to-Newick reconstruction:

```text
conda run -n ellmtrees env PYTHONPATH=src python -m pytest tests/test_reconstruction.py tests/test_cli_distance_cube.py --override-ini=addopts=
```

passed with 8 focused tests.

The full suite then passed with 105 tests:

```text
conda run -n ellmtrees env PYTHONPATH=src python -m pytest --override-ini=addopts=
```

After adding the tiny local whitebox CLI smoke:

```text
conda run -n ellmtrees env PYTHONPATH=src python -m pytest tests/test_whitebox_smoke.py --override-ini=addopts=
```

passed with 1 focused smoke test. This exercises:

```text
wt build-distance-cube -> wt reconstruct-tree -> wt score-tree -> wt aggregate-recovery
```

The full suite then passed with 106 tests:

```text
conda run -n ellmtrees env PYTHONPATH=src python -m pytest --override-ini=addopts=
```

After exercising real HF/PEFT training on Wright with Transformers 5.13.0, two executor
compatibility fixes were added:

- pass tokenizer-like objects as `processing_class` when the installed Trainer constructor uses
  that newer name, otherwise fall back to `tokenizer`;
- use `Seq2SeqTrainingArguments` for seq2seq runs when available.

Focused executor tests passed locally:

```text
conda run -n ellmtrees env PYTHONPATH=src python -m pytest tests/test_training_executor.py --override-ini=addopts=
```

with 7 tests. The full local suite then passed with 108 tests:

```text
conda run -n ellmtrees env PYTHONPATH=src python -m pytest --override-ini=addopts=
```

On Wright, after syncing the executor compatibility patch into the cloned checkout:

```text
/opt/miniforge3/bin/mamba run -n weighttraits env PYTHONPATH=src python -m pytest tests/test_training_executor.py --override-ini=addopts=
```

passed with 7 focused executor tests. The remote suite passed with 102 tests:

```text
/opt/miniforge3/bin/mamba run -n weighttraits env PYTHONPATH=src python -m pytest --override-ini=addopts=
```

After adding the two-node tiny lineage fixtures:

```text
conda run -n ellmtrees env PYTHONPATH=src python -m pytest tests/test_training_tiny_examples.py tests/test_training_runlist.py --override-ini=addopts=
```

passed locally with 14 focused tests. The full local suite passed with 110 tests:

```text
conda run -n ellmtrees env PYTHONPATH=src python -m pytest --override-ini=addopts=
```

On Wright:

```text
/opt/miniforge3/bin/mamba run -n weighttraits env PYTHONPATH=src python -m pytest tests/test_training_tiny_examples.py --override-ini=addopts=
```

passed with 4 focused fixture tests, and:

```text
/opt/miniforge3/bin/mamba run -n weighttraits env PYTHONPATH=src python -m pytest --override-ini=addopts=
```

passed with 110 tests.

After adding tiny lineage artifact distance manifests and a two-tip scoring smoke:

```text
conda run -n ellmtrees env PYTHONPATH=src python -m pytest tests/test_distance_input_manifest.py tests/test_recovery_scoring.py --override-ini=addopts=
```

passed locally with 12 focused tests. The full local suite passed with 112 tests:

```text
conda run -n ellmtrees env PYTHONPATH=src python -m pytest --override-ini=addopts=
```

On Wright:

```text
/opt/miniforge3/bin/mamba run -n weighttraits env PYTHONPATH=src python -m pytest tests/test_distance_input_manifest.py tests/test_recovery_scoring.py --override-ini=addopts=
```

passed with 12 focused tests, and:

```text
/opt/miniforge3/bin/mamba run -n weighttraits env PYTHONPATH=src python -m pytest --override-ini=addopts=
```

passed with 112 tests.

After adding the six-node tiny branching fixtures and four-leaf distance manifests:

```text
conda run -n ellmtrees env PYTHONPATH=src python -m pytest tests/test_distance_input_manifest.py tests/test_training_tiny_examples.py --override-ini=addopts=
```

passed locally with 12 focused tests. The full local suite passed with 115 tests:

```text
conda run -n ellmtrees env PYTHONPATH=src python -m pytest --override-ini=addopts=
```

On Wright:

```text
/opt/miniforge3/bin/mamba run -n weighttraits env PYTHONPATH=src python -m pytest tests/test_training_tiny_examples.py --override-ini=addopts=
```

passed with 6 focused fixture tests, and:

```text
/opt/miniforge3/bin/mamba run -n weighttraits env PYTHONPATH=src python -m pytest --override-ini=addopts=
```

passed with 115 tests.

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

Tiny local JSONL dataset smoke:

```bash
HF_DATASETS_CACHE=/tmp/weighttraits_hf_datasets \
PYTHONPATH=src python -m weighttraits.cli audit-datasets \
  --registry examples/training/tiny_dataset_registry.yaml \
  --formats examples/training/tiny_dataset_formats.yaml \
  --out /tmp/weighttraits_tiny_dataset_audit.json

HF_DATASETS_CACHE=/tmp/weighttraits_hf_datasets \
PYTHONPATH=src python -m weighttraits.cli audit-training-samples \
  --manifest examples/training/tiny_manifest.jsonl \
  --config examples/training/tiny_full_smoke.yaml \
  --registry examples/training/tiny_dataset_registry.yaml \
  --formats examples/training/tiny_dataset_formats.yaml \
  --max-samples 2 \
  --out /tmp/weighttraits_tiny_sample_audit.json
```

Those local JSONL smokes passed with 2 train rows, 1 validation row, and 2 rendered prompt samples.

Tiny full/LoRA run-list dry-runs passed:

```bash
PYTHONPATH=src python -m weighttraits.cli make-training-run-list \
  --manifest examples/training/tiny_manifest.jsonl \
  --config examples/training/tiny_full_smoke.yaml \
  --registry examples/training/tiny_dataset_registry.yaml \
  --formats examples/training/tiny_dataset_formats.yaml \
  --out /tmp/weighttraits_tiny_full_runs.jsonl \
  --allow-existing-artifacts \
  --runner-dry-run

PYTHONPATH=src python -m weighttraits.cli make-training-run-list \
  --manifest examples/training/tiny_manifest.jsonl \
  --config examples/training/tiny_lora_smoke.yaml \
  --registry examples/training/tiny_dataset_registry.yaml \
  --formats examples/training/tiny_dataset_formats.yaml \
  --out /tmp/weighttraits_tiny_lora_runs.jsonl \
  --allow-existing-artifacts \
  --runner-dry-run
```

## Wright Status

Wright is reachable, but Codex should not try to perform interactive auth itself. The reliable
workflow is:

1. Ask Shannon to run this in a normal macOS Terminal, not in Codex:

```bash
ssh -M -S /tmp/wright-codex.sock -fN wright
```

This may prompt for a passphrase or password in that Terminal. Once it succeeds, Codex can reuse
the socket with:

```bash
ssh -S /tmp/wright-codex.sock wright hostname
```

To close the socket later:

```bash
ssh -S /tmp/wright-codex.sock -O exit wright
```

Known-bad or low-value steps to skip in Codex:

```text
ssh wright hostname
ssh -o IdentitiesOnly=yes -i /Users/shannon/.ssh/id_ed25519 sgallagh@wright.hss.cmu.edu hostname
ssh-add -l
ssh-add /Users/shannon/.ssh/id_ed25519
```

Those fail or hang because Codex cannot handle the interactive auth prompt usefully. Do not ask the
user to type a passphrase "here"; ask them to run the ControlMaster command above in a regular
Terminal. After the socket is open, run the tiny real-model smoke below.

Current Wright checkout and env from 2026-07-06:

```text
repo: /home/export/sgallagh/WeightTraits
env: /home/export/sgallagh/.conda/envs/weighttraits
conda/mamba: /opt/miniforge3/bin/{conda,mamba}
cache used for tiny smoke: /home/export/sgallagh/.cache/WeightTraits
```

`/home/export/sgallagh/scratch` points to `/mnt/scratch`, but `/mnt/scratch` was not usable on the
headnode session. Use `/home/export/sgallagh/.cache/WeightTraits` for tiny smoke caches until a real
scratch path is confirmed.

Tiny full smoke on Wright or another prepared environment:

```bash
cd /home/export/sgallagh/WeightTraits
git pull
/opt/miniforge3/bin/mamba run -n weighttraits env PYTHONPATH=src HF_DATASETS_CACHE=$HOME/.cache/WeightTraits/hf_datasets \
  python -m weighttraits.cli make-training-run-list \
    --manifest examples/training/tiny_manifest.jsonl \
    --config examples/training/tiny_full_smoke.yaml \
    --registry examples/training/tiny_dataset_registry.yaml \
    --formats examples/training/tiny_dataset_formats.yaml \
    --out /tmp/weighttraits_tiny_full_runs.jsonl \
    --allow-existing-artifacts \
    --max-train-samples 2 \
    --allow-missing-eval

/opt/miniforge3/bin/mamba run -n weighttraits env PYTHONPATH=src HF_DATASETS_CACHE=$HOME/.cache/WeightTraits/hf_datasets \
  python -m weighttraits.cli run-training-row \
    --run-list /tmp/weighttraits_tiny_full_runs.jsonl \
    --index 0
```

This passed on 2026-07-06:

```text
status=completed, step=1, train_loss=7.007139682769775, eval_loss=7.006280422210693
artifacts: outputs/tiny_full_smoke/n0/model, outputs/tiny_full_smoke/n0/training_log.jsonl
ledger summary: completed_nodes=["n0"], failed_nodes=[], status_counts={"completed": 1}
```

Tiny LoRA smoke:

```bash
/opt/miniforge3/bin/mamba run -n weighttraits env PYTHONPATH=src HF_DATASETS_CACHE=$HOME/.cache/WeightTraits/hf_datasets \
  python -m weighttraits.cli make-training-run-list \
    --manifest examples/training/tiny_manifest.jsonl \
    --config examples/training/tiny_lora_smoke.yaml \
    --registry examples/training/tiny_dataset_registry.yaml \
    --formats examples/training/tiny_dataset_formats.yaml \
    --out /tmp/weighttraits_tiny_lora_runs.jsonl \
    --allow-existing-artifacts \
    --max-train-samples 2 \
    --allow-missing-eval

/opt/miniforge3/bin/mamba run -n weighttraits env PYTHONPATH=src HF_DATASETS_CACHE=$HOME/.cache/WeightTraits/hf_datasets \
  python -m weighttraits.cli run-training-row \
    --run-list /tmp/weighttraits_tiny_lora_runs.jsonl \
    --index 0
```

This passed on 2026-07-06:

```text
status=completed, step=1, train_loss=7.007139682769775, eval_loss=7.006478786468506
artifacts: outputs/tiny_lora_smoke/n0/adapter, outputs/tiny_lora_smoke/n0/merged,
           outputs/tiny_lora_smoke/n0/training_log.jsonl
ledger summary: completed_nodes=["n0"], failed_nodes=[], status_counts={"completed": 1}
```

Tiny two-node lineage smoke:

```bash
/opt/miniforge3/bin/mamba run -n weighttraits env PYTHONPATH=src HF_DATASETS_CACHE=$HOME/.cache/WeightTraits/hf_datasets \
  python -m weighttraits.cli make-training-run-list \
    --manifest examples/training/tiny_lineage_manifest.jsonl \
    --config examples/training/tiny_full_lineage_smoke.yaml \
    --registry examples/training/tiny_dataset_registry.yaml \
    --formats examples/training/tiny_dataset_formats.yaml \
    --out /tmp/weighttraits_tiny_full_lineage_runs.jsonl \
    --allow-existing-artifacts \
    --max-train-samples 2 \
    --allow-missing-eval

/opt/miniforge3/bin/mamba run -n weighttraits env PYTHONPATH=src HF_DATASETS_CACHE=$HOME/.cache/WeightTraits/hf_datasets \
  python -m weighttraits.cli run-training-row \
    --run-list /tmp/weighttraits_tiny_full_lineage_runs.jsonl \
    --index 0

/opt/miniforge3/bin/mamba run -n weighttraits env PYTHONPATH=src HF_DATASETS_CACHE=$HOME/.cache/WeightTraits/hf_datasets \
  python -m weighttraits.cli run-training-row \
    --run-list /tmp/weighttraits_tiny_full_lineage_runs.jsonl \
    --index 1
```

Full lineage passed on 2026-07-06:

```text
n1 init_from: outputs/tiny_full_lineage_smoke/n0/model
ledger summary: completed_nodes=["n0", "n1"], failed_nodes=[], status_counts={"completed": 2}
artifacts: outputs/tiny_full_lineage_smoke/n0/model, outputs/tiny_full_lineage_smoke/n1/model
```

The LoRA lineage run list uses:

```text
n1 init_from: outputs/tiny_lora_lineage_smoke/n0/merged
```

and both LoRA rows passed on 2026-07-06:

```text
ledger summary: completed_nodes=["n0", "n1"], failed_nodes=[], status_counts={"completed": 2}
artifacts: outputs/tiny_lora_lineage_smoke/n0/adapter, outputs/tiny_lora_lineage_smoke/n0/merged,
           outputs/tiny_lora_lineage_smoke/n1/adapter, outputs/tiny_lora_lineage_smoke/n1/merged
```

Tiny artifact distance plumbing smoke on Wright:

```bash
/opt/miniforge3/bin/mamba run -n weighttraits env PYTHONPATH=src \
  python -m weighttraits.cli build-distance-cube \
    --checkpoint-manifest examples/distance_inputs/tiny_full_lineage_outputs.yaml \
    --metric cosine \
    --metric l2 \
    --out outputs/tiny_full_lineage_smoke/distance_cube

/opt/miniforge3/bin/mamba run -n weighttraits env PYTHONPATH=src \
  python -m weighttraits.cli reconstruct-tree \
    --cube outputs/tiny_full_lineage_smoke/distance_cube \
    --metric l2 \
    --out outputs/tiny_full_lineage_smoke/distance_cube/tree_l2.newick \
    --audit-out outputs/tiny_full_lineage_smoke/distance_cube/tree_l2.audit.json

/opt/miniforge3/bin/mamba run -n weighttraits env PYTHONPATH=src \
  python -m weighttraits.cli score-tree \
    --truth-manifest examples/recovery/tiny_two_tip_smoke_truth_manifest.jsonl \
    --estimate-newick outputs/tiny_full_lineage_smoke/distance_cube/tree_l2.newick \
    --out outputs/tiny_full_lineage_smoke/distance_cube/score_l2.json
```

This passed on 2026-07-06. It is a command-path smoke only: the two-tip truth has no informative
splits, and the real training lineage is a chain with terminal leaf `n1`.

```text
full checkpoints: n_models=2, n_layers=110, l2 distance_mean=0.0009195269418413603
LoRA merged checkpoints: n_models=2, n_layers=110, l2 distance_mean=7.131354745703268e-05
LoRA cumulative adapters: n_models=2, n_layers=30, l2 distance_mean=0.00026147472545753084
two-tip score: n_truth_splits=0, n_estimate_splits=0, rf=0, exact_tree_recovery=true
```

Tiny branching smoke on Wright:

```text
manifest: examples/training/tiny_branching_manifest.jsonl
leaves: ["n2", "n3", "n4", "n5"]
truth splits: 1
full run list: 6 runs, 0 warnings, 0 errors
LoRA run list: 6 runs, 0 warnings, 0 errors
full training: completed_nodes=["n0", "n1", "n2", "n3", "n4", "n5"], failed_nodes=[]
LoRA training: completed_nodes=["n0", "n1", "n2", "n3", "n4", "n5"], failed_nodes=[]
```

The four-leaf distance/reconstruct/score path also passed:

```text
full leaf checkpoints: n_models=4, n_layers=110, l2 distance_mean=0.0,
  score rf=0, false_negative=0, false_positive=0, exact_tree_recovery=true
LoRA merged leaf checkpoints: n_models=4, n_layers=110, l2 distance_mean=0.00011425835655734538,
  score rf=0, false_negative=0, false_positive=0, exact_tree_recovery=true
LoRA cumulative leaf adapters: n_models=4, n_layers=30, l2 distance_mean=0.00041895296848211915,
  score rf=0, false_negative=0, false_positive=0, exact_tree_recovery=true
```

Ledger-derived distance manifest smoke:

```bash
/opt/miniforge3/bin/mamba run -n weighttraits env PYTHONPATH=src \
  python -m weighttraits.cli make-distance-input-manifest \
    --ledger outputs/tiny_lora_branching_smoke/training_ledger.jsonl \
    --truth-manifest examples/training/tiny_branching_manifest.jsonl \
    --artifact adapter_chain \
    --out outputs/tiny_lora_branching_smoke/generated_cumulative_leaf_inputs.yaml
```

Generated:

```yaml
models:
- model_id: n2
  adapter_chain:
  - n0/adapter
  - n2/adapter
- model_id: n3
  adapter_chain:
  - n0/adapter
  - n3/adapter
- model_id: n4
  adapter_chain:
  - n1/adapter
  - n4/adapter
- model_id: n5
  adapter_chain:
  - n1/adapter
  - n5/adapter
```

Building, reconstructing, and scoring from that generated manifest passed with:

```text
n_models=4, n_layers=30, l2 distance_mean=0.00041895296848211915
score rf=0, false_negative=0, false_positive=0, exact_tree_recovery=true
```

## Next Best Steps

1. Run the tiny branching contrast smoke on Wright and compare full checkpoints, LoRA merged
   checkpoints, and cumulative LoRA adapter-chain recovery against the ordinary branching smoke.
2. Add a low-rank LoRA distance accumulator that avoids dense `B @ A` slabs for large adapters.
3. Refine supervision templates:
   - prefer explicit `trainer.target_field` or `trainer.target_template`;
   - audit old prompt templates that currently include the answer in the rendered prompt.

## Important Caveats

- Do not commit generated outputs, model weights, checkpoints, caches, or reports unless they are intentional tiny examples.
- CKA is exact but still tensor-at-a-time.
- `wt reconstruct-tree` uses Biopython from the analysis extra.
- LoRA vector metrics stream dense `B @ A` row blocks; low-rank dot-product acceleration remains planned.
- The trainer execution loop has now been exercised on Wright with the tiny model
  `hf-internal-testing/tiny-random-t5` for single-row, two-node lineage, and six-node branching
  full/LoRA runs.
- Local outgoing Hugging Face traffic is disabled; run real model smoke/training on Wright or another prepared environment.
- Local `datasets.load_dataset("json", ...)` may need `HF_DATASETS_CACHE` pointed to a writable scratch directory.
- `wt audit-datasets` in load mode may require network access and the optional `datasets` dependency.
- `wt audit-training-samples` requires dataset loading and should run only in environments where downloads/cache access are intended.
- `wt make-training-run-list --runner-dry-run --slurm-out` generates a safe selector script; omit `--runner-dry-run` only when real training is intended.
