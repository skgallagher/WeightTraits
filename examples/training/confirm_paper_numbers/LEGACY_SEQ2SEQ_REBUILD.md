# Legacy Seq2seq Flan-T5 Rebuild

This document freezes the effective ELLMTrees Flan-T5 training contract used for
the corrected ordinary-cohort rebuild. It applies to the same 50 trees and 641
node assignments as `assignment_summary.json`. These are independent model
cohorts; they are not paired-tree statistical analyses.

## Shared protocol

Every cohort extends `flan_t5_legacy_seq2seq_2000_base.yaml` and therefore shares:

- `google/flan-t5-base` at revision
  `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2`;
- encoder source length 512 and decoder target/generation length 128;
- right padding to a multiple of 8, with padded decoder labels set to `-100` by
  the seq2seq collator;
- BF16 training, seed 42, exactly 2,000 required steps, batch size 8, gradient
  accumulation 4, learning rate `3e-4`, 200 warmup steps, weight decay `0.01`,
  max gradient norm `1.0`, evaluation every 250 steps, and no early stopping or
  trainer checkpoints;
- the prompts declared in the base config, including language-specific translation
  prefixes; and
- the shared `legacy_subsample` cache contract: seed 42, train cap 10,000, eval
  cap 1,000, using `dataset_registry_legacy_causal.yaml` and
  `dataset_formats_legacy_causal.yaml`.

Full fine-tuning enables gradient checkpointing. All LoRA cohorts disable it, use
rank 8, alpha 16, dropout 0.05, and merge the trained adapter before a child node
loads its parent artifact.

## Cohort overlays

| Cohort | Config | Exact adapter scope | Expected modules |
| --- | --- | --- | ---: |
| Full fine-tuning | `flan_t5_full_finetune_legacy_seq2seq_2000.yaml` | n/a | n/a |
| LoRA baseline (`lora_finetune`) | `flan_t5_lora_qv_r8_legacy_seq2seq_2000.yaml` | `q`, `v` | 72 |
| LoRA key only | `flan_t5_lora_k_r8_legacy_seq2seq_2000.yaml` | `k` | 36 |
| LoRA QKV | `flan_t5_lora_qkv_r8_legacy_seq2seq_2000.yaml` | `q`, `k`, `v` | 108 |
| LoRA full attention | `flan_t5_lora_qkvo_r8_legacy_seq2seq_2000.yaml` | `q`, `k`, `v`, `o` | 144 |
| Legacy full-FT approximation | `flan_t5_lora_full_ft_approx_actual_r8_legacy_seq2seq_2000.yaml` | `q`, `k`, `v`, `o`, `wo` | 168 |
| Corrected all projections | `flan_t5_lora_all_projections_corrected_r8_legacy_seq2seq_2000.yaml` | `q`, `k`, `v`, `o`, `wi_0`, `wi_1`, `wo` | 216 |

The legacy full-FT-approximation declaration named `wi`, but PEFT did not match
that name to Flan-T5's gated `wi_0` and `wi_1` matrices. Its executable scope was
therefore `q,k,v,o,wo`, which the `actual` overlay reproduces. The corrected
all-projection cohort is an explicit contrast and must remain separate.

## Validate and regenerate

Run from the WeightTraits repository root after building the shared legacy cache
described in [LEGACY_CAUSAL_REBUILD.md](LEGACY_CAUSAL_REBUILD.md). Regenerate all
derived run lists whenever a config, registry, format, assignment, or cache
contract changes:

```bash
stems=(
  flan_t5_full_finetune_legacy_seq2seq_2000
  flan_t5_lora_qv_r8_legacy_seq2seq_2000
  flan_t5_lora_k_r8_legacy_seq2seq_2000
  flan_t5_lora_qkv_r8_legacy_seq2seq_2000
  flan_t5_lora_qkvo_r8_legacy_seq2seq_2000
  flan_t5_lora_full_ft_approx_actual_r8_legacy_seq2seq_2000
  flan_t5_lora_all_projections_corrected_r8_legacy_seq2seq_2000
)

for stem in "${stems[@]}"; do
  PYTHONPATH=src python -m weighttraits.cli validate-training-data-set \
    --assignment-summary examples/training/confirm_paper_numbers/assignment_summary.json \
    --config "examples/training/confirm_paper_numbers/${stem}.yaml" \
    --formats examples/training/confirm_paper_numbers/dataset_formats_legacy_causal.yaml \
    --out "examples/training/confirm_paper_numbers/${stem}_data_format_validation.json"

  PYTHONPATH=src python -m weighttraits.cli make-training-run-list-set \
    --assignment-summary examples/training/confirm_paper_numbers/assignment_summary.json \
    --config "examples/training/confirm_paper_numbers/${stem}.yaml" \
    --out-dir "examples/training/confirm_paper_numbers/${stem}_training_runlists" \
    --summary-out "examples/training/confirm_paper_numbers/${stem}_training_run_list_summary.json" \
    --registry examples/training/confirm_paper_numbers/dataset_registry_legacy_causal.yaml \
    --formats examples/training/confirm_paper_numbers/dataset_formats_legacy_causal.yaml \
    --data-cache-root data/confirm_paper_numbers/legacy_causal_seed42_cache \
    --require-data-cache \
    --expected-cache-strategy legacy_subsample \
    --expected-cache-seed 42 \
    --expected-cache-train-limit 10000 \
    --expected-cache-eval-limit 1000 \
    --max-train-samples 10000 \
    --max-eval-samples 1000
done
```

Each validation must cover 641/641 jobs. Each run-list summary must report
`valid: true`, 50 trees, 641 runs, and zero errors and warnings. Inspect at least
one generated row per cohort and confirm the protocol ID, config/source hashes,
cache recipe, padding contract, exact-step requirement, output root, and—for
LoRA—the exact target-module list.

## Smoke and production launch

Before submitting any broad array, run one root and one non-root row for every
distinct adapter scope in an isolated smoke output root. The smoke must verify
source/target padding, decoder label padding of `-100`, exactly 2,000 steps in the
production row, resolved adapter-module count, parent loading, merge behavior, and
runtime/config fingerprints. A two-step diagnostic may use
`--override-max-steps`; production must leave `OVERRIDE_MAX_STEPS` unset.

After its smoke passes, submit the cohort through
`scripts/slurm/confirm_paper_tree_sequential.sbatch` with tree indices `1-50`, the
cohort-specific run-list template, a distinct run-name prefix, and an explicitly
chosen concurrency throttle. Never reuse a superseded output root or mix nodes
from different protocol IDs.
