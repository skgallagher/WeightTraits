# Legacy Causal Llama LoRA Rebuild

The two corrected Llama-3.2-1B QKV LoRA cohorts use the same causal-LM data,
prompt, padding, sampling, batching, and exact-2,000-step contract as
`llama32_1b_full_finetune_legacy_causal_2000.yaml`. Their small overlays change
only the method identity, output lineage, learning rate, and adapter definition:

| Config | Rank | Alpha | Target modules | Learning rate |
| --- | ---: | ---: | --- | ---: |
| `llama32_1b_lora_qkv_r8_legacy_causal_2000.yaml` | 8 | 16 | `q_proj`, `k_proj`, `v_proj` | `3e-4` |
| `llama32_1b_lora_qkv_r64_legacy_causal_2000.yaml` | 64 | 128 | `q_proj`, `k_proj`, `v_proj` | `3e-4` |

Both adapters are merged after each node so every child starts from its actual
parent's cumulative full-weight model. The executor requires the parent's exact
planned row, terminal completion receipt, and merged-artifact SHA-256 before a
child loads it. The resulting output roots are new and must not be mixed with the
superseded `llama32_1b_lora_qkv_r8` or `llama32_1b_lora_qkv_r64` artifacts.
Both adapter cohorts inherit PEFT `0.19.1`, Tokenizers `0.22.2`, and the complete
60-file WeightTraits source receipt from the full-fine-tuning config. A root fails
when those executable bytes differ, and a child also requires the parent's source
receipt to match its own.

From the repository root, regenerate validation reports and executable run-list
sets for each rank with:

```bash
for rank in 8 64; do
  stem="llama32_1b_lora_qkv_r${rank}_legacy_causal_2000"
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

Each result must report 50 valid trees, 641 node rows, and zero errors or
warnings. Production submission uses the same sequential-tree Slurm wrapper as
the full-fine-tuning cohort, with the rank-specific run-list template and a
distinct run-name prefix. Keep `OVERRIDE_MAX_STEPS` unset.
